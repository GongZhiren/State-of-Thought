"""Fit the lightweight SoT controller from prepared offline trajectories.

This module intentionally trains only the 582 controller parameters. Backbone
weights stay frozen and are never passed to the optimizer.
"""
from __future__ import annotations

import gzip
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
import yaml

from .artifacts import load_config
from .checkpoint import save_checkpoint
from .config import MoTConfig
from .selector import build_read_feature_vector


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source = Path(path)
    handle = (
        gzip.open(source, mode="rt", encoding="utf-8")
        if source.suffix == ".gz"
        else source.open(encoding="utf-8")
    )
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            state = row.get("m_t")
            if not isinstance(state, list) or len(state) < 4:
                raise ValueError(f"line {line_number}: m_t must contain at least four values")
            rows.append(row)
    return rows


def _selector_features(state: list[float], history: dict[str, Any]) -> list[float]:
    prior = history.get("m_j") or [0.0] * 4
    features = build_read_feature_vector(
        m_t=torch.tensor(state[:4], dtype=torch.float32),
        m_j=torch.tensor([float(value) for value in prior[:4]], dtype=torch.float32),
        z_similarity=0.0,
        age_inv=0.0,
        age_ratio=0.0,
        in_recent_window=0.0,
        read_feature_mode="geom5d",
    )
    return [float(value) for value in features.tolist()]


def _trajectory_stats(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_problem: dict[str, dict[str, Any]] = {}
    for row in records:
        problem_id = str(row.get("problem_id", ""))
        if not problem_id:
            continue
        current = by_problem.setdefault(
            problem_id,
            {
                "traj_total_tokens": 0.0,
                "traj_steps": 0.0,
                "traj_correct": 0.0,
                "traj_quality": 0.0,
                "dataset": "",
                "task_type": "",
            },
        )
        current["traj_total_tokens"] += float(row.get("step_len_tokens", 0.0))
        current["traj_steps"] += 1.0
        current["traj_correct"] = 1.0 if bool(row.get("traj_correct", False)) else current["traj_correct"]
        quality = float(
            row.get("traj_quality", row.get("quality_score", 1.0 if row.get("traj_correct") else 0.0))
        )
        current["traj_quality"] = max(float(current["traj_quality"]), max(0.0, min(1.0, quality)))
        current["dataset"] = str(row.get("dataset", ""))
        current["task_type"] = str(row.get("task_type", ""))
    references: dict[str, float] = {}
    for stats in by_problem.values():
        if float(stats["traj_quality"]) < 0.8 or float(stats["traj_total_tokens"]) <= 0:
            continue
        task_type = str(stats.get("task_type") or "unknown")
        references[task_type] = min(
            float(stats["traj_total_tokens"]), references.get(task_type, float("inf"))
        )
    for stats in by_problem.values():
        task_type = str(stats.get("task_type") or "unknown")
        stats["reference_tokens"] = float(
            references.get(task_type, max(float(stats["traj_total_tokens"]), 1.0))
        )
    return by_problem


def _read_examples(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = list(records)
    trajectory_stats = _trajectory_stats(rows)
    examples: list[dict[str, Any]] = []
    for row in rows:
        state = [float(value) for value in row["m_t"][:4]]
        stats = trajectory_stats.get(str(row.get("problem_id", "")), {})
        for history in row.get("history") or []:
            if not isinstance(history, dict) or int(history.get("retrieval_trainable", 1)) <= 0:
                continue
            hard = int(float(history.get("gate_hard_label", history.get("keep_label", 0))) > 0)
            soft = max(0.0, min(1.0, float(history.get("gate_soft_label", hard))))
            examples.append(
                {
                    "x": _selector_features(state, history),
                    "y_hard": hard,
                    "y_soft": soft,
                    "split": str(row.get("split", "train")),
                    "step_len_tokens": int(row.get("step_len_tokens", 0)),
                    "traj_quality": float(
                        row.get("traj_quality", row.get("quality_score", bool(row.get("traj_correct"))))
                    ),
                    "traj_total_tokens": float(stats.get("traj_total_tokens", 0.0)),
                    "reference_tokens": float(stats.get("reference_tokens", 0.0)),
                }
            )
    return examples


def _stop_examples(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for row in records:
        hard = int(float(row.get("stop_label", 0)) > 0)
        # The frozen training logs contain a legacy fifth slot that is always
        # zero. Retaining it during optimization reproduces the historical L2
        # normalization exactly; it is asserted inert and removed before the
        # public 4-D checkpoint is written.
        state = [float(value) for value in row["m_t"][:5]]
        if len(state) == 4:
            state.append(0.0)
        examples.append(
            {
                "x": state,
                "y_hard": hard,
                "y_soft": max(0.0, min(1.0, float(row.get("teacher_stop_score", hard)))),
                "split": str(row.get("split", "train")),
            }
        )
    return examples


def _bundle(examples: list[dict[str, Any]], split: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    selected = [example for example in examples if example["split"] == split]
    dimension = len(examples[0]["x"]) if examples else 1
    if not selected:
        return (
            torch.zeros((0, dimension), dtype=torch.float32),
            torch.zeros((0, 1), dtype=torch.float32),
            torch.zeros((0, 1), dtype=torch.float32),
            [],
        )
    return (
        torch.tensor([example["x"] for example in selected], dtype=torch.float32),
        torch.tensor([[float(example["y_hard"])] for example in selected], dtype=torch.float32),
        torch.tensor([[float(example["y_soft"])] for example in selected], dtype=torch.float32),
        selected,
    )


def _fit_stop(examples: list[dict[str, Any]], cfg: MoTConfig) -> tuple[list[float], float]:
    features, hard, soft, _ = _bundle(examples, "train")
    if features.shape[0] == 0:
        raise ValueError("no stop-head training examples")
    positive = float(soft.sum().item())
    negative = float(soft.shape[0] - positive)
    if positive <= 1e-4 or negative <= 1e-4:
        probability = max(1e-4, min(0.9999, float(soft.mean().item())))
        return [0.0] * int(features.shape[1]), float(torch.logit(torch.tensor(probability)).item())

    mean = features.mean(dim=0, keepdim=True)
    std = features.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    normalized = (features - mean) / std
    weight = torch.zeros((features.shape[1], 1), dtype=torch.float32, requires_grad=True)
    bias = torch.zeros((1,), dtype=torch.float32, requires_grad=True)
    positive_weight = min(
        max(negative / max(positive, 1.0), 1.0), float(cfg.training.fit_pos_weight_cap)
    )
    optimizer = torch.optim.Adam([weight, bias], lr=0.05)
    for _ in range(400):
        logits = normalized @ weight + bias
        sample_weight = torch.where(soft > 0.5, positive_weight, 1.0)
        loss = (F.binary_cross_entropy_with_logits(logits, soft, reduction="none") * sample_weight).mean()
        loss = loss + 1e-4 * weight.pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    original_weight = weight.detach().reshape(-1) / std.reshape(-1)
    original_bias = float(bias.detach().item()) - float((mean.reshape(-1) * original_weight).sum().item())
    fitted = [float(value) for value in original_weight.tolist()]
    if len(fitted) == 5:
        if abs(fitted[-1]) > 1e-10:
            raise ValueError("legacy fifth stop coefficient is not inert")
        fitted = fitted[:4]
    return fitted, original_bias


def _fit_gate(examples: list[dict[str, Any]], cfg: MoTConfig) -> dict[str, Any]:
    features, hard, soft, metadata = _bundle(examples, "train")
    if features.shape[0] == 0:
        # Some frozen VLM protocols train only the stopping head. Their
        # history rows remain in the shared schema with
        # retrieval_trainable=0 keeps the evidence gate fixed by protocol.
        hidden_dim = max(4, int(cfg.selector.gate_hidden_dim))
        feature_dim = 16
        return {
            "gate_hidden_dim": hidden_dim,
            "gate_w1": [0.0] * (feature_dim * hidden_dim),
            "gate_b1": [0.0] * hidden_dim,
            "gate_w2": [0.0] * hidden_dim,
            "gate_b2": 0.0,
        }
    targets = soft.reshape(-1, 1)
    hidden_dim = max(4, int(cfg.selector.gate_hidden_dim))
    dense_penalty = max(0.0, float(cfg.training.gate_dense_penalty))
    l2_penalty = max(0.0, float(cfg.training.gate_l2_penalty))
    seed = int(cfg.training.split_seed)
    generator = torch.Generator().manual_seed(seed)
    fan_in, fan_out = int(features.shape[1]), hidden_dim
    scale = (2.0 / max(fan_in + fan_out, 1)) ** 0.5

    # The matrix layout matches the frozen paper artifact exactly: training
    # uses [feature, hidden], and the flattened list is the compatibility wire
    # format consumed by the released runtime.
    w1 = (torch.randn((fan_in, hidden_dim), generator=generator) * scale).requires_grad_(True)
    b1 = torch.zeros((hidden_dim,), dtype=torch.float32, requires_grad=True)
    w2 = (torch.randn((hidden_dim, 1), generator=generator) * scale).requires_grad_(True)
    b2 = torch.zeros((1,), dtype=torch.float32, requires_grad=True)
    value_weight = (torch.randn((fan_in, 1), generator=generator) * 0.05).requires_grad_(True)
    value_bias = torch.zeros((1,), dtype=torch.float32, requires_grad=True)

    optimizer = torch.optim.Adam([w1, b1, w2, b2], lr=float(cfg.training.warmup_lr))
    for _ in range(max(1, int(cfg.training.warmup_epochs))):
        hidden = torch.tanh(features @ w1 + b1)
        logits = hidden @ w2 + b2
        probabilities = torch.sigmoid(logits)
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        loss = loss + dense_penalty * probabilities.mean()
        loss = loss + l2_penalty * (w1.pow(2).mean() + w2.pow(2).mean())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    refinement_steps = max(0, int(cfg.training.onpolicy_refine_steps))
    batch_size = max(16, int(cfg.training.onpolicy_batch_size))
    if refinement_steps:
        optimizer_refine = torch.optim.Adam(
            [w1, b1, w2, b2, value_weight, value_bias], lr=float(cfg.training.onpolicy_lr)
        )
        # Keep the column-vector auxiliary tensors used by the frozen
        # training implementation.
        step_lengths = torch.tensor(
            [[float(meta.get("step_len_tokens", 0.0))] for meta in metadata], dtype=torch.float32
        )
        step_lengths = step_lengths / max(float(step_lengths.max().item()), 1.0)
        trajectory_quality = torch.tensor(
            [[float(meta.get("traj_quality", 0.0))] for meta in metadata], dtype=torch.float32
        ).clamp(0.0, 1.0)
        trajectory_tokens = torch.tensor(
            [[max(float(meta.get("traj_total_tokens", 0.0)), 1.0)] for meta in metadata],
            dtype=torch.float32,
        )
        reference_tokens = torch.tensor(
            [[max(float(meta.get("reference_tokens", 0.0)), 1.0)] for meta in metadata],
            dtype=torch.float32,
        )
        relative_efficiency = (reference_tokens / trajectory_tokens).clamp(0.0, 2.0)
        for _ in range(refinement_steps):
            indices = torch.randint(
                0,
                features.shape[0],
                (min(batch_size, features.shape[0]),),
                dtype=torch.long,
            )
            batch = features[indices]
            target = targets[indices]
            hidden = torch.tanh(batch @ w1 + b1)
            logits = hidden @ w2 + b2
            probabilities = torch.sigmoid(logits).clamp(1e-6, 0.999999)
            distribution = torch.distributions.Bernoulli(probs=probabilities)
            action = distribution.sample()
            local_signal = 2.0 * target - 1.0
            trajectory_signal = 2.0 * trajectory_quality[indices] - 1.0
            # Historical objective: positive when the trajectory uses more
            # tokens than its high-quality task reference, negative when it
            # is shorter.  Preserve the frozen sign convention exactly.
            efficiency = relative_efficiency[indices] - 1.0
            reward = (
                0.55 * local_signal
                + 0.35 * trajectory_signal
                + 0.10 * efficiency
                - 0.02 * action
                - 0.01 * step_lengths[indices]
            )
            value = batch @ value_weight + value_bias
            advantage = reward - value.detach()
            policy_loss = -(advantage * distribution.log_prob(action)).mean()
            policy_loss = policy_loss - float(cfg.training.onpolicy_entropy_coef) * distribution.entropy().mean()
            value_loss = F.mse_loss(value, reward)
            regularizer = l2_penalty * (w1.pow(2).mean() + w2.pow(2).mean())
            loss = policy_loss + float(cfg.training.onpolicy_value_coef) * value_loss + regularizer
            optimizer_refine.zero_grad()
            loss.backward()
            if float(cfg.training.onpolicy_clip_grad) > 0:
                torch.nn.utils.clip_grad_norm_(
                    [w1, b1, w2, b2, value_weight, value_bias],
                    max_norm=float(cfg.training.onpolicy_clip_grad),
                )
            optimizer_refine.step()

    return {
        "gate_hidden_dim": hidden_dim,
        "gate_w1": [float(value) for value in w1.detach().reshape(-1).tolist()],
        "gate_b1": [float(value) for value in b1.detach().tolist()],
        "gate_w2": [float(value) for value in w2.detach().reshape(-1).tolist()],
        "gate_b2": float(b2.detach().item()),
    }


def fit_controller(
    records: list[dict[str, Any]],
    cfg: MoTConfig,
    output_path: str | Path,
) -> dict[str, Any]:
    torch.manual_seed(int(cfg.training.split_seed))
    read_examples = _read_examples(records)
    stop_examples = _stop_examples(records)
    gate = _fit_gate(read_examples, cfg)
    stop_weight, stop_bias = _fit_stop(stop_examples, cfg)
    artifact = {
        "config": asdict(cfg),
        "config_snapshot": {
            "selector": {
                "retrieval_mode": "evidence_gate",
                "read_feature_mode": "geom5d",
                "gate_prob_threshold": float(cfg.selector.gate_prob_threshold),
                "read_threshold": float(cfg.selector.read_threshold),
                "attention_mix": float(cfg.selector.attention_mix),
                "attention_temperature": float(cfg.selector.attention_temperature),
            },
            "commit": {
                "stop_threshold": float(cfg.commit.stop_threshold),
                "r_stop": int(cfg.commit.r_stop),
            },
        },
        "selector": {
            "read_weight": [0.0] * 16,
            "read_bias": 0.0,
            "read_threshold": float(cfg.selector.read_threshold),
            "retrieval_mode": "evidence_gate",
            "gate_prob_threshold": float(cfg.selector.gate_prob_threshold),
            "read_feature_mode": "geom5d",
            "lambda_token": float(cfg.selector.lambda_token),
            **gate,
            "attention_mix": float(cfg.selector.attention_mix),
            "attention_temperature": float(cfg.selector.attention_temperature),
        },
        "commit": {
            "stop_weight": stop_weight,
            "stop_bias": stop_bias,
            "stop_threshold": float(cfg.commit.stop_threshold),
            "r_stop": int(cfg.commit.r_stop),
        },
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return artifact


def train_from_config(config_path: str | Path) -> Path:
    source = Path(config_path).resolve()
    root = source.parents[2]
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    training = raw.get("training") or {}
    template = (root / str(training["template_checkpoint"])).resolve()
    records = (root / str(training["offline_records"])).resolve()
    output = (root / str(training.get("output_artifact", "outputs/training/controller.json"))).resolve()
    fit_controller(read_jsonl(records), load_config(template), output)
    output_checkpoint = (
        root / str(training.get("output_checkpoint", output.with_suffix(".pt").relative_to(root)))
    ).resolve()
    release_id = f"{raw.get('release_id', 'sot')}-refit"
    save_checkpoint(output, output_checkpoint, release_id=release_id)
    return output_checkpoint
