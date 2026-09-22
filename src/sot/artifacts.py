"""Load either the human-readable JSON artifact or public tensor checkpoint."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .checkpoint import ControllerCheckpoint, load_checkpoint
from .config import InferenceConfig, MoTConfig, load_mot_config_dict


def _merged_raw(
    *,
    config: Mapping[str, Any],
    selector: Mapping[str, Any],
    commit: Mapping[str, Any],
) -> dict[str, Any]:
    defaults = InferenceConfig()
    debt = dict(config.get("debt") or {})
    norm = dict(config.get("normalization") or {})
    context = dict(config.get("context") or {})
    inference = dict(config.get("inference") or {})
    training = dict(config.get("training") or {})
    calibration_protocol = dict(config.get("calibration_protocol") or {})
    answer_cap = int(
        inference.get(
            "answer_max_new_tokens",
            inference.get("answer_commit_max_new_tokens", defaults.answer_commit_max_new_tokens),
        )
    )
    return {
        "calibration": {
            "normalization": norm,
            "protocol": calibration_protocol,
        },
        "termination": {
            "tau_v": float(debt.get("tau_v", 0.05)),
            "t_min": int(commit.get("t_min", 0)),
            "r_stop": int(commit.get("r_stop", 1)),
        },
        "mot": {
            "debt_alpha": float(debt.get("alpha", 0.9)),
            "lambda_v": float(debt.get("lambda_v", 1.0)),
            "lambda_c": float(debt.get("lambda_c", 1.0)),
            "lambda_H": float(debt.get("lambda_H", 1.0)),
            "e_max": float(debt.get("e_max", 5.0)),
            "sentence_char_boundary_min": int(
                inference.get("sentence_char_boundary_min", defaults.sentence_char_boundary_min)
            ),
            "fallback_step_token_cap": int(
                inference.get("fallback_step_token_cap", defaults.fallback_step_token_cap)
            ),
            "degenerate_streak_limit": int(
                inference.get("degenerate_streak_limit", defaults.degenerate_streak_limit)
            ),
            "temperature": float(inference.get("temperature", defaults.temperature)),
            "context_budget_tokens": int(context.get("context_budget_tokens", 1536)),
        },
        "inference": {
            "T_max": int(inference.get("t_max_steps", inference.get("T_max", defaults.t_max_steps))),
            "max_sentence_tokens": int(
                inference.get("max_sentence_tokens", defaults.max_sentence_tokens)
            ),
            "min_sentence_tokens": int(
                inference.get("min_sentence_tokens", defaults.min_sentence_tokens)
            ),
            "answer_commit_max_new_tokens": answer_cap,
            "observe_max_input_tokens": int(
                inference.get("observe_max_input_tokens", defaults.observe_max_input_tokens)
            ),
            "degenerate_streak_limit": int(
                inference.get("degenerate_streak_limit", defaults.degenerate_streak_limit)
            ),
        },
        "sot": {
            "selector": dict(selector),
            "commit": dict(commit),
            "context": context,
            "training": training,
        },
    }


def config_from_json_artifact(path: str | Path) -> MoTConfig:
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    config = dict(artifact.get("config") or {})
    selector = dict(config.get("selector") or {})
    selector.update(artifact.get("selector") or {})
    commit = dict(config.get("commit") or {})
    commit.update(artifact.get("commit") or {})
    return load_mot_config_dict(_merged_raw(config=config, selector=selector, commit=commit))


def config_from_checkpoint(checkpoint: ControllerCheckpoint) -> MoTConfig:
    runtime = checkpoint.runtime_config
    selector = dict(runtime.get("selector") or {})
    selector.update(
        {
            "gate_w1": checkpoint.state_dict["selector.gate_w1"].reshape(-1).tolist(),
            "gate_b1": checkpoint.state_dict["selector.gate_b1"].tolist(),
            "gate_w2": checkpoint.state_dict["selector.gate_w2"].tolist(),
            "gate_b2": float(selector.get("gate_b2", checkpoint.state_dict["selector.gate_b2"].item())),
        }
    )
    commit = dict(runtime.get("commit") or {})
    commit.update(
        {
            "stop_weight": checkpoint.state_dict["commit.stop_weight"].tolist(),
            "stop_bias": float(commit.get("stop_bias", checkpoint.state_dict["commit.stop_bias"].item())),
        }
    )
    return load_mot_config_dict(_merged_raw(config=runtime, selector=selector, commit=commit))


def load_config(path: str | Path) -> MoTConfig:
    source = Path(path)
    if source.suffix.lower() == ".pt":
        return config_from_checkpoint(load_checkpoint(source))
    if source.suffix.lower() == ".json":
        return config_from_json_artifact(source)
    raise ValueError(f"unsupported controller suffix: {source.suffix}")
