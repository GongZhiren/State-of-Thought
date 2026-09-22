"""Resumable, record-first evaluation for released SoT controllers."""
from __future__ import annotations

import json
import random
import time
from collections import Counter
from hashlib import sha256
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import AutoTokenizer

from .artifacts import load_config
from .backbone.hf_backbone import HFGatedBackbone
from .backbone.hf_vlm_backbone import HFVLMBackbone
from .dataset import DatasetSample, stable_sample_id
from .grading import extract_final_answer_text, is_correct, token_level_f1
from .mechanism_config import MechanismExperimentConfig
from .text_runtime import run_sot_mechanism_on_sample
from .thresholds import apply_threshold_policy, load_threshold_policy
from .training_free import TrainingFreePolicy
from .vlm_inputs import load_pil_image
from .vlm_runtime import run_vlm_sot_on_sample

RECORD_SCHEMA_VERSION = 1


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    """Load YAML with one optional relative ``extends`` parent."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent_name = raw.pop("extends", None)
    if not parent_name:
        return raw
    parent = (path.parent / str(parent_name)).resolve()
    base = load_experiment_config(parent)
    merged = dict(base)
    for key, value in raw.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


def load_samples(path: Path, *, dataset: str, task_type: str, primary_metric: str) -> list[DatasetSample]:
    samples: list[DatasetSample] = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            answer: Any = row.get("answer", row.get("reference_answer", row.get("gold", "")))
            if isinstance(answer, list):
                answer = answer[0] if answer else ""
            extra = {
                key: value
                for key, value in row.items()
                if key not in {"id", "question", "answer", "reference_answer", "gold"}
            }
            extra.setdefault("task_type", task_type)
            extra.setdefault("primary_metric", primary_metric)
            samples.append(
                DatasetSample(
                    id=stable_sample_id(dataset, row, index),
                    question=str(row.get("question") or ""),
                    answer=str(answer or ""),
                    extra=extra,
                )
            )
    return samples


def _load_completed_id_counts(path: Path) -> Counter[str]:
    if not path.is_file():
        return Counter()
    completed: Counter[str] = Counter()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            completed[str(json.loads(line)["id"])] += 1
    return completed


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def evaluate_text_config(config_path: str | Path) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    root = config_file.parents[2]
    raw = load_experiment_config(config_file)
    if str(raw.get("modality")) != "text":
        raise ValueError("this evaluator requires modality: text")

    seed = int(raw.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    checkpoint = (root / str(raw["checkpoint"])).resolve()
    model_id = str(raw["model"])
    device_requested = str(raw.get("device", "cuda:0"))
    output_root = (root / str(raw.get("output_dir", "outputs/paper"))).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    model = HFGatedBackbone(
        model_id,
        device=device_requested,
        dtype=str(raw.get("dtype", "bfloat16")),
        attention_implementation=str(raw.get("attention_implementation", "sdpa")),
        quantization=str(raw.get("quantization", "none")),
        device_map=raw.get("device_map"),
    )
    device = model.get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=bool(raw.get("use_fast", True)))
    mot_config = load_config(checkpoint)
    threshold_policy = None
    if raw.get("threshold_policy"):
        policy_path = (root / str(raw["threshold_policy"])).resolve()
        threshold_policy = load_threshold_policy(policy_path)
    variant = str(raw.get("variant", "learned")).strip().lower()
    if variant not in {"learned", "training_free", "embedding"}:
        raise ValueError(f"unsupported text variant: {variant}")
    training_free_policy = None
    mechanism = None
    if variant == "training_free":
        policy_path = (root / str(raw["training_free_policy"])).resolve()
        training_free_policy = TrainingFreePolicy.from_path(policy_path)
    elif variant == "embedding":
        embed = dict(raw.get("embedding") or {})
        mechanism = MechanismExperimentConfig(
            sot_signal_mode="openai_embed",
            embed_pca_path=str((root / str(embed["pca"])).resolve()),
            openai_embed_model=str(embed.get("model", "text-embedding-3-small")),
            embed_h_source=str(embed.get("h_source", "spectral")),
            embed_m_source=str(embed.get("m_source", "geometry4")),
        )

    run_manifest = {
        "schema_version": 1,
        "config": str(config_file.relative_to(root)),
        "config_sha256": file_sha256(config_file),
        "checkpoint": str(checkpoint.relative_to(root)),
        "checkpoint_sha256": file_sha256(checkpoint),
        "model": model_id,
        "variant": variant,
        "seed": seed,
        "device": str(device),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
    }
    (output_root / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    summaries: dict[str, Any] = {}
    for dataset_row in raw.get("datasets") or []:
        dataset = str(dataset_row["name"])
        dataset_mot_config = (
            apply_threshold_policy(mot_config, threshold_policy, dataset=dataset)
            if threshold_policy is not None
            else mot_config
        )
        task_type = str(dataset_row["task_type"])
        primary_metric = str(dataset_row["primary_metric"])
        data_path = (root / str(dataset_row["path"])).resolve()
        samples = load_samples(
            data_path, dataset=dataset, task_type=task_type, primary_metric=primary_metric
        )
        limit = int(dataset_row.get("max_samples", 0))
        if limit > 0:
            samples = samples[:limit]
        records_path = output_root / f"{dataset}.records.jsonl"
        completed = _load_completed_id_counts(records_path)
        encountered: Counter[str] = Counter()
        with records_path.open("a", encoding="utf-8") as records:
            for sample in samples:
                encountered[sample.id] += 1
                if encountered[sample.id] <= completed[sample.id]:
                    continue
                _sync(device)
                start = time.perf_counter()
                result = run_sot_mechanism_on_sample(
                    sample,
                    mot_cfg=dataset_mot_config,
                    backbone=model,
                    tokenizer=tokenizer,
                    model_path=model_id,
                    device=device,
                    mechanism=mechanism,
                    training_free_policy=training_free_policy,
                    dataset_name=dataset,
                    commit_mode=str(raw.get("commit_mode", "fullchain")),
                )
                _sync(device)
                wall_seconds = time.perf_counter() - start
                parsed = extract_final_answer_text(
                    result.final_answer,
                    task_type=task_type,
                    ref_answer=sample.answer,
                    choices=sample.choices,
                    question=sample.question,
                )
                correct = is_correct(
                    parsed,
                    sample.answer,
                    task_type,
                    extra=sample.extra,
                    choices=sample.choices,
                )
                record = {
                    "record_schema_version": RECORD_SCHEMA_VERSION,
                    "id": sample.id,
                    "dataset": dataset,
                    "task_type": task_type,
                    "primary_metric": primary_metric,
                    "reference": sample.answer,
                    "prediction_raw": result.final_answer,
                    "prediction": parsed,
                    "correct": bool(correct),
                    "metric_value": (
                        token_level_f1(parsed, sample.answer)
                        if primary_metric == "f1"
                        else float(bool(correct))
                    ),
                    "observe_completion_tokens": int(result.generated_tokens),
                    "answer_completion_tokens": int(result.answer_completion_tokens),
                    "aggregated_completion_tokens": int(result.aggregated_completion_tokens),
                    "reasoning_steps": int(result.node_count),
                    "stopped_by": result.stopped_by,
                    "wall_seconds": wall_seconds,
                }
                records.write(json.dumps(record, ensure_ascii=False) + "\n")
                records.flush()

        records_all = [
            json.loads(line)
            for line in records_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(records_all) != len(samples):
            raise RuntimeError(
                f"{dataset}: expected {len(samples)} records, found {len(records_all)}; resume the run"
            )
        summary = {
            "dataset": dataset,
            "n": len(records_all),
            "primary_metric": primary_metric,
            "primary_metric_value": sum(float(row["metric_value"]) for row in records_all)
            / max(1, len(records_all)),
            "mean_completion_tokens": sum(
                int(row["aggregated_completion_tokens"]) for row in records_all
            )
            / max(1, len(records_all)),
            "mean_wall_seconds": sum(float(row["wall_seconds"]) for row in records_all)
            / max(1, len(records_all)),
            "records_sha256": file_sha256(records_path),
        }
        (output_root / f"{dataset}.summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        summaries[dataset] = summary
    return summaries


def evaluate_vlm_config(config_path: str | Path) -> dict[str, Any]:
    """Evaluate a frozen SoT controller on the released VLM matrix."""
    config_file = Path(config_path).resolve()
    root = config_file.parents[2]
    raw = load_experiment_config(config_file)
    if str(raw.get("modality")) != "vision-language":
        raise ValueError("this evaluator requires modality: vision-language")

    seed = int(raw.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    checkpoint = (root / str(raw["checkpoint"])).resolve()
    model_id = str(raw["model"])
    output_root = (root / str(raw.get("output_dir", "outputs/paper"))).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    backbone = HFVLMBackbone(
        model_id,
        device=str(raw.get("device", "cuda:0")),
        dtype=str(raw.get("dtype", "bfloat16")),
        quantization=str(raw.get("quantization", "none")),
        device_map=raw.get("device_map"),
        processor_use_fast=bool(raw.get("processor_use_fast", True)),
    )
    device = backbone.get_device()
    mot_config = load_config(checkpoint)
    manifest = {
        "schema_version": 1,
        "modality": "vision-language",
        "config": str(config_file.relative_to(root)),
        "config_sha256": file_sha256(config_file),
        "checkpoint": str(checkpoint.relative_to(root)),
        "checkpoint_sha256": file_sha256(checkpoint),
        "model": model_id,
        "seed": seed,
        "device": str(device),
        "dtype": str(raw.get("dtype", "bfloat16")),
        "processor_use_fast": True,
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
    }
    (output_root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    summaries: dict[str, Any] = {}
    for dataset_row in raw.get("datasets") or []:
        dataset = str(dataset_row["name"])
        task_type = str(dataset_row["task_type"])
        primary_metric = str(dataset_row.get("primary_metric", "accuracy"))
        data_path = (root / str(dataset_row["path"])).resolve()
        samples = load_samples(
            data_path, dataset=dataset, task_type=task_type, primary_metric=primary_metric
        )
        limit = int(dataset_row.get("max_samples", 0))
        if limit > 0:
            samples = samples[:limit]
        records_path = output_root / f"{dataset}.records.jsonl"
        completed = _load_completed_id_counts(records_path)
        encountered: Counter[str] = Counter()
        with records_path.open("a", encoding="utf-8") as records:
            for sample in samples:
                encountered[sample.id] += 1
                if encountered[sample.id] <= completed[sample.id]:
                    continue
                _sync(device)
                started = time.perf_counter()
                image = load_pil_image(sample.extra, data_root=root)
                try:
                    result = run_vlm_sot_on_sample(
                        sample, mot_cfg=mot_config, backbone=backbone, image=image
                    )
                finally:
                    image.close()
                _sync(device)
                wall_seconds = time.perf_counter() - started
                parsed = extract_final_answer_text(
                    result.final_answer,
                    task_type=task_type,
                    ref_answer=sample.answer,
                    choices=sample.choices,
                    question=sample.question,
                )
                correct = is_correct(
                    parsed,
                    sample.answer,
                    task_type,
                    extra=sample.extra,
                    choices=sample.choices,
                )
                record = {
                    "record_schema_version": RECORD_SCHEMA_VERSION,
                    "id": sample.id,
                    "dataset": dataset,
                    "task_type": task_type,
                    "primary_metric": primary_metric,
                    "reference": sample.answer,
                    "prediction_raw": result.final_answer,
                    "prediction": parsed,
                    "correct": bool(correct),
                    "metric_value": float(bool(correct)),
                    "observe_completion_tokens": int(result.generated_tokens),
                    "answer_completion_tokens": int(result.answer_completion_tokens),
                    "aggregated_completion_tokens": int(result.aggregated_completion_tokens),
                    "reasoning_steps": int(result.node_count),
                    "stopped_by": result.stopped_by,
                    "wall_seconds": wall_seconds,
                }
                records.write(json.dumps(record, ensure_ascii=False) + "\n")
                records.flush()
        rows = [
            json.loads(line)
            for line in records_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(rows) != len(samples):
            raise RuntimeError(
                f"{dataset}: expected {len(samples)} records, found {len(rows)}; resume the run"
            )
        summary = {
            "dataset": dataset,
            "n": len(rows),
            "primary_metric": primary_metric,
            "primary_metric_value": sum(float(row["metric_value"]) for row in rows)
            / max(1, len(rows)),
            "mean_completion_tokens": sum(int(row["aggregated_completion_tokens"]) for row in rows)
            / max(1, len(rows)),
            "mean_wall_seconds": sum(float(row["wall_seconds"]) for row in rows)
            / max(1, len(rows)),
            "records_sha256": file_sha256(records_path),
        }
        (output_root / f"{dataset}.summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        summaries[dataset] = summary
    return summaries


def evaluate_config(config_path: str | Path) -> dict[str, Any]:
    raw = load_experiment_config(Path(config_path).resolve())
    modality = str(raw.get("modality") or "text")
    if modality == "text":
        return evaluate_text_config(config_path)
    if modality == "vision-language":
        return evaluate_vlm_config(config_path)
    raise ValueError(f"unsupported modality: {modality}")
