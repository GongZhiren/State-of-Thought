#!/usr/bin/env python3
"""Fail closed on checkpoint, source-layout, and privacy release mistakes."""
from __future__ import annotations

import gzip
import json
import re
import sys
from hashlib import sha256
from pathlib import Path

import yaml

from sot.checkpoint import load_checkpoint
from sot.data_validation import ordered_id_sha256
from sot.evaluation import load_experiment_config

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".toml", ".txt", ".json", ".sh", ".cff"}
PUBLIC_TEXT_NAMES = {".env.example", ".gitattributes", ".gitignore"}
FORBIDDEN = {
    "absolute private path": re.compile("(?:/" + "scratch/|/" + "home/|/" + "Users/)"),
    "private key": re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    "access token": re.compile(r"\b(?:hf_|olp_|ghp_)[A-Za-z0-9_-]{16,}\b"),
    "cloud IPv4 address": re.compile(r"\b(?:47\.245\.|8\.215\.|172\.23\.)\d{1,3}(?:\.\d{1,3})?\b"),
    "submission-process metadata": re.compile(
        r"\b(?:" + "Neu" + "rIPS|Open" + "Review|re" + r"buttal)\b", re.IGNORECASE
    ),
}


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def privacy_failures(label: str, text: str) -> list[str]:
    return [f"{label}: contains {name}" for name, pattern in FORBIDDEN.items() if pattern.search(text)]


def main() -> int:
    failures: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in {"__pycache__", ".pytest_cache"} for part in path.parts):
            continue
        relative = path.relative_to(ROOT)
        if relative == Path("RELEASE_STATUS.md"):
            continue
        if path.suffix in {".pyc", ".pyo"}:
            failures.append(f"{relative}: bytecode is not allowed")
            continue
        if "baseline" in {part.lower() for part in relative.parts}:
            failures.append(f"{relative}: baseline implementation is outside release scope")
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in PUBLIC_TEXT_NAMES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        failures.extend(privacy_failures(str(relative), text))

    required = [
        "README.md",
        "REPRODUCIBILITY.md",
        "LICENSE",
        "CITATION.cff",
        ".env.example",
        ".gitattributes",
        ".gitignore",
        "Makefile",
        "pyproject.toml",
        ".github/workflows/release-check.yml",
        "assets/overview.png",
        "checkpoints/manifest.json",
        "results/README.md",
        "results/manifest.json",
        "results/schema/prediction-record.schema.json",
        "results/summaries/main_llm_accuracy.csv",
        "results/summaries/main_vlm.csv",
        "results/summaries/exploratory.csv",
        "configs/datasets.yaml",
        "data/README.md",
        "data/evaluation_manifest.json",
        "data/training/manifest.json",
        "scripts/build_result_summaries.py",
        "scripts/prepare_vlm_data.py",
        "src/sot/checkpoint.py",
        "src/sot/data_validation.py",
        "src/sot/inference.py",
        "src/sot/evaluation.py",
        "experiments/paper/llama-3.1-8b.yaml",
        "experiments/paper/qwen2.5-14b.yaml",
        "experiments/paper/mixtral-8x7b.yaml",
        "experiments/paper/qwen2.5-vl-7b.yaml",
        "experiments/paper/qwen2.5-vl-32b-bf16.yaml",
        "experiments/exploratory/llama-3.1-8b-training-free.yaml",
        "experiments/exploratory/llama-3.1-8b-embedding.yaml",
    ]
    failures.extend(f"missing required file: {name}" for name in required if not (ROOT / name).is_file())

    manifest_path = ROOT / "checkpoints" / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        release_ids: set[str] = set()
        for row in manifest.get("controllers") or []:
            release_id = str(row.get("release_id", ""))
            if release_id in release_ids:
                failures.append(f"duplicate checkpoint release_id: {release_id}")
            release_ids.add(release_id)
            for kind in ("json", "pt"):
                target = ROOT / "checkpoints" / str(row.get(kind, ""))
                expected = str(row.get(f"{kind}_sha256", ""))
                if not target.is_file():
                    failures.append(f"{release_id}: missing {kind} checkpoint")
                elif file_sha256(target) != expected:
                    failures.append(f"{release_id}: {kind} SHA-256 mismatch")
            target_pt = ROOT / "checkpoints" / str(row.get("pt", ""))
            if target_pt.is_file():
                try:
                    checkpoint = load_checkpoint(target_pt)
                    if checkpoint.release_id != release_id:
                        failures.append(f"{release_id}: embedded release_id mismatch")
                    if checkpoint.source_json_sha256 != str(row.get("json_sha256", "")):
                        failures.append(f"{release_id}: embedded source JSON hash mismatch")
                except Exception as error:
                    failures.append(f"{release_id}: invalid checkpoint ({error})")

        for group in (manifest.get("exploratory") or {}).values():
            for row in group:
                label = str(row.get("release_id", row.get("provider", "exploratory")))
                for key, value in row.items():
                    if not key.endswith("_sha256") and key != "sha256":
                        continue
                    path_key = key.removesuffix("_sha256") if key != "sha256" else "path"
                    target = ROOT / "checkpoints" / str(row.get(path_key, ""))
                    if not target.is_file():
                        failures.append(f"{label}: missing exploratory artifact {path_key}")
                    elif file_sha256(target) != str(value):
                        failures.append(f"{label}: exploratory {path_key} SHA-256 mismatch")
                    elif target.suffix == ".joblib":
                        try:
                            import joblib

                            artifact = joblib.load(target)
                            if not isinstance(artifact, dict) or not {"pca", "clf", "meta"}.issubset(
                                artifact
                            ):
                                failures.append(f"{label}: invalid judge artifact structure")
                            else:
                                failures.extend(
                                    privacy_failures(
                                        f"{label}: serialized metadata", repr(artifact["meta"])
                                    )
                                )
                        except Exception as error:
                            failures.append(f"{label}: unreadable judge artifact ({error})")

    result_manifest = ROOT / "results" / "manifest.json"
    if result_manifest.is_file():
        result_rows = json.loads(result_manifest.read_text(encoding="utf-8"))
        referenced_records: set[Path] = set()
        allowed_record_keys = {
            "record_schema_version",
            "id",
            "dataset",
            "task_type",
            "prediction_raw",
            "prediction",
            "reference_sha256",
            "metric",
            "metric_input",
            "metric_value",
            "completion_tokens",
            "stopped_by",
        }
        required_record_keys = allowed_record_keys - {"metric_input", "stopped_by"}
        if result_rows.get("status") != "publication-aligned":
            failures.append("result manifest is not publication-aligned")
        for summary in result_rows.get("summaries") or []:
            if not (ROOT / str(summary)).is_file():
                failures.append(f"missing generated result summary: {summary}")
        for run in result_rows.get("runs") or []:
            metrics = run.get("metrics")
            if not isinstance(metrics, list):
                if run.get("record_status") != "aggregate-only":
                    failures.append(
                        f"{run.get('release_id')}: mapping metrics require aggregate-only status"
                    )
                continue
            for row in metrics:
                if run.get("record_status") != "bundled":
                    failures.append(f"{run.get('release_id')}: record path requires bundled status")
                    continue
                target = ROOT / str(row["records"])
                referenced_records.add(target.resolve())
                if not target.is_file():
                    failures.append(f"missing result records: {row['records']}")
                    continue
                if file_sha256(target) != str(row["records_sha256"]):
                    failures.append(f"result records SHA-256 mismatch: {row['records']}")
                    continue
                records = [
                    json.loads(line)
                    for line in target.read_text(encoding="utf-8").splitlines()
                    if line
                ]
                if len(records) != int(row["n"]):
                    failures.append(f"result record count mismatch: {row['records']}")
                    continue
                ids: list[str] = []
                for index, record in enumerate(records):
                    keys = set(record)
                    if not required_record_keys.issubset(keys):
                        failures.append(
                            f"{row['records']}:{index + 1}: missing required record field"
                        )
                    if not keys.issubset(allowed_record_keys):
                        failures.append(
                            f"{row['records']}:{index + 1}: undeclared record field"
                        )
                    if record.get("record_schema_version") != 1:
                        failures.append(
                            f"{row['records']}:{index + 1}: unsupported record schema"
                        )
                    if str(record.get("dataset")) != str(row["dataset"]):
                        failures.append(f"{row['records']}:{index + 1}: dataset mismatch")
                    ids.append(str(record.get("id") or ""))
                if not all(ids):
                    failures.append(f"{row['records']}: empty record ID")
                value = sum(float(item["metric_value"]) for item in records) / len(records)
                tokens = sum(int(item["completion_tokens"]) for item in records) / len(records)
                if abs(value - float(row["value"])) > 5e-7:
                    failures.append(f"result metric mismatch: {row['records']}")
                if abs(tokens - float(row["completion_tokens"])) > 5e-7:
                    failures.append(f"result token mismatch: {row['records']}")

        public_records = {
            path.resolve() for path in (ROOT / "results" / "records").rglob("*.jsonl")
        }
        for target in sorted(public_records - referenced_records):
            failures.append(f"unreferenced public result record: {target.relative_to(ROOT)}")

    training_manifest = ROOT / "data" / "training" / "manifest.json"
    if training_manifest.is_file():
        training_rows = json.loads(training_manifest.read_text(encoding="utf-8"))
        for row in training_rows.get("runs") or []:
            target = ROOT / str(row["path"])
            label = str(row.get("release_id", target))
            if not target.is_file():
                failures.append(f"{label}: missing offline training trajectories")
                continue
            if file_sha256(target) != str(row["sha256"]):
                failures.append(f"{label}: compressed training SHA-256 mismatch")
                continue
            digest = sha256()
            count = 0
            with gzip.open(target, "rb") as handle:
                for line in handle:
                    digest.update(line)
                    count += 1
                    failures.extend(
                        privacy_failures(
                            f"{label}: decompressed row {count}",
                            line.decode("utf-8", errors="replace"),
                        )
                    )
            if digest.hexdigest() != str(row["decompressed_sha256"]):
                failures.append(f"{label}: decompressed training SHA-256 mismatch")
            if count != int(row["rows"]):
                failures.append(f"{label}: offline training row-count mismatch")

    for config_path in (ROOT / "experiments").glob("**/*.yaml"):
        config = load_experiment_config(config_path.resolve())
        checkpoint = ROOT / str(config.get("checkpoint", ""))
        if not checkpoint.is_file():
            failures.append(f"{config_path.relative_to(ROOT)}: checkpoint does not exist")
        names = [str(row.get("name")) for row in (config.get("datasets") or [])]
        if len(names) != len(set(names)):
            failures.append(f"{config_path.relative_to(ROOT)}: duplicate dataset entry")
        training = config.get("training") or {}
        offline_records = training.get("offline_records")
        if offline_records and not (ROOT / str(offline_records)).is_file():
            failures.append(
                f"{config_path.relative_to(ROOT)}: offline training records do not exist"
            )

    registry_path = ROOT / "configs" / "datasets.yaml"
    if registry_path.is_file():
        registry = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
        registered = set((registry.get("datasets") or {}).keys())
        for config_path in (ROOT / "experiments" / "paper").glob("*.yaml"):
            config = load_experiment_config(config_path.resolve())
            for row in config.get("datasets") or []:
                if str(row.get("name")) not in registered:
                    failures.append(
                        f"{config_path.relative_to(ROOT)}: unregistered dataset {row.get('name')}"
                    )

    evaluation_manifest_path = ROOT / "data" / "evaluation_manifest.json"
    if evaluation_manifest_path.is_file():
        evaluation_manifest = json.loads(evaluation_manifest_path.read_text(encoding="utf-8"))
        datasets = evaluation_manifest.get("datasets") or {}
        registered = set()
        if registry_path.is_file():
            registry = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
            registered = set((registry.get("datasets") or {}).keys())
        if set(datasets) != registered:
            failures.append("evaluation-manifest datasets do not match the dataset registry")
        for dataset, row in datasets.items():
            for key in ("source_split", "selection"):
                if not str(row.get(key) or "").strip():
                    failures.append(f"{dataset}: evaluation manifest is missing {key}")
            records = list(row.get("records") or [])
            ids = [str(record.get("id") or "") for record in records]
            if len(records) != int(row.get("n", -1)):
                failures.append(f"{dataset}: evaluation-manifest row-count mismatch")
            if not ids or any(not value for value in ids):
                failures.append(f"{dataset}: evaluation manifest contains an empty sample ID")
            if ordered_id_sha256(ids) != str(row.get("ordered_id_sha256") or ""):
                failures.append(f"{dataset}: evaluation-manifest ordered-ID hash mismatch")
            for record in records:
                reference_hash = str(record.get("reference_sha256") or "")
                if not re.fullmatch(r"[0-9a-f]{64}", reference_hash):
                    failures.append(f"{dataset}: invalid reference SHA-256")
                    break

    if failures:
        print("Release check failed:")
        print("\n".join(f"- {failure}" for failure in failures))
        return 1
    print("Release check passed: checkpoint identities, source layout, and privacy patterns are clean.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
