"""Validate locally prepared evaluation data against the released paper split."""
from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

from .dataset import DatasetSample
from .evaluation import load_experiment_config, load_samples


def value_sha256(value: Any) -> str:
    """Hash a reference value using the release manifest's canonical encoding."""
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return sha256(encoded).hexdigest()


def ordered_id_sha256(ids: list[str]) -> str:
    """Hash an ordered list of sample identifiers."""
    encoded = json.dumps(ids, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


def _image_path(sample: DatasetSample, *, root: Path) -> Path | None:
    raw = sample.extra.get("image_path")
    if not isinstance(raw, str) or not raw.strip():
        return None
    path = Path(raw)
    return path if path.is_absolute() else root / path


def validate_data_config(
    config_path: str | Path,
    *,
    manifest_path: str | Path | None = None,
    check_images: bool = True,
) -> dict[str, Any]:
    """Validate sample order, references, and VLM assets for one paper config."""
    config_file = Path(config_path).resolve()
    root = config_file.parents[2]
    config = load_experiment_config(config_file)
    manifest_file = (
        Path(manifest_path).resolve()
        if manifest_path is not None
        else root / "data" / "evaluation_manifest.json"
    )
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    expected_datasets = dict(manifest.get("datasets") or {})
    modality = str(config.get("modality") or "text")

    failures: list[str] = []
    datasets: list[dict[str, Any]] = []
    for dataset_row in config.get("datasets") or []:
        dataset = str(dataset_row["name"])
        expected = expected_datasets.get(dataset)
        if not isinstance(expected, dict):
            failures.append(f"{dataset}: missing from evaluation manifest")
            continue
        path = (root / str(dataset_row["path"])).resolve()
        if not path.is_file():
            failures.append(f"{dataset}: dataset file not found: {path}")
            continue
        samples = load_samples(
            path,
            dataset=dataset,
            task_type=str(dataset_row["task_type"]),
            primary_metric=str(dataset_row["primary_metric"]),
        )
        limit = int(dataset_row.get("max_samples", 0))
        if limit > 0:
            samples = samples[:limit]
        expected_rows = list(expected.get("records") or [])
        if len(samples) != len(expected_rows):
            failures.append(
                f"{dataset}: expected {len(expected_rows)} samples, found {len(samples)}"
            )
            continue

        ids = [sample.id for sample in samples]
        expected_ids = [str(row.get("id") or "") for row in expected_rows]
        if ids != expected_ids:
            mismatch = next(
                (index for index, pair in enumerate(zip(ids, expected_ids)) if pair[0] != pair[1]),
                None,
            )
            failures.append(f"{dataset}: ordered sample IDs differ at index {mismatch}")
        digest = ordered_id_sha256(ids)
        if digest != str(expected.get("ordered_id_sha256") or ""):
            failures.append(f"{dataset}: ordered-ID hash mismatch")

        reference_mismatches = [
            index
            for index, (sample, row) in enumerate(zip(samples, expected_rows))
            if value_sha256(sample.answer) != str(row.get("reference_sha256") or "")
        ]
        if reference_mismatches:
            preview = ", ".join(str(index) for index in reference_mismatches[:5])
            failures.append(f"{dataset}: reference mismatch at indices {preview}")

        missing_images: list[str] = []
        if modality == "vision-language" and check_images:
            for sample in samples:
                image = _image_path(sample, root=root)
                if image is None or not image.is_file():
                    missing_images.append(sample.id)
            if missing_images:
                preview = ", ".join(missing_images[:5])
                failures.append(f"{dataset}: missing images for IDs {preview}")

        datasets.append(
            {
                "dataset": dataset,
                "n": len(samples),
                "ordered_id_sha256": digest,
                "references_verified": len(reference_mismatches) == 0,
                "images_verified": modality != "vision-language" or not missing_images,
            }
        )

    return {
        "schema_version": 1,
        "config": str(config_file),
        "manifest": str(manifest_file),
        "datasets": datasets,
        "failures": failures,
        "passed": not failures,
    }


__all__ = ["ordered_id_sha256", "validate_data_config", "value_sha256"]
