from __future__ import annotations

import json
from hashlib import sha1
from pathlib import Path

from sot.data_validation import ordered_id_sha256, validate_data_config, value_sha256
from sot.dataset import stable_sample_id
from sot.evaluation import _load_completed_id_counts


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_completed_record_counts_preserve_duplicate_ids(tmp_path: Path) -> None:
    records = tmp_path / "records.jsonl"
    _write_jsonl(records, [{"id": "shared"}, {"id": "shared"}, {"id": "other"}])
    assert _load_completed_id_counts(records) == {"shared": 2, "other": 1}


def test_missing_source_id_uses_historical_question_digest() -> None:
    question = "A stable question"
    expected = sha1(question.encode("utf-8")).hexdigest()[:16]  # noqa: S324
    assert stable_sample_id("gsm8k", {"id": "", "question": question}, 9) == expected


def test_validate_data_config_checks_order_and_references(tmp_path: Path) -> None:
    root = tmp_path / "release"
    config = root / "experiments" / "paper" / "tiny.yaml"
    data = root / "data" / "eval_fixed" / "tiny.jsonl"
    manifest = root / "data" / "evaluation_manifest.json"
    rows = [
        {"id": "shared", "question": "Q1", "answer": "A1"},
        {"id": "shared", "question": "Q2", "answer": "A2"},
    ]
    _write_jsonl(data, rows)
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        "\n".join(
            [
                "release_id: tiny",
                "modality: text",
                "datasets:",
                "  - {name: tiny, task_type: binary_qa, primary_metric: accuracy, "
                "path: data/eval_fixed/tiny.jsonl, max_samples: 2}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "datasets": {
                    "tiny": {
                        "n": 2,
                        "ordered_id_sha256": ordered_id_sha256(["shared", "shared"]),
                        "records": [
                            {"id": "shared", "reference_sha256": value_sha256("A1")},
                            {"id": "shared", "reference_sha256": value_sha256("A2")},
                        ],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    report = validate_data_config(config)
    assert report["passed"] is True
    assert report["datasets"][0]["n"] == 2

    rows[1]["answer"] = "wrong"
    _write_jsonl(data, rows)
    report = validate_data_config(config)
    assert report["passed"] is False
    assert "reference mismatch at indices 1" in report["failures"][0]
