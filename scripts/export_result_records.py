#!/usr/bin/env python3
"""Export result-only records without redistributing benchmark examples."""
from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path
from typing import Any

from sot.grading import extract_final_answer_text, is_correct, token_level_f1

F1_DATASETS = {"drop", "hotpotqa", "narrativeqa", "longbench_multifieldqa"}
NUMERIC_EXACT_MATCH_DATASETS = {"gsm8k", "math"}
PASS_AT_ONE_DATASETS = {"humaneval", "mbpp"}


def _digest(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return sha256(encoded).hexdigest()


def _text_record(row: dict[str, Any], *, f1_source: str = "parsed") -> dict[str, Any]:
    dataset = str(row["dataset"])
    task_type = str(row["task_type"])
    prediction_raw = str(row.get("predicted_answer_raw_mot") or "")
    reference = row.get("reference_answer", "")
    sample_extra = dict(row.get("sample_extra") or {})
    choices = row.get("choices", sample_extra.get("choices"))
    parsed = extract_final_answer_text(
        prediction_raw,
        task_type=task_type,
        ref_answer=str(reference),
        choices=choices,
        question=str(row.get("question") or ""),
    )
    if dataset in F1_DATASETS:
        metric_input = prediction_raw if f1_source == "raw" else parsed
        metric = token_level_f1(metric_input, str(reference))
    elif dataset in {"humaneval", "mbpp", "bbh_temporal"}:
        # Code pass@1 requires the original sandbox, while the BBH temporal
        # task uses its frozen label canonicalizer. Both values are retained
        # from the audited source record.
        metric = float(bool(row.get("mot_correct")))
    else:
        metric = float(
            is_correct(
                parsed,
                str(reference),
                task_type,
                extra=sample_extra,
                choices=choices,
            )
        )
    return {
        "record_schema_version": 1,
        "id": str(row["id"]),
        "dataset": dataset,
        "task_type": task_type,
        "prediction_raw": prediction_raw,
        "prediction": parsed,
        "reference_sha256": _digest(reference),
        "metric": (
            "f1"
            if dataset in F1_DATASETS
            else "exact_match_number"
            if dataset in NUMERIC_EXACT_MATCH_DATASETS
            else "pass@1"
            if dataset in PASS_AT_ONE_DATASETS
            else "accuracy"
        ),
        "metric_input": f1_source if dataset in F1_DATASETS else "parsed",
        "metric_value": float(metric),
        "completion_tokens": int(row.get("mot_completion_tokens") or 0),
        "stopped_by": row.get("mot_stopped_by"),
    }


def _vlm_record(row: dict[str, Any], *, dataset: str) -> dict[str, Any]:
    reference = row.get("reference", row.get("gold", ""))
    return {
        "record_schema_version": 1,
        "id": str(row["id"]),
        "dataset": dataset,
        "task_type": str(row.get("task_type") or "visual_reasoning_mc"),
        "prediction_raw": str(row.get("prediction_raw") or ""),
        "prediction": str(row.get("parsed", row.get("pred_parsed", "")) or ""),
        "reference_sha256": _digest(reference),
        "metric": "accuracy",
        "metric_value": float(bool(row.get("correct"))),
        "completion_tokens": int(
            row.get("aggregated_completion_tokens", row.get("completion_tokens", 0)) or 0
        ),
        "stopped_by": row.get("stopped_by"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--modality", choices=("text", "vision-language"), required=True)
    parser.add_argument("--dataset", default="")
    parser.add_argument("--expected", type=float)
    parser.add_argument(
        "--f1-source",
        choices=("parsed", "raw"),
        default="parsed",
        help="score F1 datasets on the extracted prediction or complete decoded answer",
    )
    args = parser.parse_args()
    rows: list[dict[str, Any]] = []
    with args.source.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            source_row = json.loads(line)
            rows.append(
                _text_record(source_row, f1_source=args.f1_source)
                if args.modality == "text"
                else _vlm_record(source_row, dataset=str(args.dataset))
            )
    if not rows:
        raise ValueError("source contains no records")
    aggregate = sum(float(row["metric_value"]) for row in rows) / len(rows)
    if args.expected is not None and abs(aggregate - args.expected) > 5e-7:
        raise ValueError(f"aggregate {aggregate:.12f} != expected {args.expected:.12f}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(json.dumps({"n": len(rows), "aggregate": aggregate}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
