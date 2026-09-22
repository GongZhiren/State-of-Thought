#!/usr/bin/env python3
"""Build stable, human-readable CSV indexes for the public result bundle."""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results" / "manifest.json"
SUMMARY_DIR = ROOT / "results" / "summaries"

DOMAINS = {
    "gsm8k": "quantitative_reasoning",
    "math": "quantitative_reasoning",
    "drop": "quantitative_reasoning",
    "folio": "symbolic_and_code",
    "proofwriter": "symbolic_and_code",
    "bbh_temporal": "symbolic_and_code",
    "humaneval": "symbolic_and_code",
    "mbpp": "symbolic_and_code",
    "commonsense_qa": "general_understanding",
    "strategyqa": "general_understanding",
    "boolq": "general_understanding",
    "mmlu": "general_understanding",
    "race": "general_understanding",
    "hotpotqa": "long_context_reasoning",
    "narrativeqa": "long_context_reasoning",
    "longbench_multifieldqa": "long_context_reasoning",
    "aokvqa": "vision_language_reasoning",
    "ai2d": "vision_language_reasoning",
    "m3cot": "vision_language_reasoning",
}


def _load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = config.pop("extends", None)
    if parent:
        inherited = _load_config((path.parent / str(parent)).resolve())
        inherited.update(config)
        return inherited
    return config


def _config_metadata(relative: str) -> dict[str, dict[str, Any]]:
    config = _load_config(ROOT / relative)
    return {str(row["name"]): row for row in config.get("datasets") or []}


def _metric_rows(run: dict[str, Any]) -> list[dict[str, Any]]:
    metrics = run["metrics"]
    if isinstance(metrics, list):
        return [dict(row) for row in metrics]
    return [{"dataset": dataset, "value": value} for dataset, value in metrics.items()]


def _score(value: Any) -> str:
    return f"{100.0 * float(value):.1f}"


def _one_decimal(value: Any) -> str:
    return "" if value is None else f"{float(value):.1f}"


def _two_decimals(value: Any) -> str:
    return "" if value is None else f"{float(value):.2f}"


def _csv_bytes(fieldnames: list[str], rows: list[dict[str, Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def build() -> dict[Path, bytes]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    llm_rows: list[dict[str, Any]] = []
    vlm_rows: list[dict[str, Any]] = []
    exploratory_rows: list[dict[str, Any]] = []

    for run in manifest["runs"]:
        config_path = str(run["config"])
        metadata = _config_metadata(config_path)
        is_exploratory = config_path.startswith("experiments/exploratory/")
        is_vlm = "-vl-" in str(run["release_id"])
        for metric in _metric_rows(run):
            dataset = str(metric["dataset"])
            dataset_meta = metadata[dataset]
            common = {
                "release_id": run["release_id"],
                "domain": DOMAINS[dataset],
                "dataset": dataset,
                "metric": dataset_meta["primary_metric"],
                "n": metric.get("n", dataset_meta.get("max_samples", "")),
                "score_percent": _score(metric["value"]),
                "record_status": run["record_status"],
                "artifact": run.get("checkpoint", run.get("policy", "")),
                "records": metric.get("records", ""),
            }
            if is_exploratory:
                exploratory_rows.append(
                    {
                        **common,
                        "mean_completion_tokens": _one_decimal(
                            metric.get("completion_tokens")
                        ),
                    }
                )
            elif is_vlm:
                vlm_rows.append(
                    {
                        **common,
                        "mean_completion_tokens": _one_decimal(
                            metric.get("completion_tokens")
                        ),
                        "mean_latency_seconds": _two_decimals(metric.get("latency_seconds")),
                        "latency_samples": metric.get("latency_n", ""),
                    }
                )
            else:
                llm_rows.append(common)

    base = [
        "release_id",
        "domain",
        "dataset",
        "metric",
        "n",
        "score_percent",
        "record_status",
        "artifact",
        "records",
    ]
    return {
        SUMMARY_DIR / "main_llm_accuracy.csv": _csv_bytes(base, llm_rows),
        SUMMARY_DIR / "main_vlm.csv": _csv_bytes(
            base[:-3]
            + ["mean_completion_tokens", "mean_latency_seconds", "latency_samples"]
            + base[-3:],
            vlm_rows,
        ),
        SUMMARY_DIR / "exploratory.csv": _csv_bytes(
            base[:-3] + ["mean_completion_tokens"] + base[-3:], exploratory_rows
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check", action="store_true", help="fail if committed summaries are stale"
    )
    args = parser.parse_args()
    failures: list[str] = []
    for path, content in build().items():
        if args.check:
            if not path.is_file() or path.read_bytes() != content:
                failures.append(str(path.relative_to(ROOT)))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
    if failures:
        raise SystemExit("stale result summaries: " + ", ".join(failures))
    print("result summaries are current" if args.check else "wrote curated result summaries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
