#!/usr/bin/env python3
"""Build the paper-split identity manifest from privacy-safe result records."""
from __future__ import annotations

import json
from pathlib import Path

from sot.data_validation import ordered_id_sha256

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "bbh_temporal": "llama-3.1-8b",
    "boolq": "llama-3.1-8b",
    "commonsense_qa": "llama-3.1-8b",
    "drop": "llama-3.1-8b",
    "folio": "llama-3.1-8b",
    "gsm8k": "llama-3.1-8b",
    "hotpotqa": "llama-3.1-8b",
    "humaneval": "llama-3.1-8b",
    "longbench_multifieldqa": "llama-3.1-8b",
    "math": "llama-3.1-8b",
    "mbpp": "llama-3.1-8b",
    "mmlu": "llama-3.1-8b",
    "narrativeqa": "llama-3.1-8b",
    "proofwriter": "llama-3.1-8b",
    "race": "llama-3.1-8b",
    "strategyqa": "llama-3.1-8b",
    "aokvqa": "qwen2.5-vl-7b",
    "ai2d": "qwen2.5-vl-7b",
    "m3cot": "qwen2.5-vl-7b",
}

TEXT_PROTOCOLS = {
    "gsm8k": ("test", 500),
    "math": ("test", 400),
    "mbpp": ("test", 250),
    "humaneval": ("test", None),
    "proofwriter": ("test (depth 0)", 400),
    "folio": ("test", None),
    "bbh_temporal": ("test", None),
    "commonsense_qa": ("test", 400),
    "strategyqa": ("test", 400),
    "hotpotqa": ("test", 300),
    "drop": ("dev", 300),
    "narrativeqa": ("test", 250),
    "longbench_multifieldqa": ("test", None),
    "mmlu": ("test", 500),
    "boolq": ("test", 400),
    "race": ("test", 300),
}

VLM_PROTOCOLS = {
    "aokvqa": {
        "source": "HuggingFaceM4/A-OKVQA",
        "source_revision": "d1b0efa3a436e9101dfbde3752db7607da696c35",
        "source_split": "validation",
        "selection": "first_200_valid_in_source_order",
    },
    "ai2d": {
        "source": "lmms-lab/ai2d",
        "source_revision": "c83a9b9692933aff8349157c88a413df9d02c4e5",
        "source_split": "test",
        "selection": "first_150_in_source_order",
    },
    "m3cot": {
        "source": "LightChen2333/M3CoT",
        "source_revision": "48cf35001d595a6b0290c82c897a4b4563390821",
        "source_split": "test",
        "selection": "first_150_valid_in_source_order",
    },
}


def protocol(dataset: str) -> dict[str, object]:
    """Return the frozen source split and deterministic subset rule."""
    if dataset in VLM_PROTOCOLS:
        return dict(VLM_PROTOCOLS[dataset])
    source_split, cap = TEXT_PROTOCOLS[dataset]
    return {
        "source_split": source_split,
        "selection": "stable_sort_by_id_then_head_n",
        "selection_cap": cap,
    }


def build() -> dict[str, object]:
    datasets: dict[str, object] = {}
    for dataset, release_id in SOURCES.items():
        path = ROOT / "results" / "records" / release_id / f"{dataset}.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        records = [
            {"id": str(row["id"]), "reference_sha256": str(row["reference_sha256"])}
            for row in rows
        ]
        ids = [row["id"] for row in records]
        datasets[dataset] = {
            **protocol(dataset),
            "n": len(records),
            "ordered_id_sha256": ordered_id_sha256(ids),
            "records": records,
        }
    return {
        "schema_version": 2,
        "description": "Ordered paper-evaluation identities; references are represented only by SHA-256 hashes.",
        "datasets": datasets,
    }


def main() -> int:
    output = ROOT / "data" / "evaluation_manifest.json"
    output.write_text(json.dumps(build(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
