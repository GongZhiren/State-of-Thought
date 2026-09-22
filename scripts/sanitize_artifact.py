#!/usr/bin/env python3
"""Create a privacy-safe copy of a JSON or SoT-Judge joblib artifact."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PATH_KEYS = {
    "artifact",
    "config_path",
    "output_dir",
    "records_jsonl",
    "run_dir",
    "source_offline",
    "source_offline_jsonl",
    "threshold_policy_json",
    "train_path",
    "traj_manifest_jsonl",
}
PRIVATE_PATH_PREFIXES = tuple("/" + part for part in ("home/", "scratch/", "Users/"))


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): (None if str(key) in PATH_KEYS else _clean(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_clean(item) for item in value]
    if isinstance(value, str) and value.startswith(PRIVATE_PATH_PREFIXES):
        return None
    return value


def sanitize_json(source: Path, output: Path) -> None:
    payload = _clean(json.loads(source.read_text(encoding="utf-8")))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sanitize_joblib(source: Path, output: Path) -> None:
    try:
        import joblib
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("install the 'judge' extra to sanitize joblib artifacts") from exc
    payload = joblib.load(source)
    if not isinstance(payload, dict):
        raise TypeError("expected a dictionary SoT-Judge artifact")
    payload = dict(payload)
    payload["meta"] = _clean(dict(payload.get("meta") or {}))
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, output, compress=3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.source.suffix == ".json":
        sanitize_json(args.source, args.output)
    elif args.source.suffix == ".joblib":
        sanitize_joblib(args.source, args.output)
    else:
        raise ValueError("supported inputs: .json and .joblib")


if __name__ == "__main__":
    main()
