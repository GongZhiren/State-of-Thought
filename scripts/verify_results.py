#!/usr/bin/env python3
"""Verify the curated result bundle against its frozen manifest."""
from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    manifest = json.loads((ROOT / "results/manifest.json").read_text())
    verified = 0
    for run in manifest["runs"]:
        metrics = run["metrics"]
        if not isinstance(metrics, list):
            continue
        for expected in metrics:
            path = ROOT / expected["records"]
            raw = path.read_bytes()
            actual_hash = sha256(raw).hexdigest()
            if actual_hash != expected["records_sha256"]:
                raise ValueError(f"{path}: SHA-256 mismatch")
            rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line]
            if len(rows) != int(expected["n"]):
                raise ValueError(f"{path}: record count mismatch")
            value = sum(float(row["metric_value"]) for row in rows) / len(rows)
            tokens = sum(int(row["completion_tokens"]) for row in rows) / len(rows)
            if abs(value - float(expected["value"])) > 5e-7:
                raise ValueError(f"{path}: metric mismatch")
            if abs(tokens - float(expected["completion_tokens"])) > 5e-7:
                raise ValueError(f"{path}: token aggregate mismatch")
            verified += 1
    print(f"verified {verified} bundled paper-result cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
