"""Explicit threshold-policy loading for frozen exploratory controllers."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from .config import MoTConfig


def load_threshold_policy(path: str | Path) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("threshold policy must be a JSON object")
    return raw


def _valid_threshold(value: Any, fallback: float) -> float:
    try:
        candidate = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    return candidate if 0.0 < candidate < 1.0 else float(fallback)


def apply_threshold_policy(
    config: MoTConfig,
    policy: dict[str, Any],
    *,
    dataset: str,
) -> MoTConfig:
    """Return a config with global or per-dataset frozen thresholds applied."""

    global_row = policy.get("global") if isinstance(policy.get("global"), dict) else {}
    per_dataset = policy.get("per_dataset")
    dataset_row = (
        per_dataset.get(dataset, {})
        if isinstance(per_dataset, dict) and isinstance(per_dataset.get(dataset), dict)
        else {}
    )
    gate = float(config.selector.gate_prob_threshold)
    stop = float(config.commit.stop_threshold)
    for row in (global_row, dataset_row):
        gate = _valid_threshold(
            row.get("best_gate_prob_threshold", row.get("gate_prob_threshold", row.get("base_gate_prob_threshold"))),
            gate,
        )
        stop = _valid_threshold(
            row.get("best_stop_threshold", row.get("stop_threshold", row.get("base_stop_threshold"))),
            stop,
        )
    return replace(
        config,
        selector=replace(config.selector, gate_prob_threshold=gate),
        commit=replace(config.commit, stop_threshold=stop),
    )
