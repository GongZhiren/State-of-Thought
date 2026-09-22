from __future__ import annotations

import json
from pathlib import Path

import torch

from sot.checkpoint import build_payload, load_checkpoint, save_checkpoint


def _artifact() -> dict:
    return {
        "config": {
            "normalization": {"mu": [0, 0, 0, 0], "sigma": [1, 1, 1, 1]},
            "selector": {},
            "commit": {},
            "training": {"split_seed": 42},
        },
        "selector": {
            "gate_hidden_dim": 32,
            "gate_w1": [0.0] * 32,
            "gate_b1": [0.0] * 32,
            "gate_w2": [0.0] * 32,
            "gate_b2": 0.0,
            "gate_prob_threshold": 0.475,
        },
        "commit": {
            "stop_weight": [1.0, 2.0, 3.0, 4.0, 0.0],
            "stop_bias": -0.5,
            "stop_threshold": 0.5,
            "r_stop": 1,
        },
    }


def test_legacy_normalization_is_shape_safe(tmp_path: Path) -> None:
    source = tmp_path / "artifact.json"
    source.write_text(json.dumps(_artifact()), encoding="utf-8")
    payload = build_payload(source, release_id="test")
    assert payload["parameter_count"] == 582
    assert payload["state_dict"]["selector.gate_w1"].shape == (32, 16)
    assert payload["state_dict"]["commit.stop_weight"].shape == (4,)
    assert payload["migrations"] == [
        "expanded_zero_selector_w1",
        "removed_legacy_fifth_stop_weight",
    ]


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    source = tmp_path / "artifact.json"
    output = tmp_path / "controller.pt"
    source.write_text(json.dumps(_artifact()), encoding="utf-8")
    save_checkpoint(source, output, release_id="test")
    loaded = load_checkpoint(output)
    assert loaded.release_id == "test"
    assert loaded.parameter_count == 582
    assert torch.equal(loaded.state_dict["commit.stop_weight"], torch.tensor([1, 2, 3, 4.0]))
