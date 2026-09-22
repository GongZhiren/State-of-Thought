from __future__ import annotations

import json
from pathlib import Path

import torch

from sot.artifacts import config_from_checkpoint, config_from_json_artifact
from sot.checkpoint import load_checkpoint, save_checkpoint
from sot.commit import CommitScorer
from sot.reasoning_bank import ReasoningUnit
from sot.selector import HistoricalSelector


def _artifact() -> dict:
    w1 = [((index % 13) - 6) / 100.0 for index in range(32 * 16)]
    return {
        "config": {
            "normalization": {"mu": [0.1, 0.2, 0.3, 0.4], "sigma": [1, 2, 3, 4]},
            "debt": {"alpha": 0.8, "lambda_v": 1.2, "tau_v": 0.04},
            "context": {"context_budget_tokens": 1024},
            "inference": {"t_max_steps": 12, "temperature": 0.0},
            "training": {"split_seed": 42},
        },
        "selector": {
            "gate_hidden_dim": 32,
            "gate_w1": w1,
            "gate_b1": [0.01] * 32,
            "gate_w2": [0.02] * 32,
            "gate_b2": -0.03,
            "gate_prob_threshold": 0.51,
            "read_threshold": -0.5,
            "attention_mix": 0.7,
            "attention_temperature": 0.8,
            "read_feature_mode": "geom5d",
        },
        "commit": {
            "stop_weight": [0.1, -0.2, 0.3, -0.4, 0.0],
            "stop_bias": 0.05,
            "stop_threshold": 0.55,
            "r_stop": 1,
        },
    }


def test_json_and_pt_have_identical_decisions(tmp_path: Path) -> None:
    artifact = tmp_path / "controller.json"
    checkpoint = tmp_path / "controller.pt"
    artifact.write_text(json.dumps(_artifact()), encoding="utf-8")
    save_checkpoint(artifact, checkpoint, release_id="equivalence")

    config_json = config_from_json_artifact(artifact)
    config_pt = config_from_checkpoint(load_checkpoint(checkpoint))
    state = torch.tensor([0.2, -0.4, 0.6, -0.8])
    old_commit = CommitScorer(config_json.commit)
    new_commit = CommitScorer(config_pt.commit)
    assert old_commit.raw_logit(state) == new_commit.raw_logit(state)
    assert old_commit.update(m_t=state, consecutive=0) == new_commit.update(
        m_t=state, consecutive=0
    )

    current = ReasoningUnit(
        unit_id=5,
        text="current",
        z=torch.tensor([0.4, 0.2, -0.1]),
        m=torch.tensor([0.2, -0.4, 0.6, -0.8]),
    )
    candidates = [
        ReasoningUnit(
            unit_id=1,
            text="earlier",
            z=torch.tensor([0.1, 0.3, -0.2]),
            m=torch.tensor([-0.3, 0.5, 0.1, 0.7]),
        )
    ]
    old_score = HistoricalSelector(config_json.selector).score_candidates(
        current_unit=current, candidates=candidates
    )[0]
    new_score = HistoricalSelector(config_pt.selector).score_candidates(
        current_unit=current, candidates=candidates
    )[0]
    assert old_score.logit == new_score.logit
    assert old_score.gate_prob == new_score.gate_prob
    assert old_score.keep == new_score.keep
