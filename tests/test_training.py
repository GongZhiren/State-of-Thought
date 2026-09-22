from __future__ import annotations

import gzip
import json
from hashlib import sha256
from pathlib import Path

from sot.checkpoint import load_checkpoint
from sot.config import MoTConfig
from sot.training import fit_controller, train_from_config


def test_controller_fit_matches_frozen_golden_fixture(tmp_path: Path) -> None:
    config = MoTConfig()
    config.training.warmup_epochs = 3
    config.training.onpolicy_refine_steps = 0
    config.training.split_seed = 42
    records = []
    for index in range(8):
        records.append(
            {
                "problem_id": str(index),
                "dataset": "fixture",
                "task_type": "math_qa",
                "split": "train",
                "traj_correct": bool(index % 2),
                "step_index": index,
                "m_t": [0.1 * index, 0.2 - index * 0.03, -0.4 + index * 0.02, 0.8 - index * 0.01, 0.0],
                "history": [
                    {
                        "m_j": [0.05 * index, -0.1, 0.2, 0.3, 0.0],
                        "retrieval_trainable": 1,
                        "gate_hard_label": index % 2,
                        "keep_label": index % 2,
                        "gate_soft_label": 0.8 if index % 2 else 0.2,
                        "age": 2,
                    }
                ],
                "teacher_stop_score": 0.8 if index > 4 else 0.2,
                "stop_label": 1 if index > 4 else 0,
                "step_len_tokens": 10 + index,
            }
        )
    artifact = fit_controller(records, config, tmp_path / "controller.json")
    selector = artifact["selector"]
    commit = artifact["commit"]
    payload = {
        "w1": selector["gate_w1"],
        "b1": selector["gate_b1"],
        "w2": selector["gate_w2"],
        "b2": selector["gate_b2"],
        "sw": commit["stop_weight"],
        "sb": commit["stop_bias"],
    }
    digest = sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert digest == "0d0057b2d6022025552a7075afa17b242911140a4bed87f0e5600465d6a8ec93"
    assert len(selector["gate_w1"]) == 32 * 16
    assert len(commit["stop_weight"]) == 4


def test_onpolicy_refinement_matches_frozen_golden_fixture(tmp_path: Path) -> None:
    config = MoTConfig()
    config.training.warmup_epochs = 3
    config.training.onpolicy_refine_steps = 5
    config.training.onpolicy_batch_size = 16
    config.training.split_seed = 42
    records = []
    for index in range(8):
        records.append(
            {
                "problem_id": str(index),
                "dataset": "fixture",
                "task_type": "math_qa",
                "split": "train",
                "traj_correct": bool(index % 2),
                "step_index": index,
                "m_t": [
                    0.1 * index,
                    0.2 - index * 0.03,
                    -0.4 + index * 0.02,
                    0.8 - index * 0.01,
                    0.0,
                ],
                "history": [
                    {
                        "m_j": [0.05 * index, -0.1, 0.2, 0.3, 0.0],
                        "retrieval_trainable": 1,
                        "gate_hard_label": index % 2,
                        "keep_label": index % 2,
                        "gate_soft_label": 0.8 if index % 2 else 0.2,
                        "age": 2,
                    }
                ],
                "teacher_stop_score": 0.8 if index > 4 else 0.2,
                "stop_label": 1 if index > 4 else 0,
                "step_len_tokens": 10 + index,
            }
        )
    artifact = fit_controller(records, config, tmp_path / "controller-onpolicy.json")
    payload = {
        "w1": artifact["selector"]["gate_w1"],
        "b1": artifact["selector"]["gate_b1"],
        "w2": artifact["selector"]["gate_w2"],
        "b2": artifact["selector"]["gate_b2"],
        "sw": artifact["commit"]["stop_weight"],
        "sb": artifact["commit"]["stop_bias"],
    }
    digest = sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert digest == "34a4e64893630b0b125cd4fa66e3182a17ac8a19d3721afcff64c212b29b12ce"


def test_stop_only_training_emits_an_inert_full_shape_gate(tmp_path: Path) -> None:
    config = MoTConfig()
    records = [
        {
            "problem_id": "a",
            "split": "train",
            "m_t": [0.1, 0.2, 0.3, 0.4, 0.0],
            "history": [{"m_j": [0.0, 0.0, 0.0, 0.0], "retrieval_trainable": 0}],
            "stop_label": 0,
            "teacher_stop_score": 0.2,
        },
        {
            "problem_id": "b",
            "split": "train",
            "m_t": [0.4, 0.3, 0.2, 0.1, 0.0],
            "history": [{"m_j": [0.0, 0.0, 0.0, 0.0], "retrieval_trainable": 0}],
            "stop_label": 1,
            "teacher_stop_score": 0.8,
        },
    ]
    artifact = fit_controller(records, config, tmp_path / "controller.json")
    assert len(artifact["selector"]["gate_w1"]) == 16 * 32
    assert not any(artifact["selector"]["gate_w1"])
    assert len(artifact["commit"]["stop_weight"]) == 4


def test_config_training_reads_gzip_and_writes_public_checkpoint(tmp_path: Path) -> None:
    root = tmp_path
    config_dir = root / "experiments/paper"
    records_path = root / "data/training/toy/offline.jsonl.gz"
    template_path = root / "checkpoints/toy/controller.json"
    config_dir.mkdir(parents=True)
    records_path.parent.mkdir(parents=True)
    template_path.parent.mkdir(parents=True)
    template = {
        "config": {"training": {"warmup_epochs": 1, "onpolicy_refine_steps": 0}},
        "selector": {"gate_hidden_dim": 32},
        "commit": {},
    }
    template_path.write_text(json.dumps(template), encoding="utf-8")
    records = []
    for index in range(4):
        records.append(
            {
                "problem_id": str(index),
                "split": "train",
                "traj_correct": bool(index % 2),
                "m_t": [float(index), 0.2, -0.1, 0.4, 0.0],
                "history": [
                    {
                        "m_j": [0.0, 0.1, -0.2, 0.3],
                        "retrieval_trainable": 1,
                        "gate_hard_label": index % 2,
                        "gate_soft_label": 0.8 if index % 2 else 0.2,
                    }
                ],
                "stop_label": index % 2,
                "teacher_stop_score": 0.8 if index % 2 else 0.2,
                "step_len_tokens": 8,
            }
        )
    with gzip.open(records_path, "wt", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row) + "\n")
    config_path = config_dir / "toy.yaml"
    config_path.write_text(
        "\n".join(
            (
                "release_id: toy",
                "training:",
                "  template_checkpoint: checkpoints/toy/controller.json",
                "  offline_records: data/training/toy/offline.jsonl.gz",
                "  output_artifact: outputs/toy/controller.json",
                "  output_checkpoint: outputs/toy/controller.pt",
            )
        ),
        encoding="utf-8",
    )
    output = train_from_config(config_path)
    checkpoint = load_checkpoint(output)
    assert checkpoint.release_id == "toy-refit"
    assert checkpoint.parameter_count == 582
