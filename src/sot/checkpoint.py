"""Portable, auditable SoT controller checkpoints.

The historical experiment artifact is JSON. The public ``.pt`` representation
contains only primitive metadata and tensors, so it can be loaded with
``weights_only=True`` on recent PyTorch versions. Two legacy serialization
quirks are normalized without changing runtime behavior:

* a zero selector matrix stored as one zero per hidden unit is expanded to the
  declared ``hidden_dim x 16`` zero matrix;
* a legacy fifth stop-head coefficient is removed because the runtime used for
  the reported experiments reads only the four-dimensional internal state.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

import torch

FORMAT_NAME = "state-of-thought-controller"
FORMAT_VERSION = 1
STATE_DIM = 4
SELECTOR_FEATURE_DIM = 16


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_float_list(value: Any, *, field: str) -> list[float]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be a list")
    out = [float(item) for item in value]
    if not all(math.isfinite(item) for item in out):
        raise ValueError(f"{field} contains a non-finite value")
    return out


def _effective_sections(artifact: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    config = artifact.get("config") or {}
    selector = dict(config.get("selector") or {})
    selector.update(artifact.get("selector") or {})
    commit = dict(config.get("commit") or {})
    commit.update(artifact.get("commit") or {})
    return selector, commit


def _normalized_tensors(artifact: Mapping[str, Any]) -> tuple[dict[str, torch.Tensor], list[str]]:
    selector, commit = _effective_sections(artifact)
    migrations: list[str] = []

    hidden_dim = int(selector.get("gate_hidden_dim", 32))
    if hidden_dim <= 0:
        raise ValueError("selector.gate_hidden_dim must be positive")

    gate_w1 = _as_float_list(selector.get("gate_w1", []), field="selector.gate_w1")
    expected_w1 = hidden_dim * SELECTOR_FEATURE_DIM
    if len(gate_w1) == hidden_dim and all(value == 0.0 for value in gate_w1):
        gate_w1 = [0.0] * expected_w1
        migrations.append("expanded_zero_selector_w1")
    if len(gate_w1) != expected_w1:
        raise ValueError(
            f"selector.gate_w1 has {len(gate_w1)} values; expected {expected_w1}"
        )

    gate_b1 = _as_float_list(selector.get("gate_b1", []), field="selector.gate_b1")
    gate_w2 = _as_float_list(selector.get("gate_w2", []), field="selector.gate_w2")
    if len(gate_b1) != hidden_dim or len(gate_w2) != hidden_dim:
        raise ValueError("selector hidden-layer vector shape mismatch")

    stop_weight = _as_float_list(commit.get("stop_weight", []), field="commit.stop_weight")
    if len(stop_weight) == STATE_DIM + 1:
        stop_weight = stop_weight[:STATE_DIM]
        migrations.append("removed_legacy_fifth_stop_weight")
    if len(stop_weight) != STATE_DIM:
        raise ValueError(f"commit.stop_weight has {len(stop_weight)} values; expected 4")

    tensors = {
        "selector.gate_w1": torch.tensor(gate_w1, dtype=torch.float32).reshape(
            hidden_dim, SELECTOR_FEATURE_DIM
        ),
        "selector.gate_b1": torch.tensor(gate_b1, dtype=torch.float32),
        "selector.gate_w2": torch.tensor(gate_w2, dtype=torch.float32),
        "selector.gate_b2": torch.tensor(float(selector.get("gate_b2", 0.0)), dtype=torch.float32),
        "commit.stop_weight": torch.tensor(stop_weight, dtype=torch.float32),
        "commit.stop_bias": torch.tensor(float(commit.get("stop_bias", 0.0)), dtype=torch.float32),
    }
    if not all(torch.isfinite(tensor).all().item() for tensor in tensors.values()):
        raise ValueError("checkpoint contains a non-finite tensor")
    return tensors, migrations


def _runtime_config(artifact: Mapping[str, Any]) -> dict[str, Any]:
    selector, commit = _effective_sections(artifact)
    config = artifact.get("config") or {}
    return {
        "normalization": dict(config.get("normalization") or {}),
        "debt": dict(config.get("debt") or {}),
        "selector": {
            "read_threshold": float(selector.get("read_threshold", 0.0)),
            "retrieval_mode": str(selector.get("retrieval_mode", "evidence_gate")),
            "gate_prob_threshold": float(selector.get("gate_prob_threshold", 0.5)),
            "read_feature_mode": str(selector.get("read_feature_mode", "geom5d")),
            "gate_hidden_dim": int(selector.get("gate_hidden_dim", 32)),
            # Keep exact JSON scalars as runtime metadata. The tensor copy is
            # retained for a conventional state_dict and parameter counting.
            "gate_b2": float(selector.get("gate_b2", 0.0)),
            "attention_mix": float(selector.get("attention_mix", 0.7)),
            "attention_temperature": float(selector.get("attention_temperature", 0.8)),
        },
        "commit": {
            "stop_threshold": float(commit.get("stop_threshold", 0.5)),
            "stop_bias": float(commit.get("stop_bias", 0.0)),
            "r_stop": int(commit.get("r_stop", 1)),
            "t_min": int(commit.get("t_min", 0)),
            "long_context_tmin_tokens": int(commit.get("long_context_tmin_tokens", 512)),
            "long_context_tmin_boost": int(commit.get("long_context_tmin_boost", 1)),
        },
        "context": dict(config.get("context") or {}),
        "inference": dict(config.get("inference") or {}),
        "training": dict(config.get("training") or {}),
        "calibration_protocol": dict(config.get("calibration_protocol") or {}),
    }


def build_payload(artifact_path: str | Path, *, release_id: str) -> dict[str, Any]:
    """Convert a historical JSON artifact into the public tensor payload."""
    path = Path(artifact_path)
    artifact = json.loads(path.read_text(encoding="utf-8"))
    tensors, migrations = _normalized_tensors(artifact)
    return {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "release_id": str(release_id),
        "source_json_sha256": _file_sha256(path),
        "state_dim": STATE_DIM,
        "selector_feature_dim": SELECTOR_FEATURE_DIM,
        "parameter_count": int(sum(tensor.numel() for tensor in tensors.values())),
        "migrations": migrations,
        "runtime_config": _runtime_config(artifact),
        "state_dict": tensors,
    }


def save_checkpoint(artifact_path: str | Path, output_path: str | Path, *, release_id: str) -> dict[str, Any]:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = build_payload(artifact_path, release_id=release_id)
    torch.save(payload, output)
    return payload


@dataclass(frozen=True)
class ControllerCheckpoint:
    release_id: str
    source_json_sha256: str
    parameter_count: int
    runtime_config: dict[str, Any]
    state_dict: dict[str, torch.Tensor]
    migrations: tuple[str, ...]


def load_checkpoint(path: str | Path) -> ControllerCheckpoint:
    source = Path(path)
    try:
        raw = torch.load(source, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch < 2.4
        raw = torch.load(source, map_location="cpu")
    if not isinstance(raw, dict):
        raise ValueError("checkpoint root must be a dictionary")
    if raw.get("format") != FORMAT_NAME or int(raw.get("format_version", -1)) != FORMAT_VERSION:
        raise ValueError("unsupported SoT checkpoint format")
    state_dict = raw.get("state_dict")
    if not isinstance(state_dict, dict) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in state_dict.items()
    ):
        raise ValueError("invalid state_dict")
    checkpoint = ControllerCheckpoint(
        release_id=str(raw.get("release_id", "")),
        source_json_sha256=str(raw.get("source_json_sha256", "")),
        parameter_count=int(raw.get("parameter_count", -1)),
        runtime_config=dict(raw.get("runtime_config") or {}),
        state_dict=state_dict,
        migrations=tuple(str(item) for item in (raw.get("migrations") or [])),
    )
    verify_checkpoint(checkpoint)
    return checkpoint


def verify_checkpoint(checkpoint: ControllerCheckpoint) -> None:
    required_shapes = {
        "selector.gate_w1": (32, 16),
        "selector.gate_b1": (32,),
        "selector.gate_w2": (32,),
        "selector.gate_b2": (),
        "commit.stop_weight": (4,),
        "commit.stop_bias": (),
    }
    missing = sorted(set(required_shapes) - set(checkpoint.state_dict))
    extra = sorted(set(checkpoint.state_dict) - set(required_shapes))
    if missing or extra:
        raise ValueError(f"state_dict keys mismatch; missing={missing}, extra={extra}")
    for key, expected in required_shapes.items():
        tensor = checkpoint.state_dict[key]
        if tuple(tensor.shape) != expected:
            raise ValueError(f"{key} shape {tuple(tensor.shape)} != {expected}")
        if tensor.dtype != torch.float32 or not torch.isfinite(tensor).all().item():
            raise ValueError(f"{key} must contain finite float32 values")
    actual_count = sum(tensor.numel() for tensor in checkpoint.state_dict.values())
    if actual_count != 582 or checkpoint.parameter_count != actual_count:
        raise ValueError(
            f"controller parameter count mismatch: metadata={checkpoint.parameter_count}, actual={actual_count}"
        )
    if len(checkpoint.source_json_sha256) != 64:
        raise ValueError("missing or malformed source JSON SHA-256")


def checkpoint_summary(path: str | Path) -> dict[str, Any]:
    checkpoint = load_checkpoint(path)
    return {
        "path": str(path),
        "sha256": _file_sha256(Path(path)),
        "release_id": checkpoint.release_id,
        "source_json_sha256": checkpoint.source_json_sha256,
        "parameter_count": checkpoint.parameter_count,
        "migrations": list(checkpoint.migrations),
    }
