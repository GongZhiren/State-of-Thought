"""Training-free SoT policies over the fixed four-coordinate state space."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from .config import CommitConfig, SelectorConfig
from .logic_state import LogicSignature
from .reasoning_bank import ReasoningUnit
from .selector import HistoricalScore, build_read_feature_vector


def _v4(x: Tensor) -> Tensor:
    x = x.float().view(-1)
    if int(x.numel()) >= 4:
        return x[:4]
    if int(x.numel()) < 4:
        return torch.cat([x, torch.zeros((4 - int(x.numel()),), device=x.device, dtype=x.dtype)], dim=0)
    return x


def tf_evidence_score(m_t: Tensor, m_j: Tensor, alpha: Tuple[float, float, float, float]) -> float:
    """Compute the training-free score.

    ``s_tf = a1 <m_t,m_j> - a2 ||m_t-m_j|| + a3 c_j - a4 H_j``, where
    ``c_j`` and ``H_j`` are the final two normalized ``(delta, v, c, H)`` slots.
    """
    a1, a2, a3, a4 = (float(alpha[i]) for i in range(4))
    mt = _v4(m_t)
    mj = _v4(m_j)
    d = float(torch.dot(mt, mj).item())
    l2 = float(torch.norm(mj - mt, p=2).item())
    cj = float(mj[2].item()) if int(mj.numel()) > 2 else 0.0
    hj = float(mj[3].item()) if int(mj.numel()) > 3 else 0.0
    return a1 * d - a2 * l2 + a3 * cj - a4 * hj


@dataclass
class TrainingFreePolicy:
    """Global policy with optional dataset-specific overrides."""

    version: int = 1
    global_: Dict[str, Any] = field(
        default_factory=lambda: {
            "alpha": [1.0, 1.0, 1.0, 1.0],
            "gate_prob_threshold": 0.475,
            "gate_temp": 1.0,
            "r_stop": 1,
            "stop": {"tau_v": 0.2, "tau_H": 3.0, "tau_c": -0.5},
        }
    )
    per_dataset: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    notes: str = ""

    def resolve(self, dataset: str) -> Dict[str, Any]:
        ds = str(dataset or "").strip() or "default"
        row: Dict[str, Any] = dict(self.global_)
        ovr = self.per_dataset.get(ds) or self.per_dataset.get(str(ds).lower())
        if isinstance(ovr, dict):
            row = {**row, **{k: v for k, v in ovr.items() if v is not None}}
        return row

    @staticmethod
    def from_path(path: str | Path) -> "TrainingFreePolicy":
        p = Path(path)
        raw = json.loads(p.read_text(encoding="utf-8"))
        gd = raw.get("global") or raw.get("global_") or {}
        if not isinstance(gd, dict):
            gd = {}
        return TrainingFreePolicy(
            version=int(raw.get("version", 1)),
            global_=gd,
            per_dataset=raw.get("per_dataset") or {},
            notes=str(raw.get("notes") or ""),
        )


class TrainingFreeSelector:
    """Score prior steps with the fixed rule and a single-stage threshold."""

    def __init__(self, base_cfg: SelectorConfig, policy: TrainingFreePolicy, *, dataset: str) -> None:
        self._base = base_cfg
        self._policy = policy
        self._row = policy.resolve(dataset)
        self._alpha = tuple(float(x) for x in (self._row.get("alpha") or [1, 1, 1, 1])[:4])
        while len(self._alpha) < 4:
            self._alpha = self._alpha + (1.0,)
        self._gthr = float(self._row.get("gate_prob_threshold", getattr(base_cfg, "gate_prob_threshold", 0.5) or 0.5))
        self._temp = max(1e-3, float(self._row.get("gate_temp", 1.0) or 1.0))
        self._rthr = float(self._row.get("read_threshold", getattr(base_cfg, "read_threshold", 0.0) or 0.0))
        # The training-free variant uses no learned attention/gate mixture.
        self._attn_mix = 0.0

    def _scores_tf(self, m_t: Tensor, candidates: Sequence[ReasoningUnit]) -> List[float]:
        return [tf_evidence_score(m_t, u.m, self._alpha) for u in candidates]

    def score_candidates(
        self,
        *,
        current_unit: ReasoningUnit,
        candidates: Sequence[ReasoningUnit],
    ) -> List[HistoricalScore]:
        if not candidates:
            return []
        mt = current_unit.m
        raw_s = self._scores_tf(mt, candidates)
        mu = sum(raw_s) / len(raw_s)
        var = sum((x - mu) ** 2 for x in raw_s) / max(len(raw_s), 1)
        std = math.sqrt(max(var, 1e-8))
        logits: List[float] = []
        for s0 in raw_s:
            z = (float(s0) - mu) / std
            logits.append(float(z / self._temp))
        # Keep attention-scale diagnostics compatible with the learned selector.
        attn_logits: List[float] = []
        q = mt.float().view(-1)[:4]
        qn = max(float(torch.norm(q).item()), 1e-6)
        for u in candidates:
            k = u.m.float().view(-1)[:4]
            kn = max(float(torch.norm(k).item()), 1e-6)
            attn_logits.append(float(torch.dot(q, k).item() / (qn * kn) / max(0.1, self._base.attention_temperature)))
        p_thr = max(1e-3, min(0.999, self._gthr))
        n = max(len(logits), 1)
        mu2 = sum(logits) / n
        var2 = sum((x - mu2) ** 2 for x in logits) / n
        std2 = math.sqrt(max(var2, 1e-8))
        out: List[HistoricalScore] = []
        exp_attn = [math.exp(max(min(v, 20.0), -20.0)) for v in attn_logits]
        az = sum(exp_attn) if exp_attn else 1.0
        attn_mean = sum(attn_logits) / max(len(attn_logits), 1)
        for unit, logit, attn in zip(candidates, logits, attn_logits):
            norm_logit = (float(logit) - mu2) / std2
            sig = 1.0 / (1.0 + math.exp(-float(min(max(norm_logit, -20.0), 20.0))))
            attn_prob = math.exp(max(min(attn, 20.0), -20.0)) / az
            attn_density = min(1.0, attn_prob * n)
            gate_prob = max(0.0, min(1.0, (1.0 - self._attn_mix) * sig + self._attn_mix * attn_density))
            fused = (1.0 - self._attn_mix) * norm_logit + self._attn_mix * ((attn - attn_mean) / max(1.0, 0.1))
            d = build_read_feature_vector(
                m_t=mt,
                m_j=unit.m,
                z_similarity=0.0,
                age_inv=0.0,
                age_ratio=0.0,
                in_recent_window=0.0,
                read_feature_mode="geom5d",
            )
            out.append(
                HistoricalScore(
                    unit_id=unit.unit_id,
                    logit=float(fused),
                    keep=bool(gate_prob >= p_thr and float(fused) > self._rthr),
                    unit_kind=unit.kind,
                    gate_prob=gate_prob,
                    descriptor=[float(x) for x in d.tolist()],
                )
            )
        out.sort(key=lambda s: s.logit, reverse=True)
        return out

    def select(self, scores: Sequence[HistoricalScore]) -> List[HistoricalScore]:
        kept = [s for s in scores if s.keep]
        kept.sort(key=lambda s: s.logit, reverse=True)
        return kept

class TrainingFreeCommit:
    """Replace the learned stop head with a fixed stability criterion."""

    def __init__(self, base_cfg: CommitConfig, policy: TrainingFreePolicy, *, dataset: str) -> None:
        self._cfg = base_cfg
        self._row = policy.resolve(dataset)
        st = self._row.get("stop") or {}
        if not isinstance(st, dict):
            st = {}
        self.tau_v = float(st.get("tau_v", 0.2))
        self.tau_H = float(st.get("tau_H", 3.0))
        self.tau_c = float(st.get("tau_c", -1.0))
        self.r_stop = int(self._row.get("r_stop", base_cfg.r_stop) or 1)
        # p_stop is a trace-compatible proxy; the rule itself is deterministic.
        self._H_scale = max(0.1, float(st.get("H_scale", 1.0) or 1.0))

    def _stable_m(self, m_t: Tensor) -> bool:
        """Apply calibrated thresholds in normalized controller coordinates."""
        m = _v4(m_t)
        v = float(m[1].item()) if int(m.numel()) > 1 else 0.0
        c = float(m[2].item()) if int(m.numel()) > 2 else 0.0
        Hm = float(m[3].item()) if int(m.numel()) > 3 else 0.0
        return (v < self.tau_v) and (Hm < self.tau_H) and (c > self.tau_c)

    def _stable(self, sig: LogicSignature) -> bool:
        return (float(sig.v) < self.tau_v) and (float(sig.H) < self.tau_H) and (float(sig.c) > self.tau_c)

    @staticmethod
    def p_stop_proxy(sig: LogicSignature) -> float:
        """Map entropy to a bounded trace-compatible stop proxy."""
        h = max(0.0, min(10.0, float(sig.H)))
        return max(0.0, min(1.0, 1.0 - min(1.0, h / 5.0)))

    @staticmethod
    def p_stop_proxy_mH(H_m: float) -> float:
        """Map the calibrated entropy coordinate to a stop proxy."""
        h = float(H_m)
        if math.isnan(h) or math.isinf(h):
            h = 0.0
        h = max(0.0, min(10.0, h))
        return max(0.0, min(1.0, 1.0 - min(1.0, h / 5.0)))

    def update(
        self,
        *,
        signature: LogicSignature,
        consecutive: int,
        m_t: Optional[Tensor] = None,
    ) -> Tuple[float, float, int, bool]:
        """Return ``(p_stop, raw_logit, consecutive, should_commit)``."""
        if m_t is not None:
            st = self._stable_m(m_t)
            Hm = float(_v4(m_t)[3].item()) if int(m_t.numel()) > 3 else 0.0
            p = self.p_stop_proxy_mH(Hm) if st else 0.15
        else:
            st = self._stable(signature)
            p = self.p_stop_proxy(signature) if st else 0.15
        L = math.log(p / (1.0 - p) + 1e-6) if p < 0.99 else 4.0
        if st:
            consecutive += 1
        else:
            consecutive = 0
        sc = int(consecutive) >= int(self.r_stop)
        return float(p), float(L), int(consecutive), bool(sc)

    def raw_logit(
        self, signature: LogicSignature, *, m_t: Optional[Tensor] = None
    ) -> float:
        if m_t is not None:
            st = self._stable_m(m_t)
            Hm = float(_v4(m_t)[3].item()) if int(m_t.numel()) > 3 else 0.0
            p = self.p_stop_proxy_mH(Hm) if st else 0.15
        else:
            p = self.p_stop_proxy(signature) if self._stable(signature) else 0.15
        p = min(0.99, max(0.01, p))
        return float(math.log(p / (1.0 - p)))

    @property
    def threshold_logit(self) -> float:
        return 0.0


__all__ = [
    "TrainingFreeCommit",
    "TrainingFreePolicy",
    "TrainingFreeSelector",
    "tf_evidence_score",
]
