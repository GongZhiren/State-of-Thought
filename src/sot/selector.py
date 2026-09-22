from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import List, Sequence

import torch
from torch import Tensor

from .config import SelectorConfig
from .reasoning_bank import ReasoningUnit

_M_DIM = 4


@dataclass
class HistoricalScore:
    unit_id: int
    logit: float
    keep: bool
    unit_kind: str
    gate_prob: float = 0.0
    descriptor: List[float] | None = None


GEOM_READ_FEATURE_NAMES = [
    "m_t_0",
    "m_t_1",
    "m_t_2",
    "m_t_3",
    "m_j_0",
    "m_j_1",
    "m_j_2",
    "m_j_3",
    "delta_0",
    "delta_1",
    "delta_2",
    "delta_3",
    "prod_0",
    "prod_1",
    "prod_2",
    "prod_3",
]


# Runtime-safe context features appended in the "geom_ctx3" mode (the gate-refit fix). All three are
# computable mid-trajectory with NO future information (unlike the offline recency_prior, which leaks
# traj_step_count): age_inv=1/max(age,1); age_ratio=age/(cur_step+age+7); sim=0.5*(cos(z_t,z_j)+1).
CTX3_READ_FEATURE_NAMES = ["ctx_age_inv", "ctx_age_ratio", "ctx_sim"]


def get_read_feature_mode(mode: str) -> str:
    m = str(mode or "").strip().lower()
    if m in {"", "geom5d"}:
        return "geom5d"
    if m in {"geom_ctx3", "geom16_ctx3", "geomctx3"}:
        return "geom_ctx3"
    warnings.warn(
        f"[sot.selector] read_feature_mode={mode!r} is not supported; fallback to 'geom5d'.",
        RuntimeWarning,
        stacklevel=2,
    )
    return "geom5d"


def get_read_feature_names(mode: str) -> List[str]:
    m = get_read_feature_mode(mode)
    if m == "geom_ctx3":
        return list(GEOM_READ_FEATURE_NAMES) + list(CTX3_READ_FEATURE_NAMES)
    return list(GEOM_READ_FEATURE_NAMES)


def get_read_feature_dim(mode: str) -> int:
    return len(get_read_feature_names(mode))


def build_read_feature_vector(
    *,
    m_t: Tensor,
    m_j: Tensor,
    z_similarity: float,
    age_inv: float,
    age_ratio: float,
    in_recent_window: float,
    read_feature_mode: str = "geom5d",
) -> Tensor:
    mt = m_t.float().view(-1)[:_M_DIM]
    mj = m_j.float().view(-1)[:_M_DIM]
    if int(mt.numel()) < _M_DIM:
        mt = torch.cat([mt, torch.zeros((_M_DIM - int(mt.numel()),), dtype=mt.dtype, device=mt.device)], dim=0)
    if int(mj.numel()) < _M_DIM:
        mj = torch.cat([mj, torch.zeros((_M_DIM - int(mj.numel()),), dtype=mj.dtype, device=mj.device)], dim=0)
    delta = mt - mj
    prod = mt * mj
    parts = [mt, mj, delta, prod]
    if get_read_feature_mode(read_feature_mode) == "geom_ctx3":
        # Gate-refit fix: append runtime-safe recency + similarity context so the gate can integrate
        # internal-state geometry WITH positional/similarity signal (beats recency alone; see gate_refit).
        ctx = torch.tensor([float(age_inv), float(age_ratio), float(z_similarity)], dtype=mt.dtype, device=mt.device)
        parts.append(ctx)
    else:
        _ = (z_similarity, age_inv, age_ratio, in_recent_window)  # ignored in geom5d
    return torch.cat(parts, dim=0)


class HistoricalSelector:
    def __init__(self, cfg: SelectorConfig):
        self.cfg = cfg
        self.read_feature_mode = get_read_feature_mode(getattr(cfg, "read_feature_mode", "geom5d"))
        dim = get_read_feature_dim(self.read_feature_mode)
        w = list(cfg.read_weight)
        if not w:
            w = [0.0] * dim
        if len(w) < dim:
            w = w + [0.0] * (dim - len(w))
        if len(w) > dim:
            w = w[:dim]
        self.w = torch.tensor(w, dtype=torch.float32)
        self.b = float(cfg.read_bias)
        hidden_dim = max(0, int(getattr(cfg, "gate_hidden_dim", 0)))
        self.h_dim = hidden_dim
        self.use_mlp = False
        self.g_w1 = torch.zeros((hidden_dim, dim), dtype=torch.float32)
        self.g_b1 = torch.zeros((hidden_dim,), dtype=torch.float32)
        self.g_w2 = torch.zeros((hidden_dim,), dtype=torch.float32)
        self.g_b2 = float(getattr(cfg, "gate_b2", 0.0))
        self.attention_mix = max(0.0, min(1.0, float(getattr(cfg, "attention_mix", 0.7))))
        self.attention_temperature = max(1e-4, float(getattr(cfg, "attention_temperature", 0.8)))
        raw_w1 = list(getattr(cfg, "gate_w1", ()))
        raw_b1 = list(getattr(cfg, "gate_b1", ()))
        raw_w2 = list(getattr(cfg, "gate_w2", ()))
        legacy_dim = 20
        if len(w) == legacy_dim and dim == 16:
            # Legacy read_weight was built on 5-D m (20 features). Drop dim-4 groups.
            keep = [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18]
            w = [w[i] for i in keep]
            self.w = torch.tensor(w, dtype=torch.float32)
        if hidden_dim > 0 and len(raw_w1) == hidden_dim * legacy_dim and dim == 16:
            keep = [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18]
            w1_old = torch.tensor(raw_w1, dtype=torch.float32).view(hidden_dim, legacy_dim)
            self.g_w1 = w1_old[:, keep]
            if len(raw_b1) == hidden_dim and len(raw_w2) == hidden_dim:
                self.g_b1 = torch.tensor(raw_b1, dtype=torch.float32).view(hidden_dim)
                self.g_w2 = torch.tensor(raw_w2, dtype=torch.float32).view(hidden_dim)
                self.use_mlp = True
            return
        if hidden_dim > 0 and len(raw_w1) == hidden_dim * dim and len(raw_b1) == hidden_dim and len(raw_w2) == hidden_dim:
            self.g_w1 = torch.tensor(raw_w1, dtype=torch.float32).view(hidden_dim, dim)
            self.g_b1 = torch.tensor(raw_b1, dtype=torch.float32).view(hidden_dim)
            self.g_w2 = torch.tensor(raw_w2, dtype=torch.float32).view(hidden_dim)
            self.use_mlp = True

    def _gate_logit(self, d: Tensor) -> float:
        if self.use_mlp:
            h = torch.tanh(self.g_w1 @ d + self.g_b1)
            return float(torch.dot(self.g_w2, h).item() + self.g_b2)
        return float(torch.dot(self.w, d).item() + self.b)

    def _attention_logits(self, m_t: Tensor, candidates: Sequence[ReasoningUnit]) -> List[float]:
        if not candidates:
            return []
        q = m_t.float().view(-1)
        q = q / max(float(torch.norm(q).item()), 1e-6)
        out: List[float] = []
        for unit in candidates:
            k = unit.m.float().view(-1)
            k = k / max(float(torch.norm(k).item()), 1e-6)
            out.append(float(torch.dot(q, k).item()) / self.attention_temperature)
        return out

    def descriptor(self, current_unit: ReasoningUnit, u_j: ReasoningUnit) -> Tensor:
        age_inv = age_ratio = sim = 0.0
        if self.read_feature_mode == "geom_ctx3":
            # runtime-safe context (no future info): step distance + z-cosine similarity
            age = max(1, int(current_unit.unit_id) - int(u_j.unit_id))
            cur = int(current_unit.unit_id)
            age_inv = 1.0 / float(max(age, 1))
            age_ratio = float(age) / float(max(1, cur + age + 7))
            zt = current_unit.z.float().view(-1)
            zj = u_j.z.float().view(-1)
            n = min(int(zt.numel()), int(zj.numel()))
            if n > 0:
                a = zt[:n]
                b = zj[:n]
                na = float(torch.norm(a).item())
                nb = float(torch.norm(b).item())
                cos_z = float(torch.dot(a, b).item() / (na * nb)) if na > 1e-8 and nb > 1e-8 else 0.0
            else:
                cos_z = 0.0
            sim = 0.5 * (cos_z + 1.0)
        return build_read_feature_vector(
            m_t=current_unit.m,
            m_j=u_j.m,
            z_similarity=sim,
            age_inv=age_inv,
            age_ratio=age_ratio,
            in_recent_window=0.0,
            read_feature_mode=self.read_feature_mode,
        )

    def score_candidates(
        self,
        *,
        current_unit: ReasoningUnit,
        candidates: Sequence[ReasoningUnit],
    ) -> List[HistoricalScore]:
        """Single-stage binary memory activation over full history candidates."""
        out: List[HistoricalScore] = []
        logits: List[float] = []
        attn_logits = self._attention_logits(current_unit.m, candidates)
        descs: List[List[float]] = []
        p_thr = float(getattr(self.cfg, "gate_prob_threshold", 0.5))
        read_thr = float(self.cfg.read_threshold)
        for unit in candidates:
            d = self.descriptor(current_unit, unit)
            logit = self._gate_logit(d)
            logits.append(float(logit))
            descs.append([float(x) for x in d.tolist()])
        if logits:
            mu = sum(logits) / len(logits)
            var = sum((x - mu) ** 2 for x in logits) / len(logits)
            std = math.sqrt(max(var, 1e-8))
        else:
            mu, std = 0.0, 1.0
        if attn_logits:
            mu_a = sum(attn_logits) / len(attn_logits)
            var_a = sum((x - mu_a) ** 2 for x in attn_logits) / len(attn_logits)
            std_a = math.sqrt(max(var_a, 1e-8))
        else:
            mu_a, std_a = 0.0, 1.0
        exp_attn = [math.exp(max(min(v, 20.0), -20.0)) for v in attn_logits]
        attn_zsum = sum(exp_attn) if exp_attn else 1.0
        n = max(len(logits), 1)
        for unit, logit, attn_logit, dvals in zip(candidates, logits, attn_logits, descs):
            # Single-stage global scoring: base gate evidence + attention salience over full history.
            norm_logit = (float(logit) - mu) / std
            norm_attn = (float(attn_logit) - mu_a) / std_a
            sig = 1.0 / (1.0 + math.exp(-float(norm_logit)))
            attn_prob = math.exp(max(min(attn_logit, 20.0), -20.0)) / attn_zsum
            attn_density = min(1.0, attn_prob * n)
            gate_prob = max(0.0, min(1.0, (1.0 - self.attention_mix) * sig + self.attention_mix * attn_density))
            fused_logit = (1.0 - self.attention_mix) * norm_logit + self.attention_mix * norm_attn
            out.append(
                HistoricalScore(
                    unit_id=unit.unit_id,
                    logit=float(fused_logit),
                    keep=bool(gate_prob >= p_thr and fused_logit > read_thr),
                    unit_kind=unit.kind,
                    gate_prob=gate_prob,
                    descriptor=dvals,
                )
            )
        out.sort(key=lambda s: s.logit, reverse=True)
        return out

    def select(self, scores: Sequence[HistoricalScore]) -> List[HistoricalScore]:
        kept = [s for s in scores if s.keep]
        kept.sort(key=lambda s: s.logit, reverse=True)
        return kept
