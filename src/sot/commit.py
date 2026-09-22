from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from .config import CommitConfig


@dataclass
class CommitDecision:
    p_stop: float
    should_commit: bool
    consecutive: int


def logit_threshold(tau: float) -> float:
    """Inverse sigmoid: sigmoid(L) > tau  <=>  L > logit_threshold(tau). Clamped for tau near 0/1."""
    t = float(tau)
    eps = 1e-6
    if t <= eps:
        return -60.0
    if t >= 1.0 - eps:
        return 60.0
    return math.log(t / (1.0 - t))


def fuse_stop_threshold_into_bias(stop_bias: float, stop_threshold: float) -> float:
    """Return b' such that sigmoid(w·m + b') > 0.5  <=>  sigmoid(w·m + b) > stop_threshold (same decisions)."""
    return float(stop_bias) - logit_threshold(float(stop_threshold))


class CommitScorer:
    def __init__(self, cfg: CommitConfig):
        self.cfg = cfg
        self._feature_dim = 4
        w = list(cfg.stop_weight)
        if not w:
            w = [0.0] * self._feature_dim
        # Backward compatibility: old artifacts may carry 5-D weights (with e_bar at dim4).
        if len(w) == self._feature_dim + 1:
            w = w[: self._feature_dim]
        elif len(w) != self._feature_dim:
            raise ValueError(
                f"commit stop_weight dim mismatch: expected {self._feature_dim} (or legacy 5), got {len(w)}"
            )
        self.w = torch.tensor(w, dtype=torch.float32)
        self.b = float(cfg.stop_bias)
        tau = float(cfg.stop_threshold)
        # Logit-space threshold: identical to p_stop > stop_threshold but matches the
        # "absorb tau into bias" view (b_eff = b - logit(tau), compare to 0.5 in prob space).
        self._thr_logit = logit_threshold(tau)

    def raw_logit(self, m_t: Tensor) -> float:
        mt = m_t.float().view(-1)
        if int(mt.numel()) > self._feature_dim:
            mt = mt[: self._feature_dim]
        elif int(mt.numel()) < self._feature_dim:
            pad = torch.zeros((self._feature_dim - int(mt.numel()),), dtype=mt.dtype, device=mt.device)
            mt = torch.cat([mt, pad], dim=0)
        return float(torch.dot(self.w, mt).item() + self.b)

    @property
    def threshold_logit(self) -> float:
        return float(self._thr_logit)

    def score(self, m_t: Tensor) -> float:
        return float(torch.sigmoid(torch.tensor(self.raw_logit(m_t))).item())

    def update(self, *, m_t: Tensor, consecutive: int) -> CommitDecision:
        L = self.raw_logit(m_t)
        p_stop = float(torch.sigmoid(torch.tensor(L)).item())
        if L > self._thr_logit:
            consecutive += 1
        else:
            consecutive = 0
        should_commit = consecutive >= self.cfg.r_stop
        return CommitDecision(p_stop=p_stop, should_commit=should_commit, consecutive=consecutive)
