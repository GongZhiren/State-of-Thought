from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor

from .config import DebtConfig, NormalizationConfig


@dataclass
class LogicSignature:
    delta: float
    v: float
    c: float
    H: float

    def as_tensor(self, device: Optional[torch.device] = None) -> Tensor:
        t = torch.tensor([self.delta, self.v, self.c, self.H], dtype=torch.float32)
        return t.to(device) if device is not None else t


@dataclass
class LogicState:
    signature: LogicSignature
    hat_s: Tensor  # [4]
    e: float
    e_bar: float
    m: Tensor      # [4]


def compute_sentence_center(h: Tensor) -> Tensor:
    if h.dim() != 2:
        raise ValueError("h must have shape [n_t, D]")
    return h.mean(dim=0)


def compute_sentence_dispersion(h: Tensor, z: Tensor) -> float:
    return float((h.pow(2).sum(dim=1).mean() - z.pow(2).sum()).item())


def compute_entropy(logits: Tensor) -> float:
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    token_ent = -(probs * log_probs).sum(dim=-1)
    return float(token_ent.mean().item())


def compute_logic_signature(
    z_t: Tensor,
    z_prev: Optional[Tensor],
    z_prev_prev: Optional[Tensor],
    h_t: Tensor,
    logits_t: Tensor,
    eps: float = 1e-8,
) -> LogicSignature:
    delta = compute_sentence_dispersion(h_t, z_t)
    H = compute_entropy(logits_t)
    if z_prev is None:
        return LogicSignature(delta=delta, v=0.0, c=1.0, H=H)
    dz = z_t - z_prev
    v = float(dz.norm(p=2).item())
    if z_prev_prev is None:
        return LogicSignature(delta=delta, v=v, c=1.0, H=H)
    dz_prev = z_prev - z_prev_prev
    num = float(torch.dot(dz, dz_prev).item())
    den = float(dz.norm(p=2).item() * dz_prev.norm(p=2).item() + eps)
    c = num / den if den > 0 else 1.0
    return LogicSignature(delta=delta, v=v, c=c, H=H)


def normalize_signature(sig: LogicSignature, cfg: NormalizationConfig, device: Optional[torch.device] = None) -> Tensor:
    s = sig.as_tensor(device=device)
    mu = torch.tensor(cfg.mu, dtype=torch.float32, device=s.device)
    sigma = torch.tensor(cfg.sigma, dtype=torch.float32, device=s.device)
    return (s - mu) / (sigma + float(cfg.eps))


def low_progress(v: float, tau_v: float, eps: float = 1e-8) -> float:
    return max(0.0, (tau_v - v) / (tau_v + eps))


def turning(c: float) -> float:
    return max(0.0, -c)


def update_debt(prev_e: float, sig: LogicSignature, cfg: DebtConfig) -> Tuple[float, float]:
    u_t = cfg.lambda_v * low_progress(sig.v, cfg.tau_v) + cfg.lambda_c * turning(sig.c) + cfg.lambda_H * sig.H
    e_t = cfg.alpha * prev_e + (1.0 - cfg.alpha) * u_t
    e_bar = max(0.0, min(float(cfg.e_max), e_t)) / max(float(cfg.e_max), 1e-8)
    return e_t, e_bar


def build_logic_state(
    sig: LogicSignature,
    prev_e: float,
    norm_cfg: NormalizationConfig,
    debt_cfg: DebtConfig,
    device: Optional[torch.device] = None,
) -> LogicState:
    hat_s = normalize_signature(sig, norm_cfg, device=device)
    e, e_bar = update_debt(prev_e, sig, debt_cfg)
    # Runtime state keeps only the 4 normalized logic-signature dimensions.
    # e_bar is still computed and logged, but no longer appended into m.
    m = hat_s
    return LogicState(signature=sig, hat_s=hat_s, e=e, e_bar=e_bar, m=m)
