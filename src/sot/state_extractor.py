from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from torch import Tensor

from .config import DebtConfig, NormalizationConfig
from .logic_state import (
    LogicSignature,
    build_logic_state,
    compute_logic_signature,
    compute_sentence_center,
)


@dataclass
class SoTState:
    z: Tensor
    signature: LogicSignature
    m: Tensor
    e: float
    e_bar: float


def extract_sot_state(
    *,
    hidden: Tensor,
    logits: Tensor,
    z_prev: Optional[Tensor],
    z_prev_prev: Optional[Tensor],
    prev_e: float,
    norm_cfg: NormalizationConfig,
    debt_cfg: DebtConfig,
) -> SoTState:
    z_t = compute_sentence_center(hidden)
    sig = compute_logic_signature(
        z_t=z_t,
        z_prev=z_prev,
        z_prev_prev=z_prev_prev,
        h_t=hidden,
        logits_t=logits,
    )
    state = build_logic_state(sig=sig, prev_e=prev_e, norm_cfg=norm_cfg, debt_cfg=debt_cfg)
    return SoTState(
        z=z_t,
        signature=sig,
        m=state.m,
        e=state.e,
        e_bar=state.e_bar,
    )
