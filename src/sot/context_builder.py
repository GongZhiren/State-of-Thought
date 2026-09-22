from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

from .config import ContextConfig
from .reasoning_bank import ReasoningUnit


@dataclass
class ContextBuildResult:
    text: str
    used_tokens: int
    budget_tokens: int
    included_unit_ids: List[int]
    truncated_unit_ids: List[int]


def assemble_context(
    *,
    prompt_root: str,
    recent_units: Sequence[ReasoningUnit],
    historical_units: Sequence[ReasoningUnit],
    tokenizer_encode: Callable[[str], List[int]],
    cfg: ContextConfig,
    unit_order_mode: str = "sorted",
    shuffle_rng: Optional[random.Random] = None,
) -> ContextBuildResult:
    units = list(recent_units) + list(historical_units)
    mode = str(unit_order_mode or "sorted").strip().lower()
    if mode == "sorted":
        units.sort(key=lambda u: (u.sort_key, u.unit_id))
    elif mode == "reversed":
        units.sort(key=lambda u: (u.sort_key, u.unit_id))
        units.reverse()
    elif mode == "shuffled":
        units.sort(key=lambda u: (u.sort_key, u.unit_id))
        rng = shuffle_rng if shuffle_rng is not None else random.Random(0)
        rng.shuffle(units)
    else:
        units.sort(key=lambda u: (u.sort_key, u.unit_id))
    chunks: List[str] = [prompt_root]
    used = len(tokenizer_encode(prompt_root))
    budget = max(64, int(cfg.context_budget_tokens))
    included_unit_ids: List[int] = []
    truncated_unit_ids: List[int] = []
    for unit in units:
        txt = unit.text
        n_tok = len(tokenizer_encode(txt))
        if used + n_tok > budget and len(chunks) > 1:
            truncated_unit_ids.append(int(unit.unit_id))
            continue
        chunks.append(txt)
        used += n_tok
        included_unit_ids.append(int(unit.unit_id))
        if used >= budget:
            break
    return ContextBuildResult(
        text="\n".join(chunks),
        used_tokens=used,
        budget_tokens=budget,
        included_unit_ids=included_unit_ids,
        truncated_unit_ids=truncated_unit_ids,
    )
