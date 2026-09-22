from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from torch import Tensor


@dataclass
class ReasoningUnit:
    unit_id: int
    text: str
    z: Tensor
    m: Tensor
    kind: str = "step"
    sort_key: int = 0
    citation_count: int = 0
    source_unit_ids: List[int] | None = None


class ReasoningBank:
    def __init__(self, prompt_root: str) -> None:
        self.prompt_root = prompt_root
        self._step_units: List[ReasoningUnit] = []
        self._memory_index: Dict[int, ReasoningUnit] = {}

    @property
    def units(self) -> List[ReasoningUnit]:
        return self._step_units

    def add_unit(self, text: str, z: Tensor, m: Tensor) -> ReasoningUnit:
        unit = ReasoningUnit(
            unit_id=len(self._step_units) + 1,
            text=text,
            z=z.detach().clone(),
            m=m.detach().clone(),
            kind="step",
            sort_key=len(self._step_units) + 1,
            source_unit_ids=[len(self._step_units) + 1],
        )
        self._step_units.append(unit)
        self._memory_index[unit.unit_id] = unit
        return unit

    def recent_trace(self, window_size: int) -> List[ReasoningUnit]:
        _ = window_size
        return []

    def historical_candidates(self, window_size: int) -> List[ReasoningUnit]:
        _ = window_size
        return list(self._step_units)

    def all_history_candidates(self) -> List[ReasoningUnit]:
        return list(self._step_units)

    def promoted_candidates(self) -> List[ReasoningUnit]:
        return []

    def summary_candidates(self) -> List[ReasoningUnit]:
        return []

    def candidate_pool(self, window_size: int, *, use_all_history: bool = False) -> List[ReasoningUnit]:
        _ = (window_size, use_all_history)
        return list(self._step_units)

    def note_selected(self, unit_ids: List[int]) -> None:
        for unit_id in unit_ids:
            unit = self._memory_index.get(int(unit_id))
            if unit is not None:
                unit.citation_count += 1

    def get_units_by_id(self, unit_ids: List[int]) -> List[ReasoningUnit]:
        out: List[ReasoningUnit] = []
        for unit_id in unit_ids:
            unit = self._memory_index.get(int(unit_id))
            if unit is not None:
                out.append(unit)
        return out

    def _maybe_add_summary(self) -> None:
        return
