"""Optional interventions for mechanism / paper experiments (does not affect default SoT)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class MechanismExperimentConfig:
    """All fields off by default — identical to legacy behavior."""

    # assemble_context: how to order reasoning units before packing into the prompt
    context_unit_order: str = "sorted"  # sorted | shuffled | reversed
    shuffle_seed: int = 42

    # Additive noise before state → selector (controller path)
    m_noise_std: float = 0.0

    # Noise on sentence hidden states before extract_sot_state
    hidden_noise_std: float = 0.0

    # At step t, replace m with m_replay[t] if provided (cross-sample / counterfactual)
    m_replay: Optional[List[torch.Tensor]] = None

    # Attach full candidate gate dump to StepTrace.mechanism_extras (can be large)
    log_all_candidate_scores: bool = False

    # Rich per-step snapshot (m_t, per-candidate m_j / gate / cosines) for open offline collectors
    # that must match LLM build_offline_logs.jsonl history supervision (not training-free).
    offline_collect_snapshots: bool = False

    # Offline collection only: bypass sparse selector decisions when assembling next-step context.
    # This keeps rollout close to plain CoT continuation while still logging per-candidate scores.
    offline_force_full_history: bool = False

    # Raw offline rollout: do not end early via CommitScorer (weights are usually empty / untrained).
    # Stop timing is delegated to HQ / teacher relabel — same philosophy as ``stop_label=0`` placeholders.
    offline_ignore_commit_stop: bool = False

    # After each observe, if cumulative observe completion tokens reach this budget, run one final
    # ``answer_commit_fn`` and stop (``stopped_by=observe_token_cap``). 0 = disabled.
    offline_max_observe_tokens: int = 0

    # Exploratory embedding-based SoT. The embedding service and PCA artifact
    # replace hidden-state sentence centers; the downstream four-coordinate
    # geometry and controller interface are unchanged.
    sot_signal_mode: str = "hidden"  # hidden | openai_embed
    embed_pca_path: str = ""
    openai_embed_model: str = "text-embedding-3-small"
    embed_h_source: str = "spectral"  # spectral | logits
    embed_m_source: str = "geometry4"  # geometry4 | raw_pca


__all__ = ["MechanismExperimentConfig"]
