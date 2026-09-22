"""State-of-Thought multi-step inference orchestration."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch

from .commit import CommitDecision, CommitScorer
from .config import (
    CommitConfig,
    ContextConfig,
    DebtConfig,
    InferenceConfig,
    NormalizationConfig,
    SelectorConfig,
)
from .context_builder import assemble_context
from .mechanism_config import MechanismExperimentConfig
from .reasoning_bank import ReasoningBank
from .selector import HistoricalSelector
from .state_extractor import SoTState, extract_sot_state
from .training_free import TrainingFreeCommit, TrainingFreePolicy, TrainingFreeSelector


def _cosine_vec(a: torch.Tensor, b: torch.Tensor) -> float:
    a1 = a.detach().float().reshape(-1)
    b1 = b.detach().float().reshape(-1)
    n = min(int(a1.numel()), int(b1.numel()))
    if n <= 0:
        return 0.0
    a1 = a1[:n]
    b1 = b1[:n]
    na = float(torch.norm(a1).item())
    nb = float(torch.norm(b1).item())
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float((torch.dot(a1, b1) / (na * nb)).item())


@dataclass
class StepObservation:
    text: str
    token_ids: List[int]
    hidden: torch.Tensor
    logits: torch.Tensor


@dataclass
class StepTrace:
    step: int = 0
    p_stop: float = 0.0
    commit_logit: float = 0.0
    commit_threshold_logit: float = 0.0
    commit_crossed_threshold: bool = False
    allow_stop_now: bool = False
    commit_consecutive: int = 0
    degenerate_streak: int = 0
    candidate_pool_size: int = 0
    recalled_candidate_count: int = 0
    selected_history_count: int = 0
    selected_history_unit_ids: List[int] = field(default_factory=list)
    selected_history_kinds: List[str] = field(default_factory=list)
    selected_summary_count: int = 0
    active_memory_step: bool = False
    low_signal_recall: bool = False
    reused_selected_ratio: float = 0.0
    gate_density: float = 0.0
    history_scores_top: List[Dict[str, Any]] = field(default_factory=list)
    context_used_tokens: int = 0
    context_budget_tokens: int = 0
    context_unit_ids: List[int] = field(default_factory=list)
    context_truncated_unit_ids: List[int] = field(default_factory=list)
    mechanism_extras: Dict[str, Any] = field(default_factory=dict)
    m_controller: List[float] = field(default_factory=list)
    # len(obs.token_ids) for this outer step (baseline-style completion slice; not context_used_tokens delta).
    observe_completion_tokens: int = 0


@dataclass
class MotInferenceResult:
    final_answer: str
    """Sum of token ids emitted across observe-step forwards (excludes final answer commit)."""
    generated_tokens: int
    traces: List[StepTrace]
    stopped_by: str
    node_count: int
    """Token count of the final answer string under ``tokenizer_encode`` (commit phase)."""
    answer_completion_tokens: int = 0
    """Observe completions + answer commit (closer to baseline ``aggregated_completion_tokens``)."""
    aggregated_completion_tokens: int = 0
    """Maximum assembled context-token estimate across SoT reasoning steps."""
    prompt_context_tokens: int = 0


def _offline_ignore_commit_stop(mechanism: Optional[MechanismExperimentConfig]) -> bool:
    return bool(mechanism and getattr(mechanism, "offline_ignore_commit_stop", False))


def _offline_max_observe_tokens(mechanism: Optional[MechanismExperimentConfig]) -> int:
    v = int(getattr(mechanism, "offline_max_observe_tokens", 0) or 0)
    return max(0, v)


def _is_degenerate_text(text: str) -> bool:
    t = (text or "").strip()
    return t in ("", ".", "[empty step]")


def run_mot_inference(
    *,
    prompt: str,
    observe_step_fn: Callable[[str], StepObservation],
    answer_commit_fn: Callable[[str], str],
    tokenizer_encode: Callable[[str], List[int]],
    selector_cfg: SelectorConfig,
    commit_cfg: CommitConfig,
    context_cfg: ContextConfig,
    normalization: NormalizationConfig,
    debt: DebtConfig,
    inference_cfg: InferenceConfig,
    mechanism: Optional[MechanismExperimentConfig] = None,
    training_free_policy: Optional[TrainingFreePolicy] = None,
    dataset_name: str = "default",
    commit_mode: str = "fullchain",
) -> MotInferenceResult:
    bank = ReasoningBank(prompt_root=prompt)
    commit_tf: Optional[TrainingFreeCommit] = None
    if training_free_policy is not None:
        selector = TrainingFreeSelector(
            selector_cfg,
            training_free_policy,
            dataset=dataset_name,
        )
        commit_tf = TrainingFreeCommit(
            commit_cfg,
            training_free_policy,
            dataset=dataset_name,
        )
        commit_scorer: Optional[CommitScorer] = None
    else:
        selector = HistoricalSelector(selector_cfg)
        commit_scorer = CommitScorer(commit_cfg)

    traces: List[StepTrace] = []
    z_prev: Optional[torch.Tensor] = None
    z_prev_prev: Optional[torch.Tensor] = None
    prev_e = 0.0
    commit_consecutive = 0
    degenerate_streak = 0
    generated_tokens = 0
    context_text = ""
    prompt_len_tokens = len(tokenizer_encode(prompt))
    observe_tokens_cum = 0

    def _assemble_kw(si: int) -> Dict[str, Any]:
        if not mechanism:
            return {"unit_order_mode": "sorted", "shuffle_rng": None}
        mode = str(mechanism.context_unit_order or "sorted").strip().lower()
        rng = None
        if mode == "shuffled":
            rng = random.Random(int(mechanism.shuffle_seed) + int(si) * 10009 + 17)
        return {"unit_order_mode": mode, "shuffle_rng": rng}

    def _finalize_rollout(si: int, *, stopped_by: str) -> MotInferenceResult:
        kw_f = _assemble_kw(si)
        # The paper protocol commits from the complete trajectory. Evidence
        # activation still controls the iterative reasoning context; this
        # choice makes terminal answer generation independent of stop reason.
        effective_commit_mode = str(commit_mode or "fullchain").strip().lower()
        if effective_commit_mode in ("selected", "gate", "gated"):
            hist_units = bank.get_units_by_id([s.unit_id for s in selected])
        else:
            last_id = int(all_units[-1].unit_id) if all_units else -1
            hist_units = [u for u in all_units if int(u.unit_id) != last_id]
        final_ctx = assemble_context(
            prompt_root=prompt,
            recent_units=all_units[-1:],
            historical_units=hist_units,
            tokenizer_encode=tokenizer_encode,
            cfg=context_cfg,
            unit_order_mode=str(kw_f["unit_order_mode"]),
            shuffle_rng=kw_f["shuffle_rng"],
        )
        ans = answer_commit_fn(final_ctx.text)
        ans_plain = str(ans or "").strip()
        ans_tok = len(tokenizer_encode(ans_plain))
        obs_tok = int(generated_tokens)
        ctx_tok = max((t.context_used_tokens for t in traces), default=0)
        return MotInferenceResult(
            final_answer=ans_plain,
            generated_tokens=obs_tok,
            traces=traces,
            stopped_by=stopped_by,
            node_count=si + 1,
            answer_completion_tokens=int(ans_tok),
            aggregated_completion_tokens=int(obs_tok + ans_tok),
            prompt_context_tokens=int(ctx_tok),
        )

    for step_idx in range(int(inference_cfg.t_max_steps)):
        obs = observe_step_fn(context_text)
        observe_tokens_cum += int(len(obs.token_ids))
        generated_tokens += len(obs.token_ids)

        h_step = obs.hidden
        if mechanism and float(mechanism.hidden_noise_std) > 0.0:
            h_step = h_step + torch.randn_like(h_step, dtype=h_step.dtype, device=h_step.device) * float(
                mechanism.hidden_noise_std
            )

        use_embed = bool(
            mechanism is not None
            and str(mechanism.sot_signal_mode or "hidden") == "openai_embed"
        )
        if use_embed:
            if not mechanism or not str(mechanism.embed_pca_path or "").strip():
                raise RuntimeError("embedding SoT requires an explicit PCA artifact")
            from .embedding_sot import get_embed_runtime

            sot = get_embed_runtime(
                mechanism.embed_pca_path,
                mechanism.openai_embed_model,
                h_source=mechanism.embed_h_source,
                m_source=mechanism.embed_m_source,
            ).to_sot_state(
                sentence=str(obs.text or ""),
                logits_t=obs.logits,
                z_prev=z_prev,
                z_prev_prev=z_prev_prev,
                prev_e=prev_e,
                norm_cfg=normalization,
                debt_cfg=debt,
            )
        else:
            sot = extract_sot_state(
                hidden=h_step,
                logits=obs.logits,
                z_prev=z_prev,
                z_prev_prev=z_prev_prev,
                prev_e=prev_e,
                norm_cfg=normalization,
                debt_cfg=debt,
            )
        if mechanism and mechanism.m_replay is not None and step_idx < len(mechanism.m_replay):
            mr = mechanism.m_replay[step_idx].to(device=sot.m.device, dtype=sot.m.dtype)
            sot = SoTState(z=sot.z, signature=sot.signature, m=mr, e=sot.e, e_bar=sot.e_bar)
        if mechanism and float(mechanism.m_noise_std) > 0.0:
            m2 = sot.m + torch.randn_like(sot.m) * float(mechanism.m_noise_std)
            sot = SoTState(z=sot.z, signature=sot.signature, m=m2, e=sot.e, e_bar=sot.e_bar)
        z_prev_prev = z_prev
        z_prev = sot.z
        prev_e = sot.e

        unit = bank.add_unit(obs.text, sot.z, sot.m)
        all_units = bank.units
        candidates = [u for u in all_units if u.unit_id != unit.unit_id]
        scores = selector.score_candidates(current_unit=unit, candidates=candidates)
        selected = selector.select(scores)
        if mechanism and getattr(mechanism, "offline_collect_snapshots", False) and bool(
            getattr(mechanism, "offline_force_full_history", False)
        ):
            # Collection-time full-history replay: keep rollout behavior close to natural CoT
            # (no sparse gating intervention before training), but still record score snapshots.
            selected = [
                s
                for s in scores
                if (
                    hasattr(s, "unit_id")
                    and isinstance(getattr(s, "unit_id", None), int)
                    and int(getattr(s, "unit_id", 0)) != int(unit.unit_id)
                )
            ]

        pre_cited = 0
        for s in selected:
            us = bank.get_units_by_id([s.unit_id])
            if us and us[0].citation_count > 0:
                pre_cited += 1
        reused_ratio = (pre_cited / len(selected)) if selected else 0.0
        selected_unit_ids = [int(s.unit_id) for s in selected]
        selected_units = bank.get_units_by_id(selected_unit_ids)
        bank.note_selected(selected_unit_ids)

        top_hs: List[Dict[str, Any]] = []
        for s in selected[:8]:
            top_hs.append({"unit_id": int(s.unit_id), "gate_prob": float(s.gate_prob), "logit": float(s.logit)})
        gate_density = sum(float(s.gate_prob) for s in selected) / max(len(selected), 1) if selected else 0.0

        kw_asm = _assemble_kw(step_idx)
        ctx = assemble_context(
            prompt_root=prompt,
            recent_units=[unit],
            historical_units=bank.get_units_by_id([s.unit_id for s in selected]),
            tokenizer_encode=tokenizer_encode,
            cfg=context_cfg,
            unit_order_mode=str(kw_asm["unit_order_mode"]),
            shuffle_rng=kw_asm["shuffle_rng"],
        )
        context_text = ctx.text

        if commit_tf is not None:
            p_stop, commit_logit, commit_consecutive, should_commit = commit_tf.update(
                signature=sot.signature,
                consecutive=commit_consecutive,
                m_t=sot.m,
            )
            commit_decision = CommitDecision(
                p_stop=p_stop,
                should_commit=should_commit,
                consecutive=commit_consecutive,
            )
            commit_thr_logit = commit_tf.threshold_logit
        else:
            assert commit_scorer is not None
            commit_logit = float(commit_scorer.raw_logit(sot.m))
            commit_thr_logit = float(commit_scorer.threshold_logit)
            commit_decision = commit_scorer.update(
                m_t=sot.m,
                consecutive=commit_consecutive,
            )
            commit_consecutive = commit_decision.consecutive

        if _is_degenerate_text(obs.text):
            degenerate_streak += 1
        else:
            degenerate_streak = 0

        low_signal = len(selected) == 0 and len(candidates) > 0

        mech_ex: Dict[str, Any] = {}
        if mechanism and mechanism.log_all_candidate_scores:
            mech_ex["candidates"] = [
                {
                    "unit_id": int(s.unit_id),
                    "gate_prob": float(s.gate_prob),
                    "logit": float(s.logit),
                    "keep": bool(s.keep),
                    "descriptor": [float(x) for x in (s.descriptor or [])],
                }
                for s in scores
            ]

        if mechanism and getattr(mechanism, "offline_collect_snapshots", False):
            snap_cands: List[Dict[str, Any]] = []
            for s in scores:
                us2 = bank.get_units_by_id([int(s.unit_id)])
                if not us2:
                    continue
                uj = us2[0]
                mj = [float(x) for x in uj.m.detach().float().cpu().view(-1).tolist()]
                cos_z = _cosine_vec(sot.z, uj.z)
                cos_m = _cosine_vec(sot.m, uj.m)
                snap_cands.append(
                    {
                        "unit_id": int(s.unit_id),
                        "step_j": int(max(0, int(uj.sort_key) - 1)),
                        "loop_idx_j": int(uj.unit_id) - 1,
                        "m_j": mj,
                        "gate_prob": float(s.gate_prob),
                        "logit": float(s.logit),
                        "keep": bool(s.keep),
                        "cos_z": float(cos_z),
                        "cos_m": float(cos_m),
                        "text_stub": str(uj.text or "")[:400],
                        "unit_kind": str(uj.kind or "step"),
                    }
                )
            mech_ex["offline_snapshot"] = {
                "loop_idx": int(step_idx),
                "trace_step": int(step_idx + 1),
                "m_t": [float(x) for x in sot.m.detach().float().cpu().view(-1).tolist()],
                "candidates": snap_cands,
            }

        m_snap = [float(x) for x in sot.m.detach().float().cpu().view(-1).tolist()]
        traces.append(
            StepTrace(
                step=step_idx + 1,
                p_stop=float(commit_decision.p_stop),
                commit_logit=float(commit_logit),
                commit_threshold_logit=float(commit_thr_logit),
                commit_crossed_threshold=bool(commit_logit > commit_thr_logit),
                degenerate_streak=int(degenerate_streak),
                candidate_pool_size=len(candidates),
                recalled_candidate_count=len(selected),
                selected_history_count=len(selected),
                selected_history_unit_ids=selected_unit_ids,
                selected_history_kinds=[str(u.kind) for u in selected_units],
                selected_summary_count=0,
                active_memory_step=bool(len(selected) > 0),
                low_signal_recall=bool(low_signal),
                reused_selected_ratio=float(reused_ratio),
                gate_density=float(gate_density),
                history_scores_top=top_hs,
                context_used_tokens=int(ctx.used_tokens),
                context_budget_tokens=int(ctx.budget_tokens),
                context_unit_ids=[int(x) for x in ctx.included_unit_ids],
                context_truncated_unit_ids=[int(x) for x in ctx.truncated_unit_ids],
                mechanism_extras=mech_ex,
                m_controller=m_snap,
                observe_completion_tokens=int(len(obs.token_ids)),
            )
        )

        force_deg = degenerate_streak >= int(inference_cfg.degenerate_streak_limit)
        min_step_gate = int(getattr(commit_cfg, "t_min", 0) or 0)
        if obs.token_ids:
            min_tokens = int(getattr(commit_cfg, "long_context_tmin_tokens", 0) or 0)
            if min_tokens > 0 and prompt_len_tokens >= min_tokens:
                min_step_gate += int(getattr(commit_cfg, "long_context_tmin_boost", 0) or 0)
        allow_stop_now = (step_idx + 1) >= max(0, min_step_gate)
        traces[-1].allow_stop_now = bool(allow_stop_now)
        traces[-1].commit_consecutive = int(commit_consecutive)
        effective_commit = bool(commit_decision.should_commit)
        if _offline_ignore_commit_stop(mechanism):
            effective_commit = False

        if force_deg:
            return _finalize_rollout(step_idx, stopped_by="degenerate")

        cap = _offline_max_observe_tokens(mechanism)
        if cap > 0:
            worst = int(inference_cfg.t_max_steps) * max(1, int(inference_cfg.max_sentence_tokens))
            if worst > 0:
                # Otherwise a loose env default (e.g. 8192) never beats max_steps when T_max=40 and cap=192.
                cap = min(int(cap), max(1, worst - 1))
        if cap > 0 and observe_tokens_cum >= cap:
            return _finalize_rollout(step_idx, stopped_by="observe_token_cap")

        if allow_stop_now and effective_commit:
            return _finalize_rollout(step_idx, stopped_by="stop")

    if bank.units:
        kw_t = _assemble_kw(len(bank.units) - 1)
        tail_ctx = assemble_context(
            prompt_root=prompt,
            recent_units=[bank.units[-1]],
            historical_units=bank.units[:-1],
            tokenizer_encode=tokenizer_encode,
            cfg=context_cfg,
            unit_order_mode=str(kw_t["unit_order_mode"]),
            shuffle_rng=kw_t["shuffle_rng"],
        )
        ans = answer_commit_fn(tail_ctx.text)
    else:
        root_ctx = assemble_context(
            prompt_root=prompt,
            recent_units=[],
            historical_units=[],
            tokenizer_encode=tokenizer_encode,
            cfg=context_cfg,
            unit_order_mode="sorted",
            shuffle_rng=None,
        )
        ans = answer_commit_fn(root_ctx.text)
    ans_plain = str(ans or "").strip()
    ans_tok = len(tokenizer_encode(ans_plain))
    obs_tok = int(generated_tokens)
    ctx_tok = max((t.context_used_tokens for t in traces), default=0)
    return MotInferenceResult(
        final_answer=ans_plain,
        generated_tokens=obs_tok,
        traces=traces,
        stopped_by="max_steps",
        node_count=len(bank.units),
        answer_completion_tokens=int(ans_tok),
        aggregated_completion_tokens=int(obs_tok + ans_tok),
        prompt_context_tokens=int(ctx_tok),
    )


__all__ = [
    "MotInferenceResult",
    "StepObservation",
    "StepTrace",
    "run_mot_inference",
]
