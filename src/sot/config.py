from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


@dataclass
class NormalizationConfig:
    mu: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    sigma: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    eps: float = 1e-8


@dataclass
class CalibrationProtocolConfig:
    protocol: str = "certified_fast"
    certify_noninferiority_margin: float = 0.0
    certify_stop_delta: float = 0.06
    certify_gate_delta: float = 0.05
    nonregression_eval_anchor: bool = True


@dataclass
class DebtConfig:
    alpha: float = 0.9
    lambda_v: float = 1.0
    lambda_c: float = 1.0
    lambda_H: float = 1.0
    tau_v: float = 0.05
    e_max: float = 5.0


def normalize_retrieval_mode(mode: str) -> str:
    """Single-stage evidence activation is the only supported mode."""
    m = str(mode or "").strip().lower()
    if m in {"", "evidence_gate", "single_stage", "direct", "direct_m_gate", "m_evidence"}:
        return "evidence_gate"
    return "evidence_gate"


@dataclass
class SelectorConfig:
    read_threshold: float = 0.0
    retrieval_mode: str = "evidence_gate"
    gate_prob_threshold: float = 0.5
    # Legacy name kept for compatibility; runtime currently uses first 4 m-dimensions.
    read_feature_mode: str = "geom5d"
    # Legacy compatibility fields (kept inert for old run_eval.pyc readers).
    hist_top_k: int = -1
    min_hist_keep: int = 0
    working_window: int = 0
    age_clip: int = 64
    recall_top_m: int = 0
    fallback_when_empty: str = "none"
    summary_stride: int = 0
    summary_window: int = 0
    max_summary_units: int = 0
    promote_after_citations: int = 0
    selection_policy: str = "binary_gate"
    min_effective_gain: float = 0.0
    gain_cost_penalty: float = 0.0
    include_recent_trace: bool = False
    use_all_history_candidates: bool = True
    lambda_token: float = 0.0
    lambda_ctx: float = 0.0
    lambda_dense: float = 0.0
    gate_hidden_dim: int = 32
    gate_w1: Tuple[float, ...] = field(default_factory=tuple)  # len=gate_hidden_dim*read_feature_dim
    gate_b1: Tuple[float, ...] = field(default_factory=tuple)  # len=gate_hidden_dim
    gate_w2: Tuple[float, ...] = field(default_factory=tuple)  # len=gate_hidden_dim
    gate_b2: float = 0.0
    attention_mix: float = 0.7
    attention_temperature: float = 0.8
    read_weight: Tuple[float, ...] = field(default_factory=tuple)  # len=read_feature_dim
    read_bias: float = 0.0


@dataclass
class CommitConfig:
    # Legacy compatibility field; stop logic is controlled by stop_threshold + r_stop.
    # r_stop > 1: hysteresis (consecutive steps above threshold). r_stop=1 matches per-step stop head semantics.
    t_min: int = 0
    long_context_tmin_tokens: int = 512
    long_context_tmin_boost: int = 1
    r_stop: int = 1
    stop_threshold: float = 0.5
    # Runtime uses 4-D m; legacy 5-D weights are truncated during load.
    stop_weight: Tuple[float, ...] = field(default_factory=tuple)
    stop_bias: float = 0.0


@dataclass
class ContextConfig:
    context_budget_tokens: int = 1536


@dataclass
class InferenceConfig:
    # SoT outer step budget; raise with max_sentence_tokens to reduce max_steps-only rollouts.
    t_max_steps: int = 40
    # Per-step observe inner-loop cap (token-wise). 64 forces frequent hard cuts and hurts stop/teaching.
    max_sentence_tokens: int = 192
    min_sentence_tokens: int = 6
    sentence_char_boundary_min: int = 24
    fallback_step_token_cap: int = 192
    temperature: float = 0.0
    # Final answer after commit; align with strong CoT-style baselines (fair_eval cot uses 768).
    answer_commit_max_new_tokens: int = 768
    # Left-truncate text prompts before backbone forward in LLM harness (0 = unlimited). VLM harness
    # keeps cap=0 to avoid dropping vision tokens; this still documents intended headroom for parity YAML.
    observe_max_input_tokens: int = 8192
    # Consecutive degenerate steps before forcing answer_commit (avoid false positives on "." placeholders)
    degenerate_streak_limit: int = 8


@dataclass
class TrainingConfig:
    # lightweight replay-derived targets used by pipeline fitting
    keep_similarity_threshold: float = 0.62
    keep_debt_threshold: float = 0.75
    stop_correct_only: bool = True
    val_ratio: float = 0.1
    holdout_ratio: float = 0.1
    split_seed: int = 42
    teacher_future_horizon: int = 2
    teacher_counterfactual_horizon: int = 2
    fit_pos_weight_cap: float = 10.0
    # Stop supervision controls:
    # - correct trajectory: stop when confidence is high enough and progress is sufficiently late.
    # - incorrect trajectory: stop only when clearly stagnating at later steps.
    stop_min_step_index: int = 2
    stop_success_min_confidence: float = 0.58
    stop_success_min_progress: float = 0.50
    stop_failure_min_progress: float = 0.70
    stop_failure_min_stagnation: float = 0.55
    # Retrieval supervision should focus on long trajectories where memory selection actually matters.
    read_min_traj_steps: int = 6
    read_min_step_index: int = 3
    read_min_history_age: int = 4
    # Blend teacher targets toward counterfactual gain (0=relevance-only, 1=gain-only).
    read_teacher_gain_weight: float = 0.7
    gate_label_threshold: float = 0.5
    gate_dense_penalty: float = 0.02
    gate_l2_penalty: float = 1e-4
    warmup_epochs: int = 8
    warmup_lr: float = 1e-3
    onpolicy_refine_steps: int = 0
    onpolicy_batch_size: int = 512
    onpolicy_lr: float = 2e-4
    onpolicy_entropy_coef: float = 0.001
    onpolicy_value_coef: float = 0.5
    onpolicy_clip_grad: float = 1.0
    # Periodic read-head diagnostics during on-policy refine (subsample of train bundle).
    onpolicy_log_interval: int = 100
    onpolicy_log_sample_cap: int = 16384


@dataclass
class MoTConfig:
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    calibration_protocol: CalibrationProtocolConfig = field(default_factory=CalibrationProtocolConfig)
    debt: DebtConfig = field(default_factory=DebtConfig)
    selector: SelectorConfig = field(default_factory=SelectorConfig)
    commit: CommitConfig = field(default_factory=CommitConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def _tuple4(x: List[float], default: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    if not isinstance(x, list) or len(x) != 4:
        return default
    return (float(x[0]), float(x[1]), float(x[2]), float(x[3]))


def load_mot_config_dict(raw: Dict[str, Any]) -> MoTConfig:
    norm_raw = ((raw or {}).get("calibration") or {}).get("normalization") or {}
    calib_proto_raw = ((raw or {}).get("calibration") or {}).get("protocol") or {}
    calibration_protocol = CalibrationProtocolConfig(
        protocol=str(calib_proto_raw.get("name", calib_proto_raw.get("protocol", "certified_fast"))),
        certify_noninferiority_margin=float(calib_proto_raw.get("certify_noninferiority_margin", 0.0)),
        certify_stop_delta=float(calib_proto_raw.get("certify_stop_delta", 0.06)),
        certify_gate_delta=float(calib_proto_raw.get("certify_gate_delta", 0.05)),
        nonregression_eval_anchor=bool(calib_proto_raw.get("nonregression_eval_anchor", True)),
    )

    term_raw = (raw or {}).get("termination") or {}
    inf_raw = (raw or {}).get("inference") or {}
    mot_raw = (raw or {}).get("mot") or {}
    sot_raw = (raw or {}).get("sot") or {}
    selector_raw = sot_raw.get("selector") or {}
    commit_raw = sot_raw.get("commit") or {}
    context_raw = sot_raw.get("context") or {}
    training_raw = sot_raw.get("training") or {}

    normalization = NormalizationConfig(
        mu=_tuple4(norm_raw.get("mu"), (0.0, 0.0, 0.0, 0.0)),
        sigma=_tuple4(norm_raw.get("sigma"), (1.0, 1.0, 1.0, 1.0)),
        eps=float(norm_raw.get("eps", 1e-8)),
    )
    debt = DebtConfig(
        tau_v=float(term_raw.get("tau_v", 0.05)),
        alpha=float(mot_raw.get("debt_alpha", 0.9)),
        lambda_v=float(mot_raw.get("lambda_v", 1.0)),
        lambda_c=float(mot_raw.get("lambda_c", 1.0)),
        lambda_H=float(mot_raw.get("lambda_H", 1.0)),
        e_max=float(mot_raw.get("e_max", 5.0)),
    )

    requested_read_feature_mode = str(selector_raw.get("read_feature_mode", "geom5d"))
    if requested_read_feature_mode.strip().lower() not in {"", "geom5d", "geom_ctx3", "geom16_ctx3", "geomctx3"}:
        warnings.warn(
            f"[sot.config] selector.read_feature_mode={requested_read_feature_mode!r} is unsupported; "
            "fallback to 'geom5d'.",
            RuntimeWarning,
            stacklevel=2,
        )
    inert_keys = (
        "hist_top_k",
        "min_hist_keep",
        "working_window",
        "age_clip",
        "recall_top_m",
        "fallback_when_empty",
        "summary_stride",
        "summary_window",
        "max_summary_units",
        "promote_after_citations",
        "selection_policy",
        "min_effective_gain",
        "gain_cost_penalty",
        "include_recent_trace",
        "use_all_history_candidates",
        "lambda_token",
        "lambda_ctx",
        "lambda_dense",
    )
    configured_inert = [k for k in inert_keys if k in selector_raw]
    if configured_inert:
        warnings.warn(
            "[sot.config] selector contains compatibility fields not used by the current runtime: "
            + ", ".join(configured_inert),
            RuntimeWarning,
            stacklevel=2,
        )

    selector = SelectorConfig(
        read_threshold=float(selector_raw.get("read_threshold", 0.0)),
        retrieval_mode=normalize_retrieval_mode(str(selector_raw.get("retrieval_mode", "evidence_gate"))),
        gate_prob_threshold=float(selector_raw.get("gate_prob_threshold", 0.5)),
        read_feature_mode=(requested_read_feature_mode.strip().lower()
                           if requested_read_feature_mode.strip().lower() in {"geom_ctx3", "geom16_ctx3", "geomctx3"}
                           else "geom5d"),
        hist_top_k=int(selector_raw.get("hist_top_k", -1)),
        min_hist_keep=int(selector_raw.get("min_hist_keep", 0)),
        working_window=int(selector_raw.get("working_window", 0)),
        age_clip=int(selector_raw.get("age_clip", 64)),
        recall_top_m=int(selector_raw.get("recall_top_m", 0)),
        fallback_when_empty=str(selector_raw.get("fallback_when_empty", "none")),
        summary_stride=int(selector_raw.get("summary_stride", 0)),
        summary_window=int(selector_raw.get("summary_window", 0)),
        max_summary_units=int(selector_raw.get("max_summary_units", 0)),
        promote_after_citations=int(selector_raw.get("promote_after_citations", 0)),
        selection_policy=str(selector_raw.get("selection_policy", "binary_gate")),
        min_effective_gain=float(selector_raw.get("min_effective_gain", 0.0)),
        gain_cost_penalty=float(selector_raw.get("gain_cost_penalty", 0.0)),
        include_recent_trace=bool(selector_raw.get("include_recent_trace", False)),
        use_all_history_candidates=bool(selector_raw.get("use_all_history_candidates", True)),
        lambda_token=float(selector_raw.get("lambda_token", 0.0)),
        lambda_ctx=float(selector_raw.get("lambda_ctx", 0.0)),
        lambda_dense=float(selector_raw.get("lambda_dense", 0.0)),
        gate_hidden_dim=int(selector_raw.get("gate_hidden_dim", 32)),
        gate_w1=tuple(float(x) for x in selector_raw.get("gate_w1", [])),
        gate_b1=tuple(float(x) for x in selector_raw.get("gate_b1", [])),
        gate_w2=tuple(float(x) for x in selector_raw.get("gate_w2", [])),
        gate_b2=float(selector_raw.get("gate_b2", 0.0)),
        attention_mix=float(selector_raw.get("attention_mix", 0.7)),
        attention_temperature=float(selector_raw.get("attention_temperature", 0.8)),
        read_weight=tuple(float(x) for x in selector_raw.get("read_weight", [])),
        read_bias=float(selector_raw.get("read_bias", 0.0)),
    )
    commit = CommitConfig(
        t_min=int(commit_raw.get("t_min", term_raw.get("t_min", 0))),
        long_context_tmin_tokens=int(commit_raw.get("long_context_tmin_tokens", 512)),
        long_context_tmin_boost=int(commit_raw.get("long_context_tmin_boost", 1)),
        r_stop=int(commit_raw.get("r_stop", term_raw.get("r_stop", 1))),
        stop_threshold=float(commit_raw.get("stop_threshold", mot_raw.get("tau_e", 0.5))),
        stop_weight=tuple(float(x) for x in commit_raw.get("stop_weight", [])),
        stop_bias=float(commit_raw.get("stop_bias", 0.0)),
    )
    context = ContextConfig(
        context_budget_tokens=int(context_raw.get("context_budget_tokens", mot_raw.get("context_budget_tokens", 1536))),
    )
    training = TrainingConfig(
        keep_similarity_threshold=float(training_raw.get("keep_similarity_threshold", 0.62)),
        keep_debt_threshold=float(training_raw.get("keep_debt_threshold", 0.75)),
        stop_correct_only=bool(training_raw.get("stop_correct_only", True)),
        val_ratio=float(training_raw.get("val_ratio", 0.1)),
        holdout_ratio=float(training_raw.get("holdout_ratio", 0.1)),
        split_seed=int(training_raw.get("split_seed", 42)),
        teacher_future_horizon=int(training_raw.get("teacher_future_horizon", 2)),
        teacher_counterfactual_horizon=int(training_raw.get("teacher_counterfactual_horizon", 2)),
        fit_pos_weight_cap=float(training_raw.get("fit_pos_weight_cap", 10.0)),
        stop_min_step_index=int(training_raw.get("stop_min_step_index", 2)),
        stop_success_min_confidence=float(training_raw.get("stop_success_min_confidence", 0.58)),
        stop_success_min_progress=float(training_raw.get("stop_success_min_progress", 0.50)),
        stop_failure_min_progress=float(training_raw.get("stop_failure_min_progress", 0.70)),
        stop_failure_min_stagnation=float(training_raw.get("stop_failure_min_stagnation", 0.55)),
        read_min_traj_steps=int(training_raw.get("read_min_traj_steps", 6)),
        read_min_step_index=int(training_raw.get("read_min_step_index", 3)),
        read_min_history_age=int(training_raw.get("read_min_history_age", 4)),
        read_teacher_gain_weight=float(training_raw.get("read_teacher_gain_weight", 0.7)),
        gate_label_threshold=float(training_raw.get("gate_label_threshold", 0.5)),
        gate_dense_penalty=float(training_raw.get("gate_dense_penalty", 0.02)),
        gate_l2_penalty=float(training_raw.get("gate_l2_penalty", 1e-4)),
        warmup_epochs=int(training_raw.get("warmup_epochs", 8)),
        warmup_lr=float(training_raw.get("warmup_lr", 1e-3)),
        onpolicy_refine_steps=int(training_raw.get("onpolicy_refine_steps", 0)),
        onpolicy_batch_size=int(training_raw.get("onpolicy_batch_size", 512)),
        onpolicy_lr=float(training_raw.get("onpolicy_lr", 2e-4)),
        onpolicy_entropy_coef=float(training_raw.get("onpolicy_entropy_coef", 0.001)),
        onpolicy_value_coef=float(training_raw.get("onpolicy_value_coef", 0.5)),
        onpolicy_clip_grad=float(training_raw.get("onpolicy_clip_grad", 1.0)),
        onpolicy_log_interval=int(training_raw.get("onpolicy_log_interval", 100)),
        onpolicy_log_sample_cap=int(training_raw.get("onpolicy_log_sample_cap", 16384)),
    )
    _def_inf = InferenceConfig()
    inference = InferenceConfig(
        t_max_steps=int(inf_raw.get("T_max", _def_inf.t_max_steps)),
        max_sentence_tokens=int(inf_raw.get("max_sentence_tokens", _def_inf.max_sentence_tokens)),
        min_sentence_tokens=int(inf_raw.get("min_sentence_tokens", _def_inf.min_sentence_tokens)),
        sentence_char_boundary_min=int(mot_raw.get("sentence_char_boundary_min", _def_inf.sentence_char_boundary_min)),
        fallback_step_token_cap=int(mot_raw.get("fallback_step_token_cap", _def_inf.fallback_step_token_cap)),
        temperature=float(mot_raw.get("temperature", _def_inf.temperature)),
        answer_commit_max_new_tokens=(
            int(inf_raw["answer_max_new_tokens"])
            if inf_raw.get("answer_max_new_tokens") is not None
            else int(inf_raw["answer_commit_max_new_tokens"])
            if inf_raw.get("answer_commit_max_new_tokens") is not None
            else _def_inf.answer_commit_max_new_tokens
        ),
        degenerate_streak_limit=int(
            mot_raw.get("degenerate_streak_limit", inf_raw.get("degenerate_streak_limit", _def_inf.degenerate_streak_limit))
        ),
        observe_max_input_tokens=int(
            inf_raw.get(
                "observe_max_input_tokens",
                mot_raw.get("observe_max_input_tokens", _def_inf.observe_max_input_tokens),
            )
        ),
    )
    return MoTConfig(
        normalization=normalization,
        calibration_protocol=calibration_protocol,
        debt=debt,
        selector=selector,
        commit=commit,
        context=context,
        inference=inference,
        training=training,
    )


def inference_from_mot_yaml_file(yaml_path: str | Path) -> InferenceConfig:
    """Parse a repository SoT YAML and return its resolved ``InferenceConfig``."""
    p = Path(yaml_path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return load_mot_config_dict(raw).inference


def apply_mot_inference_yaml_overlay(cfg: MoTConfig, yaml_path: str | Path) -> MoTConfig:
    """Refresh generation/decoding limits from YAML while keeping the rest of ``cfg`` unchanged.

    Typical flow: ``cfg = load_mot_config_from_artifact_path(artifact)`` then overlay so an older
    artifact does not pin aggressive ``max_sentence_tokens`` / ``answer_commit_max_new_tokens``.
    """
    p = Path(yaml_path)
    if not p.is_file():
        return cfg
    return replace(cfg, inference=inference_from_mot_yaml_file(p))
