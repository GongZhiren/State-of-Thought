"""Readable text-backbone runtime used by training and evaluation."""
from __future__ import annotations

from typing import Callable, List, Optional

import torch
from torch import Tensor

from sot.backbone.hf_backbone import HFGatedBackbone
from sot.chat_templates import build_prompt_and_tokenize
from sot.config import InferenceConfig, MoTConfig
from sot.dataset import DatasetSample
from sot.grading import extract_final_answer_text, is_correct
from sot.inference import MotInferenceResult, StepObservation, run_mot_inference
from sot.mechanism_config import MechanismExperimentConfig
from sot.prompts import build_user_root_text
from sot.step_boundary import find_step_boundaries, observe_inner_loop_should_stop
from sot.training_free import TrainingFreePolicy


def sample_next_token(logits: Tensor, *, temperature: float) -> int:
    """Greedy decode at zero temperature; otherwise sample a finite distribution."""
    values = torch.nan_to_num(logits.float().reshape(-1), nan=0.0, posinf=1e4, neginf=-1e4)
    if temperature <= 0.0:
        return int(values.argmax().item())
    scaled = torch.clamp(values / max(float(temperature), 1e-8), min=-80.0, max=80.0)
    probabilities = torch.softmax(scaled, dim=-1)
    probabilities = torch.nan_to_num(probabilities, nan=0.0).clamp(min=0.0)
    total = probabilities.sum()
    if not torch.isfinite(total) or float(total) <= 0.0:
        return int(values.argmax().item())
    return int(torch.multinomial(probabilities / total, num_samples=1).item())


def build_aligned_user_text(sample: DatasetSample) -> str:
    root = build_user_root_text(
        question=sample.question,
        task_type=str(sample.extra.get("task_type") or "math_qa"),
        choices=sample.choices,
    )
    return str(root["content"])


def _truncate_token_ids(seq: List[int], max_len: int) -> List[int]:
    if max_len <= 0 or len(seq) <= max_len:
        return seq
    return seq[-max_len:]


def _make_observe_step(
    *,
    backbone: HFGatedBackbone,
    tokenizer,
    model_path: str,
    device: torch.device,
    prompt_root: str,
    inf: InferenceConfig,
    hidden_size: int,
    vocab_size: int,
) -> Callable[[str], StepObservation]:
    icfg = inf
    cap = int(getattr(icfg, "observe_max_input_tokens", 0) or 0)

    def _observe(context_text: str) -> StepObservation:
        body = (context_text or "").strip()
        if not body:
            body = prompt_root
        _, input_ids = build_prompt_and_tokenize(
            tokenizer, model_path, body, return_tensors=True, device=device
        )
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        seq: List[int] = _truncate_token_ids(input_ids[0].tolist(), cap)
        comp: List[int] = []
        for _ in range(max(1, int(icfg.max_sentence_tokens))):
            batch = torch.tensor([seq], device=device, dtype=torch.long)
            out = backbone.forward_with_gates(batch)
            nid = sample_next_token(
                out.logits[0, -1, :], temperature=float(icfg.temperature)
            )
            seq.append(int(nid))
            comp.append(int(nid))
            bd = find_step_boundaries(
                comp,
                tokenizer,
                min_step_tokens=int(icfg.min_sentence_tokens),
                max_step_tokens=int(icfg.max_sentence_tokens),
                min_step_chars=int(icfg.sentence_char_boundary_min),
            )
            if observe_inner_loop_should_stop(
                boundaries=bd,
                generated_len=len(comp),
                max_step_tokens=int(icfg.max_sentence_tokens),
            ):
                break
            seq = _truncate_token_ids(seq, cap)
        if not comp:
            h = torch.zeros(1, hidden_size, device=device, dtype=torch.float32)
            lg = torch.zeros(1, vocab_size, device=device, dtype=torch.float32)
            return StepObservation(text="[empty step]", token_ids=[], hidden=h, logits=lg)
        batch = torch.tensor([seq], device=device, dtype=torch.long)
        out = backbone.forward_with_gates(batch)
        h = out.hidden[0, -len(comp) :, :].contiguous()
        lg = out.logits[0, -len(comp) :, :].contiguous()
        text = tokenizer.decode(comp, skip_special_tokens=True)
        return StepObservation(text=text, token_ids=comp, hidden=h, logits=lg)

    return _observe


def _make_answer_commit(
    *,
    backbone: HFGatedBackbone,
    tokenizer,
    model_path: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    observe_max_input_tokens: int = 0,
) -> Callable[[str], str]:
    eos = getattr(tokenizer, "eos_token_id", None)
    cap = int(observe_max_input_tokens or 0)

    def _commit(ctx: str) -> str:
        _, input_ids = build_prompt_and_tokenize(
            tokenizer, model_path, str(ctx or "").strip(), return_tensors=True, device=device
        )
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        seq = _truncate_token_ids(input_ids[0].tolist(), cap)
        # Track generated ids independently from the sliding context window.
        # If the prompt already fills ``cap``, slicing the final capped sequence
        # at the original prompt length incorrectly returns an empty answer.
        generated_ids: List[int] = []
        for _ in range(max(1, int(max_new_tokens))):
            seq = _truncate_token_ids(seq, cap)
            batch = torch.tensor([seq], device=device, dtype=torch.long)
            out = backbone.forward_with_gates(batch)
            nid = sample_next_token(out.logits[0, -1, :], temperature=float(temperature))
            seq.append(int(nid))
            generated_ids.append(int(nid))
            if eos is not None and int(nid) == int(eos):
                break
        return tokenizer.decode(generated_ids, skip_special_tokens=True)

    return _commit


def run_sot_mechanism_on_sample(
    sample: DatasetSample,
    *,
    mot_cfg: MoTConfig,
    backbone: HFGatedBackbone,
    tokenizer,
    model_path: str,
    device: torch.device,
    mechanism: Optional[MechanismExperimentConfig] = None,
    training_free_policy: Optional[TrainingFreePolicy] = None,
    dataset_name: str = "default",
    commit_mode: str = "fullchain",
) -> MotInferenceResult:
    prompt = build_aligned_user_text(sample)
    inf = mot_cfg.inference
    observe = _make_observe_step(
        backbone=backbone,
        tokenizer=tokenizer,
        model_path=model_path,
        device=device,
        prompt_root=prompt,
        inf=inf,
        hidden_size=backbone.hidden_size,
        vocab_size=backbone.vocab_size,
    )
    commit = _make_answer_commit(
        backbone=backbone,
        tokenizer=tokenizer,
        model_path=model_path,
        device=device,
        max_new_tokens=inf.answer_commit_max_new_tokens,
        temperature=float(inf.temperature),
        observe_max_input_tokens=int(getattr(inf, "observe_max_input_tokens", 0) or 0),
    )

    def tok_enc(s: str) -> List[int]:
        return tokenizer.encode(str(s or ""), add_special_tokens=False)

    return run_mot_inference(
        prompt=prompt,
        observe_step_fn=observe,
        answer_commit_fn=commit,
        tokenizer_encode=tok_enc,
        selector_cfg=mot_cfg.selector,
        commit_cfg=mot_cfg.commit,
        context_cfg=mot_cfg.context,
        normalization=mot_cfg.normalization,
        debt=mot_cfg.debt,
        inference_cfg=inf,
        mechanism=mechanism,
        training_free_policy=training_free_policy,
        dataset_name=dataset_name,
        commit_mode=commit_mode,
    )


def grade_sample(sample: DatasetSample, pred_raw: str, *, task_type: str) -> bool:
    gold = str(sample.answer or "")
    ex = dict(sample.extra)
    ex.setdefault("question", sample.question)
    ex.setdefault("answer", sample.answer)
    ch = sample.choices
    parsed = extract_final_answer_text(
        pred_raw,
        task_type=task_type,
        ref_answer=gold,
        choices=ch,
        question=str(sample.question or ""),
    )
    return is_correct(parsed, gold, str(task_type), extra=ex, choices=ch)


def extract_m_replay_from_result(res: MotInferenceResult) -> List[Tensor]:
    out: List[Tensor] = []
    for tr in res.traces:
        if tr.m_controller:
            out.append(torch.tensor(tr.m_controller, dtype=torch.float32))
    return out


__all__ = [
    "extract_m_replay_from_result",
    "grade_sample",
    "run_sot_mechanism_on_sample",
]
