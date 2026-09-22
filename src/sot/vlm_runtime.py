"""Qwen2.5-VL runtime for the paper-aligned SoT evaluation."""
from __future__ import annotations

from typing import Any, Callable, List, Optional

import torch
from PIL import Image

from .backbone.hf_vlm_backbone import HFVLMBackbone
from .dataset import DatasetSample
from .grading import extract_final_answer_text
from .inference import MotInferenceResult, StepObservation, run_mot_inference
from .mechanism_config import MechanismExperimentConfig
from .prompts import build_user_root_text
from .step_boundary import find_step_boundaries, observe_inner_loop_should_stop
from .text_runtime import sample_next_token
from .vlm_inputs import apply_chat_to_model_inputs, qwen_text_user_messages, qwen_vl_user_messages

VLM_REASONING_SUFFIX = "Let's think step by step. End with a clear final answer."


def _prompt(sample: DatasetSample) -> str:
    root = build_user_root_text(
        question=sample.question,
        task_type=str(sample.extra.get("task_type") or "visual_reasoning_mc"),
        choices=sample.choices,
    )
    return f"{root['content']}\n\n{VLM_REASONING_SUFFIX}"


def _observe_factory(
    *,
    backbone: HFVLMBackbone,
    image: Image.Image,
    prompt_root: str,
    config: Any,
) -> Callable[[str], StepObservation]:
    processor = backbone.processor
    device = backbone.get_device()

    def observe(context: str) -> StepObservation:
        body = str(context or "").strip() or prompt_root
        inputs = apply_chat_to_model_inputs(
            processor,
            qwen_vl_user_messages(pil=image, user_text=body),
            add_generation_prompt=True,
            device=device,
        )
        sequence = inputs["input_ids"][0].tolist()
        attention = inputs.get("attention_mask")
        mask = attention[0].tolist() if attention is not None else [1] * len(sequence)
        backbone.clear_mm_extras()
        backbone.set_mm_extras(inputs)
        prefill, cache = backbone.forward_with_cache(
            torch.tensor([sequence], device=device, dtype=torch.long),
            attention_mask=torch.tensor([mask], device=device, dtype=torch.long),
            output_hidden_states=False,
        )
        pending = sample_next_token(
            prefill.logits[0, -1, :], temperature=float(config.temperature)
        )
        completion: List[int] = []
        hidden_states: List[torch.Tensor] = []
        logits: List[torch.Tensor] = []
        for _ in range(max(1, int(config.max_sentence_tokens))):
            sequence.append(int(pending))
            mask.append(1)
            completion.append(int(pending))
            boundaries = find_step_boundaries(
                completion,
                processor.tokenizer,
                min_step_tokens=int(config.min_sentence_tokens),
                max_step_tokens=int(config.max_sentence_tokens),
                min_step_chars=int(config.sentence_char_boundary_min),
            )
            decoded, cache = backbone.forward_with_cache(
                torch.tensor([[pending]], device=device, dtype=torch.long),
                attention_mask=torch.tensor([mask], device=device, dtype=torch.long),
                past_key_values=cache,
                output_hidden_states=True,
            )
            hidden_states.append(decoded.hidden[0, -1, :].contiguous())
            logits.append(decoded.logits[0, -1, :].contiguous())
            if observe_inner_loop_should_stop(
                boundaries=boundaries,
                generated_len=len(completion),
                max_step_tokens=int(config.max_sentence_tokens),
            ):
                break
            pending = sample_next_token(
                decoded.logits[0, -1, :], temperature=float(config.temperature)
            )
        backbone.clear_mm_extras()
        if not completion:
            return StepObservation(
                text="[empty step]",
                token_ids=[],
                hidden=torch.zeros(1, backbone.hidden_size, device=device),
                logits=torch.zeros(1, backbone.vocab_size, device=device),
            )
        return StepObservation(
            text=processor.tokenizer.decode(completion, skip_special_tokens=True),
            token_ids=completion,
            hidden=torch.stack(hidden_states),
            logits=torch.stack(logits),
        )

    return observe


def _answer_factory(
    *,
    backbone: HFVLMBackbone,
    image: Image.Image,
    sample: DatasetSample,
    max_new_tokens: int,
    temperature: float,
) -> Callable[[str], str]:
    processor = backbone.processor
    tokenizer = processor.tokenizer
    device = backbone.get_device()
    task_type = str(sample.extra.get("task_type") or "visual_reasoning_mc")
    eos = getattr(tokenizer, "eos_token_id", None)

    def generate(messages: list[dict[str, Any]], limit: int, temp: float) -> str:
        inputs = apply_chat_to_model_inputs(
            processor, messages, add_generation_prompt=True, device=device
        )
        sequence = inputs["input_ids"][0].tolist()
        mask_tensor = inputs.get("attention_mask")
        mask = mask_tensor[0].tolist() if mask_tensor is not None else [1] * len(sequence)
        prompt_length = len(sequence)
        backbone.clear_mm_extras()
        backbone.set_mm_extras(inputs)
        output, cache = backbone.forward_with_cache(
            torch.tensor([sequence], device=device, dtype=torch.long),
            attention_mask=torch.tensor([mask], device=device, dtype=torch.long),
            output_hidden_states=False,
        )
        next_id = sample_next_token(output.logits[0, -1, :], temperature=temp)
        for _ in range(max(1, int(limit))):
            sequence.append(int(next_id))
            mask.append(1)
            if eos is not None and int(next_id) == int(eos):
                break
            output, cache = backbone.forward_with_cache(
                torch.tensor([[next_id]], device=device, dtype=torch.long),
                attention_mask=torch.tensor([mask], device=device, dtype=torch.long),
                past_key_values=cache,
                output_hidden_states=False,
            )
            next_id = sample_next_token(output.logits[0, -1, :], temperature=temp)
        generated = sequence[prompt_length:]
        backbone.clear_mm_extras()
        return tokenizer.decode(generated, skip_special_tokens=True)

    def answer(context: str) -> str:
        guard = (
            "Now provide only the final answer. Do not output reasoning steps or repeated filler. "
            "For a multiple-choice task, return only the option letter."
        )
        raw = generate(
            qwen_vl_user_messages(
                pil=image, user_text=f"{str(context or '').strip()}\n\n{guard}"
            ),
            max_new_tokens,
            temperature,
        )
        parsed = extract_final_answer_text(
            raw,
            task_type=task_type,
            ref_answer=sample.answer,
            choices=sample.choices,
            question=sample.question,
        ).strip()
        # The frozen protocol uses a short deterministic normalization pass for
        # open-ended VQA only. Multiple-choice reasoning is returned directly.
        if task_type == "visual_qa" and parsed:
            polished = generate(
                qwen_text_user_messages(
                    user_text=(
                        "Return only the final short answer span, with no explanation.\n"
                        f"Question: {sample.question}\nDraft answer: {parsed}"
                    )
                ),
                24,
                0.0,
            )
            parsed = extract_final_answer_text(
                polished,
                task_type=task_type,
                ref_answer=sample.answer,
                choices=sample.choices,
                question=sample.question,
            ).strip() or parsed
        return parsed if task_type in {"visual_qa", "image_classification"} and parsed else raw

    return answer


def run_vlm_sot_on_sample(
    sample: DatasetSample,
    *,
    mot_cfg: Any,
    backbone: HFVLMBackbone,
    image: Image.Image,
    mechanism: Optional[MechanismExperimentConfig] = None,
) -> MotInferenceResult:
    prompt = _prompt(sample)
    processor = backbone.processor
    return run_mot_inference(
        prompt=prompt,
        observe_step_fn=_observe_factory(
            backbone=backbone, image=image, prompt_root=prompt, config=mot_cfg.inference
        ),
        answer_commit_fn=_answer_factory(
            backbone=backbone,
            image=image,
            sample=sample,
            max_new_tokens=int(mot_cfg.inference.answer_commit_max_new_tokens),
            temperature=float(mot_cfg.inference.temperature),
        ),
        tokenizer_encode=lambda text: processor.tokenizer.encode(
            str(text or ""), add_special_tokens=False
        ),
        selector_cfg=mot_cfg.selector,
        commit_cfg=mot_cfg.commit,
        context_cfg=mot_cfg.context,
        normalization=mot_cfg.normalization,
        debt=mot_cfg.debt,
        inference_cfg=mot_cfg.inference,
        mechanism=mechanism,
    )


__all__ = ["run_vlm_sot_on_sample"]
