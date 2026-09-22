"""Frozen Qwen2.5-VL / Qwen3-VL backbone: forward expects cached multimodal tensors + input_ids."""
from __future__ import annotations

from typing import Any, Optional

import torch
from torch import Tensor
from transformers import AutoConfig, AutoModelForVision2Seq, AutoProcessor

from sot.backbone.base import BackboneForward, BackboneOutput
from sot.vlm_inputs import apply_chat_to_model_inputs, qwen_vl_user_messages


def load_vlm_model_cls(model_path: str) -> type:
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    arch = (getattr(cfg, "architectures", None) or ["AutoModelForVision2Seq"])[0]
    if arch == "Qwen3VLForConditionalGeneration":
        from transformers import Qwen3VLForConditionalGeneration

        return Qwen3VLForConditionalGeneration
    if arch in {"Qwen2_5_VLForConditionalGeneration", "Qwen2VLForConditionalGeneration"}:
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration

            return Qwen2_5_VLForConditionalGeneration
        except Exception:  # pragma: no cover
            pass
    return AutoModelForVision2Seq


class HFVLMBackbone(BackboneForward):
    """Vision-language model with per-session ``mm_extras`` (pixel_values, image_grid_thw, …)."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str | torch.device,
        dtype: str | torch.dtype = "bfloat16",
        quantization: str = "none",
        device_map: str | dict[str, Any] | None = None,
        processor_use_fast: bool = True,
    ) -> None:
        dev = torch.device(device) if isinstance(device, str) else device
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        resolved_dtype = dtype if isinstance(dtype, torch.dtype) else dtype_map[str(dtype).lower()]
        use_4bit = str(quantization or "none").lower() in {"4bit", "int4", "nf4"}
        model_cls = load_vlm_model_cls(model_path)
        kw: dict[str, Any] = {"dtype": resolved_dtype, "trust_remote_code": True}
        if use_4bit and dev.type == "cuda":
            from transformers import BitsAndBytesConfig
            kw.pop("dtype", None)
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=resolved_dtype,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
            kw["device_map"] = {"": (dev.index if dev.index is not None else 0)}
            self._model = model_cls.from_pretrained(model_path, **kw)
            self._device = torch.device(next(self._model.parameters()).device)
        elif device_map is not None and dev.type == "cuda":
            kw["device_map"] = device_map
            self._model = model_cls.from_pretrained(model_path, **kw)
            self._device = torch.device(next(self._model.parameters()).device)
        else:
            self._model = model_cls.from_pretrained(model_path, **kw).to(dev)
            self._device = dev
        self._model.eval()
        # Freeze the processor choice used by the camera-ready reruns. Recent
        # Transformers versions changed Qwen-VL's implicit default to fast;
        # making it explicit prevents a future version-dependent protocol.
        self._processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True, use_fast=bool(processor_use_fast)
        )
        cfg = self._model.config
        text_cfg = getattr(cfg, "text_config", None) or cfg
        self._n_layer = int(getattr(text_cfg, "num_hidden_layers", 1))
        self._n_head = int(getattr(text_cfg, "num_attention_heads", 1))
        self._hidden = int(getattr(text_cfg, "hidden_size", getattr(text_cfg, "d_model", 4096)))
        self._vocab = int(getattr(text_cfg, "vocab_size", 151936))
        self._mm_extras: dict[str, Tensor] = {}

    @property
    def processor(self):
        return self._processor

    def clear_mm_extras(self) -> None:
        self._mm_extras = {}

    def set_mm_extras(self, batch: dict[str, Any]) -> None:
        """Freeze pixel_values / image_grid_thw / … from a processor batch (exclude input_ids, attention_mask)."""
        ex: dict[str, Tensor] = {}
        for k, v in batch.items():
            if k in {"input_ids", "attention_mask"}:
                continue
            if isinstance(v, Tensor):
                ex[k] = v.to(self._device)
        self._mm_extras = ex

    @property
    def num_layers(self) -> int:
        return self._n_layer

    @property
    def num_heads(self) -> int:
        return self._n_head

    @property
    def hidden_size(self) -> int:
        return self._hidden

    @property
    def vocab_size(self) -> int:
        return self._vocab

    def get_device(self) -> torch.device:
        return self._device

    def forward_with_gates(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        head_gates: Optional[Tensor] = None,
    ) -> BackboneOutput:
        _ = head_gates
        input_ids = input_ids.to(self._device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._device)
        else:
            attention_mask = torch.ones_like(input_ids, device=self._device)
        kwargs: dict[str, Any] = {
            **self._mm_extras,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": False,
            "output_hidden_states": True,
            "return_dict": True,
        }
        with torch.inference_mode():
            out = self._model(**kwargs)
        hidden = out.hidden_states[-1]
        logits = out.logits
        return BackboneOutput(
            hidden=hidden,
            logits=logits,
            hidden_states_tuple=out.hidden_states,
            hidden_states_all=None,
        )

    def forward_with_cache(
        self,
        input_ids: Tensor,
        *,
        attention_mask: Optional[Tensor] = None,
        past_key_values: Any = None,
        output_hidden_states: bool = True,
    ) -> tuple[BackboneOutput, Any]:
        """VLM incremental forward with KV cache.

        - ``past_key_values is None``: prefill pass; include cached multimodal extras.
        - ``past_key_values is not None``: decode pass; only pass text token(s) + cache.
        """
        input_ids = input_ids.to(self._device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._device)
        else:
            attention_mask = torch.ones_like(input_ids, device=self._device)
        kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": True,
            "output_hidden_states": bool(output_hidden_states),
            "return_dict": True,
            "past_key_values": past_key_values,
        }
        if past_key_values is None:
            kwargs.update(self._mm_extras)
        with torch.inference_mode():
            out = self._model(**kwargs)
        hidden = out.hidden_states[-1] if bool(output_hidden_states) and out.hidden_states else torch.empty(
            input_ids.shape[0], input_ids.shape[1], self._hidden, device=self._device, dtype=torch.float32
        )
        logits = out.logits
        return (
            BackboneOutput(
                hidden=hidden,
                logits=logits,
                hidden_states_tuple=out.hidden_states if bool(output_hidden_states) else None,
                hidden_states_all=None,
            ),
            out.past_key_values,
        )

    def generate_answer(
        self,
        pil: Any,
        *,
        user_text: str,
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        """Single-turn multimodal generation (for VLM baselines). ``pil`` is a ``PIL.Image.Image``."""
        messages = qwen_vl_user_messages(pil=pil, user_text=str(user_text or "").strip())
        inputs = apply_chat_to_model_inputs(
            self._processor,
            messages,
            add_generation_prompt=True,
            device=self._device,
        )
        do_sample = float(temperature) > 0.0
        gen_kw: dict[str, Any] = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": do_sample,
        }
        if do_sample:
            gen_kw["temperature"] = float(temperature)
            gen_kw["top_p"] = 0.9
        with torch.inference_mode():
            gen_ids = self._model.generate(**inputs, **gen_kw)
        in_len = int(inputs["input_ids"].shape[1])
        trimmed = gen_ids[:, in_len:]
        texts = self._processor.batch_decode(
            trimmed.cpu(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return (texts[0] if texts else "").strip()


__all__ = ["HFVLMBackbone", "load_vlm_model_cls"]
