"""HF CausalLM wrapper exposing ``BackboneForward`` for baseline token loops."""
from __future__ import annotations

from typing import Any, Optional

import torch
from torch import Tensor
from transformers import AutoModelForCausalLM

from sot.backbone.base import BackboneForward, BackboneOutput


class HFGatedBackbone(BackboneForward):
    """Frozen LM; ``head_gates`` accepted for API compatibility but ignored."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str | torch.device,
        dtype: str | torch.dtype = "bfloat16",
        attention_implementation: str = "sdpa",
        quantization: str = "none",
        device_map: str | dict[str, Any] | None = None,
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
        quant_mode = str(quantization or "none").strip().lower()
        attn_impl = str(attention_implementation or "sdpa").strip().lower()

        def _build_load_kw(mode: str) -> dict:
            kw: dict[str, Any] = {"dtype": resolved_dtype}
            if attn_impl and attn_impl not in {"none", "default", "auto"}:
                kw["attn_implementation"] = attn_impl
            m = str(mode or "").strip().lower()
            if m in {"8bit", "int8"} and dev.type == "cuda":
                kw["device_map"] = device_map or "auto"
                kw["load_in_8bit"] = True
                kw.pop("dtype", None)
                return kw
            if m in {"4bit", "int4"} and dev.type == "cuda":
                kw["device_map"] = device_map or "auto"
                kw["load_in_4bit"] = True
                kw["bnb_4bit_compute_dtype"] = resolved_dtype
                kw.pop("dtype", None)
                return kw
            if device_map is not None and dev.type == "cuda":
                kw["device_map"] = device_map
            return kw

        def _load(mode: str):
            kw = _build_load_kw(mode)
            if "device_map" in kw:
                model = AutoModelForCausalLM.from_pretrained(model_path, **kw)
                d = torch.device(next(model.parameters()).device)
                return model, d
            model = AutoModelForCausalLM.from_pretrained(model_path, **kw).to(dev)
            return model, dev

        self._model, self._device = _load(quant_mode)
        self._model.eval()
        cfg = self._model.config
        self._n_layer = int(getattr(cfg, "num_hidden_layers", 1))
        self._n_head = int(getattr(cfg, "num_attention_heads", 1))
        self._hidden = int(getattr(cfg, "hidden_size", getattr(cfg, "d_model", 4096)))
        self._vocab = int(getattr(cfg, "vocab_size", 32000))

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
        with torch.no_grad():
            out = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
        hidden = out.hidden_states[-1]
        logits = out.logits
        return BackboneOutput(
            hidden=hidden,
            logits=logits,
            hidden_states_tuple=out.hidden_states,
            hidden_states_all=None,
        )


__all__ = ["HFGatedBackbone"]
