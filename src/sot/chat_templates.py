"""Tokenizer/chat-template helpers compatible with run_eval.pyc."""
from __future__ import annotations

from typing import Tuple

import torch


def build_prompt(tokenizer, model_path: str, user_text: str) -> str:
    _ = model_path
    text = str(user_text or "")
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    return text


def build_prompt_and_tokenize(
    tokenizer,
    model_path: str,
    user_text: str,
    *,
    return_tensors: bool = True,
    device: torch.device | None = None,
) -> Tuple[str, torch.Tensor]:
    prompt = build_prompt(tokenizer, model_path, user_text)
    if tokenizer is None:
        ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
        return prompt, ids
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = enc["input_ids"]
    if device is not None:
        input_ids = input_ids.to(device)
    if not return_tensors:
        return prompt, input_ids[0]
    return prompt, input_ids


__all__ = ["build_prompt", "build_prompt_and_tokenize"]
