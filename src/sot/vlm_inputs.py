"""Qwen2.5-VL / Qwen3-VL multimodal chat inputs (single image per sample)."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PIL import Image


def load_pil_image(
    sample_extra: dict[str, Any], *, data_root: str | Path | None = None
) -> Image.Image:
    """Load one RGB image, resolving relative paths from the experiment root.

    ``data_root`` is passed explicitly by the evaluator so this also works from
    an installed wheel; it does not depend on the package's installation path.
    """
    rel = str(sample_extra.get("image_path") or "").strip()
    if not rel:
        raise ValueError("sample.extra missing image_path for VLM")
    p = Path(rel).expanduser()
    if not p.is_absolute():
        p = (Path(data_root) if data_root is not None else Path.cwd()) / p
    p = p.resolve()
    if not p.is_file():
        raise FileNotFoundError(f"VLM image not found: {p}")
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError("install the 'vlm' extra to evaluate image inputs") from exc
    img = Image.open(p).convert("RGB")
    return img


def qwen_vl_user_messages(*, pil: Image.Image, user_text: str) -> list[dict[str, Any]]:
    """Single-turn user message with interleaved image + text (HF Qwen-VL chat template)."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil},
                {"type": "text", "text": str(user_text or "").strip()},
            ],
        }
    ]


def qwen_text_user_messages(*, user_text: str) -> list[dict[str, Any]]:
    """Single-turn text-only user message (no image content)."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": str(user_text or "").strip()},
            ],
        }
    ]


def apply_chat_to_model_inputs(
    processor,
    messages: list[dict[str, Any]],
    *,
    add_generation_prompt: bool,
    device: Any,
) -> dict[str, Any]:
    out = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        return_dict=True,
        return_tensors="pt",
    )
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in out.items()}


__all__ = [
    "apply_chat_to_model_inputs",
    "load_pil_image",
    "qwen_text_user_messages",
    "qwen_vl_user_messages",
]
