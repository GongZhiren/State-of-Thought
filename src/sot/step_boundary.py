from __future__ import annotations

import re
from typing import List, Sequence

_SENTENCE_END_RE = re.compile(r"[.!?。！？][\"')\]）】」』]?\s*$")


def has_sentence_boundary(text: str) -> bool:
    t = text.rstrip()
    if not t:
        return False
    return bool(_SENTENCE_END_RE.search(t) or t.endswith("\n"))


def find_step_boundaries(
    gen_ids: Sequence[int],
    tokenizer,
    *,
    min_step_tokens: int,
    max_step_tokens: int,
    min_step_chars: int,
) -> List[int]:
    boundaries = [0]
    buf: List[int] = []
    for i, tid in enumerate(gen_ids):
        buf.append(int(tid))
        if len(buf) >= max(1, int(min_step_tokens)):
            seg_text = tokenizer.decode(buf, skip_special_tokens=True).rstrip()
            if len(seg_text) >= max(1, int(min_step_chars)) and has_sentence_boundary(seg_text):
                boundaries.append(i + 1)
                buf = []
        if len(buf) >= max(1, int(max_step_tokens)):
            boundaries.append(i + 1)
            buf = []
    if boundaries[-1] != len(gen_ids):
        boundaries.append(len(gen_ids))
    return boundaries


def observe_inner_loop_should_stop(
    *,
    boundaries: Sequence[int],
    generated_len: int,
    max_step_tokens: int,
) -> bool:
    """Whether the token-by-token observe sub-loop should stop for this SoT step.

    ``find_step_boundaries`` always ends with ``[..., n]`` for ``n == len(comp)``, so
    ``len(boundaries) > 1`` holds for **every** non-empty ``comp`` (typically ``[0, n]``).
    Using ``len(bd) > 1`` therefore stops after the **first** generated token — starving
    SoT of real step text (especially harmful for ``visual_qa`` / long answers).

    Stop when there is a **true internal** cut (``len(boundaries) > 2``) or the chunk hits
    ``max_step_tokens`` (hard cap aligned with ``InferenceConfig.max_sentence_tokens``).
    """
    n = int(generated_len)
    if n <= 0:
        return False
    if int(len(boundaries)) > 2:
        return True
    cap = max(1, int(max_step_tokens))
    return n >= cap


def split_first_step(
    gen_ids: Sequence[int],
    tokenizer,
    *,
    min_step_tokens: int,
    max_step_tokens: int,
    min_step_chars: int,
) -> List[int]:
    if not gen_ids:
        return []
    boundaries = find_step_boundaries(
        gen_ids,
        tokenizer,
        min_step_tokens=min_step_tokens,
        max_step_tokens=max_step_tokens,
        min_step_chars=min_step_chars,
    )
    end = boundaries[1] if len(boundaries) > 1 else len(gen_ids)
    return list(gen_ids[:end])
