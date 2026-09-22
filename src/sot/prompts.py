"""Shared user root for SoT and all baselines."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

_TASK_FORMAT_SUFFIX: Dict[str, str] = {
    "binary_qa": "Final answer format: output only yes or no.",
    "multiple_choice": "Final answer format: output only one option letter (A-E).",
    "logical_label": (
        "Final answer format: output only one valid label "
        "(True/False/Unknown or one option letter)."
    ),
    "math_qa": "Final answer format: output one final numeric answer.",
    "math_competition": "Final answer format: output one final numeric answer.",
    "symbolic_sequence": "Final answer format: output only the final sequence on one line.",
    "relation_extraction": (
        "Final answer format: output only one relation label from this set: "
        "aunt, brother, daughter, daughter-in-law, father, father-in-law, "
        "granddaughter, grandfather, grandmother, grandson, mother, mother-in-law, "
        "nephew, niece, sister, son, son-in-law, uncle. "
        "For query pair ('A', 'B'), output how B is related to A."
    ),
    # VLM: align with eval.metrics._extract_visual_qa_final_answer / baseline grading.
    "visual_qa": (
        "Final answer format: end your entire reply with exactly one line of the form "
        "Final Answer: <short answer> (no quotes; keep the answer on the same line)."
    ),
    "image_classification": (
        "Final answer format: output only the chosen class label "
        "(copy the exact option text when options are listed, or a single option letter A–E)."
    ),
}


def build_user_root_text(*, question: str, task_type: str, choices: Optional[List[Any]] = None) -> Dict[str, Any]:
    q = str(question or "").strip()
    tt = str(task_type or "").strip()
    tt_norm = tt.lower()
    lines: List[str] = []
    lines.append(
        "Work through the problem carefully. Use clear intermediate reasoning, "
        "then give a concise final answer that matches the task format."
    )
    if choices:
        lines.append("Options:")
        for i, c in enumerate(choices):
            label = chr(ord("A") + i) if i < 26 else str(i)
            lines.append(f"  {label}. {c}")
    lines.append(f"Task type: {tt}")
    fmt = _TASK_FORMAT_SUFFIX.get(tt_norm)
    if fmt:
        lines.append(fmt)
    lines.append(f"Question:\n{q}")
    content = "\n".join(lines)
    return {"question": q, "task_type": tt, "choices": choices, "content": content}


def build_vanilla_floor_user_text(*, question: str, task_type: str, choices: Optional[List[Any]] = None) -> str:
    """One-shot lower bound: optional options list + format line + question only.

    Omits the CoT scaffold and **does not** emit ``Task type:`` meta-hint (weaker than other baselines).
    Keeps only a short format line when known so grading can still run.
    """
    q = str(question or "").strip()
    tt = str(task_type or "").strip()
    tt_norm = tt.lower()
    lines: List[str] = []
    if choices:
        lines.append("Options:")
        for i, c in enumerate(choices):
            label = chr(ord("A") + i) if i < 26 else str(i)
            lines.append(f"  {label}. {c}")
    fmt = _TASK_FORMAT_SUFFIX.get(tt_norm)
    if fmt:
        lines.append(fmt)
    lines.append(f"Question:\n{q}")
    return "\n".join(lines)


__all__ = ["build_user_root_text", "build_vanilla_floor_user_text"]
