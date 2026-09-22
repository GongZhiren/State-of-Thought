"""Grading helpers and metric aggregation for eval scripts (.pyc) and baselines."""
from __future__ import annotations

import io
import os
import re
import signal
import threading
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, Optional, Sequence

_RELATION_LABELS = {
    "aunt",
    "brother",
    "daughter",
    "daughter-in-law",
    "father",
    "father-in-law",
    "granddaughter",
    "grandfather",
    "grandmother",
    "grandson",
    "mother",
    "mother-in-law",
    "nephew",
    "niece",
    "sister",
    "son",
    "son-in-law",
    "uncle",
}
_RELATION_PLURAL_TO_SINGULAR = {
    "sisters": "sister",
    "brothers": "brother",
    "daughters": "daughter",
    "sons": "son",
    "mothers": "mother",
    "fathers": "father",
    "aunts": "aunt",
    "uncles": "uncle",
    "nieces": "niece",
    "nephews": "nephew",
    "granddaughters": "granddaughter",
    "grandsons": "grandson",
    "grandmothers": "grandmother",
    "grandfathers": "grandfather",
}


def _normalize_answer(text: str) -> str:
    s = str(text or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _normalize(text: str) -> str:
    return _normalize_answer(text)


def token_level_f1(pred: str, ref: str) -> float:
    p = _normalize(pred).split()
    r = _normalize(ref).split()
    if not r:
        return 1.0 if not p else 0.0
    pc, rc = Counter(p), Counter(r)
    common = sum((pc & rc).values())
    if common == 0:
        return 0.0
    prec = common / max(len(p), 1)
    rec = common / max(len(r), 1)
    return 2.0 * prec * rec / (prec + rec + 1e-8)


def _extract_last_number(text: str) -> str:
    s = re.sub(r"(?<=\d),(?=\d)", "", str(text or ""))
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if not nums:
        return ""
    return nums[-1]


def _extract_choice(pred: str, choices: Optional[Sequence[Any]] = None) -> str:
    t = str(pred or "")
    # Prefer the terminal answer commitment. Reasoning traces commonly enumerate
    # A/B/C/D before committing, so taking the first standalone letter silently
    # grades many correct verbose answers as option A.
    explicit = re.findall(
        r"(?:final\s+answer|answer|option|choice)\s*(?:is|:)?\s*[`*_]*\s*[\[\(\{]?\s*([A-E])\s*[\]\)\}]?",
        t,
        flags=re.IGNORECASE,
    )
    if explicit:
        return explicit[-1].upper()
    lines = [ln.strip().strip("`*_ ") for ln in t.splitlines() if ln.strip()]
    if lines:
        m_last = re.fullmatch(r"(?:[\[\(\{]\s*)?([A-Ea-e])(?:\s*[\]\)\}])?[\.:]?", lines[-1])
        if m_last:
            return m_last.group(1).upper()
    # Accept lower-case option letters only in explicit answer contexts.
    m_low = re.search(r"^\s*([a-e])\s*$", t)
    if m_low:
        return m_low.group(1).upper()
    # Require "final answer" (not bare "final"), otherwise "final answer is: True"
    # matches "final " + ([a-e])->"a" from "answer", yielding bogus multiple-choice "A".
    m_ctx = re.findall(
        r"(?:answer|option|choice|final\s+answer)\s*(?:is|:)?\s*[`*_]*\s*[\[\(\{]?\s*([a-e])\s*[\]\)\}]?",
        t,
        flags=re.IGNORECASE,
    )
    if m_ctx:
        return m_ctx[-1].upper()
    m_bracket = re.findall(r"[\[\(\{]\s*([a-e])\s*[\]\)\}]", t, flags=re.IGNORECASE)
    if m_bracket:
        return m_bracket[-1].upper()
    # Last-resort uppercase match, also terminal rather than first-enumerated.
    standalone = re.findall(r"\b([A-E])\b", t)
    if standalone:
        # An output that merely reproduces several labelled alternatives has
        # not committed to a choice.  Selecting its first or last label would
        # turn prompt echo / truncation into accidental credit.
        if len(set(standalone)) > 1:
            return ""
        return standalone[-1]
    m2 = re.search(r"\b(true|false|unknown)\b", t, flags=re.IGNORECASE)
    if m2:
        w = m2.group(1).lower()
        return {"true": "True", "false": "False", "unknown": "Unknown"}[w]
    m3 = re.search(r"\b(yes|no)\b", t, flags=re.IGNORECASE)
    if m3:
        return m3.group(1).lower()
    if choices:
        hits = [(t.rfind(str(c).strip()), i) for i, c in enumerate(choices) if str(c).strip()]
        hits = [x for x in hits if x[0] >= 0]
        if hits:
            _, i = max(hits)
            return chr(ord("A") + i) if i < 26 else str(i)
    return t.strip()


# Fashion-MNIST canonical strings (align with data/vlm/fashion_mnist/eval_fixed.jsonl).
_FASHION_MNIST_LABELS: tuple[str, ...] = (
    "T - shirt / top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)

# Common alias -> canonical label-key for image classification outputs.
# This helps VLM outputs that are semantically correct but not in canonical dataset surface form.
_CLASS_LABEL_ALIASES: dict[str, str] = {
    "pants": "trouser",
    "pant": "trouser",
    "sweater": "pullover",
    "sweatshirt": "pullover",
    "hoodie": "pullover",
    "tshirt": "tshirttop",
    "tee": "tshirttop",
    "top": "tshirttop",
    "boots": "ankleboot",
    "boot": "ankleboot",
    "sneakers": "sneaker",
    "jacket": "coat",
    "wallet": "bag",
    "tanktop": "tshirttop",
    "tank": "tshirttop",
    "shoe": "sandal",
    "s": "sandal",
}


def _parse_allowed_labels_from_classification_question(question: str) -> list[str]:
    """Parse enumerated class names from prompts like '... one label: a, b, or c.'."""
    q = str(question or "")
    m = re.search(
        r"(?:exactly\s+one\s+)?(?:label|labels|classes)\s*(?:is|are)?\s*:\s*([^\n]+)",
        q,
        flags=re.IGNORECASE,
    )
    if not m:
        return []
    frag = m.group(1).strip().rstrip(".)\"'")
    parts = re.split(r",\s*|\s+or\s+", frag, flags=re.IGNORECASE)
    return [p.strip().strip("`\"' ") for p in parts if p.strip()]


def _rfind_allowed_label(haystack: str, label: str) -> int:
    """Return start index of last case-insensitive occurrence of full label, or -1."""
    h = str(haystack or "")
    lab = str(label or "")
    if not lab:
        return -1
    if "_" in lab and re.fullmatch(r"[a-z0-9_]+", lab, flags=re.I):
        pat = r"(?<![a-z0-9_])" + re.escape(lab) + r"(?![a-z0-9_])"
        last = -1
        for m in re.finditer(pat, h, flags=re.IGNORECASE):
            last = m.start()
        return last
    # Single-token class names (e.g. healthy): avoid matching inside "unhealthy".
    if re.fullmatch(r"[A-Za-z]+", lab):
        last = -1
        for m in re.finditer(
            r"(?<![A-Za-z0-9_])" + re.escape(lab) + r"(?![A-Za-z0-9_])",
            h,
            flags=re.IGNORECASE,
        ):
            last = m.start()
        return last
    lo = h.lower()
    needle = lab.lower()
    return lo.rfind(needle)


def _extract_image_classification_label(
    raw: str,
    *,
    allowed_labels: Sequence[str],
    ref_answer: str,
) -> str:
    """Pull a single class label from CoT-heavy VLM text (see image_classification in datasets.yaml)."""
    text = str(raw or "").strip()
    if not text:
        return ""

    labels = [str(x).strip() for x in allowed_labels if str(x).strip()]
    if not labels and ref_answer:
        labels = [str(ref_answer).strip()]

    if len(labels) == 1:
        only = labels[0]
        if _normalize(text) == _normalize(only):
            return only

    if labels:
        # Prefer an exact canonical surface form before substring matching.
        # This matters for overlapping label sets such as Fashion-MNIST, where
        # ``Shirt`` is a substring of ``T - shirt / top``.
        for lab in sorted(set(labels), key=len, reverse=True):
            if _normalize(text) == _normalize(lab) or _classification_key(text) == _classification_key(lab):
                return lab

        best_score = (-1, -1)
        best_lab = ""
        for lab in sorted(set(labels), key=len, reverse=True):
            pos = _rfind_allowed_label(text, lab)
            # Rank by the end of the match, then by label length. A longer
            # overlapping label should beat a suffix label starting later.
            score = (pos + len(lab), len(lab)) if pos >= 0 else (-1, -1)
            if score > best_score:
                best_score = score
                best_lab = lab
        if best_lab:
            return best_lab

    rk = _classification_key(text)
    if labels and rk:
        rk_alias = _CLASS_LABEL_ALIASES.get(rk, rk)
        for lab in sorted(set(labels), key=lambda x: len(_classification_key(x)), reverse=True):
            lk = _classification_key(lab)
            if lk == rk_alias:
                return lab
        for lab in sorted(set(labels), key=lambda x: len(_classification_key(x)), reverse=True):
            lk = _classification_key(lab)
            if len(lk) >= 4 and lk in rk:
                return lab

    tail = _extract_relation_title(text)
    if labels:
        for lab in sorted(set(labels), key=len, reverse=True):
            if _normalize(tail) == _normalize(lab) or _classification_key(tail) == _classification_key(lab):
                return lab
    return tail.strip() if tail else text.strip()


def _classification_key(s: str) -> str:
    """Loose key for English / snake_case class names (Fashion-MNIST spacing variants)."""
    t = re.sub(r"[\s_\-/]+", "", str(s or "").lower())
    return t


def _extract_relation_title(raw: str) -> str:
    """Pull a short kinship / relation label from CoT-heavy model outputs."""
    text = str(raw or "").strip()
    if not text:
        return ""
    fence = re.search(r"```(?:text|markdown)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    if fence:
        inner = fence.group(1).strip()
        if inner and len(inner) < 120:
            text = inner
    patterns = [
        r"(?:the\s+)?final\s+answer\s+is\s*[:\s]+([^\n#]+?)(?=\n|\.(?:\s|$)|##|\Z)",
        r"relationship\s+title\s+(?:is\s*)?[:\s]+([^\n#]+?)(?=\n|\.(?:\s|$)|##|\Z)",
        r"\banswer\s*[:\s]+([^\n#]+?)(?=\n|\.(?:\s|$)|##|\Z)",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            frag = m.group(1).strip().strip("`\"'").strip()
            frag = re.split(r"[\n]", frag, maxsplit=1)[0].strip()
            frag = re.split(r"(?:\.|;)\s*", frag, maxsplit=1)[0].strip()
            frag = frag.strip(" `'\"")
            if frag:
                return frag
    mq = re.findall(r"['\"]([a-zA-Z][a-zA-Z\s\-]{0,46})['\"]", text)
    if mq:
        cand = mq[-1].strip()
        if len(cand.split()) <= 4:
            return cand
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        tail = lines[-1].rstrip(".,;:")
        tail = tail.strip("`\"'")
        if len(tail) <= 72 and tail.count(" ") <= 5 and not tail.startswith("#"):
            return tail
    return text.strip()


def _canonical_relation_label(raw: str) -> str:
    text = str(raw or "").strip().lower()
    if not text:
        return ""
    t = text.strip(" `\"'.,;:[](){}")
    t = t.replace("_", "-").replace(" ", "-")
    t = (
        t.replace("motherinlaw", "mother-in-law")
        .replace("fatherinlaw", "father-in-law")
        .replace("soninlaw", "son-in-law")
        .replace("daughterinlaw", "daughter-in-law")
    )
    t = _RELATION_PLURAL_TO_SINGULAR.get(t, t)
    if t in _RELATION_LABELS:
        return t
    normalized_text = (
        str(raw or "")
        .lower()
        .replace("_", "-")
        .replace("mother in law", "mother-in-law")
        .replace("father in law", "father-in-law")
        .replace("son in law", "son-in-law")
        .replace("daughter in law", "daughter-in-law")
    )
    found: list[tuple[int, str]] = []
    for lb in sorted(_RELATION_LABELS, key=len, reverse=True):
        for m in re.finditer(r"(?<![a-z-])" + re.escape(lb) + r"(?![a-z-])", normalized_text):
            found.append((m.start(), lb))
    if found:
        found.sort(key=lambda x: x[0])
        return found[-1][1]
    return t


def _extract_visual_qa_final_answer(text: str, *, question: str = "") -> str:
    """Extract the short final answer used by visual-QA grading after a long rationale."""
    t = str(text or "").strip()
    if not t:
        return ""
    low = t.lower()
    key = "final answer:"
    idx = low.rfind(key)
    if idx >= 0:
        rest = t[idx + len(key) :].lstrip()
        rest = rest.split("\n")[0].strip()
        rest = rest.strip("*`\"' ")
        if rest:
            return rest
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    cand = lines[-1] if lines else t
    cand = cand.strip()
    # Collapse obvious degeneration tails, e.g., "444444444444" -> "4", "21 21 21" -> "21".
    cand = re.sub(r"(\d)\1{4,}", r"\1", cand)
    cand = re.sub(r"([A-Za-z])\1{4,}", r"\1", cand)
    cand = re.sub(r"(\b\d{1,4}%?\b)(?:\s*\1){2,}", r"\1", cand, flags=re.IGNORECASE)
    # If question is asking for percentage and answer is bare number, keep the unit.
    ql = str(question or "").lower()
    if re.fullmatch(r"\d{1,3}(?:\.\d+)?", cand) and any(k in ql for k in ("percent", "percentage", "%")):
        cand = f"{cand}%"
    return cand


def extract_final_answer_text(
    raw: str,
    *,
    task_type: str,
    ref_answer: str,
    choices: Optional[Sequence[Any]] = None,
    question: Optional[str] = None,
    **_: Any,
) -> str:
    tt = str(task_type or "")
    text = str(raw or "")
    _ = ref_answer
    if tt in ("math_qa", "math_competition"):
        n = _extract_last_number(text)
        return n if n else text.strip()
    if tt == "relation_extraction":
        return _canonical_relation_label(_extract_relation_title(text))
    if tt in ("binary_qa", "multiple_choice", "logical_label"):
        return _extract_choice(text, choices)
    if tt == "code_generation":
        m = re.search(r"```(?:python)?\s*([\s\S]*?)```", text)
        if m:
            return m.group(1).strip()
        return text.strip()
    if tt == "symbolic_sequence":
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        return lines[-1] if lines else text.strip()
    if tt == "image_classification":
        q = str(question or "")
        allowed = _parse_allowed_labels_from_classification_question(q)
        if not allowed and choices:
            allowed = [str(c).strip() for c in choices if str(c).strip()]
        if not allowed:
            ql = re.sub(r"\s+", "", q.lower())
            if "fashion-mnist" in ql or ("fashion" in q.lower() and "mnist" in q.lower()):
                allowed = list(_FASHION_MNIST_LABELS)
        if not allowed and str(ref_answer or "").strip():
            allowed = [str(ref_answer).strip()]
        return _extract_image_classification_label(
            text,
            allowed_labels=allowed,
            ref_answer=str(ref_answer or ""),
        )
    if tt == "visual_qa":
        return _extract_visual_qa_final_answer(text, question=str(question or ""))
    return text.strip()


def _pair_correct(pred: str, gold: str, task_type: str) -> bool:
    tt = str(task_type or "")
    if tt == "relation_extraction":
        return _canonical_relation_label(pred) == _canonical_relation_label(gold)
    if tt == "multihop_qa":
        return token_level_f1(pred, gold) >= 0.5
    if tt == "image_classification":
        if _normalize(pred) == _normalize(gold):
            return True
        return _classification_key(pred) == _classification_key(gold)
    if tt == "symbolic_sequence":
        return _normalize(pred) == _normalize(gold)
    if tt in ("math_qa", "math_competition"):
        pn, gn = _extract_last_number(pred), _extract_last_number(gold)
        if pn and gn:
            try:
                return abs(float(pn) - float(gn)) < 1e-6
            except ValueError:
                return pn.strip() == gn.strip()
        return _normalize(pred) == _normalize(gold)
    gnorm = gold.strip().lower()
    if gnorm == "uncertain":
        gnorm = "unknown"
    if gnorm in ("yes", "no"):
        return _extract_choice(pred, None).lower() == gnorm
    if gnorm in ("a", "b", "c", "d", "e"):
        return _extract_choice(pred, None).lower() == gnorm
    if gnorm in ("true", "false", "unknown"):
        pnorm = _extract_choice(pred, None)
        if gnorm == "true":
            return pnorm == "True"
        if gnorm == "false":
            return pnorm == "False"
        return pnorm == "Unknown"
    return _normalize(pred) == _normalize(gold)


def exact_match_numbers(pred: str, gold: str) -> bool:
    pn, gn = _extract_last_number(pred), _extract_last_number(gold)
    if pn and gn:
        try:
            return abs(float(pn) - float(gn)) < 1e-6
        except ValueError:
            return pn.strip() == gn.strip()
    return _normalize(pred) == _normalize(gold)


def exact_match_strings(pred: str, gold: str) -> bool:
    return _normalize(pred) == _normalize(gold)


def f1(pred: str, ref: str) -> float:
    return token_level_f1(pred, ref)


def accuracy(
    preds: Sequence[str],
    refs: Sequence[str],
    task_type: str = "multiple_choice",
    **_: Any,
) -> float:
    """Batch accuracy / exact-match rate compatible with offline log builders."""
    p_list = list(preds)
    r_list = list(refs)
    if len(p_list) != len(r_list):
        raise ValueError("preds and refs length mismatch")
    m = compute_metrics(str(task_type), p_list, r_list)
    for k in ("accuracy", "exact_match", "f1"):
        if k in m:
            return float(m[k])
    return 0.0


def compute_metrics(task_type: str, parsed: Sequence[str], refs: Sequence[str]) -> dict:
    p_list = list(parsed)
    r_list = list(refs)
    if len(p_list) != len(r_list):
        raise ValueError("parsed and refs length mismatch")
    n = len(p_list)
    if n == 0:
        return {"accuracy": 0.0}
    if str(task_type) == "multihop_qa":
        f1s = [token_level_f1(a, b) for a, b in zip(p_list, r_list)]
        return {"f1": float(sum(f1s)) / float(n)}
    tot = sum(1 for a, b in zip(p_list, r_list) if _pair_correct(a, b, str(task_type)))
    key = "exact_match" if str(task_type) in ("math_qa", "math_competition", "symbolic_sequence") else "accuracy"
    return {key: float(tot) / float(n)}


def _code_timeout(_signum: int, _frame: Any) -> None:
    raise TimeoutError("generated-code test timed out")


def _run_code_tests(predicted_code: str, extra: dict[str, Any]) -> bool:
    """Run benchmark tests only after the caller explicitly accepts the risk.

    Python ``exec`` is required to reproduce HumanEval/MBPP pass@1 but is not a
    security sandbox. Public documentation instructs users to run this path in
    an isolated container with no secrets or network access.
    """
    if os.environ.get("SOT_ALLOW_UNSAFE_CODE_EVAL", "").lower() not in {"1", "true", "yes"}:
        raise RuntimeError(
            "code evaluation executes model-generated Python; run inside an isolated container "
            "and set SOT_ALLOW_UNSAFE_CODE_EVAL=1"
        )
    code = str(predicted_code or "").strip()
    raw_tests = extra.get("tests")
    tests = (
        "\n".join(str(item).strip() for item in raw_tests if str(item).strip())
        if isinstance(raw_tests, (list, tuple))
        else str(raw_tests or "").strip()
    )
    if not code or not tests:
        return False
    entry_point = extra.get("entry_point")
    if not entry_point:
        match = re.search(r"def\s+([a-zA-Z_]\w*)\s*\(", code)
        if not match:
            return False
        entry_point = match.group(1)
    namespace: dict[str, Any] = {}
    timeout = int(os.environ.get("SOT_CODE_TEST_TIMEOUT_SEC", "60"))
    use_alarm = timeout > 0 and hasattr(signal, "SIGALRM") and threading.current_thread() is threading.main_thread()
    previous = signal.signal(signal.SIGALRM, _code_timeout) if use_alarm else None
    if use_alarm:
        signal.alarm(timeout)
    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            exec(code, namespace)
            function = namespace.get(str(entry_point))
            if function is None:
                return False
            exec(tests, namespace)
            check = namespace.get("check")
            if callable(check):
                check(function)
            return True
    except Exception:
        return False
    finally:
        if use_alarm:
            signal.alarm(0)
            if previous is not None:
                signal.signal(signal.SIGALRM, previous)


def is_correct(
    prediction: str,
    reference: str,
    task_type: str,
    *,
    extra: Optional[dict[str, Any]] = None,
    choices: Optional[Sequence[Any]] = None,
) -> bool:
    """Paper-aligned single-example grading."""
    task = str(task_type or "")
    metadata = dict(extra or {})
    if task == "code_generation":
        return _run_code_tests(str(prediction or ""), metadata)
    if task in {"binary_qa", "multiple_choice", "logical_label", "visual_reasoning_mc"}:
        gold = str(reference or "").strip().lower()
        if gold == "uncertain":
            gold = "unknown"
        parsed = _extract_choice(prediction, choices)
        if gold in {"a", "b", "c", "d", "e", "yes", "no"}:
            return parsed.lower() == gold
        if gold in {"true", "false", "unknown"}:
            return parsed.lower() == gold
    return _pair_correct(prediction, reference, task)


__all__ = [
    "_extract_choice",
    "_extract_relation_title",
    "_extract_last_number",
    "_normalize_answer",
    "_normalize",
    "token_level_f1",
    "accuracy",
    "exact_match_numbers",
    "exact_match_strings",
    "f1",
    "compute_metrics",
    "extract_final_answer_text",
    "is_correct",
]
