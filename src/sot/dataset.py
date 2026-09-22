"""Load datasets from ``data/<dataset>/<split>.jsonl`` and ``data/eval_fixed``."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]


def _coerce_gold_answer(row: Dict[str, Any]) -> str:
    """Best-effort gold string from common jsonl conventions (fixes empty test splits, HF keys)."""
    for key in ("answer", "reference_answer", "references", "gold"):
        v = row.get(key)
        if v is None:
            continue
        if isinstance(v, list) and v:
            v = v[0]
        s = str(v).strip()
        if s:
            return s
    lab = row.get("label")
    if lab is not None and str(lab).strip():
        return str(lab).strip()
    meta = row.get("metadata")
    if isinstance(meta, dict):
        for mk in ("answerKey", "answer_key", "correct_answer"):
            ak = meta.get(mk)
            if ak is not None and str(ak).strip():
                return str(ak).strip()
    return ""


def _dataset_yaml_row(dataset: str) -> Dict[str, str]:
    """Return a dataset registry row, including an optional fixed-evaluation filename."""
    candidates = [
        ROOT / "scripts" / "configs" / "datasets.yaml",
        ROOT / "configs" / "datasets.yaml",
    ]
    p = next((c for c in candidates if c.is_file()), None)
    if p is None:
        return {"task_type": "math_qa", "primary_metric": "accuracy"}
    try:
        import yaml

        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:
        raw = {}
    row = (raw.get("datasets") or {}).get(str(dataset).strip(), {}) or {}
    out: Dict[str, str] = {
        "task_type": str(row.get("task_type") or "math_qa"),
        "primary_metric": str(row.get("primary_metric") or "accuracy"),
    }
    ej = row.get("eval_fixed_jsonl")
    if isinstance(ej, str) and ej.strip():
        # Accept only a basename so registry entries cannot escape the data directory.
        out["eval_fixed_jsonl"] = Path(ej.strip()).name
    return out


@dataclass
class DatasetSample:
    id: str
    question: str
    answer: str
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def choices(self) -> Optional[List[Any]]:
        ch = self.extra.get("choices")
        return ch if isinstance(ch, list) else None

    @property
    def tests(self) -> Optional[str]:
        t = self.extra.get("tests")
        return str(t) if t is not None else None

    @property
    def entry_point(self) -> Optional[str]:
        ep = self.extra.get("entry_point")
        return str(ep) if ep is not None else None


def stable_sample_id(dataset: str, row: Dict[str, Any], line_index: int) -> str:
    """Return the source ID or the historical question digest used by paper runs."""
    sid = str(row.get("id") or "").strip()
    if sid:
        return sid
    q = str(row.get("question") or "")
    # SHA-1 is used only as a non-security content identifier. Keeping this
    # historical encoding makes new records line up exactly with the frozen
    # paper records for datasets (notably GSM8K) that omit source IDs.
    return hashlib.sha1(q.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


def _infer_entry_point(question: str) -> Optional[str]:
    import re

    m = re.search(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", question or "")
    return m.group(1) if m else None


def load_dataset(dataset: str, split: str) -> List[DatasetSample]:
    ds = str(dataset).strip()
    sp = str(split).strip()
    if sp == "eval_fixed" and ds.startswith("vlm_"):
        slug = ds[len("vlm_") :].strip()
        path = ROOT / "data" / "vlm" / slug / "eval_fixed.jsonl"
    elif sp == "eval_fixed":
        yrow = _dataset_yaml_row(ds)
        rel = str(yrow.get("eval_fixed_jsonl") or "").strip() or f"{ds}.jsonl"
        path = ROOT / "data" / "eval_fixed" / Path(rel).name
    else:
        path = ROOT / "data" / ds / f"{sp}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"dataset file not found: {path}")

    out: List[DatasetSample] = []
    with path.open(encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            q = str(row.get("question") or "")
            a = _coerce_gold_answer(row)
            extra = {k: v for k, v in row.items() if k not in ("id", "question", "answer", "reference_answer")}
            meta = _dataset_yaml_row(ds)
            extra.setdefault("task_type", meta["task_type"])
            extra.setdefault("primary_metric", meta["primary_metric"])
            if not extra.get("entry_point") and extra.get("tests"):
                ep = _infer_entry_point(q)
                if ep:
                    extra["entry_point"] = ep
            out.append(
                DatasetSample(
                    id=stable_sample_id(ds, row, idx),
                    question=q,
                    answer=a,
                    extra=extra,
                )
            )
    return out


__all__ = ["DatasetSample", "load_dataset", "stable_sample_id"]
