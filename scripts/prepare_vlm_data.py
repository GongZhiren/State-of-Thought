#!/usr/bin/env python3
"""Materialize the three pinned vision-language evaluation subsets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
LETTERS = "ABCDEFGH"
SPECS = {
    "aokvqa_val": {
        "hub": "HuggingFaceM4/A-OKVQA",
        "revision": "d1b0efa3a436e9101dfbde3752db7607da696c35",
        "split": "validation",
        "n": 200,
    },
    "ai2d": {
        "hub": "lmms-lab/ai2d",
        "revision": "c83a9b9692933aff8349157c88a413df9d02c4e5",
        "split": "test",
        "n": 150,
    },
    "m3cot": {
        "hub": "LightChen2333/M3CoT",
        "revision": "48cf35001d595a6b0290c82c897a4b4563390821",
        "split": "test",
        "n": 150,
    },
}


def _answer_index(answer: Any, choices: list[Any]) -> int | None:
    if isinstance(answer, int):
        return answer
    value = str(answer).strip()
    if value.isdigit():
        return int(value)
    if len(value) == 1 and value.upper() in LETTERS:
        return LETTERS.index(value.upper())
    try:
        return choices.index(answer)
    except ValueError:
        return None


def _iter_rows(slug: str, source: Iterable[dict[str, Any]]) -> Iterable[tuple[int, dict[str, Any]]]:
    for index, example in enumerate(source):
        image = example.get("image")
        choices = list(example.get("choices") or example.get("options") or [])
        if slug == "aokvqa_val":
            answer_index = example.get("correct_choice_idx")
        else:
            answer_index = _answer_index(example.get("answer"), choices)
        if image is None or not choices or not isinstance(answer_index, int):
            continue
        if not 0 <= answer_index < len(choices):
            continue
        yield index, {"example": example, "image": image, "choices": choices, "answer_index": answer_index}


def prepare(slug: str, output_root: Path, *, overwrite: bool) -> Path:
    spec = SPECS[slug]
    output_dir = output_root / slug
    output_file = output_dir / "eval_fixed.jsonl"
    if output_file.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {output_file}; pass --overwrite explicitly")
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    source = load_dataset(
        str(spec["hub"]),
        split=str(spec["split"]),
        revision=str(spec["revision"]),
    )
    rows: list[dict[str, Any]] = []
    for source_index, item in _iter_rows(slug, source):
        if len(rows) >= int(spec["n"]):
            break
        example = item["example"]
        choices = item["choices"]
        sample_id = f"{slug}_{source_index:05d}"
        image_path = image_dir / f"{sample_id}.png"
        item["image"].convert("RGB").save(image_path)
        question = str(example.get("question") or "").strip()
        context = ""
        if slug == "m3cot":
            context = str(example.get("context") or example.get("hint") or "").strip()
        options = "\n".join(f"{LETTERS[i]}. {choice}" for i, choice in enumerate(choices))
        answer_instruction = (
            "Think step by step, then answer with the single letter."
            if slug == "ai2d"
            else "Think step by step, then answer with the single letter of the correct option."
        )
        prompt = (
            question
            + "\n"
            + (f"Context: {context}\n" if context else "")
            + f"Options:\n{options}\n{answer_instruction}"
        )
        rows.append(
            {
                "id": sample_id,
                "question": prompt,
                "answer": LETTERS[item["answer_index"]],
                "image_path": str(image_path.relative_to(ROOT)),
                "metadata": {
                    "dataset": slug,
                    "hub": spec["hub"],
                    "revision": spec["revision"],
                    "split": spec["split"],
                    "source_index": source_index,
                    "task_type": "visual_reasoning_mc",
                    "n_choices": len(choices),
                },
            }
        )
    if len(rows) != int(spec["n"]):
        raise RuntimeError(f"{slug}: expected {spec['n']} valid rows, found {len(rows)}")
    output_file.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return output_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["all", *SPECS], default="all")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data" / "vlm")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    selected = list(SPECS) if args.dataset == "all" else [args.dataset]
    for slug in selected:
        print(f"wrote {prepare(slug, args.output_root.resolve(), overwrite=args.overwrite)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
