"""Train and evaluate the trajectory-level SoT-Judge classifier.

The released judge represents a reasoning trajectory by embedding its sentence
units, projecting them to a PCA subspace, and concatenating the coordinate-wise
mean and standard deviation with three inexpensive length features.  The
backbone that produced the trajectory is not needed at judge time.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

EmbeddingFunction = Callable[[Sequence[str]], np.ndarray]


@dataclass(frozen=True)
class JudgeExample:
    """One labeled reasoning trajectory."""

    example_id: str
    dataset: str
    sentences: tuple[str, ...]
    correct: bool


def split_sentences(text: str) -> list[str]:
    """Split a generated trajectory without requiring an NLP model."""

    text = str(text or "").strip()
    if not text:
        return []
    pieces = re.split(r"(?<=[.!?。！？])\s+|\n+", text)
    return [piece.strip() for piece in pieces if piece.strip()]


def read_judge_records(path: str | Path) -> list[JudgeExample]:
    """Read labeled JSONL records accepted by the public judge pipeline."""

    examples: list[JudgeExample] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("error"):
                continue
            units = row.get("sentence_units")
            if isinstance(units, list):
                sentences = [str(unit).strip() for unit in units if str(unit).strip()]
            else:
                sentences = split_sentences(str(row.get("prediction_raw") or row.get("prediction") or ""))
            if not sentences:
                continue
            if "correct" not in row:
                raise ValueError(f"line {line_number}: missing boolean 'correct' label")
            examples.append(
                JudgeExample(
                    example_id=str(row.get("id", row.get("example_id", line_number))),
                    dataset=str(row.get("dataset", "")),
                    sentences=tuple(sentences),
                    correct=bool(row["correct"]),
                )
            )
    if not examples:
        raise ValueError(f"no usable trajectories in {path}")
    return examples


def sentence_scalar_features(sentences: Sequence[str]) -> np.ndarray:
    """Return sentence-count, total-character, and mean-length features."""

    nonempty = [str(sentence) for sentence in sentences if str(sentence).strip()]
    count = max(len(nonempty), 1)
    total_characters = float(sum(len(sentence) for sentence in nonempty))
    return np.asarray(
        [np.log1p(float(count)), np.log1p(total_characters), total_characters / count],
        dtype=np.float32,
    )


def trajectory_features(
    projected_sentences: np.ndarray,
    sentences: Sequence[str],
    *,
    include_scalars: bool = True,
) -> np.ndarray:
    """Aggregate a variable-length projected trajectory into one vector."""

    projected = np.asarray(projected_sentences, dtype=np.float32)
    if projected.ndim != 2 or projected.shape[0] == 0:
        raise ValueError("projected_sentences must have shape [steps, dimensions]")
    base = np.concatenate((projected.mean(axis=0), projected.std(axis=0) + 1e-8))
    if include_scalars:
        base = np.concatenate((base, sentence_scalar_features(sentences)))
    return np.asarray(base, dtype=np.float32)


def openai_embedder(model: str = "text-embedding-3-large", batch_size: int = 64) -> EmbeddingFunction:
    """Create a batched OpenAI embedding function.

    The API key is read at call time from ``OPENAI_API_KEY`` and is never
    serialized into an artifact.
    """

    def embed(texts: Sequence[str]) -> np.ndarray:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise RuntimeError("install the 'judge' extra to use OpenAI embeddings") from exc
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI embeddings")
        client = OpenAI(api_key=api_key)
        vectors: list[list[float]] = []
        for start in range(0, len(texts), max(1, int(batch_size))):
            chunk = list(texts[start : start + max(1, int(batch_size))])
            response = client.embeddings.create(model=model, input=chunk)
            ordered = sorted(response.data, key=lambda item: item.index)
            vectors.extend(item.embedding for item in ordered)
        return np.asarray(vectors, dtype=np.float32)

    return embed


def _flatten_examples(
    examples: Sequence[JudgeExample],
) -> tuple[list[str], list[tuple[int, int]]]:
    sentences: list[str] = []
    spans: list[tuple[int, int]] = []
    for example in examples:
        start = len(sentences)
        sentences.extend(example.sentences)
        spans.append((start, len(sentences)))
    return sentences, spans


def _balance_examples(examples: Sequence[JudgeExample], seed: int) -> list[JudgeExample]:
    """Oversample whole trajectories to match the selected paper protocol."""

    positive = [example for example in examples if example.correct]
    negative = [example for example in examples if not example.correct]
    if not positive or not negative or len(positive) == len(negative):
        return list(examples)
    rng = np.random.default_rng(int(seed))
    minority, majority = (positive, negative) if len(positive) < len(negative) else (negative, positive)
    additions = list(rng.choice(minority, size=len(majority) - len(minority), replace=True))
    balanced = list(examples) + additions
    rng.shuffle(balanced)
    return balanced


def fit_judge(
    examples: Sequence[JudgeExample],
    output_path: str | Path,
    *,
    embed: EmbeddingFunction,
    embedding_model: str = "text-embedding-3-large",
    pca_dim: int = 32,
    seed: int = 42,
    balance: bool = True,
) -> dict[str, Any]:
    """Fit the PCA-trajectory MLP used by the released SoT-Judge variants."""

    try:
        import joblib
        from sklearn.decomposition import PCA
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.utils.class_weight import compute_sample_weight
    except ImportError as exc:  # pragma: no cover - depends on optional packages
        raise RuntimeError("install the 'judge' extra to train SoT-Judge") from exc

    work = _balance_examples(examples, seed) if balance else list(examples)
    if len(work) < 2 or len({example.correct for example in work}) != 2:
        raise ValueError("judge training requires both correct and incorrect trajectories")
    all_sentences, spans = _flatten_examples(work)
    embeddings = np.asarray(embed(all_sentences), dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(all_sentences):
        raise ValueError("embedding function returned an incompatible array")
    components = min(int(pca_dim), embeddings.shape[0], embeddings.shape[1])
    if components < 1:
        raise ValueError("pca_dim must be positive")
    pca = PCA(n_components=components, random_state=int(seed))
    projected = pca.fit_transform(embeddings)
    features = np.stack(
        [trajectory_features(projected[start:end], example.sentences) for example, (start, end) in zip(work, spans)]
    )
    labels = np.asarray([int(example.correct) for example in work], dtype=np.int32)
    classifier = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    alpha=1e-3,
                    max_iter=800,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=25,
                    random_state=int(seed),
                ),
            ),
        ]
    )
    weights = compute_sample_weight("balanced", labels)
    classifier.fit(features, labels, mlp__sample_weight=weights)
    artifact = {
        "pca": pca,
        "clf": classifier,
        "meta": {
            "trajectory_repr": "pca",
            "pca_dim": int(pca_dim),
            "pca_n_components_": int(pca.n_components_),
            "use_scalar_features": True,
            "n_scalars": 3,
            "feature_dim_expected": int(features.shape[1]),
            "classifier": "mlp",
            "balance_oversample": bool(balance),
            "embedding_backend": "openai",
            "openai_embed_model": embedding_model,
            "single_class_train": False,
            "train_n_positive": int(labels.sum()),
            "train_n_negative": int((labels == 0).sum()),
        },
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, destination)
    return {
        "artifact": str(destination),
        "examples": int(len(labels)),
        "feature_dim": int(features.shape[1]),
        "training_accuracy": float((classifier.predict(features) == labels).mean()),
    }


def evaluate_judge(
    artifact_path: str | Path,
    examples: Sequence[JudgeExample],
    *,
    embed: EmbeddingFunction | None = None,
    batch_size: int = 64,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate a released or newly trained judge artifact."""

    try:
        import joblib
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError("install the 'judge' extra to evaluate SoT-Judge") from exc
    artifact = joblib.load(artifact_path)
    metadata = artifact.get("meta") or {}
    pca = artifact.get("pca")
    classifier = artifact.get("clf")
    if pca is None or classifier is None:
        raise ValueError("judge artifact must contain 'pca' and 'clf'")
    if str(metadata.get("trajectory_repr", "pca")) != "pca":
        raise ValueError("this release supports PCA-trajectory judge artifacts")
    model = str(metadata.get("openai_embed_model") or "text-embedding-3-large")
    embed_function = embed or openai_embedder(model=model, batch_size=batch_size)
    all_sentences, spans = _flatten_examples(examples)
    embeddings = np.asarray(embed_function(all_sentences), dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(all_sentences):
        raise ValueError("embedding function returned an incompatible array")
    projected = pca.transform(embeddings)
    include_scalars = bool(metadata.get("use_scalar_features", False))
    features = np.stack(
        [
            trajectory_features(
                projected[start:end], example.sentences, include_scalars=include_scalars
            )
            for example, (start, end) in zip(examples, spans)
        ]
    )
    expected = metadata.get("feature_dim_expected")
    if expected is not None and features.shape[1] != int(expected):
        raise ValueError(f"artifact expects {expected} features, built {features.shape[1]}")
    predictions = np.asarray(classifier.predict(features), dtype=np.int32)
    labels = np.asarray([int(example.correct) for example in examples], dtype=np.int32)
    rows = [
        {
            "id": example.example_id,
            "dataset": example.dataset,
            "gold_correct": bool(label),
            "pred_correct_class": bool(prediction),
        }
        for example, label, prediction in zip(examples, labels, predictions)
    ]
    summary = {
        "artifact": str(artifact_path),
        "n_examples": int(len(labels)),
        "accuracy": float((predictions == labels).mean()),
        "embedding_model": model,
        "feature_dim": int(features.shape[1]),
    }
    return summary, rows


def evaluate_judge_files(
    artifact_path: str | Path,
    records_path: str | Path,
    output_path: str | Path,
    *,
    batch_size: int = 64,
) -> dict[str, Any]:
    """File-oriented wrapper used by the CLI."""

    examples = read_judge_records(records_path)
    summary, rows = evaluate_judge(
        artifact_path, examples, batch_size=batch_size
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary_path = destination.with_name(f"{destination.stem}_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def fit_judge_files(
    records_path: str | Path,
    output_path: str | Path,
    *,
    embedding_model: str = "text-embedding-3-large",
    batch_size: int = 64,
    pca_dim: int = 32,
    seed: int = 42,
) -> dict[str, Any]:
    """File-oriented training wrapper used by the CLI."""

    examples = read_judge_records(records_path)
    return fit_judge(
        examples,
        output_path,
        embed=openai_embedder(model=embedding_model, batch_size=batch_size),
        embedding_model=embedding_model,
        pca_dim=pca_dim,
        seed=seed,
    )
