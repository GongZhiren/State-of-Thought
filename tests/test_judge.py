from pathlib import Path

import joblib
import numpy as np

from sot.judge import JudgeExample, evaluate_judge, sentence_scalar_features, trajectory_features

ROOT = Path(__file__).resolve().parents[1]


def test_scalar_and_trajectory_feature_layout() -> None:
    sentences = ("First step.", "Second step is longer.")
    projected = np.asarray([[1.0, 3.0], [5.0, 7.0]], dtype=np.float32)
    features = trajectory_features(projected, sentences)
    expected = np.concatenate(
        (
            projected.mean(axis=0),
            projected.std(axis=0) + 1e-8,
            sentence_scalar_features(sentences),
        )
    )
    np.testing.assert_allclose(features, expected, rtol=0, atol=1e-7)
    assert features.shape == (7,)


def test_released_judge_matches_direct_sklearn_pipeline() -> None:
    artifact_path = ROOT / "checkpoints/exploratory/judge/openai.joblib"
    artifact = joblib.load(artifact_path)
    examples = [
        JudgeExample("a", "toy", ("One.", "Two."), True),
        JudgeExample("b", "toy", ("A longer single reasoning step." ,), False),
    ]
    rng = np.random.default_rng(7)
    embeddings = rng.normal(size=(3, 3072)).astype(np.float32)

    def embed(_sentences: list[str] | tuple[str, ...]) -> np.ndarray:
        return embeddings

    summary, rows = evaluate_judge(artifact_path, examples, embed=embed)
    projected = artifact["pca"].transform(embeddings)
    direct_features = np.stack(
        (
            trajectory_features(projected[0:2], examples[0].sentences),
            trajectory_features(projected[2:3], examples[1].sentences),
        )
    )
    expected = artifact["clf"].predict(direct_features)
    assert [row["pred_correct_class"] for row in rows] == [bool(value) for value in expected]
    assert summary["feature_dim"] == 67
