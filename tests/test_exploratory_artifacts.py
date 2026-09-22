from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
JUDGE_ROOT = ROOT / "checkpoints" / "exploratory" / "judge"

EXPECTED = {
    "anthropic.joblib": {
        "sha256": "93becf04a48bba4a0ba3cdeeeb5553c85f80258acde4b7d02e6a1b5e0ea2ff17",
        "positive": 6,
    },
    "openai.joblib": {
        "sha256": "81c3b528ae2aa8de6a4b71d6a596f74423f5ad7594440e9043a081615524c742",
        "positive": 7,
    },
    "qwen-max.joblib": {
        "sha256": "bb91cb337e44fd7f278351509d3c087beb53bf9260c427537af9af7355a4299a",
        "positive": 7,
    },
}


def _file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("filename", sorted(EXPECTED))
def test_sanitized_judge_checkpoint(filename: str) -> None:
    joblib = pytest.importorskip("joblib")
    payload = joblib.load(JUDGE_ROOT / filename)
    expected = EXPECTED[filename]
    assert _file_hash(JUDGE_ROOT / filename) == expected["sha256"]
    assert payload["meta"]["train_path"] is None
    assert payload["meta"]["feature_dim_expected"] == 67
    assert payload["pca"].n_features_in_ == 3072

    # This fixed probe protects the released estimators against accidental
    # mutation without requiring private training records in the repository.
    rng = np.random.default_rng(90210)
    probe = rng.normal(size=(19, 67))
    predictions = payload["clf"].predict(probe)
    assert int(np.sum(predictions)) == expected["positive"]
