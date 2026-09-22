from importlib.metadata import version

import sot


def test_runtime_version_matches_distribution_metadata() -> None:
    assert sot.__version__ == version("state-of-thought")
