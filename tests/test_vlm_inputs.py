from pathlib import Path

import pytest
from PIL import Image

from sot.vlm_inputs import load_pil_image


def test_image_path_is_resolved_from_explicit_data_root(tmp_path: Path) -> None:
    image_path = tmp_path / "data" / "example.png"
    image_path.parent.mkdir()
    Image.new("L", (3, 2), color=127).save(image_path)

    with load_pil_image({"image_path": "data/example.png"}, data_root=tmp_path) as image:
        assert image.mode == "RGB"
        assert image.size == (3, 2)


def test_missing_image_path_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="VLM image not found"):
        load_pil_image({"image_path": "data/missing.png"}, data_root=tmp_path)
