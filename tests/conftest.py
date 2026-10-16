# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 The Jungle Fruit
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def _sharp(size=(1200, 1200), seed=0) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


def _blurry(size=(1200, 1200)) -> Image.Image:
    arr = np.full((size[1], size[0], 3), 128, dtype=np.uint8)
    arr[:, : size[0] // 2] = 120  # a soft step -> low Laplacian variance
    return Image.fromarray(arr, "RGB")


@pytest.fixture
def dataset_dir(tmp_path: Path) -> Path:
    """Five synthetic images (noise, no people): two good, one duplicate, one blurry, one too small."""
    d = tmp_path / "raw"
    d.mkdir()
    _sharp(seed=1).save(d / "a.png")
    _sharp(seed=2).save(d / "b.png")
    _sharp(seed=1).save(d / "c.png")                 # duplicate of a.png
    _blurry().save(d / "d.png")                      # blurry
    _sharp((300, 300), seed=3).save(d / "e.png")     # low resolution
    return d
