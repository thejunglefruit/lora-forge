# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 The Jungle Fruit
from __future__ import annotations

from pathlib import Path

from forge_core.validate import validate_dataset


def test_flags_lowres_blur_and_dupes(dataset_dir: Path):
    imgs = sorted(dataset_dir.glob("*.png"))
    reqs = {"imgs_min": 15, "imgs_rec": 25, "res_floor": 1024}
    report = validate_dataset(imgs, res_floor=1024, lora_type="character",
                              requirements=reqs, detect_faces=False)

    assert report.n == 5
    by_name = {im.path.name: im for im in report.images}

    # low resolution -> flag + ok=False
    assert any("low resolution" in f for f in by_name["e.png"].flags)
    assert by_name["e.png"].ok is False

    # blur -> flag
    assert any("blurry" in f for f in by_name["d.png"].flags)

    # duplicate a <-> c
    assert any("near-duplicate" in f for f in by_name["a.png"].flags)
    assert any("near-duplicate" in f for f in by_name["c.png"].flags)

    # clean image -> no flags
    assert by_name["b.png"].flags == []

    # small dataset -> count warning, and the flags BLOCK the flow
    assert "count_warning" in report.summary
    assert report.blocked is True
