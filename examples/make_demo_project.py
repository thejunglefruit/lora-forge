# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 The Jungle Fruit
"""Builds a demo project with synthetic images (drawn in code: no photos, no people) and runs
the mock flow up to a config, so the web app has something to show.

    python examples/make_demo_project.py <projects_folder>
    FORGE_PROJECTS=<projects_folder> python forge_web/api.py
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from typer.testing import CliRunner

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forge_cli.main import app  # noqa: E402


def vase(seed: int, size: int = 1024) -> Image.Image:
    """A flat-shaded 'ceramic vase' on a coloured backdrop, with film grain."""
    rnd = random.Random(seed)
    bg = tuple(rnd.randint(60, 200) for _ in range(3))
    im = Image.new("RGB", (size, size), bg)
    d = ImageDraw.Draw(im)
    d.rectangle([0, int(size * 0.72), size, size], fill=tuple(max(0, c - 40) for c in bg))
    cx = size // 2 + rnd.randint(-120, 120)
    body = (rnd.randint(150, 240), rnd.randint(80, 160), rnd.randint(40, 120))
    w, h = rnd.randint(220, 320), rnd.randint(300, 420)
    d.ellipse([cx - w // 2, 700 - h, cx + w // 2, 760], fill=body)
    d.rectangle([cx - w // 6, 700 - h - 140, cx + w // 6, 700 - h + 40], fill=body)
    d.ellipse([cx - w // 5, 700 - h - 170, cx + w // 5, 700 - h - 110], fill=tuple(c - 30 for c in body))
    arr = np.asarray(im).astype(np.int16)
    arr += np.random.default_rng(seed).integers(-18, 18, arr.shape, dtype=np.int16)
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


def main() -> None:
    projects = Path(sys.argv[1]).resolve()
    src = projects.parent / "demo_src"
    src.mkdir(parents=True, exist_ok=True)
    for i in range(7):
        vase(i).save(src / f"vase_{i:02d}.png")
    vase(3).save(src / "vase_03_copy.png")                                  # near-duplicate -> flagged
    vase(9).filter(ImageFilter.GaussianBlur(14)).save(src / "vase_blurry.png")  # blurry -> flagged
    vase(8, 512).save(src / "vase_small.png")                               # low resolution -> flagged

    projects.mkdir(parents=True, exist_ok=True)
    os.chdir(projects)
    runner = CliRunner()
    for args in (["new", "ceramic-vase", "--type", "object", "--model", "flux"],):
        print(runner.invoke(app, args).output)
    os.chdir(projects / "ceramic-vase")
    for args in (["ingest", str(src)], ["caption"], ["config", "--gpu", "RTX 4090", "--gpu-vram", "24"]):
        print(runner.invoke(app, args).output)


if __name__ == "__main__":
    main()
