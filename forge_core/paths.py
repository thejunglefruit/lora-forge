# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 The Jungle Fruit
"""On-disk layout of a LoRA Forge project.

One "project" == one LoRA. Layout:

    <project>/
      forge.yaml                 descriptor (type, model, trigger, eval prompts)
      forge.db                   SQLite state (gitignored; lives on the GPU volume)
      calibration.json           export of the calibration state (committed)
      dataset/
        <img>.<ext> + <img>.txt  ai-toolkit convention (caption with the same stem)
        manifest.json
        dataset_report.md
      configs/<run_id>.yaml
      runs/<run_id>/train.log, samples/, report.md
      outputs/<run_id>/*.safetensors   (gitignored)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def inside(base: Path, name: str) -> Path:
    """Join `name` onto `base` and refuse anything that escapes it (path traversal)."""
    base_r = base.resolve()
    target = (base_r / name).resolve()
    if os.path.commonpath([str(base_r), str(target)]) != str(base_r) or target == base_r:
        raise ValueError(f"path escapes its folder: {name!r}")
    return target


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @property
    def descriptor(self) -> Path:
        return self.root / "forge.yaml"

    @property
    def db(self) -> Path:
        return self.root / "forge.db"

    @property
    def calibration(self) -> Path:
        return self.root / "calibration.json"

    @property
    def dataset(self) -> Path:
        return self.root / "dataset"

    @property
    def manifest(self) -> Path:
        return self.dataset / "manifest.json"

    @property
    def dataset_report(self) -> Path:
        return self.dataset / "dataset_report.md"

    @property
    def configs(self) -> Path:
        return self.root / "configs"

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def outputs(self) -> Path:
        return self.root / "outputs"

    def config_file(self, run_id: str) -> Path:
        return self.configs / f"{run_id}.yaml"

    def run_dir(self, run_id: str) -> Path:
        return self.runs / run_id

    def train_log(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "train.log"

    def samples_dir(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "samples"

    def run_report(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "report.md"

    def output_dir(self, run_id: str) -> Path:
        return self.outputs / run_id

    def ensure_base(self) -> None:
        for p in (self.root, self.dataset, self.configs, self.runs, self.outputs):
            p.mkdir(parents=True, exist_ok=True)

    def list_images(self) -> list[Path]:
        if not self.dataset.exists():
            return []
        return sorted(
            p for p in self.dataset.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )


def find_project(start: Path | None = None) -> ProjectPaths | None:
    """Walk up the tree looking for forge.yaml, the way git looks for .git."""
    start = (start or Path.cwd()).resolve()
    for cand in (start, *start.parents):
        if (cand / "forge.yaml").exists():
            return ProjectPaths(cand)
    return None
