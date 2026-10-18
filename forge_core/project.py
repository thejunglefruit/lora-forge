# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 The Jungle Fruit
"""Project descriptor (forge.yaml), dataset manifest and reproducibility hashes."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import secrets
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .paths import ProjectPaths


@dataclass
class ProjectDescriptor:
    name: str
    lora_type: str
    model: str
    trigger: str | None
    eval_prompts: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: _dt.datetime.now().isoformat(timespec="seconds"))

    def save(self, paths: ProjectPaths) -> None:
        paths.descriptor.write_text(
            yaml.safe_dump(asdict(self), sort_keys=False, allow_unicode=True), encoding="utf-8"
        )

    @classmethod
    def load(cls, paths: ProjectPaths) -> "ProjectDescriptor":
        data = yaml.safe_load(paths.descriptor.read_text(encoding="utf-8"))
        return cls(**data)


def suggest_trigger(name: str) -> str:
    """A rare trigger token, unlikely to collide with the model's vocabulary."""
    slug = "".join(ch for ch in name.lower() if ch.isalnum())[:6] or "subj"
    return f"{slug}{secrets.token_hex(2)}"


# --- manifest ---------------------------------------------------------------
def write_manifest(paths: ProjectPaths, images: list[dict[str, Any]], meta: dict[str, Any]) -> None:
    payload = {"meta": meta, "images": images}
    paths.manifest.parent.mkdir(parents=True, exist_ok=True)
    paths.manifest.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_manifest(paths: ProjectPaths) -> dict[str, Any]:
    if not paths.manifest.exists():
        return {"meta": {}, "images": []}
    return json.loads(paths.manifest.read_text(encoding="utf-8"))


def dataset_report_markdown(name: str, model: str, lora_type: str, report: Any) -> str:
    """The human-readable dataset report (shared by the CLI and the web app)."""
    lines = [f"# Dataset report · {name}", "",
             f"- model: {model}", f"- type: {lora_type}",
             f"- images: {report.n}", f"- flagged: {report.summary['flagged']}",
             f"- face detection: {'available' if report.face_detection_available else 'no (GPU box only)'}",
             "",
             "| image | WxH | AR | blur_var | faces | flags |",
             "|---|---|---|---|---|---|"]
    for im in report.images:
        lines.append(f"| {im.path.name} | {im.width}x{im.height} | {im.aspect_ratio:.2f} | "
                     f"{im.blur_var:.0f} | {im.faces if im.faces is not None else '-'} | "
                     f"{'; '.join(im.flags) or '-'} |")
    return "\n".join(lines) + "\n"


# --- reproducibility hashes -------------------------------------------------
def hash_manifest(paths: ProjectPaths) -> str:
    if not paths.manifest.exists():
        return "none"
    return hashlib.sha256(paths.manifest.read_bytes()).hexdigest()[:12]


def hash_captions(paths: ProjectPaths) -> str:
    """Hash of the set of captions (.txt). Captions are edited by hand, so they must be
    pinned too for a run to be reproducible."""
    h = hashlib.sha256()
    txts = sorted(paths.dataset.glob("*.txt")) if paths.dataset.exists() else []
    if not txts:
        return "none"
    for t in txts:
        h.update(t.name.encode("utf-8"))
        h.update(t.read_bytes())
    return h.hexdigest()[:12]
