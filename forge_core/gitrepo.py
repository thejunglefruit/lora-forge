# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 The Jungle Fruit
"""Commits METADATA to git (the source of truth). Never heavy binaries.

Only these are added: forge.yaml, manifest.json, dataset_report.md, configs/*.yaml,
runs/**/report.md, calibration.json and the caption .txt files. The .gitignore keeps
out safetensors, forge.db and the images. A thin wrapper over the git CLI.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

# Metadata patterns that ARE versioned.
METADATA_GLOBS = [
    "forge.yaml",
    "calibration.json",
    "dataset/manifest.json",
    "dataset/dataset_report.md",
    "dataset/*.txt",
    "configs/*.yaml",
    "runs/*/report.md",
]


class GitError(Exception):
    pass


def _run(args: list[str], cwd: Path) -> str:
    r = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True)
    if r.returncode != 0:
        raise GitError(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout.strip()


def is_repo(root: Path) -> bool:
    return (root / ".git").exists()


def ensure_identity(root: Path, name: str = "LoRA Forge", email: str = "forge@localhost") -> None:
    """Sets a repo-local identity when there is no global one (avoids a failed commit)."""
    try:
        _run(["config", "user.name"], root)
    except GitError:
        _run(["config", "user.name", name], root)
    try:
        _run(["config", "user.email"], root)
    except GitError:
        _run(["config", "user.email", email], root)


def commit_metadata(root: Path, message: str) -> str | None:
    """Adds only metadata and commits. Returns the short hash, or None if nothing changed."""
    if not is_repo(root):
        raise GitError(f"{root} is not a git repo (run git init first).")
    ensure_identity(root)
    for glob in METADATA_GLOBS:
        matches = list(root.glob(glob))
        if matches:
            _run(["add", "--", *[str(m.relative_to(root)) for m in matches]], root)
    status = _run(["status", "--porcelain"], root)
    if not status:
        return None
    _run(["commit", "-m", message], root)
    return _run(["rev-parse", "--short", "HEAD"], root)
