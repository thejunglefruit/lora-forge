# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 The Jungle Fruit
"""Preflight checks for the GPU box.

Before any heavy operation: GPU/VRAM, free disk and credentials. Degrades cleanly on
a local machine without a GPU instead of failing. Every credential is read from the
environment; nothing is stored in the repo.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field


@dataclass
class PodEnv:
    on_pod: bool
    gpu_name: str | None = None
    vram_total_gb: int | None = None
    vram_free_gb: int | None = None
    disk_free_gb: float | None = None
    has_hf_token: bool = False
    has_object_storage: bool = False
    notes: list[str] = field(default_factory=list)

    def render(self) -> str:
        lines = ["Environment:"]
        lines.append(f"  GPU box: {'yes' if self.on_pod else 'no (local)'}")
        if self.gpu_name:
            lines.append(f"  GPU: {self.gpu_name}  VRAM {self.vram_free_gb}/{self.vram_total_gb} GB free")
        else:
            lines.append("  GPU: not detected (no nvidia-smi)")
        if self.disk_free_gb is not None:
            lines.append(f"  Free disk: {self.disk_free_gb:.0f} GB")
        lines.append(f"  HF token: {'yes' if self.has_hf_token else 'NO'}   "
                     f"object storage: {'yes' if self.has_object_storage else 'NO'}")
        for n in self.notes:
            lines.append(f"  [!] {n}")
        return "\n".join(lines)


def _nvidia_smi() -> tuple[str, int, int] | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return None
        first = out.stdout.strip().splitlines()[0]
        name, total, free = (x.strip() for x in first.split(","))
        return name, int(round(int(total) / 1024)), int(round(int(free) / 1024))
    except Exception:
        return None


def detect_pod_env() -> PodEnv:
    # FORGE_ON_POD=1 marks a rented GPU box; RUNPOD_POD_ID is set automatically on RunPod.
    on_pod = bool(os.environ.get("FORGE_ON_POD") or os.environ.get("RUNPOD_POD_ID"))
    env = PodEnv(on_pod=on_pod)

    smi = _nvidia_smi()
    if smi:
        env.gpu_name, env.vram_total_gb, env.vram_free_gb = smi

    try:
        env.disk_free_gb = shutil.disk_usage("/").free / (1024 ** 3)
    except Exception:
        pass

    env.has_hf_token = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    # Any S3-compatible bucket for outputs, configured with the standard variables.
    env.has_object_storage = bool(os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"))

    if on_pod and env.gpu_name is None:
        env.notes.append("On a GPU box but no GPU visible: check the template.")
    if env.disk_free_gb is not None and env.disk_free_gb < 20:
        env.notes.append(f"Low disk ({env.disk_free_gb:.0f}GB): risky for cached latents/weights.")
    return env


def check_vram_for(vram_needed_gb: int, env: PodEnv | None = None) -> tuple[bool | None, str]:
    env = env or detect_pod_env()
    if env.vram_total_gb is None:
        return None, "VRAM unknown (no local GPU); the real check happens on the GPU box."
    ok = env.vram_total_gb >= vram_needed_gb
    verdict = "OK" if ok else "NOT ENOUGH"
    return ok, f"VRAM {env.vram_total_gb}GB vs needs {vram_needed_gb}GB -> {verdict}"
