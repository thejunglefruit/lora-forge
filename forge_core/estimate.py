# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 The Jungle Fruit
"""VRAM, time and cost estimate BEFORE launching a run.

Uses the sec/step measured per GPU (registry / SQLite). When there is no measurement
for that GPU it falls back to a rough, explicitly uncalibrated value. The first real
run on a GPU measures sec/step and stores it, so estimates converge over time.
"""
from __future__ import annotations

from dataclasses import dataclass

from .registry import ModelSpec

# Rough, uncalibrated sec/step per architecture. Only used until a real measurement exists.
_FALLBACK_SEC_PER_STEP = {
    "flux": 3.0,
    "flux2": 4.0,
    "sdxl": 1.2,
    "z_image": 1.0,
    "qwen_image": 3.5,
}


@dataclass
class Estimate:
    steps: int
    sec_per_step: float
    calibrated: bool
    gpu_model: str | None
    vram_needed_gb: int
    vram_available_gb: int | None
    hourly_rate_usd: float

    @property
    def seconds(self) -> float:
        return self.steps * self.sec_per_step

    @property
    def minutes(self) -> float:
        return self.seconds / 60.0

    @property
    def cost_usd(self) -> float:
        return (self.seconds / 3600.0) * self.hourly_rate_usd

    @property
    def vram_ok(self) -> bool | None:
        if self.vram_available_gb is None:
            return None
        return self.vram_available_gb >= self.vram_needed_gb

    def render(self) -> str:
        cal = "measured" if self.calibrated else "UNCALIBRATED (fallback)"
        lines = [
            f"Estimate  ·  {self.steps} steps @ {self.sec_per_step:.2f}s/step ({cal})",
            f"  Time:    ~{self.minutes:.0f} min",
            f"  Cost:    ~${self.cost_usd:.2f}  (rate ${self.hourly_rate_usd:.2f}/h)",
            f"  VRAM:    needs ~{self.vram_needed_gb}GB",
        ]
        if self.vram_available_gb is not None:
            verdict = "OK" if self.vram_ok else "NOT ENOUGH"
            lines.append(f"           available {self.vram_available_gb}GB -> {verdict}")
        else:
            lines.append("           (GPU VRAM unknown; pass --gpu-vram to check)")
        if not self.calibrated:
            lines.append("  [!] sec/step not measured on this GPU; the error can be large until the first run.")
        return "\n".join(lines)


def estimate_run(
    *,
    model: ModelSpec,
    steps: int,
    gpu_model: str | None,
    measured_sec_per_step: float | None,
    hourly_rate_usd: float,
    vram_available_gb: int | None = None,
) -> Estimate:
    if measured_sec_per_step is not None:
        sps, calibrated = measured_sec_per_step, True
    else:
        sps = _FALLBACK_SEC_PER_STEP.get(model.arch, 2.5)
        calibrated = False
    return Estimate(
        steps=steps,
        sec_per_step=sps,
        calibrated=calibrated,
        gpu_model=gpu_model,
        vram_needed_gb=model.vram_gb,
        vram_available_gb=vram_available_gb,
        hourly_rate_usd=hourly_rate_usd,
    )
