# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 The Jungle Fruit
"""Loads and validates the model registry (models.json).

It is the only source of model-specific data: arch, paths, presets, VRAM, thresholds,
requirements and measured sec/step. config_gen / estimate / intake / caption read from here.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REGISTRY_PATH = Path(__file__).with_name("models.json")

LORA_TYPES = ("character", "style", "object")


class RegistryError(Exception):
    pass


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    raw: dict[str, Any]

    @property
    def status(self) -> str:
        return self.raw.get("status", "unverified")

    @property
    def display_name(self) -> str:
        return self.raw.get("display_name", self.model_id)

    @property
    def arch(self) -> str:
        return self.raw["arch"]

    @property
    def name_or_path(self) -> str:
        return self.raw["name_or_path"]

    @property
    def backend(self) -> str:
        return self.raw["backend"]

    @property
    def vram_gb(self) -> int:
        return int(self.raw["vram_gb"])

    @property
    def caption_engine(self) -> str:
        return self.raw["caption_engine"]

    @property
    def trigger_required(self) -> bool:
        return bool(self.raw.get("trigger_required", False))

    @property
    def preset(self) -> dict[str, Any]:
        return self.raw["preset"]

    @property
    def license(self) -> dict[str, Any]:
        return self.raw.get("license", {})

    def sec_per_step(self, gpu_model: str | None = None) -> float | None:
        table = self.raw.get("sec_per_step", {})
        if gpu_model and gpu_model in table:
            return float(table[gpu_model])
        return None

    def requirements(self, lora_type: str) -> dict[str, Any]:
        reqs = self.raw.get("requirements", {})
        if lora_type not in reqs:
            raise RegistryError(
                f"Model '{self.model_id}' defines no requirements for type '{lora_type}'."
            )
        return reqs[lora_type]

    def is_verified(self) -> bool:
        return self.status == "active" and self.name_or_path != "VERIFY"


@dataclass(frozen=True)
class Registry:
    meta: dict[str, Any]
    _models: dict[str, dict[str, Any]]

    @classmethod
    def load(cls, path: Path | None = None) -> "Registry":
        path = path or _REGISTRY_PATH
        if not path.exists():
            raise RegistryError(f"Model registry not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        models = data.get("models")
        if not isinstance(models, dict) or not models:
            raise RegistryError("models.json has no valid 'models' section.")
        reg = cls(meta=data.get("_meta", {}), _models=models)
        reg.validate()
        return reg

    def validate(self) -> None:
        required = {"arch", "name_or_path", "backend", "vram_gb", "caption_engine", "preset"}
        for mid, m in self._models.items():
            missing = required - set(m)
            if missing:
                raise RegistryError(f"Model '{mid}' is incomplete: missing {sorted(missing)}.")
            if m["caption_engine"] not in ("tags", "natural", "dual"):
                raise RegistryError(
                    f"Model '{mid}': invalid caption_engine '{m['caption_engine']}'."
                )

    def ids(self) -> list[str]:
        return list(self._models)

    def active_ids(self) -> list[str]:
        return [k for k, v in self._models.items() if v.get("status") == "active"]

    def get(self, model_id: str) -> ModelSpec:
        if model_id not in self._models:
            raise RegistryError(
                f"Unknown model '{model_id}'. Available: {', '.join(self.ids())}."
            )
        return ModelSpec(model_id, self._models[model_id])

    @property
    def pod_hourly_rate_usd(self) -> float:
        return float(self.meta.get("pod_hourly_rate_usd", 0.0))
