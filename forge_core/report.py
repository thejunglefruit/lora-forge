# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 The Jungle Fruit
"""Run note in Markdown with YAML frontmatter (works in Obsidian and any Markdown tool).

Includes a FINDINGS section: each finding names the place in the code or registry where
it should be frozen, so what a run teaches is not lost.
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Finding:
    finding: str
    freeze_in: str

    def render(self) -> str:
        return f"- **{self.finding}** → freeze in `{self.freeze_in}`"


@dataclass
class RunNote:
    run_id: str
    model: str
    lora_type: str
    trigger: str | None
    dataset_stats: dict[str, Any]
    config_hash: str
    manifest_hash: str
    caption_hash: str
    steps: int
    gpu_model: str | None
    duration_min: float | None
    cost_usd: float | None
    commercial_use: Any = None
    license_id: str | None = None
    sample_images: list[str] = field(default_factory=list)
    eval_prompts: list[str] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)

    def _frontmatter(self) -> dict[str, Any]:
        return {
            "type": "lora-run",
            "run_id": self.run_id,
            "model": self.model,
            "lora_type": self.lora_type,
            "trigger": self.trigger,
            "steps": self.steps,
            "gpu": self.gpu_model,
            "duration_min": round(self.duration_min, 1) if self.duration_min else None,
            "cost_usd": round(self.cost_usd, 2) if self.cost_usd is not None else None,
            "config_hash": self.config_hash,
            "manifest_hash": self.manifest_hash,
            "caption_hash": self.caption_hash,
            "license": self.license_id,
            "commercial_use": self.commercial_use,
            "date": _dt.date.today().isoformat(),
            "tags": ["lora-forge", f"model/{self.model}", f"type/{self.lora_type}"],
        }

    def render(self) -> str:
        fm = yaml.safe_dump(self._frontmatter(), sort_keys=False, allow_unicode=True)
        parts = [f"---\n{fm}---\n", f"# LoRA run · {self.run_id}\n"]

        if self.commercial_use is False:
            parts.append(
                f"> [!warning] NON-commercial licence ({self.license_id}). "
                f"The resulting LoRA inherits that restriction: do not sell it without checking.\n"
            )
        elif self.commercial_use is None:
            parts.append(
                "> [!note] Commercial use TO VERIFY. Confirm the base model's licence.\n"
            )

        parts.append("## Dataset\n")
        for k, v in self.dataset_stats.items():
            parts.append(f"- **{k}**: {v}")
        parts.append("")

        parts.append("## Reproducibility")
        parts.append(f"- manifest_hash: `{self.manifest_hash}`")
        parts.append(f"- caption_hash: `{self.caption_hash}`")
        parts.append(f"- config_hash: `{self.config_hash}`\n")

        if self.eval_prompts:
            parts.append("## Evaluation prompts (fixed)")
            for i, p in enumerate(self.eval_prompts, 1):
                parts.append(f"{i}. {p}")
            parts.append("")

        parts.append("## Samples")
        if self.sample_images:
            for s in self.sample_images:
                parts.append(f"![[{s}]]")
        else:
            parts.append("_(no samples)_")
        parts.append("")

        parts.append("## FINDINGS TO FREEZE")
        if self.findings:
            parts += [f.render() for f in self.findings]
        else:
            parts.append("_(none)_")
        parts.append("")

        parts.append("## Subjective notes")
        parts.append("<!-- fill in by hand: identity, flexibility, artefacts, bleed -->\n")
        return "\n".join(parts)


def write_run_note(note: RunNote, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(note.render(), encoding="utf-8")
    return path
