# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 The Jungle Fruit
"""Captioning doctrine per (type, model).

Golden rule: whatever you caption becomes promptable (variable); whatever you leave out
is absorbed into the trigger. This module does NOT write captions (that is caption/);
it defines WHAT to describe and WHAT to omit, and gives the captioner its instruction.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CaptionDoctrine:
    lora_type: str
    caption_engine: str          # tags | natural | dual (from the registry, per model)
    trigger_required: bool
    include: list[str]           # what to describe
    omit: list[str]              # what NOT to describe (goes to the trigger / residual)
    instruction: str             # guide prompt for the VLM / captioner


_BY_TYPE = {
    "character": {
        "include": ["clothing", "pose", "background", "lighting", "expression", "framing"],
        "omit": ["facial structure", "fixed identity traits"],
        "instruction": (
            "Describe clothing, pose, background, lighting, expression and framing. "
            "Do NOT describe the facial structure or identity traits of the person "
            "(the trigger learns those)."
        ),
    },
    "style": {
        "include": ["subjects", "composition", "concrete scene content"],
        "omit": ["the style itself", "texture/palette/technique of the style"],
        "instruction": (
            "Describe ONLY the content: subjects, objects and composition. "
            "Do not mention the style, technique, palette or 'look' (it is learnt as the residual)."
        ),
    },
    "object": {
        "include": ["context", "angle", "lighting", "background"],
        "omit": ["the object's own defining attributes"],
        "instruction": (
            "Describe context, angle, lighting and background. "
            "Do NOT describe the object's own defining attributes (the trigger learns those)."
        ),
    },
}


def doctrine_for(lora_type: str, caption_engine: str, trigger_required: bool) -> CaptionDoctrine:
    if lora_type not in _BY_TYPE:
        raise ValueError(f"Unknown LoRA type: {lora_type}")
    d = _BY_TYPE[lora_type]
    return CaptionDoctrine(
        lora_type=lora_type,
        caption_engine=caption_engine,
        trigger_required=trigger_required,
        include=d["include"],
        omit=d["omit"],
        instruction=d["instruction"],
    )
