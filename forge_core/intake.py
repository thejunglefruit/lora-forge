# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 The Jungle Fruit
"""Requirements cards per (type, model), read from the registry.

Printed BEFORE any image is accepted. Plain human text, not telegraphic.
"""
from __future__ import annotations

from dataclasses import dataclass

from .registry import ModelSpec

_TYPE_ADVICE = {
    "character": [
        "VARIED angles, expressions, lighting and backgrounds.",
        "Exactly ONE subject per image.",
        "<=40% close-ups: include medium and full-body shots.",
        "Avoid sunglasses, heavy occlusion of the face and motion blur.",
        "The trigger absorbs the identity; the face is NOT captioned.",
    ],
    "style": [
        "DIVERSE content that shares one coherent style.",
        "Caption the CONTENT of each image, never the style (it is learnt as the residual).",
        "Do not mix different styles in the same dataset.",
    ],
    "object": [
        "Full coverage of the object's angles.",
        "Varied backgrounds and lighting; the object always recognisable.",
        "Caption the trigger + context, NOT the object's own attributes.",
    ],
}


@dataclass
class RequirementsCard:
    lora_type: str
    model: ModelSpec
    imgs_min: int
    imgs_rec: int
    res_floor: int
    advice: list[str]
    max_closeup_ratio: float | None = None

    def render(self) -> str:
        lic = self.model.license
        commercial = lic.get("commercial_use")
        if commercial is True:
            lic_line = f"Licence: {lic.get('id', '?')} (commercial use: YES, verify)"
        elif commercial is False:
            lic_line = f"Licence: {lic.get('id', '?')} (commercial use: NO)"
        else:
            lic_line = f"Licence: {lic.get('id', 'to verify')} (commercial use: TO VERIFY)"

        lines = [
            f"REQUIREMENTS CARD  ·  {self.lora_type.upper()} → {self.model.display_name}",
            "-" * 60,
            f"Images: minimum {self.imgs_min}, recommended {self.imgs_rec}",
            f"Resolution: short side >= {self.res_floor}px",
        ]
        if self.max_closeup_ratio is not None:
            lines.append(f"Close-ups: <= {int(self.max_closeup_ratio * 100)}%")
        lines.append("")
        lines += [f"  - {a}" for a in self.advice]
        lines += ["", lic_line]
        if self.model.status != "active":
            lines.append(f"[!] Model '{self.model.model_id}' status={self.model.status}: "
                         f"verify arch/name_or_path against ai-toolkit before training.")
        return "\n".join(lines)


def build_card(model: ModelSpec, lora_type: str) -> RequirementsCard:
    reqs = model.requirements(lora_type)
    return RequirementsCard(
        lora_type=lora_type,
        model=model,
        imgs_min=int(reqs["imgs_min"]),
        imgs_rec=int(reqs["imgs_rec"]),
        res_floor=int(reqs["res_floor"]),
        advice=_TYPE_ADVICE.get(lora_type, []),
        max_closeup_ratio=reqs.get("max_closeup_ratio"),
    )
