# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 The Jungle Fruit
"""Captioner interface + caption writing helpers.

Idempotence: rewriting captions never overwrites manual edits unless forced.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ..captionrules import CaptionDoctrine


class Captioner(Protocol):
    name: str

    def caption(self, image_path: Path, doctrine: CaptionDoctrine) -> tuple[str, str]:
        """Returns (natural_language, tags). Depending on the engine one may be empty."""
        ...


class MockCaptioner:
    """Development captioner (no GPU). Produces plausible, deterministic text so the
    whole flow can be tested locally. Not for real training."""

    name = "mock"

    def caption(self, image_path: Path, doctrine: CaptionDoctrine) -> tuple[str, str]:
        stem = image_path.stem
        nl = (
            f"a photo, {', '.join(doctrine.include)}, neutral lighting "
            f"[mock caption of {stem}]"
        )
        tags = ", ".join(doctrine.include)
        return nl, tags


def get_captioner(engine: str, prefer_real: bool = False) -> Captioner:
    """Picks a captioner. Without prefer_real it is always the mock.

    With prefer_real it uses a vision model served by Ollama when one is reachable.
    Planned GPU captioners per engine:
      - 'tags'    -> WD14 tagger
      - 'natural' -> JoyCaption / Florence-2 / Qwen2.5-VL
      - 'dual'    -> JoyCaption (NL) + WD14 (tags)
    They are imported lazily so torch is never needed locally.
    """
    if not prefer_real:
        return MockCaptioner()
    from .ollama import OllamaCaptioner  # lazy import

    if OllamaCaptioner.available():
        return OllamaCaptioner()
    raise NotImplementedError(
        "No real captioner available: start Ollama with a vision model "
        "(set FORGE_CAPTION_MODEL) or use the mock captioner."
    )


def build_caption(nl: str, tags: str, engine: str) -> str:
    """Combines natural language + tags according to the model's engine."""
    nl = nl.strip()
    tags = tags.strip()
    if engine == "tags":
        return tags
    if engine == "natural":
        return nl
    # dual: natural language as the body, tags as the last line.
    return f"{nl}\n{tags}" if tags else nl


def inject_trigger(caption: str, trigger: str | None) -> str:
    if not trigger:
        return caption
    if trigger in caption:
        return caption
    return f"{trigger}, {caption}"


def caption_path_for(image_path: Path) -> Path:
    return image_path.with_suffix(".txt")


def write_caption(image_path: Path, text: str) -> Path:
    out = caption_path_for(image_path)
    out.write_text(text.rstrip() + "\n", encoding="utf-8")
    return out
