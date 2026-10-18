# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 The Jungle Fruit
"""Captioners. Common interface + writing the .txt files (ai-toolkit convention).

Heavy GPU captioners (JoyCaption, Florence-2, WD14, Qwen2.5-VL) are planned for the GPU
box. Locally there are two options: the MockCaptioner, to exercise the flow without
downloading anything, and a real vision model served by a local Ollama instance.
The engine (tags / natural / dual) comes from the registry.
"""
from __future__ import annotations

from .base import (
    Captioner,
    MockCaptioner,
    build_caption,
    caption_path_for,
    get_captioner,
    inject_trigger,
    write_caption,
)

__all__ = [
    "Captioner",
    "MockCaptioner",
    "build_caption",
    "caption_path_for",
    "get_captioner",
    "inject_trigger",
    "write_caption",
]
