# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 The Jungle Fruit
"""Real captioner backed by a local vision model served by Ollama (default: minicpm-v4.5:8b,
override with FORGE_CAPTION_MODEL). It follows the doctrine (what to describe / what to omit)
and returns (natural_language, tags). Without a free GPU it runs on CPU: slow, but it works.
"""
from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

from ..captionrules import CaptionDoctrine

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
MODEL = os.environ.get("FORGE_CAPTION_MODEL", "minicpm-v4.5:8b")


class OllamaCaptioner:
    name = f"ollama:{MODEL}"

    def __init__(self, model: str = MODEL, url: str = OLLAMA_URL, timeout: int = 300):
        self.model = model
        self.url = url
        self.timeout = timeout
        self.name = f"ollama:{model}"

    @staticmethod
    def available(url: str = OLLAMA_URL, model: str = MODEL) -> bool:
        try:
            with urllib.request.urlopen(f"{url}/api/tags", timeout=5) as r:
                names = [m.get("name", "") for m in json.loads(r.read()).get("models", [])]
            return any(n == model or n.split(":")[0] == model.split(":")[0] for n in names)
        except Exception:  # noqa: BLE001
            return False

    def _ask(self, image_b64: str, prompt: str) -> str:
        body = {"model": self.model, "stream": False, "options": {"temperature": 0.2, "num_predict": 220},
                "messages": [{"role": "user", "content": prompt, "images": [image_b64]}]}
        req = urllib.request.Request(f"{self.url}/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())["message"]["content"].strip()

    def caption(self, image_path: Path, doctrine: CaptionDoctrine) -> tuple[str, str]:
        b64 = base64.b64encode(image_path.read_bytes()).decode()
        nl = ""
        tags = ""
        if doctrine.caption_engine in ("natural", "dual"):
            nl = self._ask(b64, "Describe this image in one English sentence, for training an image model. "
                                + doctrine.instruction + " No preamble, no quotes, no lists: just the sentence.")
            nl = nl.replace("\n", " ").strip().strip('"')
        if doctrine.caption_engine in ("tags", "dual"):
            tags = self._ask(b64, "Give 8 to 15 English tags, lowercase, comma-separated, booru style, "
                                  "describing ONLY the following: " + ", ".join(doctrine.include) + ". "
                                  "Do not describe: " + ", ".join(doctrine.omit) + ". Only the list, no sentences.")
            tags = ", ".join(t.strip().strip(".") for t in tags.replace("\n", ",").split(",") if t.strip())
        return nl, tags
