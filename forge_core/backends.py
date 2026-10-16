# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 The Jungle Fruit
"""Detecting and launching training backends.

Current backend: ai-toolkit. The process is launched DETACHED (start_new_session) and
writes its log to a file and its state to SQLite; neither the CLI nor the web app owns
the subprocess, so it survives a closed tab. Without a backend, a MOCK run exercises the
whole flow locally.

Contract: idempotence (never relaunch a finished run), measure the real sec/step so it
can be fed back into the estimates, and never fail silently.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Where ai-toolkit may be installed on the GPU box.
_AITOOLKIT_ENV = "AI_TOOLKIT_PATH"
_AITOOLKIT_CANDIDATES = ("/workspace/ai-toolkit", "/root/ai-toolkit", "./ai-toolkit")

_STEP_RE = re.compile(r"(\d+)\s*/\s*(\d+)")  # heuristic to parse "step/total" from the log


class BackendError(Exception):
    pass


@dataclass
class Backend:
    name: str
    path: Path | None
    available: bool

    def install_hint(self) -> str:
        if self.name == "ai-toolkit":
            return ("ai-toolkit not found. Install it on the GPU box:\n"
                    "  git clone https://github.com/ostris/ai-toolkit /workspace/ai-toolkit\n"
                    f"  export {_AITOOLKIT_ENV}=/workspace/ai-toolkit\n"
                    "(or pass --mock to test the flow without training).")
        return f"Backend '{self.name}' not available."


def detect_aitoolkit() -> Backend:
    env_path = os.environ.get(_AITOOLKIT_ENV)
    candidates = [env_path] if env_path else []
    candidates += list(_AITOOLKIT_CANDIDATES)
    for c in candidates:
        if not c:
            continue
        p = Path(c)
        if (p / "run.py").exists():
            return Backend("ai-toolkit", p, True)
    return Backend("ai-toolkit", None, False)


def launch_training(
    *,
    backend: Backend,
    config_path: Path,
    log_path: Path,
    mock: bool = False,
    mock_steps: int = 40,
) -> "TrainingProcess":
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if mock or not backend.available:
        return _launch_mock(config_path, log_path, mock_steps)
    return _launch_real(backend, config_path, log_path)


def _popen_logged(args: list[str], log_path: Path, **kwargs) -> subprocess.Popen:
    # The child gets its own copy of the handle; the parent closes its copy right away so
    # no open file outlives the launch (on Windows that kept the test process from exiting).
    with open(log_path, "w", encoding="utf-8") as log_f:
        return subprocess.Popen(args, stdout=log_f, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, **kwargs)


def _launch_real(backend: Backend, config_path: Path, log_path: Path) -> "TrainingProcess":
    assert backend.path is not None
    kwargs: dict = {"cwd": str(backend.path)}
    if os.name == "posix":
        kwargs["start_new_session"] = True  # detach from the parent process
    proc = _popen_logged([sys.executable, str(backend.path / "run.py"), str(config_path)], log_path, **kwargs)
    return TrainingProcess(proc=proc, log_path=log_path, mock=False, started=time.time())


def _launch_mock(config_path: Path, log_path: Path, steps: int) -> "TrainingProcess":
    """Simulates a run by writing an ai-toolkit-like log (for local tests)."""
    script = (
        "import time,sys\n"
        f"total={steps}\n"
        "for i in range(1,total+1):\n"
        "    print(f'step {i}/{total} loss={0.5/i:.4f}',flush=True)\n"
        "    time.sleep(0.02)\n"
        "print('saving checkpoint',flush=True)\n"
        "print('done',flush=True)\n"
    )
    proc = _popen_logged([sys.executable, "-c", script], log_path)
    return TrainingProcess(proc=proc, log_path=log_path, mock=True, started=time.time())


@dataclass
class TrainingProcess:
    proc: subprocess.Popen
    log_path: Path
    mock: bool
    started: float

    def wait(self, timeout: float | None = None) -> int:
        return self.proc.wait(timeout=timeout)

    def poll(self) -> int | None:
        return self.proc.poll()

    def last_step(self) -> tuple[int, int] | None:
        """Last (step, total) parsed from the log. Used for progress and to measure sec/step."""
        try:
            text = self.log_path.read_text(encoding="utf-8", errors="ignore")
        except FileNotFoundError:
            return None
        last = None
        for m in _STEP_RE.finditer(text):
            last = (int(m.group(1)), int(m.group(2)))
        return last

    def measured_sec_per_step(self) -> float | None:
        """Real measured sec/step (fed back into the estimates)."""
        ls = self.last_step()
        if not ls or ls[0] <= 0:
            return None
        elapsed = time.time() - self.started
        return elapsed / ls[0]
