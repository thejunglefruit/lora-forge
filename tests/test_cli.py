# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 The Jungle Fruit
"""End-to-end CLI flow in mock mode: new -> ingest -> caption -> config -> train --mock -> report -> pick."""
from __future__ import annotations

import os
import re
from pathlib import Path

from PIL import Image
from typer.testing import CliRunner

from forge_cli.main import app


def test_cli_mock_flow(tmp_path: Path, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    for i in range(4):  # gradient tiles generated in code
        im = Image.linear_gradient("L").resize((1024, 1024)).rotate(i * 90).convert("RGB")
        im.save(src / f"tile_{i}.png")
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    r = runner.invoke(app, ["new", "teapot", "--type", "object", "--model", "flux"])
    assert r.exit_code == 0, r.output
    os.chdir(tmp_path / "teapot")
    assert runner.invoke(app, ["ingest", str(src)]).exit_code == 0
    r = runner.invoke(app, ["caption"])
    assert r.exit_code == 0 and "Captions written: 4" in r.output
    assert len(list((tmp_path / "teapot" / "dataset").glob("*.txt"))) == 4
    r = runner.invoke(app, ["config", "--gpu", "RTX 4090", "--gpu-vram", "24"])
    assert r.exit_code == 0, r.output
    run_id = re.search(r"run-\d{8}-\d{6}", r.output).group(0)
    r = runner.invoke(app, ["train", run_id, "--mock"])
    assert r.exit_code == 0, r.output
    assert runner.invoke(app, ["report", run_id]).exit_code == 0
    assert (tmp_path / "teapot" / "runs" / run_id / "report.md").exists()
    assert runner.invoke(app, ["pick", run_id, "1000"]).exit_code == 0
    assert (tmp_path / "teapot" / "calibration.json").exists()
