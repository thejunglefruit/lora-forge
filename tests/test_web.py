# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 The Jungle Fruit
"""Web layer tests: the same flow as the CLI (new -> ingest -> mock caption -> config -> mock train ->
report -> pick), plus the local-only guards (Host, Origin, path traversal)."""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TMP = Path(tempfile.mkdtemp(prefix="forge_web_test_"))
os.environ["FORGE_PROJECTS"] = str(TMP / "projects")
sys.path.insert(0, str(ROOT / "forge_web"))
import api  # noqa: E402

LOCAL = "http://127.0.0.1:8800"


@pytest.fixture(scope="module", autouse=True)
def images():
    (TMP / "src").mkdir(parents=True, exist_ok=True)
    for i in range(6):  # flat colour fields generated in code: no real photos, no people
        Image.new("RGB", (1024, 1280), (i * 40, 120, 200 - i * 20)).save(TMP / "src" / f"swatch_{i}.png")
    yield
    shutil.rmtree(TMP, ignore_errors=True)


@pytest.fixture
def client():
    with TestClient(api.app, base_url=LOCAL) as c:
        yield c


def test_full_flow(client):
    assert client.get("/api/health").json()["ok"]
    mods = client.get("/api/models").json()
    assert "flux" in [m["key"] for m in mods["models"]] and "character" in mods["types"]
    r = client.post("/api/projects", json={"name": "web test", "type": "object", "model": "flux"})
    assert r.status_code == 200, r.text
    folder = r.json()["folder"]
    assert r.json()["trigger"]
    r = client.post(f"/api/projects/{folder}/ingest", json={"folder": str(TMP / "src")})
    assert r.status_code == 200, r.text
    assert r.json()["n"] == 6
    d = client.get(f"/api/projects/{folder}").json()
    assert len(d["images"]) == 6 and d["dataset_report"]
    r = client.post(f"/api/projects/{folder}/captions", json={"real": False})
    assert r.json()["written"] == 6
    client.put(f"/api/projects/{folder}/caption/swatch_0.png", json={"text": "manual edit"})
    r = client.post(f"/api/projects/{folder}/captions", json={"real": False})
    assert r.json()["kept"] == 6
    assert client.get(f"/api/projects/{folder}").json()["captions"]["swatch_0.png"] == "manual edit"
    r = client.post(f"/api/projects/{folder}/config", json={"gpu": "RTX 4090", "gpu_vram": 24})
    assert r.status_code == 200, r.text
    run_id = r.json()["run_id"]
    assert r.json()["steps"] > 0 and "sd_trainer" in r.json()["config"]
    r = client.post(f"/api/projects/{folder}/train/{run_id}", json={"mock": True})
    assert r.json()["launched"]
    for _ in range(100):
        time.sleep(0.3)
        v = client.get(f"/api/projects/{folder}/runs/{run_id}").json()
        if v["run"]["status"] in ("done", "failed"):
            break
    assert v["run"]["status"] == "done", v
    r = client.post(f"/api/projects/{folder}/report/{run_id}")
    assert r.status_code == 200 and "run_id" in r.json()["text"]
    r = client.post(f"/api/projects/{folder}/pick/{run_id}", json={"checkpoint": "1000"})
    assert r.json()["ok"] and "rolling_agreement" in r.json()["calibration"]
    assert (api.PROJECTS / folder / "calibration.json").exists()


def test_rejects_foreign_origin_and_null(client):
    assert client.get("/api/projects", headers={"Origin": "https://evil.example"}).status_code == 403
    assert client.get("/api/projects", headers={"Origin": "null"}).status_code == 403
    assert client.get("/api/projects", headers={"Origin": LOCAL}).status_code == 200


def test_rejects_foreign_host():
    with TestClient(api.app, base_url="http://attacker.example") as c:
        assert c.get("/api/projects").status_code == 403


def test_path_traversal_is_refused(client):
    client.post("/api/projects", json={"name": "guard", "type": "style", "model": "flux"})
    (TMP / "secret.png").write_bytes(b"not for you")
    for bad in ("..%2F..%2Fsecret.png", "..%5C..%5Csecret.png", "%2E%2E%5Csecret.png"):
        r = client.get(f"/api/projects/guard/image/{bad}")
        assert r.status_code in (400, 404), (bad, r.status_code)
        assert b"not for you" not in r.content
    r = client.put("/api/projects/guard/caption/..%5C..%5Cforge.yaml", json={"text": "x"})
    assert r.status_code in (400, 404)
    assert client.get("/api/projects/guard/runs/..%5Cx").status_code in (400, 404)
