# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 The Jungle Fruit
"""LoRA Forge web layer (FastAPI) over forge_core. A thin frontend: every endpoint calls the SAME
functions as the CLI (new / ingest / caption / config / train / report / pick) and serves the
compiled React app.

Local-only by design: it listens on 127.0.0.1, rejects requests whose Host is not local
(DNS rebinding) and whose Origin is foreign or "null" (cross-site requests from other tabs),
and every file name coming from a URL is resolved inside its folder (commonpath check).
On a rented GPU box, reach it through an SSH tunnel instead of exposing the port.

Environment: FORGE_PROJECTS (projects folder, default ~/lora-forge-projects), FORGE_PORT (8800).
"""
from __future__ import annotations

import os
import re
import shutil
import sys
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import UploadFile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from forge_cli.main import build_run_note, default_eval_prompts  # noqa: E402
from forge_core import backends, config_gen, estimate as est, intake, preflight, project as proj, report as rep  # noqa: E402
from forge_core.caption import build_caption, caption_path_for, get_captioner, inject_trigger, write_caption  # noqa: E402
from forge_core.captionrules import doctrine_for  # noqa: E402
from forge_core.paths import IMAGE_EXTS, ProjectPaths, inside  # noqa: E402
from forge_core.registry import LORA_TYPES, Registry  # noqa: E402
from forge_core.runstate import RunState  # noqa: E402
from forge_core.validate import validate_dataset  # noqa: E402

PROJECTS = Path(os.environ.get("FORGE_PROJECTS", str(Path.home() / "lora-forge-projects")))
PROJECTS.mkdir(parents=True, exist_ok=True)
PORT = int(os.environ.get("FORGE_PORT", "8800"))
LOCAL_HOSTS = {"127.0.0.1", "localhost"}
ALLOWED_ORIGINS = {f"http://{h}:{p}" for h in LOCAL_HOSTS for p in (PORT, 5175)}  # 5175 = Vite dev server
RUN_ID_RE = re.compile(r"^run-\d{8}-\d{6}$")

app = FastAPI(title="LoRA Forge")
RUNNING: dict[str, dict] = {}  # run_id -> {status, step, ...}


@app.middleware("http")
async def local_only(request: Request, call_next):
    host = (request.headers.get("host") or "").rsplit(":", 1)[0].strip("[]").lower()
    if host not in LOCAL_HOSTS:
        return JSONResponse({"detail": "host not allowed"}, status_code=403)
    origin = request.headers.get("origin")
    if origin is not None and origin not in ALLOWED_ORIGINS:  # includes the literal "null"
        return JSONResponse({"detail": "origin not allowed"}, status_code=403)
    resp = await call_next(request)
    if resp.headers.get("content-type", "").startswith("text/html"):
        resp.headers["Cache-Control"] = "no-store"
    resp.headers["X-Content-Type-Options"] = "nosniff"
    return resp


def _reg() -> Registry:
    return Registry.load()


def _safe_name(name: str) -> str:
    return re.sub(r"[^\w\-]", "_", name)


def _paths(name: str) -> ProjectPaths:
    p = ProjectPaths(inside(PROJECTS, _safe_name(name)))
    if not p.descriptor.exists():
        raise HTTPException(404, f"No such project: {name}")
    return p


def _dataset_file(paths: ProjectPaths, filename: str) -> Path:
    try:
        p = inside(paths.dataset, filename)
    except ValueError:
        raise HTTPException(400, "invalid file name")
    if p.parent != paths.dataset.resolve():
        raise HTTPException(400, "invalid file name")
    return p


def _run_id(run_id: str) -> str:
    if not RUN_ID_RE.match(run_id):
        raise HTTPException(400, "invalid run_id")
    return run_id


def _summary(paths: ProjectPaths) -> dict:
    d = proj.ProjectDescriptor.load(paths)
    imgs = paths.list_images()
    manifest = proj.read_manifest(paths) if paths.manifest.exists() else {}
    captions = {}
    for im in imgs:
        cp = caption_path_for(im)
        captions[im.name] = cp.read_text(encoding="utf-8").strip() if cp.exists() else None
    runs = RunState(paths.db).list_runs() if paths.db.exists() else []
    for r in runs:
        r["running"] = RUNNING.get(r["run_id"], {}).get("status") == "running"
    m = _reg().get(d.model)
    return {
        "name": d.name, "type": d.lora_type, "model": d.model, "trigger": d.trigger, "eval_prompts": d.eval_prompts,
        "images": [im.name for im in imgs], "manifest": manifest, "captions": captions,
        "runs": runs, "doctrine": doctrine_for(d.lora_type, m.caption_engine, m.trigger_required).__dict__,
        "requirements": m.requirements(d.lora_type), "license": m.license,
        "dataset_report": paths.dataset_report.read_text(encoding="utf-8") if paths.dataset_report.exists() else None,
    }


@app.get("/api/health")
def health():
    from forge_core.caption.ollama import OllamaCaptioner

    env = preflight.detect_pod_env()
    return {"ok": True, "gpu": env.gpu_name, "real_captioner": OllamaCaptioner.available(),
            "aitoolkit_backend": backends.detect_aitoolkit().available}


@app.get("/api/models")
def models():
    reg = _reg()
    out = []
    for key in reg.ids():
        m = reg.get(key)
        out.append({"key": key, "arch": m.arch, "status": m.status, "caption_engine": m.caption_engine,
                    "trigger_required": m.trigger_required, "license": m.license, "vram_gb": m.vram_gb})
    return {"models": out, "types": list(LORA_TYPES), "pod_rate_usd_h": reg.pod_hourly_rate_usd}


@app.get("/api/projects")
def list_projects():
    out = []
    for d in sorted(PROJECTS.iterdir()):
        if (d / "forge.yaml").exists():
            try:
                desc = proj.ProjectDescriptor.load(ProjectPaths(d))
                out.append({"name": desc.name, "folder": d.name, "type": desc.lora_type, "model": desc.model,
                            "images": len(ProjectPaths(d).list_images())})
            except Exception as e:  # noqa: BLE001
                out.append({"name": d.name, "folder": d.name, "error": str(e)})
    return out


@app.post("/api/projects")
async def create_project(request: Request):
    a = await request.json()
    name = str(a.get("name", "")).strip()
    if not name:
        raise HTTPException(400, "empty name")
    lora_type = a.get("type", "character")
    model = a.get("model", "flux")
    if lora_type not in LORA_TYPES:
        raise HTTPException(400, f"invalid type; options: {', '.join(LORA_TYPES)}")
    reg = _reg()
    if model not in reg.ids():
        raise HTTPException(400, f"unknown model; options: {', '.join(reg.ids())}")
    m = reg.get(model)
    root = inside(PROJECTS, _safe_name(name))
    if (root / "forge.yaml").exists():
        raise HTTPException(409, "already exists")
    paths = ProjectPaths(root)
    paths.ensure_base()
    trigger = a.get("trigger") or (proj.suggest_trigger(name) if m.trigger_required else None)
    card = intake.build_card(m, lora_type)
    prompts = a.get("eval_prompts") or default_eval_prompts(lora_type, trigger)
    proj.ProjectDescriptor(name=name, lora_type=lora_type, model=model, trigger=trigger, eval_prompts=prompts).save(paths)
    RunState(paths.db).init()
    return {"ok": True, "folder": root.name, "trigger": trigger, "card": card.render()}


@app.get("/api/projects/{name}")
def show_project(name: str):
    return _summary(_paths(name))


@app.get("/api/projects/{name}/image/{filename}")
def image(name: str, filename: str):
    p = _dataset_file(_paths(name), filename)
    if not p.exists() or p.suffix.lower() not in IMAGE_EXTS:
        raise HTTPException(404)
    return FileResponse(p)


@app.post("/api/projects/{name}/ingest")
async def ingest(name: str, request: Request):
    """Uploads files (multipart 'files') or copies a local folder ('folder' in JSON). Validates and blocks."""
    paths = _paths(name)
    d = proj.ProjectDescriptor.load(paths)
    reqs = _reg().get(d.model).requirements(d.lora_type)
    paths.dataset.mkdir(parents=True, exist_ok=True)
    added = 0
    if request.headers.get("content-type", "").startswith("multipart/"):
        form = await request.form()
        for f in form.getlist("files"):
            if not isinstance(f, UploadFile):
                continue
            clean = re.sub(r"[^\w\-. ]", "_", Path(f.filename or "").name).lstrip(".")
            if Path(clean).suffix.lower() in IMAGE_EXTS:
                _dataset_file(paths, clean).write_bytes(await f.read())
                added += 1
    else:
        a = await request.json()
        src = Path(str(a.get("folder", "")))
        if not src.is_dir():
            raise HTTPException(400, f"Not a folder: {src}")
        for p in sorted(src.iterdir()):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS and not (paths.dataset / p.name).exists():
                shutil.copy2(p, _dataset_file(paths, p.name))
                added += 1
    imgs = paths.list_images()
    if not imgs:
        raise HTTPException(400, "no images")
    report = validate_dataset(images=imgs, res_floor=int(reqs["res_floor"]), lora_type=d.lora_type, requirements=reqs)
    proj.write_manifest(paths, images=[im.to_dict() for im in report.images],
                        meta={"model": d.model, "lora_type": d.lora_type, "summary": report.summary})
    paths.dataset_report.write_text(proj.dataset_report_markdown(d.name, d.model, d.lora_type, report), encoding="utf-8")
    return {"ok": True, "added": added, "n": report.n, "blocked": report.blocked, "summary": report.summary,
            "images": [im.to_dict() for im in report.images]}


@app.delete("/api/projects/{name}/image/{filename}")
def remove_image(name: str, filename: str):
    p = _dataset_file(_paths(name), filename)
    if p.exists() and p.suffix.lower() in IMAGE_EXTS:
        p.unlink()
        cp = caption_path_for(p)
        if cp.exists():
            cp.unlink()
    return {"ok": True}


@app.post("/api/projects/{name}/captions")
async def captions(name: str, request: Request):
    a = await request.json()
    paths = _paths(name)
    d = proj.ProjectDescriptor.load(paths)
    m = _reg().get(d.model)
    doctrine = doctrine_for(d.lora_type, m.caption_engine, m.trigger_required)
    try:
        captioner = get_captioner(m.caption_engine, prefer_real=bool(a.get("real")))
    except NotImplementedError as e:
        raise HTTPException(400, str(e))
    only = a.get("only")  # list of file names, or None
    written, kept = 0, 0
    t0 = time.time()
    for img in paths.list_images():
        if only and img.name not in only:
            continue
        cp = caption_path_for(img)
        if cp.exists() and not a.get("force") and not only:
            kept += 1
            continue
        nl, tags = captioner.caption(img, doctrine)
        write_caption(img, inject_trigger(build_caption(nl, tags, m.caption_engine), d.trigger))
        written += 1
    return {"ok": True, "written": written, "kept": kept, "engine": captioner.name, "seconds": round(time.time() - t0, 1)}


@app.put("/api/projects/{name}/caption/{filename}")
async def edit_caption(name: str, filename: str, request: Request):
    a = await request.json()
    p = _dataset_file(_paths(name), filename)
    if not p.exists() or p.suffix.lower() not in IMAGE_EXTS:
        raise HTTPException(404)
    write_caption(p, str(a.get("text", "")))
    return {"ok": True}


@app.post("/api/projects/{name}/config")
async def config(name: str, request: Request):
    a = await request.json()
    paths = _paths(name)
    d = proj.ProjectDescriptor.load(paths)
    reg = _reg()
    m = reg.get(d.model)
    n = len(paths.list_images())
    if n == 0:
        raise HTTPException(400, "no images")
    run_id = time.strftime("run-%Y%m%d-%H%M%S")
    cfg = config_gen.build_aitoolkit_config(run_id=run_id, model=m, lora_type=d.lora_type, trigger=d.trigger,
                                            dataset_folder=str(paths.dataset), output_folder=str(paths.output_dir(run_id)),
                                            n_images=n, eval_prompts=d.eval_prompts)
    cfg_hash = config_gen.write_config(cfg, paths.config_file(run_id))
    state = RunState(paths.db)
    gpu = a.get("gpu") or None
    state.create_run(run_id, d.model, d.lora_type, d.trigger)
    state.update_run(run_id, config_hash=cfg_hash, steps=cfg.steps, manifest_hash=proj.hash_manifest(paths),
                     caption_hash=proj.hash_captions(paths), gpu_model=gpu)
    measured = state.get_measurement(gpu, d.model) if gpu else None
    vram = a.get("gpu_vram")
    e = est.estimate_run(model=m, steps=cfg.steps, gpu_model=gpu, measured_sec_per_step=measured,
                         hourly_rate_usd=reg.pod_hourly_rate_usd, vram_available_gb=int(vram) if vram else None)
    return {"ok": True, "run_id": run_id, "steps": cfg.steps, "config": cfg.yaml_text, "hash": cfg_hash,
            "estimate": dict(e.__dict__, minutes=e.minutes, cost_usd=e.cost_usd, vram_ok=e.vram_ok),
            "estimate_text": e.render()}


def _train(paths: ProjectPaths, run_id: str, mock: bool):
    d = proj.ProjectDescriptor.load(paths)
    state = RunState(paths.db)
    run = state.get_run(run_id)
    backend = backends.detect_aitoolkit()
    paths.samples_dir(run_id).mkdir(parents=True, exist_ok=True)
    state.update_run(run_id, status="running", started_at=time.time())
    RUNNING[run_id] = {"status": "running", "step": None}
    try:
        tp = backends.launch_training(backend=backend, config_path=paths.config_file(run_id),
                                      log_path=paths.train_log(run_id), mock=mock or not backend.available)
        while tp.poll() is None:
            ls = tp.last_step()
            if ls:
                RUNNING[run_id]["step"] = ls
            time.sleep(0.3)
        rc = tp.wait()
        sps = tp.measured_sec_per_step()
        if sps and run.get("gpu_model"):
            state.record_measurement(run["gpu_model"], d.model, sps)
        state.update_run(run_id, status="done" if rc == 0 else "failed", finished_at=time.time())
        RUNNING[run_id] = {"status": "done" if rc == 0 else "failed", "sec_per_step": sps}
    except Exception as e:  # noqa: BLE001
        state.update_run(run_id, status="failed", finished_at=time.time())
        RUNNING[run_id] = {"status": "failed", "error": str(e)}


@app.post("/api/projects/{name}/train/{run_id}")
async def train(name: str, run_id: str, request: Request):
    a = await request.json()
    paths = _paths(name)
    run = RunState(paths.db).get_run(_run_id(run_id))
    if run is None:
        raise HTTPException(404, "unknown run")
    if run["status"] == "done":
        return {"ok": True, "note": "already finished"}
    if RUNNING.get(run_id, {}).get("status") == "running":
        return {"ok": True, "note": "already running"}
    mock = bool(a.get("mock"))
    if not mock and not backends.detect_aitoolkit().available:
        raise HTTPException(400, backends.detect_aitoolkit().install_hint())
    threading.Thread(target=_train, args=(paths, run_id, mock), daemon=True).start()
    return {"ok": True, "launched": True, "mock": mock}


@app.get("/api/projects/{name}/runs/{run_id}")
def show_run(name: str, run_id: str):
    paths = _paths(name)
    run = RunState(paths.db).get_run(_run_id(run_id))
    if run is None:
        raise HTTPException(404)
    log = paths.train_log(run_id)
    tail = log.read_text(encoding="utf-8", errors="replace")[-6000:] if log.exists() else ""
    note = paths.run_report(run_id)
    samples_dir = paths.samples_dir(run_id)
    samples = [p.name for p in sorted(samples_dir.glob("*.png"))] if samples_dir.exists() else []
    cfg = paths.config_file(run_id)
    return {"run": run, "progress": RUNNING.get(run_id), "log": tail,
            "report": note.read_text(encoding="utf-8") if note.exists() else None, "samples": samples,
            "config": cfg.read_text(encoding="utf-8") if cfg.exists() else None}


@app.post("/api/projects/{name}/report/{run_id}")
def report(name: str, run_id: str):
    paths = _paths(name)
    run = RunState(paths.db).get_run(_run_id(run_id))
    if run is None:
        raise HTTPException(404)
    out = rep.write_run_note(build_run_note(paths, run_id, run), paths.run_report(run_id))
    return {"ok": True, "text": out.read_text(encoding="utf-8")}


@app.post("/api/projects/{name}/pick/{run_id}")
async def pick(name: str, run_id: str, request: Request):
    a = await request.json()
    paths = _paths(name)
    state = RunState(paths.db)
    if state.get_run(_run_id(run_id)) is None:
        raise HTTPException(404)
    state.record_pick(run_id, human_pick=str(a.get("checkpoint", "")))
    state.export_calibration(paths.calibration)
    return {"ok": True, "calibration": state.calibration_stats()}


DIST = ROOT / "forge_web" / "frontend" / "dist"
if DIST.exists():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="web")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
