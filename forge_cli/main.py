# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 The Jungle Fruit
"""LoRA Forge CLI (Typer). A thin wrapper: all the logic lives in forge_core.

Flow:  new -> ingest -> caption -> config -> train -> report -> pick
"""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path
from typing import Optional

# UTF-8 console on legacy Windows code pages too (cp1252 cannot print arrows or middots).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

import typer
from rich.console import Console
from rich.table import Table

from forge_core import (
    backends,
    config_gen,
    estimate as est,
    intake,
    preflight,
    project as proj,
    report as rep,
)
from forge_core.caption import build_caption, caption_path_for, get_captioner, inject_trigger, write_caption
from forge_core.captionrules import doctrine_for
from forge_core.paths import IMAGE_EXTS, ProjectPaths, find_project
from forge_core.registry import LORA_TYPES, Registry
from forge_core.runstate import RunState
from forge_core.validate import validate_dataset

app = typer.Typer(add_completion=False, help="LoRA Forge: dataset + LoRA training orchestrator.")
console = Console()


def _registry() -> Registry:
    return Registry.load()


def _require_project() -> ProjectPaths:
    p = find_project()
    if p is None:
        console.print("[red]Not inside a Forge project (no forge.yaml). Use 'forge new'.[/]")
        raise typer.Exit(1)
    return p


def _run_id() -> str:
    return time.strftime("run-%Y%m%d-%H%M%S")


def default_eval_prompts(lora_type: str, trigger: str | None) -> list[str]:
    t = trigger or "the subject"
    if lora_type == "character":
        return [
            f"{t}, portrait, soft studio lighting, plain background",
            f"{t} walking in a busy city street, daytime",
            f"{t} sitting in a cafe, warm light, candid",
            f"{t} full body, standing outdoors, natural light",
        ]
    if lora_type == "style":
        return ["a mountain landscape", "a portrait of a person",
                "a city at night", "a still life with fruit"]
    return [f"{t} on a wooden table, studio light",
            f"{t} outdoors on grass", f"{t} held in a hand", f"{t} on a shelf, indoor light"]


# --------------------------------------------------------------------------- #
@app.command()
def models() -> None:
    """Lists the models in the registry."""
    reg = _registry()
    table = Table(title="Models (models.json)")
    for col in ("id", "status", "arch", "backend", "VRAM", "caption", "trigger", "commercial"):
        table.add_column(col)
    for mid in reg.ids():
        m = reg.get(mid)
        lic = m.license.get("commercial_use")
        comm = "yes" if lic is True else ("no" if lic is False else "?")
        table.add_row(mid, m.status, m.arch, m.backend, f"{m.vram_gb}GB",
                      m.caption_engine, "yes" if m.trigger_required else "no", comm)
    console.print(table)


@app.command()
def env() -> None:
    """Shows the environment (GPU/VRAM/disk/credentials)."""
    console.print(preflight.detect_pod_env().render())


@app.command()
def new(
    name: str = typer.Argument(..., help="Project / LoRA name."),
    lora_type: str = typer.Option(..., "--type", "-t", help=f"One of: {', '.join(LORA_TYPES)}"),
    model: str = typer.Option(..., "--model", "-m", help="Registry model (e.g. flux)."),
    trigger: Optional[str] = typer.Option(None, help="Trigger token (generated when needed)."),
    path: Optional[Path] = typer.Option(None, help="Target folder (default ./<name>)."),
) -> None:
    """Creates a project, prints the requirements card and initialises the state."""
    reg = _registry()
    if lora_type not in LORA_TYPES:
        console.print(f"[red]Invalid type '{lora_type}'. Options: {', '.join(LORA_TYPES)}[/]")
        raise typer.Exit(1)
    m = reg.get(model)

    root = (path or Path.cwd() / name).resolve()
    if (root / "forge.yaml").exists():
        console.print(f"[red]A project already exists in {root}[/]")
        raise typer.Exit(1)
    paths = ProjectPaths(root)
    paths.ensure_base()

    if trigger is None and m.trigger_required:
        trigger = proj.suggest_trigger(name)

    card = intake.build_card(m, lora_type)
    prompts = default_eval_prompts(lora_type, trigger)

    descriptor = proj.ProjectDescriptor(
        name=name, lora_type=lora_type, model=model, trigger=trigger, eval_prompts=prompts
    )
    descriptor.save(paths)
    RunState(paths.db).init()

    console.print(card.render())
    console.print()
    if trigger:
        console.print(f"[green]Trigger:[/] [bold]{trigger}[/]")
    console.print(f"[green]Project created in[/] {root}")
    console.print("Next: [bold]forge ingest <image_folder>[/]")


@app.command()
def ingest(
    src: Path = typer.Argument(..., help="Folder with the raw images."),
    move: bool = typer.Option(False, help="Move instead of copy."),
) -> None:
    """Validates the dataset, writes manifest + dataset_report.md and BLOCKS on flags."""
    paths = _require_project()
    d = proj.ProjectDescriptor.load(paths)
    reg = _registry()
    m = reg.get(d.model)
    reqs = m.requirements(d.lora_type)

    src = src.resolve()
    if not src.is_dir():
        console.print(f"[red]Not a folder: {src}[/]")
        raise typer.Exit(1)

    srcs = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not srcs:
        console.print(f"[red]No images in {src}[/]")
        raise typer.Exit(1)

    paths.dataset.mkdir(parents=True, exist_ok=True)
    dst_imgs: list[Path] = []
    for p in srcs:
        dst = paths.dataset / p.name
        if not dst.exists():
            (shutil.move if move else shutil.copy2)(str(p), str(dst))
        dst_imgs.append(dst)

    report = validate_dataset(
        images=dst_imgs, res_floor=int(reqs["res_floor"]),
        lora_type=d.lora_type, requirements=reqs,
    )

    proj.write_manifest(
        paths,
        images=[im.to_dict() for im in report.images],
        meta={"model": d.model, "lora_type": d.lora_type, "summary": report.summary},
    )
    paths.dataset_report.write_text(
        proj.dataset_report_markdown(d.name, d.model, d.lora_type, report), encoding="utf-8")

    console.print(f"Images: {report.n}  ·  flagged: {report.summary['flagged']}  "
                  f"·  faces: {'yes' if report.face_detection_available else 'no (GPU box only)'}")
    if "count_warning" in report.summary:
        console.print(f"[yellow]{report.summary['count_warning']}[/]")
    for im in report.flagged:
        console.print(f"  [yellow]![/] {im.path.name}: {'; '.join(im.flags)}")

    console.print(f"\nReport: {paths.dataset_report}")
    if report.blocked:
        console.print("[bold yellow]BLOCKED:[/] review the flags. Fix or remove images and ingest again,")
        console.print("or go on knowingly with [bold]forge caption[/] if you accept the warnings.")
    else:
        console.print("[green]No flags. Next: forge caption[/]")


@app.command()
def caption(
    force: bool = typer.Option(False, help="Regenerate even existing captions (overwrites manual edits)."),
    real: bool = typer.Option(False, help="Use a real captioner (Ollama vision model)."),
) -> None:
    """Auto-captioning by (type, model). Idempotent: never overwrites edits without --force."""
    paths = _require_project()
    d = proj.ProjectDescriptor.load(paths)
    reg = _registry()
    m = reg.get(d.model)
    doctrine = doctrine_for(d.lora_type, m.caption_engine, m.trigger_required)
    captioner = get_captioner(m.caption_engine, prefer_real=real)

    imgs = paths.list_images()
    if not imgs:
        console.print("[red]No ingested images. Run 'forge ingest' first.[/]")
        raise typer.Exit(1)

    written, skipped = 0, 0
    for img in imgs:
        cap_path = caption_path_for(img)
        if cap_path.exists() and not force:
            skipped += 1
            continue
        nl, tags = captioner.caption(img, doctrine)
        text = inject_trigger(build_caption(nl, tags, m.caption_engine), d.trigger)
        write_caption(img, text)
        written += 1

    console.print(f"Captions written: {written}  ·  kept (manual edits): {skipped}")
    console.print(f"Engine: {captioner.name} ({m.caption_engine})  ·  trigger: {d.trigger or '-'}")
    if skipped and not force:
        console.print("[dim](use --force to regenerate the kept ones)[/]")
    console.print("Review/edit the .txt files, then: [bold]forge config[/]")


@app.command()
def config(
    gpu: Optional[str] = typer.Option(None, help="GPU model (to estimate time/cost)."),
    gpu_vram: Optional[int] = typer.Option(None, help="Available VRAM in GB (check)."),
) -> None:
    """Generates the ai-toolkit YAML + VRAM/time/cost estimate BEFORE launching."""
    paths = _require_project()
    d = proj.ProjectDescriptor.load(paths)
    reg = _registry()
    m = reg.get(d.model)

    imgs = paths.list_images()
    n = len(imgs)
    if n == 0:
        console.print("[red]No images. Ingest first.[/]")
        raise typer.Exit(1)

    run_id = _run_id()
    cfg = config_gen.build_aitoolkit_config(
        run_id=run_id, model=m, lora_type=d.lora_type, trigger=d.trigger,
        dataset_folder=str(paths.dataset), output_folder=str(paths.output_dir(run_id)),
        n_images=n, eval_prompts=d.eval_prompts,
    )
    cfg_hash = config_gen.write_config(cfg, paths.config_file(run_id))

    state = RunState(paths.db)
    state.create_run(run_id, d.model, d.lora_type, d.trigger)
    state.update_run(run_id, config_hash=cfg_hash, steps=cfg.steps,
                     manifest_hash=proj.hash_manifest(paths), caption_hash=proj.hash_captions(paths),
                     gpu_model=gpu)

    measured = state.get_measurement(gpu, d.model) if gpu else None
    estimate = est.estimate_run(
        model=m, steps=cfg.steps, gpu_model=gpu, measured_sec_per_step=measured,
        hourly_rate_usd=reg.pod_hourly_rate_usd, vram_available_gb=gpu_vram,
    )

    console.print(f"[green]Config:[/] {paths.config_file(run_id)}  (hash {cfg_hash})")
    console.print(f"run_id: [bold]{run_id}[/]  ·  steps: {cfg.steps}\n")
    console.print(estimate.render())
    if estimate.vram_ok is False:
        console.print("[bold red]NOT ENOUGH VRAM: do not train on this GPU.[/]")
    console.print(f"\nNext: [bold]forge train {run_id}[/]  (or --mock to test without a GPU)")


@app.command()
def train(
    run_id: str = typer.Argument(..., help="run_id returned by 'forge config'."),
    mock: bool = typer.Option(False, help="Simulate the training (no GPU, to test the flow)."),
) -> None:
    """Launches the detached backend, tails the log and measures the real sec/step."""
    paths = _require_project()
    d = proj.ProjectDescriptor.load(paths)
    state = RunState(paths.db)
    run = state.get_run(run_id)
    if run is None:
        console.print(f"[red]Unknown run_id: {run_id}. Run 'forge config' first.[/]")
        raise typer.Exit(1)
    if run["status"] == "done":
        console.print(f"[yellow]{run_id} is already finished (idempotent). Nothing to do.[/]")
        raise typer.Exit(0)

    cfg_path = paths.config_file(run_id)
    if not cfg_path.exists():
        console.print(f"[red]Config not found: {cfg_path}[/]")
        raise typer.Exit(1)

    backend = backends.detect_aitoolkit()
    if not backend.available and not mock:
        console.print(f"[yellow]{backend.install_hint()}[/]")
        raise typer.Exit(1)

    log_path = paths.train_log(run_id)
    paths.samples_dir(run_id).mkdir(parents=True, exist_ok=True)
    console.print(f"Launching ({'MOCK' if (mock or not backend.available) else backend.name})… log: {log_path}")
    state.update_run(run_id, status="running", started_at=time.time())

    tp = backends.launch_training(backend=backend, config_path=cfg_path, log_path=log_path, mock=mock)
    with console.status("training…") as status:
        while tp.poll() is None:
            ls = tp.last_step()
            if ls:
                status.update(f"training… step {ls[0]}/{ls[1]}")
            time.sleep(0.2)
    rc = tp.wait()

    sps = tp.measured_sec_per_step()
    gpu = run.get("gpu_model")
    if sps and gpu:
        state.record_measurement(gpu, d.model, sps)  # freeze the measured sec/step

    if rc == 0:
        state.update_run(run_id, status="done", finished_at=time.time())
        console.print(f"[green]Training OK.[/] measured sec/step: "
                      f"{f'{sps:.2f}s' if sps else 'n/a'}")
        console.print(f"Next: [bold]forge report {run_id}[/]")
    else:
        state.update_run(run_id, status="failed", finished_at=time.time())
        console.print(f"[red]Backend exited with code {rc}. Check {log_path}[/]")
        raise typer.Exit(rc)


def build_run_note(paths: ProjectPaths, run_id: str, run: dict) -> rep.RunNote:
    """Collects everything a run note needs (shared by the CLI and the web app)."""
    d = proj.ProjectDescriptor.load(paths)
    m = _registry().get(d.model)
    manifest = proj.read_manifest(paths)
    duration = None
    if run.get("started_at") and run.get("finished_at"):
        duration = (run["finished_at"] - run["started_at"]) / 60.0
    samples_dir = paths.samples_dir(run_id)
    samples = [p.name for p in sorted(samples_dir.glob("*.png"))] if samples_dir.exists() else []
    lic = m.license
    return rep.RunNote(
        run_id=run_id, model=d.model, lora_type=d.lora_type, trigger=d.trigger,
        dataset_stats=manifest.get("meta", {}).get("summary", {"images": len(paths.list_images())}),
        config_hash=run.get("config_hash") or "?", manifest_hash=run.get("manifest_hash") or "?",
        caption_hash=run.get("caption_hash") or "?", steps=run.get("steps") or 0,
        gpu_model=run.get("gpu_model"), duration_min=duration, cost_usd=None,
        commercial_use=lic.get("commercial_use"), license_id=lic.get("id"),
        sample_images=samples, eval_prompts=d.eval_prompts,
        findings=[rep.Finding("sec/step measured on this GPU",
                              f"models.json > {d.model} > sec_per_step")] if run.get("gpu_model") else [],
    )


@app.command()
def report(run_id: str = typer.Argument(...)) -> None:
    """Writes the run note (Markdown with frontmatter + hashes + findings)."""
    paths = _require_project()
    run = RunState(paths.db).get_run(run_id)
    if run is None:
        console.print(f"[red]Unknown run_id: {run_id}[/]")
        raise typer.Exit(1)
    out = rep.write_run_note(build_run_note(paths, run_id, run), paths.run_report(run_id))
    console.print(f"[green]Run note:[/] {out}")
    console.print("Next: pick a checkpoint with [bold]forge pick[/]")


@app.command()
def pick(
    run_id: str = typer.Argument(...),
    checkpoint: str = typer.Argument(..., help="Name/step of the chosen checkpoint."),
) -> None:
    """Records your checkpoint choice (feeds the evaluator calibration)."""
    paths = _require_project()
    state = RunState(paths.db)
    if state.get_run(run_id) is None:
        console.print(f"[red]Unknown run_id: {run_id}[/]")
        raise typer.Exit(1)
    state.record_pick(run_id, human_pick=checkpoint)
    state.export_calibration(paths.calibration)
    stats = state.calibration_stats()
    console.print(f"[green]Pick saved:[/] {checkpoint}")
    console.print(f"Calibration: {stats['n']} runs with a pick · rolling agreement "
                  f"{stats['rolling_agreement']*100:.0f}% · gate {'MET' if stats['gate_met'] else 'pending'}")


@app.command()
def status() -> None:
    """Lists the project's runs."""
    paths = _require_project()
    runs = RunState(paths.db).list_runs()
    if not runs:
        console.print("No runs yet.")
        return
    table = Table(title="Runs")
    for col in ("run_id", "model", "type", "steps", "status", "gpu"):
        table.add_column(col)
    for r in runs:
        table.add_row(r["run_id"], r["model"], r["lora_type"], str(r.get("steps") or "-"),
                      r["status"], r.get("gpu_model") or "-")
    console.print(table)


if __name__ == "__main__":
    app()
