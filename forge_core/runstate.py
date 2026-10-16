# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 The Jungle Fruit
"""SQLite state: runs, checkpoints, calibration and sec/step measurements.

The source of truth for PROGRESS (the CLI and the web app read and write the same forge.db).
WAL mode for concurrent access. The calibration state is exported to a separate JSON file
(calibration.json) that IS committed, so it survives the GPU box being torn down.
"""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id        TEXT PRIMARY KEY,
    model         TEXT NOT NULL,
    lora_type     TEXT NOT NULL,
    trigger       TEXT,
    config_hash   TEXT,
    manifest_hash TEXT,
    caption_hash  TEXT,
    gpu_model     TEXT,
    steps         INTEGER,
    status        TEXT NOT NULL DEFAULT 'created',
    created_at    REAL NOT NULL,
    started_at    REAL,
    finished_at   REAL
);
CREATE TABLE IF NOT EXISTS checkpoints (
    run_id  TEXT NOT NULL,
    step    INTEGER NOT NULL,
    path    TEXT,
    created_at REAL NOT NULL,
    PRIMARY KEY (run_id, step)
);
CREATE TABLE IF NOT EXISTS picks (
    run_id         TEXT PRIMARY KEY,
    human_pick     TEXT,
    evaluator_pick TEXT,
    evaluator_top2 TEXT,
    agreed         INTEGER,
    created_at     REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS measurements (
    gpu_model    TEXT NOT NULL,
    model        TEXT NOT NULL,
    sec_per_step REAL NOT NULL,
    measured_at  REAL NOT NULL,
    PRIMARY KEY (gpu_model, model)
);
"""

_RUN_COLUMNS = {"model", "lora_type", "trigger", "config_hash", "manifest_hash", "caption_hash",
                "gpu_model", "steps", "status", "started_at", "finished_at"}


@dataclass
class RunState:
    db_path: Path

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        # sqlite3's own context manager commits but never closes; close explicitly so no
        # file handle outlives the call (Windows keeps the files locked otherwise).
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA foreign_keys=ON;")
            with conn:
                yield conn
        finally:
            conn.close()

    def init(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as c:
            c.executescript(_SCHEMA)

    # --- runs ---------------------------------------------------------------
    def create_run(self, run_id: str, model: str, lora_type: str, trigger: str | None) -> None:
        with self._connect() as c:
            c.execute(
                "INSERT INTO runs (run_id, model, lora_type, trigger, status, created_at) "
                "VALUES (?,?,?,?,'created',?)",
                (run_id, model, lora_type, trigger, time.time()),
            )

    def update_run(self, run_id: str, **fields: Any) -> None:
        if not fields:
            return
        unknown = set(fields) - _RUN_COLUMNS
        if unknown:
            raise ValueError(f"unknown run columns: {sorted(unknown)}")
        cols = ", ".join(f"{k}=?" for k in fields)
        with self._connect() as c:
            c.execute(f"UPDATE runs SET {cols} WHERE run_id=?", (*fields.values(), run_id))

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as c:
            row = c.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            return dict(row) if row else None

    def list_runs(self) -> list[dict[str, Any]]:
        with self._connect() as c:
            rows = c.execute("SELECT * FROM runs ORDER BY created_at DESC").fetchall()
            return [dict(r) for r in rows]

    # --- checkpoints --------------------------------------------------------
    def add_checkpoint(self, run_id: str, step: int, path: str | None) -> None:
        with self._connect() as c:
            c.execute(
                "INSERT OR REPLACE INTO checkpoints (run_id, step, path, created_at) VALUES (?,?,?,?)",
                (run_id, step, path, time.time()),
            )

    def checkpoints(self, run_id: str) -> list[dict[str, Any]]:
        with self._connect() as c:
            rows = c.execute(
                "SELECT * FROM checkpoints WHERE run_id=? ORDER BY step", (run_id,)
            ).fetchall()
            return [dict(r) for r in rows]

    # --- picks / calibration ------------------------------------------------
    def record_pick(
        self,
        run_id: str,
        human_pick: str,
        evaluator_pick: str | None = None,
        evaluator_top2: list[str] | None = None,
    ) -> None:
        agreed = None
        if evaluator_top2 is not None:
            agreed = 1 if human_pick in evaluator_top2 else 0
        with self._connect() as c:
            c.execute(
                "INSERT OR REPLACE INTO picks "
                "(run_id, human_pick, evaluator_pick, evaluator_top2, agreed, created_at) "
                "VALUES (?,?,?,?,?,?)",
                (run_id, human_pick, evaluator_pick,
                 json.dumps(evaluator_top2 or []), agreed, time.time()),
            )

    def calibration_stats(self, window: int = 10) -> dict[str, Any]:
        """Cumulative and rolling agreement (last `window` picks). Basis of the double gate."""
        with self._connect() as c:
            rows = c.execute(
                "SELECT agreed FROM picks WHERE agreed IS NOT NULL ORDER BY created_at"
            ).fetchall()
        agreed = [r["agreed"] for r in rows]
        n = len(agreed)
        total = sum(agreed)
        recent = agreed[-window:]
        rolling = (sum(recent) / len(recent)) if recent else 0.0
        return {
            "n": n,
            "agreed_total": total,
            "agreement": (total / n) if n else 0.0,
            "rolling_agreement": rolling,
            "rolling_window": len(recent),
            "gate_met": n >= 10 and rolling >= 0.8,
        }

    def export_calibration(self, out_path: Path) -> dict[str, Any]:
        """Dumps picks + stats to a committable JSON file."""
        with self._connect() as c:
            rows = c.execute("SELECT * FROM picks ORDER BY created_at").fetchall()
        payload = {"picks": [dict(r) for r in rows], "stats": self.calibration_stats()}
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return payload

    # --- measurements (sec/step) -------------------------------------------
    def record_measurement(self, gpu_model: str, model: str, sec_per_step: float) -> None:
        with self._connect() as c:
            c.execute(
                "INSERT OR REPLACE INTO measurements (gpu_model, model, sec_per_step, measured_at) "
                "VALUES (?,?,?,?)",
                (gpu_model, model, sec_per_step, time.time()),
            )

    def get_measurement(self, gpu_model: str, model: str) -> float | None:
        with self._connect() as c:
            row = c.execute(
                "SELECT sec_per_step FROM measurements WHERE gpu_model=? AND model=?",
                (gpu_model, model),
            ).fetchone()
            return float(row["sec_per_step"]) if row else None
