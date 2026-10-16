# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 The Jungle Fruit
from __future__ import annotations

import yaml

from forge_core.config_gen import build_aitoolkit_config, scale_steps
from forge_core.registry import Registry


def test_scale_steps_respects_bounds():
    preset = {"steps_per_image": 100, "base_steps": 2000, "steps_min": 1200, "steps_max": 3500}
    assert scale_steps(preset, 1) == 2000        # base wins with few images
    assert scale_steps(preset, 40) == 3500       # ceiling
    assert scale_steps(preset, 25) == 2500       # 25 * 100


def test_flux_config_is_valid_yaml_and_has_flowmatch():
    m = Registry.load().get("flux")
    cfg = build_aitoolkit_config(
        run_id="run-test", model=m, lora_type="character", trigger="subjab12",
        dataset_folder="/ds", output_folder="/out", n_images=20,
        eval_prompts=["subjab12 portrait"],
    )
    # the header is a comment; the body must parse
    data = yaml.safe_load(cfg.yaml_text)
    proc = data["config"]["process"][0]
    assert proc["type"] == "sd_trainer"
    assert proc["train"]["noise_scheduler"] == "flowmatch"  # flux uses flowmatch
    assert proc["network"]["linear"] == m.preset["rank"]
    assert proc["trigger_word"] == "subjab12"
    assert cfg.steps == 2000
    assert len(cfg.config_hash) == 12
