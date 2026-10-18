# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 The Jungle Fruit
from __future__ import annotations

import pytest

from forge_core.registry import Registry, RegistryError


def test_registry_loads_and_validates():
    reg = Registry.load()
    assert "flux" in reg.ids()
    assert "flux" in reg.active_ids()


def test_flux_spec_is_coherent():
    m = Registry.load().get("flux")
    assert m.backend == "ai-toolkit"
    assert m.arch == "flux"
    assert m.trigger_required is True
    assert m.caption_engine == "dual"
    # FLUX.1 dev is non-commercial (it matters if you want to sell the LoRA)
    assert m.license.get("commercial_use") is False
    reqs = m.requirements("character")
    assert reqs["res_floor"] == 1024
    assert reqs["imgs_min"] >= 1


def test_unknown_model_raises():
    with pytest.raises(RegistryError):
        Registry.load().get("does-not-exist")
