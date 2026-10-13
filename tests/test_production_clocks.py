"""controller.production_clocks: canary (H, default) or max (ablation A2: no DVFS)."""

import asyncio
import dataclasses

import pytest
from test_router_controller import MAX, H, published, setup

from canatune.config import load_config
from canatune.controller.tier_controller import ControllerConfigError, ControllerSettings
from canatune.domain.groups import Tier


def published_groups(production_clocks: str):
    _, _, groups, _, router, controller, actuator = setup()
    controller.settings = dataclasses.replace(
        controller.settings, production_clocks=production_clocks
    )
    asyncio.run(controller.start())
    asyncio.run(controller.publish(published(), "test"))
    return groups[1:], router, actuator


def test_max_keeps_production_at_max_clocks_while_admission_uses_the_table() -> None:
    production, router, actuator = published_groups("max")
    assert all(g.tier is Tier.H and g.effective == MAX for g in production)
    assert not router.open_admission  # the Canary's table still drives admission
    assert {mhz for _, mhz in actuator.calls} <= {MAX.prefill_mhz, MAX.decode_mhz}


def test_canary_default_moves_production_to_h() -> None:
    production, _, _ = published_groups("canary")
    assert all(g.tier is Tier.H and g.effective == H for g in production)


def test_production_clocks_is_validated() -> None:
    config = load_config()
    config["controller"]["production_clocks"] = "driver"
    with pytest.raises(ControllerConfigError):
        ControllerSettings.from_config(config)
