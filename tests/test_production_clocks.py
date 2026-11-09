"""Production policies: Canary H, locked MAX, or driver-managed (no production locks)."""

import asyncio
import dataclasses

import httpx
import pytest
from test_router_controller import MAX, H, published, setup

from canatune.config import load_config
from canatune.controller.tier_controller import ControllerConfigError, ControllerSettings
from canatune.domain.groups import GroupState, Tier
from canatune.infrastructure.clocks import AgentClockActuator, GpuRef, NullClockActuator


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
    assert ControllerSettings.from_config(config).production_clocks == "driver"
    config["controller"]["production_clocks"] = "invalid"
    with pytest.raises(ControllerConfigError):
        ControllerSettings.from_config(config)


def driver_controller():
    values = setup(table=published(), overload="best_effort", router={"group_choice": "room"})
    controller = values[5]
    controller.settings = dataclasses.replace(controller.settings, production_clocks="driver")
    controller.gpu_refs = {name: GpuRef("http://agent", i)
                           for i, name in enumerate(e for g in values[2]
                                                    for e in (g.prefill, g.decode))}
    return values


def test_driver_releases_locks_for_every_tier_without_changing_admission():
    _, _, groups, _, router, controller, actuator = driver_controller()
    asyncio.run(controller.start())
    assert not router.open_admission
    assert controller.state()["driver_clocks_ready"]
    assert actuator.resets and not actuator.calls
    for tier in (Tier.MAX, Tier.H, Tier.PARK, Tier.H):
        asyncio.run(controller.set_tier(groups[1], tier, "test"))
        assert groups[1].effective == (None if tier is Tier.PARK else MAX)
    assert not actuator.calls  # entering/exiting pressure or parking never locks a clock


def test_canary_probe_can_lock_then_return_to_driver_management():
    _, _, groups, _, _, controller, actuator = driver_controller()
    canary = groups[0]
    canary.state = GroupState.EXPLORING
    ref = controller.gpu_refs[canary.prefill]
    asyncio.run(actuator.lock(ref, 900))  # the probe owns this lock, not set_tier
    canary.state = GroupState.ACTIVE
    asyncio.run(controller.set_tier(canary, Tier.H, "canary_done"))
    assert ref in actuator.resets and actuator.calls == [(ref, 900)]
    assert canary.effective == MAX


def test_failed_driver_reset_is_not_ready_and_is_retried():
    class FailingReset(NullClockActuator):
        fail = True

        async def reset(self, ref):
            if self.fail:
                raise RuntimeError("reset failed")
            return await super().reset(ref)

    clock, _, groups, _, _, controller, _ = driver_controller()
    actuator = controller.actuator = FailingReset()
    group = groups[1]
    group.state = GroupState.ACTIVE
    asyncio.run(controller.set_tier(group, Tier.H, "test"))
    assert not controller.state()["driver_clocks_ready"]
    assert group.effective is None
    actuator.fail = False
    clock.now += 30
    asyncio.run(controller._retry_locks(clock.now))
    assert controller.state()["driver_clocks_ready"] and not actuator.calls


def test_agent_reset_calls_reset_endpoint_without_lock():
    calls = []

    def respond(request):
        calls.append((request.method, request.url.path))
        return httpx.Response(200, json={"ok": True, "gpu": 2})

    actuator = AgentClockActuator(lambda: httpx.AsyncClient(transport=httpx.MockTransport(respond)))
    assert asyncio.run(actuator.reset(GpuRef("http://agent", 2)))["ok"]
    assert calls == [("POST", "/gpus/2/reset")]
