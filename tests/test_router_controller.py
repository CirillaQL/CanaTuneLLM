"""Router admission and Controller tiering with fake clocks (no GPUs, no vLLM)."""

import asyncio

import pytest

from canatune.config import load_config
from canatune.controller.router import CanaTuneRouter, RouterSettings, prompt_tokens
from canatune.controller.tier_controller import ControllerSettings, TierController
from canatune.domain.groups import (
    ClockPoint,
    Group,
    GroupState,
    Tier,
    TierState,
    TierStore,
    TierTable,
)
from canatune.domain.load import LengthStats
from canatune.domain.risk import RiskTable
from canatune.infrastructure.clocks import GpuRef, NullClockActuator
from canatune.infrastructure.telemetry import EndpointSnapshot, Telemetry
from canatune.service import build_groups, identity

MAX = ClockPoint(2520, 1500)
H = ClockPoint(1815, 1050)
L = ClockPoint(1305, 1050)
PARK = ClockPoint(900, 450)


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def published(**kwargs) -> TierTable:
    values = {"park": PARK, "h": H, "capacity_h": 3000.0, "alpha_tokens": 460.0}
    values.update(kwargs)
    return TierTable(**values)


def setup(table=None, telemetry=None, store=None):
    config = load_config()
    config["controller"]["stagger_s"] = 0.0
    clock = FakeClock()
    risk = RiskTable.from_config(config["risk"], identity(config))
    groups = build_groups(config)
    tiers = TierState(max_point=MAX, table=table)
    lengths = LengthStats([(512, 64)], min_samples=1)
    router = CanaTuneRouter(
        groups,
        risk,
        RouterSettings.from_config(config),
        tiers,
        lengths=lengths,
        telemetry=telemetry,
        clock=clock,
    )
    actuator = NullClockActuator()
    refs = {e: GpuRef("http://agent", i) for i, e in enumerate(("P0", "D0", "P1", "D1"))}
    controller = TierController(
        groups,
        router,
        ControllerSettings.from_config(config),
        actuator,
        refs,
        tiers,
        store=store,
        clock=clock,
    )
    return clock, risk, groups, tiers, router, controller, actuator


def activate(groups, point=H, tier=Tier.H) -> None:
    for g in groups:
        g.state, g.tier, g.effective = GroupState.ACTIVE, tier, point


def make_safe(risk: RiskTable, point: ClockPoint, n_await: int = 3, prompt: int = 2048) -> None:
    """Enough clean samples in a heavy cell so every lighter cell is bounded by 0."""
    for d_busy in (False, True):
        for _ in range(20):
            risk.record(risk.cell(point, n_await, prompt, d_busy), False)


def test_prompt_tokens() -> None:
    assert prompt_tokens({"prompt": [1, 2, 3]}, 4) == (3, True)
    assert prompt_tokens({"prompt": "abcdefgh"}, 4) == (2, False)
    with pytest.raises(ValueError):
        prompt_tokens({"prompt": None}, 4)


def test_cold_start_max_clocks_open_admission_and_canary_calibrates() -> None:
    _, _, groups, _, router, controller, actuator = setup()
    asyncio.run(controller.start())
    assert groups[0].state is GroupState.EXPLORING  # Canary calibrates at once
    assert all(g.effective == MAX for g in groups[1:])
    assert (GpuRef("http://agent", 2), 2520) in actuator.calls
    assert router.open_admission
    # Everything is admitted, even an unknown long prompt, spread over production.
    names = [router.try_admit(4000, True).group.name for _ in range(4)]
    assert names == ["G1", "G2", "G3", "G1"]
    asyncio.run(controller.tick())  # no table: no consolidation, no switching
    assert all(g.state is GroupState.ACTIVE for g in groups[1:])


def test_publish_moves_groups_to_h_and_switches_admission(tmp_path) -> None:
    store = TierStore(tmp_path / "tiers.json", {"model": "m"})
    _, risk, groups, tiers, router, controller, _ = setup(store=store)
    asyncio.run(controller.start())
    asyncio.run(controller.publish(published(), "test"))
    assert all(g.effective == H for g in groups[1:])
    assert not router.open_admission
    assert store.load().h == H  # persisted with the identity
    # Unknown cells at H are unsafe: rejected until the Canary filled them.
    assert router.try_admit(128, True) is None
    make_safe(risk, H)
    assert router.try_admit(128, True).group.name == "G1"


def test_router_concentrates_on_most_loaded_feasible_group() -> None:
    _, risk, groups, _, router, _, _ = setup(table=published())
    activate(groups)
    risk.load_seed(H, [[0.0] * 5, [0.12] * 5] + [[1.0] * 5] * 4)
    first = router.try_admit(128, True)
    second = router.try_admit(128, True)
    # G1 wins the tie (production before Canary); with N_await=1 the seeded risk
    # is 12 % > 10 %, so the second request goes to the next group.
    assert first.group.name == "G1"
    assert second.group.name == "G2"
    router.first_token(first)
    assert router.try_admit(512, True).group.name == "G1"  # most in flight again


def test_decode_wall_blocks_admission() -> None:
    _, risk, groups, _, router, _, _ = setup(table=published(decode_max_running=2))
    activate(groups[1:2])
    make_safe(risk, H)
    a = router.try_admit(128, True)
    router.first_token(a)
    b = router.try_admit(128, True)
    router.first_token(b)
    assert router.try_admit(128, True) is None  # D would exceed B* = 2
    router.finish(a, status="ok", ttft_ms=100, tpot_ms=50, output_tokens=8)
    assert router.try_admit(128, True) is not None


def test_decode_kv_limit_from_telemetry() -> None:
    clock = FakeClock()
    telemetry = Telemetry({}, period_s=1, client_factory=lambda: None, clock=clock)
    _, risk, groups, _, router, _, _ = setup(
        table=published(decode_kv_limit=0.9), telemetry=telemetry
    )
    router._clock = clock
    activate(groups[1:2])
    make_safe(risk, H)
    telemetry.update("D1", EndpointSnapshot(clock.now, 5, 0, 0.95, 0, True))
    assert router.try_admit(128, True) is None
    telemetry.update("D1", EndpointSnapshot(clock.now, 5, 0, 0.5, 0, True))
    assert router.try_admit(128, True) is not None
    clock.now += 5  # stale snapshot: unknown state is unsafe
    assert router.try_admit(128, True) is None


def test_bounded_wait_admits_once_a_slot_frees() -> None:
    _, risk, groups, tiers, _, _, _ = setup(table=published(decode_max_running=1))
    config = load_config()
    router = CanaTuneRouter(groups[1:2], risk, RouterSettings.from_config(config), tiers)
    activate(groups[1:2])
    make_safe(risk, H)
    blocker = router.try_admit(128, True)

    async def run():
        pending = asyncio.create_task(router.admit(128, True))
        await asyncio.sleep(0.03)
        router.finish(blocker, status="ok", ttft_ms=50, tpot_ms=50, output_tokens=4)
        return await pending

    ticket = asyncio.run(run())
    assert ticket is not None and ticket.wait_ms > 0


def test_rejection_after_wait_budget() -> None:
    _, _, groups, _, router, _, _ = setup(table=published())
    activate(groups)
    assert asyncio.run(router.admit(128, True)) is None  # empty table at H: unknown
    assert router.rejections == 1


def test_finish_records_only_clean_samples_and_lengths() -> None:
    _, risk, groups, _, router, _, _ = setup(table=published())
    activate(groups)
    make_safe(risk, H)
    t1 = router.try_admit(128, True)
    router.first_token(t1)
    router.finish(t1, status="ok", ttft_ms=100, tpot_ms=60, output_tokens=8)
    assert risk.to_json()["cells"][t1.cell.key()]["admitted"] == 1
    t2 = router.try_admit(128, True)
    router.clock_epochs[t2.group.name] += 1
    router.finish(t2, status="ok", ttft_ms=900, tpot_ms=60, output_tokens=8)
    t3 = router.try_admit(128, False)
    router.finish(t3, status="ok", ttft_ms=100, tpot_ms=60, output_tokens=8)
    t4 = router.try_admit(128, True)
    router.finish(t4, status="decode_error", ttft_ms=None, tpot_ms=None, output_tokens=0)
    skips = [r["table_skip"] for r in router.log.recent]
    assert skips == [None, "clock_changed", "prompt_length_estimated", "request_failed"]
    assert all(g.n_await == 0 and g.n_inflight == 0 and g.t_await == 0 for g in groups)
    assert list(router.outcomes) == [False]
    assert router.lengths.pairs()[-1] == (128, 8)


def test_consolidation_parks_at_park_clocks_and_pressure_wakes() -> None:
    clock, _, groups, _, router, controller, actuator = setup()
    asyncio.run(controller.start())
    groups[0].state = GroupState.ACTIVE  # Canary back in service
    asyncio.run(controller.publish(published(), "test"))
    for _ in range(4):  # idle for longer than t_down: park one group per window
        clock.now += 31
        asyncio.run(controller.tick())
        asyncio.run(controller.tick())
    active = [g for g in groups if g.state is GroupState.ACTIVE]
    assert len(active) == 1 and not active[0].canary
    assert groups[0].state is GroupState.PARK  # Canary parked first
    assert (GpuRef("http://agent", 0), 900) in actuator.calls
    assert (GpuRef("http://agent", 1), 450) in actuator.calls
    router.rejections += 1
    asyncio.run(controller.tick())
    assert len([g for g in groups if g.state is GroupState.ACTIVE]) == 2


def test_pressure_with_nothing_parked_aborts_canary() -> None:
    clock, _, groups, _, _, controller, _ = setup(table=published(capacity_h=100.0))
    activate(groups[1:])
    groups[0].state = GroupState.EXPLORING
    reasons = []
    controller.on_pressure = reasons.append
    for g in groups[1:]:
        for _ in range(10):
            g.record_admission(clock.now, 512, 10)  # ~ 972 eq tokens/s >> 0.85 * 100
    asyncio.run(controller.tick())
    assert reasons == ["load_above_wake"]


def test_pressure_without_experiment_boosts_to_max_then_returns_to_h() -> None:
    clock, _, groups, _, _, controller, _ = setup(table=published(capacity_h=100.0))
    activate(groups)
    controller.on_pressure = lambda reason: False  # no experiment to abort
    for g in groups:
        for _ in range(10):
            g.record_admission(clock.now, 512, 10)
    asyncio.run(controller.tick())
    assert [g.tier for g in groups].count(Tier.MAX) == 1  # one group per tick
    for _ in groups:
        asyncio.run(controller.tick())
    assert all(g.tier is Tier.MAX and g.effective == MAX for g in groups)
    clock.now += 60  # load gone
    for _ in range(2 * len(groups) + 2):
        asyncio.run(controller.tick())
        clock.now += 31
    assert all(g.tier is Tier.H for g in groups if g.state is GroupState.ACTIVE)


def test_pressure_aborting_the_canary_does_not_boost() -> None:
    clock, _, groups, _, _, controller, _ = setup(table=published(capacity_h=100.0))
    activate(groups)
    controller.on_pressure = lambda reason: True  # the Canary comes back instead
    for g in groups:
        for _ in range(10):
            g.record_admission(clock.now, 512, 10)
    asyncio.run(controller.tick())
    assert all(g.tier is Tier.H for g in groups)


def test_l_h_switching_with_hysteresis() -> None:
    clock, risk, groups, _, _, controller, _ = setup(
        table=published(l=L, tau_up=2000.0, tau_down=1000.0)
    )
    group = groups[1]
    controller.groups = [group]
    activate([group])
    asyncio.run(controller.tick())
    assert group.tier is Tier.H
    clock.now += 31
    asyncio.run(controller.tick())
    assert group.tier is Tier.L and group.effective == L
    for _ in range(40):  # 40 x (512 + 460) / 10 s = 3888 eq tokens/s > tau_up
        group.record_admission(clock.now, 512, 10)
    asyncio.run(controller.tick())
    assert group.tier is Tier.H


def test_router_uses_slower_point_while_clock_changes() -> None:
    _, _, groups, _, _, controller, _ = setup(table=published(l=L, tau_up=2.0, tau_down=1.0))
    group = groups[1]
    activate([group], L, Tier.L)
    seen = []

    class SlowActuator(NullClockActuator):
        async def lock(self, ref, mhz):
            seen.append(group.effective)
            return await super().lock(ref, mhz)

    controller.actuator = SlowActuator()
    asyncio.run(controller.set_tier(group, Tier.H, "test"))
    assert seen == [L, L]
    assert group.effective == H


def test_group_load_in_equivalent_tokens() -> None:
    g = Group("G", "P", "D")
    for t in range(10):
        g.record_admission(100.0 + t, 100, 5)
    # 5 requests in the last 5 s, each 100 prompt tokens + alpha 460.
    assert g.load(109.5, 5, 460.0) == pytest.approx(5 * 560 / 5)
    assert g.load(109.5, 5, 0.0) == pytest.approx(100.0)
