"""Router admission and Controller tiering with fake clocks (no GPUs, no vLLM)."""

import asyncio
import dataclasses
import json

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


# What a Canary calibration publishes (fixtures standing in for its measurements).
CALIBRATION = {
    "predictor_coef": [150.0, 1.0, 1.0, 800.0, 2.0],
    "slack_counts": {str(b): [50, 0] for b in range(9, 14)},  # >= 400 ms slack: safe
    "kv_gate_fraction": 0.5,
}
MODEL = {
    "prefill": {str(f): {"128": 40.0, "2048": 150.0 * 2520 / f} for f in (1815, 2520)},
    "residence": {"128": 100.0, "2048": 200.0},
    "decode": {"1050": [50.0, 1.0], "1500": [45.0, 0.9]},
    "power_prefill": {"1815": [60.0, 200.0], "2520": [70.0, 320.0]},
    "power_decode": {"1050": [20.0, 30.0], "1500": [22.0, 50.0]},
    "park_power_w": 70.0,
    "rho_prefill": 0.5,
    "ttft_slo_ms": 1000.0,
    "tpot_slo_ms": 200.0,
    "kv_bytes_per_token": 131072,
    "kv_gate_bytes": None,
    "kv_capacity_tokens": 30000,
    "b_star": {},
    "prompt_limit": None,
}


def published(model: bool = False, calibrated: bool = True, **kwargs) -> TierTable:
    values = {"park": PARK, "h": H, "capacity_h": 3000.0, "alpha_tokens": 460.0}
    values.update(kwargs)
    table = TierTable(**values)
    if calibrated:
        table.evidence["prefill_ms_by_clock"] = {"1815": {"128": 40.0, "2048": 200.0}}
        table.evidence["admission"] = dict(CALIBRATION)
    if model:
        table.evidence["model"] = json.loads(json.dumps(MODEL))
    return table


def setup(
    table=None, telemetry=None, store=None, admission="cells", overload="reject", solver=False
):
    config = load_config()
    config["controller"]["stagger_s"] = 0.0
    config["controller"]["solver"] = solver  # the optional solver; off by default
    config["router"]["admission"] = admission
    config["router"]["overload"] = overload
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
    for d_busy in (False, True):
        for i in range(100):  # one queued request ahead: 12 % violated > theta
            risk.record(risk.cell(H, 1, 128, d_busy), i < 12)
        for _ in range(30):  # idle P: clean, bounds every lighter cell
            risk.record(risk.cell(H, 0, 2048, d_busy), False)
    first = router.try_admit(128, True)
    second = router.try_admit(128, True)
    # G1 wins the tie (production before Canary); with N_await=1 the observed risk
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
    config["router"]["overload"] = "reject"
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
    skips = [r["table_skip"] for r in router.log.recent if r.get("event") == "request"]
    assert skips == [None, "clock_changed", "prompt_length_estimated", "request_failed"]
    assert all(g.n_await == 0 and g.n_inflight == 0 and g.t_await == 0 for g in groups)
    assert list(router.outcomes) == [False]
    assert router.lengths.pairs()[-1] == (128, 8)


def test_plan_drains_the_canary_first_then_parks_down_to_the_minimum() -> None:
    clock, _, groups, _, router, controller, actuator = setup(solver=True)
    asyncio.run(controller.start())
    groups[0].state = GroupState.ACTIVE  # Canary back in service
    asyncio.run(controller.publish(published(model=True), "test"))
    asyncio.run(controller.tick())  # no demand: the solver wants one group
    assert controller.plan.n == 1
    assert all(g.state is GroupState.ACTIVE for g in groups)  # not before t_down
    for _ in range(4):  # one group drained per step
        clock.now += 31
        asyncio.run(controller.tick())
        asyncio.run(controller.tick())
    active = [g for g in groups if g.state is GroupState.ACTIVE]
    assert len(active) == 1 and not active[0].canary
    assert groups[0].state is GroupState.PARK  # Canary parked first
    assert (GpuRef("http://agent", 0), 900) in actuator.calls
    assert (GpuRef("http://agent", 1), 450) in actuator.calls
    router.rejections += 1  # pressure wakes a group at once
    asyncio.run(controller.tick())
    assert len([g for g in groups if g.state is GroupState.ACTIVE]) == 2


def test_no_scale_down_within_t_down_of_pressure() -> None:
    clock, _, groups, _, router, controller, _ = setup(table=published(model=True), solver=True)
    activate(groups)
    controller.on_pressure = lambda reason: True
    router.rejections += 5
    asyncio.run(controller.tick())
    for _ in range(3):  # 27 s < t_down
        clock.now += 9
        asyncio.run(controller.tick())
    assert all(g.state is GroupState.ACTIVE for g in groups)
    clock.now += 31
    asyncio.run(controller.tick())
    asyncio.run(controller.tick())
    assert groups[0].state is not GroupState.ACTIVE  # then the Canary goes first


def test_plan_scales_up_at_once_and_down_only_after_the_dwell() -> None:
    clock, _, groups, tiers, router, controller, _ = setup(
        table=published(model=True), solver=True
    )
    activate(groups[1:2])
    for g in groups[2:] + groups[:1]:
        g.state, g.tier, g.effective = GroupState.PARK, Tier.PARK, None
    for k in range(400):  # 40 req/s over the 10 s load window
        router.arrivals.append(clock.now - 10 + k * 0.025)
    asyncio.run(controller.tick())
    active = [g for g in groups if g.state is GroupState.ACTIVE]
    assert controller.plan.n > 1 and len(active) == controller.plan.n  # up: at once
    working = tiers.working
    router.arrivals.clear()  # demand gone
    clock.now += 1
    asyncio.run(controller.tick())
    assert len([g for g in groups if g.state is GroupState.ACTIVE]) == len(active)
    assert tiers.working == working  # down waits for t_down
    clock.now += 31  # the peak left the dwell window: a cheaper target appears ...
    asyncio.run(controller.tick())
    assert tiers.working == working
    clock.now += 31  # ... and must hold for t_down before the switch
    asyncio.run(controller.tick())
    assert tiers.working != working or any(g.state is GroupState.DRAINING for g in groups)


def test_static_comparison_keeps_every_group_at_h_without_a_solver() -> None:
    clock, _, groups, _, router, controller, _ = setup()
    controller.settings = dataclasses.replace(controller.settings, solver=False)
    asyncio.run(controller.start())
    groups[0].state = GroupState.ACTIVE
    asyncio.run(controller.publish(published(model=True), "test"))
    assert controller.solver() is None
    for _ in range(4):
        clock.now += 31
        asyncio.run(controller.tick())
    assert controller.plan is None
    assert all(g.state is GroupState.ACTIVE and g.tier is Tier.H for g in groups)


def test_failed_lock_is_retried_and_the_group_is_not_counted_meanwhile() -> None:
    clock, _, groups, _, router, controller, actuator = setup()
    asyncio.run(controller.start())
    groups[0].state = GroupState.ACTIVE
    asyncio.run(controller.publish(published(model=True), "test"))
    g = groups[1]
    g.state = GroupState.PARK
    asyncio.run(controller.set_tier(g, Tier.PARK, "test"))
    real = actuator.lock

    async def down(ref, mhz):
        raise RuntimeError("agent unreachable")

    actuator.lock = down
    assert asyncio.run(controller.wake("test")) is g
    assert g.state is GroupState.ACTIVE and g.effective is None  # not routable
    asyncio.run(controller.tick())  # retried and failed again: backing off
    assert g.effective is None and g.name in controller._retry
    routable = [x for x in groups if x.state is GroupState.ACTIVE and x.effective is not None]
    assert g not in routable
    actuator.lock = real  # the agent is back
    for _ in range(6):
        clock.now += controller.settings.t_down_s
        asyncio.run(controller.tick())
        if g.effective is not None:
            break
    assert g.effective is not None and g.name not in controller._retry


def test_failed_lock_on_a_cold_start_is_retried_before_any_table() -> None:
    clock, _, groups, _, router, controller, actuator = setup()
    real = actuator.lock

    async def down(ref, mhz):
        raise RuntimeError("agent unreachable")

    actuator.lock = down
    asyncio.run(controller.start())
    g = next(x for x in groups if x.state is GroupState.ACTIVE)
    assert g.effective is None and g.name in controller._retry
    actuator.lock = real  # the agent is back; still no table
    for _ in range(6):
        clock.now += controller.settings.t_down_s
        asyncio.run(controller.tick())
    assert controller.tiers.table is None
    assert g.effective == MAX and g.name not in controller._retry


def test_pressure_with_nothing_parked_aborts_canary() -> None:
    clock, _, groups, _, router, controller, _ = setup(table=published())
    activate(groups[1:])
    groups[0].state = GroupState.EXPLORING
    reasons = []
    controller.on_pressure = reasons.append
    router.rejections += 1
    asyncio.run(controller.tick())
    assert reasons == ["pressure"]


def test_pressure_without_experiment_boosts_to_max_then_returns_to_h() -> None:
    clock, _, groups, _, router, controller, _ = setup(table=published())
    activate(groups)
    controller.on_pressure = lambda reason: False  # no experiment to abort
    router.rejections += 1
    asyncio.run(controller.tick())
    assert [g.tier for g in groups].count(Tier.MAX) == 1  # one group per tick
    for _ in groups:
        router.rejections += 1
        asyncio.run(controller.tick())
    assert all(g.tier is Tier.MAX and g.effective == MAX for g in groups)
    for _ in range(2 * len(groups) + 2):  # calm
        asyncio.run(controller.tick())
        clock.now += 31
    assert all(g.tier is Tier.H for g in groups if g.state is GroupState.ACTIVE)


def test_pressure_aborting_the_canary_does_not_boost() -> None:
    clock, _, groups, _, router, controller, _ = setup(table=published())
    activate(groups)
    controller.on_pressure = lambda reason: True  # the Canary comes back instead
    router.rejections += 1
    asyncio.run(controller.tick())
    assert all(g.tier is Tier.H for g in groups)


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


def test_length_shift_ignores_median_flips_of_a_discrete_mix() -> None:
    reference = LengthStats([(128, 64), (512, 64), (1024, 64), (2048, 64)], min_samples=1)
    live = LengthStats([(1, 1)], min_samples=1)
    for prompt in [128, 512, 512, 1024, 2048] * 40:  # median 512 instead of 1024
        live.record(prompt, 64)
    assert live.summary().prompt_p50 != reference.summary().prompt_p50
    assert not live.summary().shifted(reference.summary(), 0.30)
    for _ in range(400):
        live.record(2048, 256)
    assert live.summary().shifted(reference.summary(), 0.30)


# ---- state-based (slack) admission -------------------------------------------------


def test_stage_accounting_follows_a_request_through_p_transfer_and_decode() -> None:
    _, _, groups, _, router, _, _ = setup(table=published(), admission="slack")
    activate(groups)
    t = router.try_admit(1024, True)
    g = t.group
    assert (g.n_at_p, g.n_decoding, g.inflight_bytes) == (1, 0, 0)
    assert g.pending_ms == pytest.approx(t.s_own_ms) and t.s_own_ms > 0
    router.prefill_done(t)
    assert (g.n_at_p, g.pending_ms) == (0, 0) and g.inflight_bytes == 1024 * 131072
    router.first_token(t)
    assert g.inflight_bytes == 0 and g.n_decoding == 1
    router.finish(t, status="ok", ttft_ms=300, tpot_ms=60, output_tokens=8)
    assert (g.n_at_p, g.n_decoding, g.inflight_bytes, g.n_inflight) == (0, 0, 0, 0)
    # a failure in the transfer stage releases the in-flight bytes too
    t2 = router.try_admit(512, True)
    router.prefill_done(t2)
    router.finish(t2, status="decode_error", ttft_ms=None, tpot_ms=None, output_tokens=0)
    assert t2.group.inflight_bytes == 0


def test_canary_seeded_slack_admits_then_a_violating_bucket_closes() -> None:
    _, _, groups, _, router, _, _ = setup(table=published(), admission="slack")
    activate(groups)
    t = router.try_admit(128, True)
    assert t is not None and t.estimate.source == "slack_observed"
    refused_after = None
    for i in range(60):  # requests in that slack bucket keep violating
        tk = router.try_admit(128, True)
        if tk is None:
            refused_after = i
            break
        router.first_token(tk)
        router.finish(tk, status="ok", ttft_ms=1500, tpot_ms=60, output_tokens=8)
    assert refused_after is not None and refused_after <= 30


def test_kv_in_flight_gate_blocks_the_group_until_its_d_takes_the_kv() -> None:
    _, _, groups, _, router, _, _ = setup(table=published(), admission="slack")
    activate(groups)
    for g in groups:
        g.inflight_bytes = 0.6e9  # > 0.5 x 1 GB buffer
    assert router.try_admit(128, True) is None
    groups[1].inflight_bytes = 0.1e9
    t = router.try_admit(128, True)
    assert t is not None and t.group is groups[1]


def test_predictor_refits_towards_observed_ttft() -> None:
    from canatune.domain.admission import TtftPredictor

    p = TtftPredictor((100.0, 1.0, 0.6, 800.0, 1.0), refit_every=10, ridge=0.01)
    for pending in range(0, 500, 5):
        x = p.features(50, pending, 0, 0)
        p.record(x, 200 + 2.0 * pending)  # true pending weight 2.0 (prior 0.6)
    assert p.coef[2] == pytest.approx(2.0, rel=0.1)
    assert p.predict(p.features(50, 300, 0, 0)) == pytest.approx(800, rel=0.1)


def test_slack_risk_bounds_sparse_buckets_by_less_slack(tmp_path) -> None:
    from canatune.domain.admission import SlackRisk

    s = SlackRisk(min_samples=20, path=tmp_path / "s.json")
    for _ in range(30):
        s.record(50, True)  # little slack: violated
    for _ in range(30):
        s.record(650, False)  # plenty of slack: clean
    for _ in range(5):
        s.record(150, False)  # sparse: bounded by the risky bucket below it
    assert s.estimate(150).source == "bounded" and s.estimate(150).risk > 0.1
    assert s.estimate(750).source == "bounded" and s.estimate(750).risk < 0.1
    assert s.estimate(-1000).source == "unknown" and s.estimate(-1000).risk == 1.0
    s.save()
    t = SlackRisk(min_samples=20, path=tmp_path / "s.json")
    t.load()
    assert t.to_json() == s.to_json()


def test_kv_bytes_per_token_from_model_config() -> None:
    from canatune.domain.admission import kv_bytes_per_token

    mistral = {
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "hidden_size": 4096,
    }
    assert kv_bytes_per_token(mistral) == 131072


def test_serve_rescues_to_the_lowest_predicted_group_and_counts_pressure() -> None:
    _, _, groups, _, router, _, _ = setup(table=published(), admission="slack", overload="serve")
    activate(groups)
    for g in groups:
        g.pending_ms = 4000.0
    groups[1].pending_ms = 300.0  # within the SLO, but its slack bucket is risky
    for _ in range(30):
        router.slack.record(1000 - 400 - router._predict(groups[1], 33)[1], True)
    t = router.step(router.new_waiter(), 128, True, 400.0)
    assert t.group is groups[1] and t.overflow == "rescue"
    assert t.estimate.source == "overflow_rescue" and router.pressure >= 1
    assert t.slack_ms == pytest.approx(1000 - 400 - t.predicted_ms)
    # overflow outcomes stay out of the drift signal; the predictor learns TTFT - wait
    router.first_token(t)
    t.wait_ms = 400.0
    router.finish(t, status="ok", ttft_ms=1500, tpot_ms=60, output_tokens=8)
    assert len(router.outcomes) == 0
    assert router.predictor.samples[-1][1] == pytest.approx(1100)


def test_serve_doomed_backfills_spare_capacity_or_dispatches_by_setting() -> None:
    _, _, groups, _, router, _, _ = setup(table=published(), admission="slack", overload="serve")
    activate(groups)
    for g in groups:
        g.pending_ms = 4000.0  # SLO lost everywhere
    waiter = router.new_waiter()
    assert router.step(waiter, 128, True, 1000.0) == "wait"  # no spare capacity: hold
    groups[2].pending_ms = 0.0  # feasible for a fresh request: backfill it
    t = router.step(waiter, 128, True, 1100.0)
    assert t.group is groups[2] and t.overflow == "backfill"
    assert router.state()["holding"] == 0

    _, _, groups, _, router, _, _ = setup(table=published(), admission="slack", overload="serve")
    router = CanaTuneRouter(
        groups,
        router.table,
        dataclasses.replace(router.settings, doomed="dispatch"),
        router.tiers,
    )
    activate(groups)
    for g in groups:
        g.pending_ms = 4000.0
    groups[1].pending_ms = 2500.0
    t = router.step(router.new_waiter(), 128, True, 1000.0)
    assert t.group is groups[1] and t.overflow == "doomed"


def test_serve_holds_first_come_first_served_while_every_group_is_at_a_hard_limit() -> None:
    _, _, groups, _, router, _, _ = setup(table=published(), admission="slack", overload="serve")
    activate(groups)
    for g in groups:
        g.inflight_bytes = 0.6e9  # KV gate: a hard limit, never overflowed
    first, second = router.new_waiter(), router.new_waiter()
    assert router.step(first, 128, True, 1000.0) == "wait"
    assert router.step(second, 128, True, 1000.0) == "wait"
    groups[0].inflight_bytes = 0.0  # free again
    assert router.step(second, 128, True, 1000.0) == "wait"  # not the oldest holder
    t = router.step(first, 128, True, 1000.0)
    assert t.group is groups[0] and t.overflow == "backfill"
    for g in groups:
        g.inflight_bytes = 0.6e9
    late = router.new_waiter()
    assert router.step(late, 128, True, router.settings.hold_max_ms) == "reject"
    assert router.rejections == 1 and router.state()["holding"] == 1  # `second` still waits


def test_doomed_waits_for_a_rescue_until_its_slo_deadline_then_best_effort() -> None:
    table = published()
    table.evidence["idle_ttft_ms"] = {"128": 150.0, "2048": 250.0}  # the Canary's idle TTFT
    clock, _, groups, _, router, _, _ = setup(table=table, admission="slack", overload="serve")
    activate(groups)
    assert router.deadline_ms(128) == pytest.approx(850)  # SLO - idle TTFT
    assert router.deadline_ms(1088) == pytest.approx(800)  # interpolated
    for g in groups:
        g.inflight_bytes = 0.6e9  # every group at the KV gate
    old, young = router.new_waiter(), router.new_waiter()
    assert router.step(old, 128, True, 840.0) == "wait"  # deadline 850 ms: savable
    assert router.step(young, 128, True, 200.0) == "wait"
    clock.now += 0.1  # old passes its deadline, young is still within its own
    assert router.state()["holding_expired"] == 1
    groups[0].inflight_bytes = 0.0
    assert router.step(old, 128, True, 940.0) == "wait"  # a savable holder goes first
    t = router.step(young, 128, True, 300.0)
    assert t.group is groups[0] and t.overflow is None  # met the SLO
    router.prefill_done(t)
    router.first_token(t)  # young's prompt is done: the group has spare capacity again
    groups[0].inflight_bytes = 0.0
    t = router.step(old, 128, True, 950.0)
    assert t is not None and t.overflow == "backfill"  # then the late one, best effort
    late = router.new_waiter()
    assert router.step(late, 128, True, router.settings.hold_max_ms) == "reject"  # timeout


def test_controller_boosts_on_overflow_pressure() -> None:
    clock, _, groups, _, router, controller, actuator = setup(
        table=published(), admission="slack", overload="serve"
    )
    activate(groups)
    router.overflows["doomed"] += 3
    asyncio.run(controller.tick())
    assert any(g.tier is Tier.MAX for g in groups)


def test_best_effort_signals_before_wait_budget_and_serves_late_requests():
    _, _, groups, _, router, _, _ = setup(
        table=published(), admission="slack", overload="best_effort"
    )
    activate(groups)
    for g in groups:
        g.pending_ms = 4000
    waiter = router.new_waiter()
    ticket = router.step(waiter, 128, True, 0)
    assert router.pressure_event.is_set() and router.risk_signals == 1
    assert ticket.overflow == "best_effort"
    assert router.rejections == 0


def test_best_effort_hard_limits_queue_fifo_and_cancellation_releases():
    _, _, groups, _, router, _, _ = setup(
        table=published(), admission="slack", overload="best_effort"
    )
    activate(groups)
    router.full_effort = True
    for g in groups:
        g.inflight_bytes = 0.6e9
    first, second = router.new_waiter(), router.new_waiter()
    assert router.step(first, 128, True, 1500) == "wait"
    assert router.step(second, 128, True, 1500) == "wait"
    groups[1].inflight_bytes = 0
    assert router.step(second, 128, True, 1520) == "wait"
    router._release(first)  # same cleanup used by admit() when cancelled
    assert router.step(second, 128, True, 1540).group is groups[1]


def test_full_effort_wakes_all_and_prevents_energy_plans_until_calm():
    clock, _, groups, _, router, controller, _ = setup(
        table=published(model=True), admission="slack", overload="best_effort"
    )
    activate(groups[:1])
    controller.settings = dataclasses.replace(controller.settings, confirm_s=0)
    ticket = router.try_admit(128, True)
    ticket.admitted_at -= 3  # unfinished production request, severe measured TTFT
    asyncio.run(controller.tick())
    router.finish(ticket, status="error", ttft_ms=None, tpot_ms=None, output_tokens=0)
    assert router.full_effort
    assert all(g.routable and g.tier is Tier.MAX for g in groups)
    clock.now += 5
    asyncio.run(controller.tick())
    assert all(g.tier is Tier.MAX for g in groups)
    clock.now += controller.settings.t_down_s
    asyncio.run(controller.tick())
    assert not router.full_effort
    assert all(g.state is GroupState.ACTIVE and g.tier is Tier.H for g in groups)


def test_router_pressure_wakes_controller_before_period():
    async def run():
        _, _, groups, _, router, controller, _ = setup(
            table=published(), admission="slack", overload="best_effort"
        )
        activate(groups)
        controller.settings = dataclasses.replace(controller.settings, period_s=30)
        stop = asyncio.Event()
        task = asyncio.create_task(controller.run(stop))
        await asyncio.sleep(0.01)
        router.signal_pressure("test")
        for _ in range(50):
            if controller.mode == "warning":
                break
            await asyncio.sleep(0.01)
        stop.set()
        await asyncio.wait_for(task, 1)
        assert controller.mode == "warning" and not router.full_effort
    asyncio.run(run())


def test_best_effort_service_timeout_is_enforced_even_when_capacity_frees():
    _, _, groups, _, router, _, _ = setup(
        table=published(), admission="slack", overload="best_effort"
    )
    activate(groups)
    router.full_effort = True
    assert router.step(router.new_waiter(), 128, True, router.settings.hold_max_ms) == "reject"
    assert router.admitted == 0 and router.state()["queued"] == 0


def test_decode_tpot_prediction_warns_but_is_not_a_hard_wall():
    table = published(model=True)
    table.evidence["model"]["decode"] = {"1500": [250, 0]}
    _, _, groups, _, router, _, _ = setup(
        table=table, admission="slack", overload="best_effort"
    )
    activate(groups)
    waiter = router.new_waiter()
    assert router.step(waiter, 128, True, 0).overflow == "best_effort"
    assert router.risk_signals == 1 and not router.full_effort


def test_cold_full_effort_returns_to_calibration_after_idle_cooldown():
    clock, _, groups, _, router, controller, _ = setup(
        table=None, admission="slack", overload="best_effort"
    )
    activate(groups)
    controller.settings = dataclasses.replace(controller.settings, confirm_s=0)
    ticket = router.try_admit(128, True)
    ticket.admitted_at -= 3  # unfinished production request, severe measured TTFT
    asyncio.run(controller.tick())
    router.finish(ticket, status="error", ttft_ms=None, tpot_ms=None, output_tokens=0)
    assert router.full_effort
    clock.now += controller.settings.t_down_s + 1
    asyncio.run(controller.tick())
    assert not router.full_effort
    assert all(g.tier is Tier.MAX for g in groups)


def record_production(router, ttft=950, tpot=100, count=5, tokens=128):
    for _ in range(count):
        ticket = router.try_admit(tokens, True, best_effort=True)
        assert ticket is not None
        router.finish(ticket, status="ok", ttft_ms=ttft, tpot_ms=tpot, output_tokens=64)


def test_prediction_alone_warns_without_expansion_or_max():
    clock, _, groups, _, router, controller, _ = setup(
        table=published(model=True), admission="slack", overload="best_effort"
    )
    activate(groups[1:2])
    groups[1].pending_ms = 4000
    ticket = router.step(router.new_waiter(), 128, True, 0)
    router.finish(ticket, status="ok", ttft_ms=100, tpot_ms=60, output_tokens=64)
    asyncio.run(controller.tick())
    clock.now += 5
    asyncio.run(controller.tick())
    assert controller.mode == "warning" and not router.full_effort
    assert len([g for g in groups if g.routable]) == 1
    assert groups[1].tier is Tier.H


def test_production_confirmation_expands_then_persistent_pressure_enters_max():
    clock, _, groups, _, router, controller, _ = setup(
        table=published(model=True), admission="slack", overload="best_effort"
    )
    activate(groups[1:2])
    router.signal_pressure("prediction")
    record_production(router)
    asyncio.run(controller.tick())
    assert controller.mode == "confirming" and not router.full_effort
    clock.now += controller.settings.confirm_s
    asyncio.run(controller.tick())
    assert controller.mode == "expanding" and not router.full_effort
    assert sum(g.routable for g in groups) == 2
    assert all(g.tier is Tier.H for g in groups if g.routable)
    # Old pre-expansion samples alone cannot trigger MAX, even after the grace period.
    clock.now += controller.settings.expansion_grace_s
    asyncio.run(controller.tick())
    assert not router.full_effort
    record_production(router)
    asyncio.run(controller.tick())
    clock.now += controller.settings.expansion_grace_s
    asyncio.run(controller.tick())
    assert controller.mode == "full_effort" and router.full_effort
    assert all(g.routable and g.tier is Tier.MAX for g in groups)


def test_static_expands_using_measured_equivalent_token_capacity():
    clock, _, groups, _, router, controller, _ = setup(
        table=published(capacity_h=1000), admission="slack", overload="best_effort"
    )
    controller.settings = dataclasses.replace(controller.settings, solver=False)
    activate(groups[1:2])
    for _ in range(3):
        router.new_waiter()  # 3 req/s; C_H/(512+460) = 1.029 req/s/group
    record_production(router, tokens=512)
    asyncio.run(controller.tick())
    clock.now += controller.settings.confirm_s
    for _ in range(3):
        router.new_waiter()
    asyncio.run(controller.tick())
    assert controller.mode == "expanding"
    assert sum(g.routable for g in groups) == 3
    assert not router.full_effort


def test_one_violation_does_not_confirm_and_intrinsically_long_ttft_is_excluded():
    table = published()
    table.evidence["idle_ttft_ms"] = {"128": 100, "2048": 1200}
    clock, _, groups, _, router, controller, _ = setup(
        table=table, admission="slack", overload="best_effort"
    )
    activate(groups)
    record_production(router, ttft=1500, count=1)
    asyncio.run(controller.tick())
    assert controller.mode == "energy"
    clock.now += controller.settings.feedback_window_s + 1
    record_production(router, ttft=1500, tokens=2048)
    clock.now += 3
    asyncio.run(controller.tick())
    assert controller.mode == "energy" and not router.full_effort


def test_live_decode_stall_confirms_without_waiting_for_request_completion():
    clock, _, groups, _, router, controller, _ = setup(
        table=published(), admission="slack", overload="best_effort"
    )
    activate(groups[1:2])  # the others parked: expansion has groups to wake
    ticket = router.try_admit(128, True)
    router.first_token(ticket)
    clock.now += 0.25
    asyncio.run(controller.tick())
    assert controller.mode == "confirming" and not ticket.finished
    clock.now += controller.settings.confirm_s
    asyncio.run(controller.tick())
    assert controller.mode == "expanding" and not router.full_effort
    clock.now += controller.settings.confirm_s
    asyncio.run(controller.tick())
    assert router.full_effort and not ticket.finished


def test_confirmed_pressure_with_every_group_serving_goes_straight_to_max():
    clock, _, groups, _, router, controller, _ = setup(
        table=published(), admission="slack", overload="best_effort"
    )
    controller.settings = dataclasses.replace(controller.settings, solver=False)
    activate(groups)  # static: nothing parked, expansion cannot add capacity
    record_production(router)
    asyncio.run(controller.tick())
    assert controller.mode == "confirming"
    clock.now += controller.settings.confirm_s
    asyncio.run(controller.tick())
    assert controller.mode == "full_effort" and router.full_effort  # no expansion grace
    assert all(g.tier is Tier.MAX for g in groups)


def test_stream_feedback_counts_requests_and_cancellation_cleans_it_up():
    _, _, groups, _, router, _, _ = setup(
        table=published(), admission="slack", overload="best_effort"
    )
    activate(groups)
    ticket = router.try_admit(128, True)
    router.first_token(ticket)
    for _ in range(20):
        router.token_progress(ticket, 100)
    assert router.production_feedback(10, 5, 0.9)["samples"] == 1
    router.finish(ticket, status="client_disconnected", ttft_ms=None, tpot_ms=None, output_tokens=0)
    assert not router._live and not router._production


def test_solver_capacity_prediction_cannot_expand_without_production_confirmation():
    _, _, groups, _, router, controller, _ = setup(
        table=published(model=True), admission="slack", overload="best_effort", solver=True
    )
    activate(groups[1:2])
    for _ in range(100):
        router.new_waiter()
    asyncio.run(controller.tick())
    assert controller.mode == "warning" and not router.full_effort
    assert sum(g.routable for g in groups) == 1


def test_old_first_token_latency_is_not_refreshed_by_new_tokens_or_completion():
    clock, _, groups, _, router, _, _ = setup(
        table=published(), admission="slack", overload="best_effort"
    )
    activate(groups)
    ticket = router.try_admit(128, True)
    router.first_token(ticket)
    router.observe_latency(ticket, 1200, 300)
    cutoff = clock.now + 1
    clock.now += 2
    router.token_progress(ticket, 50)
    router.finish(ticket, status="ok", ttft_ms=1200, tpot_ms=300, output_tokens=64)
    feedback = router.production_feedback(10, 1, 0.9, since=cutoff)
    assert feedback["samples"] == 1 and not feedback["pressure"]


def test_expansion_recovers_without_max_if_fresh_feedback_is_healthy():
    clock, _, groups, _, router, controller, _ = setup(
        table=published(model=True), admission="slack", overload="best_effort"
    )
    activate(groups[1:2])
    record_production(router)
    asyncio.run(controller.tick())
    clock.now += controller.settings.confirm_s
    asyncio.run(controller.tick())
    assert controller.mode == "expanding"
    clock.now += 1
    record_production(router, ttft=150, tpot=50)
    asyncio.run(controller.tick())
    clock.now += controller.settings.t_down_s
    asyncio.run(controller.tick())
    assert controller.mode == "energy" and not router.full_effort
    assert all(g.tier is Tier.H for g in groups if g.routable)


def test_severe_production_wait_must_persist_before_emergency_max():
    clock, _, groups, _, router, controller, _ = setup(
        table=published(), admission="slack", overload="best_effort"
    )
    activate(groups)
    ticket = router.try_admit(128, True)
    clock.now += 2.1
    asyncio.run(controller.tick())
    assert controller.mode == "confirming" and not router.full_effort
    clock.now += controller.settings.confirm_s
    asyncio.run(controller.tick())
    assert router.full_effort
    assert all(g.tier is Tier.MAX for g in groups)
    router.finish(ticket, status="client_disconnected", ttft_ms=None, tpot_ms=None, output_tokens=0)


@pytest.mark.parametrize("kwargs", [
    {"feedback_window_s": 0}, {"feedback_window_s": float("nan")},
    {"feedback_min_samples": 0}, {"feedback_near_slo": 1.1},
    {"confirm_s": -1}, {"expansion_grace_s": float("inf")},
])
def test_invalid_feedback_settings_are_rejected(kwargs):
    from canatune.controller.tier_controller import ControllerConfigError

    with pytest.raises(ControllerConfigError):
        ControllerSettings(**kwargs)


def test_a_prompt_larger_than_the_kv_gate_is_admitted_on_an_empty_link() -> None:
    clock, _, groups, _, router, controller, _ = setup(
        table=published(model=True), admission="slack", overload="best_effort"
    )
    activate(groups)
    router._apply_table()
    router.kv_fraction = 0.2  # gate 0.2 GB: a 2048-token prompt (0.27 GB) exceeds it
    big = 2048
    assert big * router.settings.kv_bytes_per_token > 0.2 * router.settings.kv_buffer_bytes
    assert not router._kv_blocked(groups[1], big)  # alone on an empty link
    groups[1].inflight_bytes = 0.1e9
    assert router._kv_blocked(groups[1], big)  # would pass the gate with others in flight
    groups[1].inflight_bytes = 0.0
    assert router.try_admit(big, True, 0.0, best_effort=True) is not None


@pytest.mark.parametrize("full_effort", [False, True])
def test_best_effort_b_star_warns_but_does_not_limit_decode_throughput(full_effort):
    clock = FakeClock()
    telemetry = Telemetry({}, period_s=1, client_factory=lambda: None, clock=clock)
    _, _, groups, _, router, _, _ = setup(
        table=published(decode_max_running=24, decode_kv_limit=0.9),
        telemetry=telemetry, admission="slack", overload="best_effort",
    )
    activate(groups[1:2])
    group = groups[1]
    group.n_inflight = group.n_decoding = 28
    if full_effort:
        group.tier, group.effective = Tier.MAX, MAX
    router.full_effort = full_effort
    telemetry.update(group.decode, EndpointSnapshot(clock.now, 28, 4, 0.5, 0, True))
    ticket = router.step(router.new_waiter(), 128, True, 0)
    assert ticket.group is group and ticket.overflow == "best_effort"
    assert group.n_inflight == 29 and router.state()["queued"] == 0
    assert router.risk_signals == 1


def test_prefill_reservations_are_not_counted_as_b_star_decode_sequences():
    _, _, groups, _, router, _, _ = setup(
        table=published(decode_max_running=24), admission="slack", overload="best_effort"
    )
    activate(groups[1:2])
    group = groups[1]
    group.n_inflight = group.n_await = group.n_at_p = 24
    ticket = router.step(router.new_waiter(), 128, True, 0)
    assert ticket.group is group and ticket.overflow is None
    assert router.risk_signals == 0


@pytest.mark.parametrize("blocked", ["kv_usage", "transfer_buffer", "stale"])
def test_full_effort_still_enforces_kv_and_telemetry_guards(blocked):
    clock = FakeClock()
    telemetry = Telemetry({}, period_s=1, client_factory=lambda: None, clock=clock)
    _, _, groups, _, router, _, _ = setup(
        table=published(decode_max_running=24, decode_kv_limit=0.9),
        telemetry=telemetry, admission="slack", overload="best_effort",
    )
    activate(groups[1:2])
    group = groups[1]
    router.full_effort = True
    snapshot_at = clock.now - 5 if blocked == "stale" else clock.now
    telemetry.update(group.decode, EndpointSnapshot(
        snapshot_at, 28, 0, 0.95 if blocked == "kv_usage" else 0.5, 0, True,
    ))
    if blocked == "transfer_buffer":
        group.inflight_bytes = 0.6e9
    assert router.step(router.new_waiter(), 128, True, 0) == "wait"
    assert router.admitted == 0


def test_repeated_prediction_warnings_do_not_starve_solver_or_restart_target_hold():
    clock, _, groups, _, router, controller, _ = setup(
        table=published(model=True), admission="slack", overload="best_effort", solver=True
    )
    activate(groups)
    target_since = None
    for _ in range(7):
        router.signal_pressure("prediction_only")
        asyncio.run(controller.tick())
        if target_since is None:
            target_since = controller._target_since
        assert controller._pressure_at == float("-inf")
        if clock.now < target_since + controller.settings.t_down_s:
            assert controller._target_since == target_since
        clock.now += 5
    plans = [e for e in controller.log.recent if e.get("event") == "plan"]
    assert plans and plans[0]["to"]["n"] == 1
    assert not router.full_effort
    assert sum(g.state is GroupState.PARK for g in groups) >= 1


@pytest.mark.parametrize("mode", ["warning", "confirming"])
def test_feedback_observation_modes_do_not_block_draining_cleanup(mode):
    clock, _, groups, _, router, controller, _ = setup(
        table=published(), admission="slack", overload="best_effort"
    )
    activate(groups)
    groups[2].state = GroupState.DRAINING
    if mode == "warning":
        router.signal_pressure("prediction")
    else:
        record_production(router)
    asyncio.run(controller.tick())
    assert controller.mode == mode
    assert groups[2].state is GroupState.PARK and groups[2].tier is Tier.PARK


def test_a_non_streaming_request_past_prefill_is_no_ttft_backlog():
    clock, _, groups, _, router, controller, _ = setup(
        table=published(), admission="slack", overload="best_effort"
    )
    controller.settings = dataclasses.replace(controller.settings, solver=False)
    activate(groups)
    plain = router.try_admit(128, True)
    plain.streaming = False  # as the proxy marks a non-streaming response
    router.prefill_done(plain)
    for _ in range(6):  # a long generation: its end-to-end time is no TTFT
        clock.now += 1
        asyncio.run(controller.tick())
    assert controller.mode == "energy" and not router.full_effort
    assert controller.feedback["live_ratio"] == 0
    router.finish(plain, status="ok", ttft_ms=None, tpot_ms=None, output_tokens=0)
    # A streaming request still shows a missing first token, and so does a
    # non-streaming one stuck at P.
    for streaming in (True, False):
        ticket = router.try_admit(128, True)
        ticket.streaming = streaming
        if streaming:
            router.prefill_done(ticket)
        clock.now += 1
        asyncio.run(controller.tick())
        assert controller.feedback["live_ratio"] >= 1
        router.finish(ticket, status="ok", ttft_ms=None, tpot_ms=None, output_tokens=0)


def test_arrivals_during_slow_clock_changes_do_not_break_the_tick():
    clock, _, groups, _, router, controller, actuator = setup(
        table=published(model=True), admission="slack", overload="best_effort"
    )
    controller.settings = dataclasses.replace(controller.settings, confirm_s=0)
    activate(groups)
    real = actuator.lock

    async def slow(ref, mhz):  # each lock takes 0.6 s while requests keep arriving
        clock.now += 0.6
        router.new_waiter()
        return await real(ref, mhz)

    actuator.lock = slow
    record_production(router)
    asyncio.run(controller.tick())
    asyncio.run(controller.tick())
    assert router.full_effort
    clock.now += controller.settings.t_down_s + controller.settings.feedback_window_s + 1
    asyncio.run(controller.tick())  # recovery locks every group back to H, then plans
    assert not router.full_effort and controller.mode == "energy"
    rate, burst = router.offered(clock.now - 2.0, 1.0, 1.0)  # arrivals after `now`
    assert rate > 0 and burst >= 1.0


def test_the_solver_is_off_by_default() -> None:
    config = load_config()
    assert ControllerSettings.from_config(config).solver is False
    assert ControllerSettings().solver is False
    _, _, groups, _, _, controller, _ = setup(table=published(model=True))
    activate(groups)
    assert controller.solver() is None  # every group stays at the Canary's H


def length_router(running, kv_usage, coef=(50.0, 0.4, 8e-4, 2.0), n_decoding=None):
    clock = FakeClock()
    telemetry = Telemetry({}, period_s=1, client_factory=lambda: None, clock=clock)
    table = published(decode_max_running=24, decode_kv_limit=0.9,
                      decode_length={"1050": list(coef)}, decode_kv_tokens=30000)  # fmt: skip
    _, _, groups, _, router, _, _ = setup(
        table=table, telemetry=telemetry, admission="slack", overload="best_effort"
    )
    activate(groups[1:2])
    group = groups[1]
    group.n_inflight = group.n_decoding = running if n_decoding is None else n_decoding
    telemetry.update(group.decode, EndpointSnapshot(clock.now, running, 0, kv_usage, 0, True))
    return router, group


def test_length_model_lets_short_sequences_pass_the_count_bound():
    # 28 sequences (above B* = 24) holding 9000 tokens: 50 + 0.4 x 29 + 8e-4 x 9160 + 2
    router, group = length_router(28, 0.3)
    ticket = router.step(router.new_waiter(), 128, True, 0)
    assert ticket.group is group and ticket.overflow is None and router.risk_signals == 0
    assert ticket.snapshot["tpot_bound_ms"] == pytest.approx(70.9, abs=0.1)


def test_length_model_warns_on_the_kv_wall_of_long_contexts():
    # 10 sequences holding 25500 tokens: a 2048-token prompt (+64 output) passes 0.9 x 30000
    router, group = length_router(10, 0.85)
    ticket = router.step(router.new_waiter(), 2048, True, 0)
    assert ticket.group is group and ticket.overflow == "best_effort"
    assert router.risk_signals == 1


def test_length_model_warns_when_the_tpot_bound_passes_the_slo():
    # alpha 150 + 2 ms per sequence: 21 sequences put the bound above 200 ms
    router, group = length_router(20, 0.1, coef=(150.0, 2.0, 1e-3, 5.0))
    ticket = router.step(router.new_waiter(), 128, True, 0)
    assert ticket.overflow == "best_effort" and router.risk_signals == 1
    assert ticket.snapshot["tpot_bound_ms"] > 200


def test_length_model_counts_requests_still_at_prefill():
    router, group = length_router(0, 0.0, coef=(150.0, 2.0, 1e-3, 5.0))
    group.n_inflight = group.n_await = group.n_at_p = 24
    ticket = router.step(router.new_waiter(), 128, True, 0)
    assert ticket.overflow == "best_effort"  # 25 sequences on their way: 150 + 50 + ...
