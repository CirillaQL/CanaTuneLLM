"""Tier locator and Canary scheduler against a surrogate cluster.

The surrogate is the one used for the offline replay: P energy/latency
calibrated on K2 (job 267571: s(f) = 39.5 + 73428/f ms per 1024-token request,
power cap at ~2040 MHz, busy and idle power as measured), M/D/1 queueing plus a
180 ms P/D overhead for TTFT; D behaves like K3b (job 267654): a KV wall at
~44 sequences that no clock moves, TPOT barely depends on the clock.
"""

import asyncio
import dataclasses
import math
import random

import pytest

from canatune.config import load_config
from canatune.controller.canary import CanaryScheduler, SchedulerSettings
from canatune.controller.locator import (
    Hardware,
    LocatorError,
    LocatorRun,
    LocatorSettings,
    TierLocator,
    WindowResult,
    fit_alpha,
    spread,
    wilson_ucb,
)
from canatune.controller.router import CanaTuneRouter, RouterSettings
from canatune.controller.tier_controller import ControllerSettings, TierController
from canatune.domain.admission import CostBook
from canatune.domain.calibration import ProbeSample
from canatune.domain.groups import ClockPoint, GroupState, Tier, TierState, TierTable
from canatune.domain.load import LengthStats
from canatune.domain.risk import RiskTable
from canatune.infrastructure.clocks import GpuRef, NullClockActuator
from canatune.service import build_groups, identity

PROMPT = 1024
ALPHA = 28.1 / 0.0615  # surrogate single requests: prefill = 28 ms + 0.0615 ms/token


def interp(x, xs, ys):
    if x <= xs[0]:
        return ys[0]
    for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return ys[-1]


class SurrogateBackend:
    def __init__(self, seed=0, noise=0.01, fail_after=None):
        self.rng = random.Random(seed)
        self.noise = noise
        self.fail_after = fail_after
        self.windows = 0
        self.f_cap = 2040
        self.long_penalty = 0.0  # extra idle TTFT per token above 1500 (ms)
        self.limit = None

    async def hardware(self):
        return Hardware(tuple(range(600, 2521, 15)), tuple(range(300, 1501, 15)))

    def _tick(self):
        self.windows += 1
        if self.fail_after is not None and self.windows > self.fail_after:
            raise asyncio.CancelledError  # simulated abort in the middle of a run

    # --- P model ---
    def eff(self, f):
        return min(f, self.f_cap)

    def s_ms(self, f):
        return 39.5 + 73428 / self.eff(f)

    def pb(self, f):
        return interp(f, [900, 1305, 1815, 2040, 2520], [266.0, 287.6, 316.0, 322.0, 345.0])

    def pi(self, f):
        return interp(f, [600, 900, 1305, 1815, 2520], [62.3, 62.6, 63.2, 64.5, 78.2])

    def limited(self, f):
        return min(0.63, max(0.0, (f - 1400) / (2115 - 1400) * 0.38))

    def model(self, f, rps):
        s = self.s_ms(f)
        rho = rps * s / 1000
        if rho >= 1:
            return math.inf, 1.0, math.inf
        wait = rho * s / (2 * (1 - rho))
        x = 500 - 180 - s
        viol = 1.0 if x <= 0 else (min(1.0, rho * math.exp(-rho * x / wait)) if wait else 0.0)
        energy = self.pb(f) * s / 1000 + self.pi(f) * max(1 / rps - s / 1000, 0)
        return 180 + s + 3 * wait, viol, energy

    async def idle_power(self, clock, seconds):
        self._tick()
        return self.pi(clock.prefill_mhz), 21.7 + 10 * (clock.decode_mhz / 1500) ** 3

    async def service_times(self, clock, prompts):
        self._tick()
        scale = self.s_ms(clock.prefill_mhz) / self.s_ms(2040)
        out = []
        for p in prompts:
            prefill = (28.1 + 0.0615 * p) * scale * (1 + self.rng.gauss(0, 0.01))
            out.append((p, prefill, 180 + prefill + self.long_penalty * max(0, p - 1500)))
        return out

    def set_prompt_limit(self, max_prompt):
        self.limit = max_prompt
        return PROMPT

    async def open_window(self, clock, eq_tps, alpha, seconds, abort_above, output_tokens=None,
                          prompt_lengths=None):
        self._tick()
        f = clock.prefill_mhz
        rps = eq_tps / (PROMPT + alpha)
        ttft, viol, energy = self.model(f, rps)
        n = max(1, int(rps * seconds))
        k = sum(self.rng.random() < viol for _ in range(n))
        aborted = k / n > abort_above
        j = energy * (1 + self.rng.gauss(0, self.noise))
        return WindowResult(
            clock=clock,
            kind="open",
            load=eq_tps,
            duration_s=seconds,
            requests=n,
            violations=k,
            ttft_p95_ms=ttft * (1 + self.rng.gauss(0, 0.03)),
            prefill_j_per_request=j,
            prefill_mhz_median=float(self.eff(f)),
            prefill_limited_fraction=self.limited(f),
            aborted=aborted,
            samples=[
                ProbeSample(f, clock.decode_mhz, PROMPT, (), 0, 0, ttft, i < k) for i in range(n)
            ],
            prefill_avg_w=j * n / seconds,
        )

    async def closed_window(self, clock, concurrency, seconds, rep=0, mix="all", **_):
        self._tick()
        f = clock.decode_mhz
        tpot = (55 + 1.4 * concurrency) * (1050 / f) ** 0.15
        wall = concurrency > 44
        power = 21.7 + 50 * (f / 1500) ** 2
        tps = concurrency / (tpot / 1000)
        return WindowResult(
            clock=clock,
            kind="closed",
            load=concurrency,
            duration_s=seconds,
            requests=int(tps * seconds / 64),
            tpot_p95_ms=tpot * (1.6 if wall else 1.0),
            decode_j_per_token=power / tps * (1 + self.rng.gauss(0, self.noise)),
            decode_preemptions=5.0 if wall else 0.0,
            decode_waiting_max=3.0 if wall else 0.0,
            decode_avg_w=power,
            decode_busy_fraction=1.0,
            decode_running_mean=float(concurrency),
            tpot_p50_ms=tpot,
        )


def settings(**kwargs):
    return LocatorSettings(**kwargs)


def test_helpers() -> None:
    grid = list(range(600, 2521, 15))
    assert spread(grid, 600, 2040, 5) == [2040, 1680, 1320, 960, 600]
    assert wilson_ucb(0, 100, 1.28) < 0.02 < wilson_ucb(1, 20, 1.28)
    assert fit_alpha([(128, 36.0), (1024, 85.0), (2048, 154.0)]) == pytest.approx(455, rel=0.1)


def test_locate_on_surrogate_matches_manual_experiments() -> None:
    backend = SurrogateBackend(seed=1)
    locator = TierLocator(backend, settings())
    table = asyncio.run(locator.locate(PROMPT, [128, 512, 1024, 2048]))
    ev = table.evidence
    # alpha from the K2-shaped service times (~457 tokens)
    assert table.alpha_tokens == pytest.approx(ALPHA, rel=0.1)
    # the power cap is found under load: nothing above 2040 is searched
    assert ev["f_eff"] == 2040
    # H inside the measured 2 % band (1590-2145 at 0.8 C0), not power-limited > 30 %
    assert 1590 <= table.h.prefill_mhz <= 2040
    assert backend.limited(table.h.prefill_mhz) <= 0.3
    # one working tier on this cluster (bands at 0.3 C0 and 0.8 C0 overlap)
    assert table.l is None
    # D: KV wall (B* equal at the chosen and the highest clock), low D clock chosen
    assert table.decode_wall and 32 <= table.decode_max_running <= 44
    assert table.h.decode_mhz < 1500
    # park: lowest idle power -> lowest clocks
    assert table.park == ClockPoint(600, 300)
    assert table.capacity_h > 0.5 * ev["capacity_c0"]
    # Original calibration plus independent searches of the five P distributions.
    assert len(ev["room"]["p_profiles"]) == 5
    assert locator.run.windows <= 90


def test_aborted_run_resumes_from_cached_windows() -> None:
    backend = SurrogateBackend(seed=2, fail_after=12)
    locator = TierLocator(backend, settings())
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(locator.locate(PROMPT, [128, 512, 1024, 2048]))
    backend.fail_after = None
    before = backend.windows
    table = asyncio.run(locator.locate(PROMPT, [128, 512, 1024, 2048]))
    assert locator.run.reused >= 10  # finished windows were not measured again
    fresh = TierLocator(SurrogateBackend(seed=2), settings())
    asyncio.run(fresh.locate(PROMPT, [128, 512, 1024, 2048]))
    assert backend.windows - before < fresh.run.windows
    assert table.h.prefill_mhz >= 1590


def test_prompts_too_long_even_when_idle_are_excluded_from_probes() -> None:
    backend = SurrogateBackend(seed=7)
    backend.long_penalty = 1.0  # 2048 tokens: idle TTFT ~ 880 ms > 500 ms SLO
    locator = TierLocator(backend, settings())
    table = asyncio.run(locator.locate(PROMPT, [128, 512, 1024, 2048]))
    assert table.evidence["prompt_limit"] == 1024 and backend.limit == 1024
    assert table.evidence["idle_ttft_ms"][2048] > 500 >= table.evidence["idle_ttft_ms"][1024]


def test_locate_publishes_a_cluster_model_the_solver_can_use() -> None:
    from canatune.domain.models import ClusterModel, Solver

    table = asyncio.run(
        TierLocator(SurrogateBackend(seed=3), settings()).locate(PROMPT, [128, 1024, 2048])
    )
    model = ClusterModel.from_json(table.evidence["model"])
    assert len(model.prefill) >= 3 and model.rho_prefill > 0
    assert len(model.decode) >= 2 and all(d >= 0 for _, d in model.decode.values())
    assert model.power_prefill and model.power_decode and model.park_power_w > 0
    solver = Solver(model, groups=3)
    low = solver.solve(0.5, [(PROMPT, 64)])
    high = solver.solve(3 * table.evidence["capacity_rps"], [(PROMPT, 64)])
    assert low.feasible and low.n == 1
    assert high.n > low.n  # more load needs more groups


def build(backend, table=None, solver=False):
    config = load_config()
    config["router"]["overload"] = "serve"
    config["controller"]["stagger_s"] = 0.0
    config["controller"]["solver"] = solver  # the optional solver; off by default
    risk = RiskTable.from_config(config["risk"], identity(config))
    groups = build_groups(config)
    tiers = TierState(max_point=ClockPoint(2520, 1500), table=table)
    lengths = LengthStats([(PROMPT, 64)], min_samples=1)
    router = CanaTuneRouter(groups, risk, RouterSettings.from_config(config), tiers)
    refs = {e: GpuRef("http://agent", i) for i, e in enumerate(("P0", "D0", "P1", "D1"))}
    controller = TierController(
        groups, router, ControllerSettings.from_config(config), NullClockActuator(), refs, tiers
    )
    locator = TierLocator(backend, settings())
    scheduler = CanaryScheduler(
        controller,
        locator,
        lengths,
        SchedulerSettings(quiet_s=0, min_interval_s=0, periodic_s=1e9, max_duty=1.0),
        theta=0.1,
    )
    return groups, tiers, router, controller, scheduler


def test_scheduler_cold_start_calibrates_publishes_and_returns_canary() -> None:
    groups, tiers, router, controller, scheduler = build(SurrogateBackend(seed=4))

    async def run():
        await controller.start()
        assert groups[0].state is GroupState.EXPLORING and router.open_admission
        await scheduler.step()
        assert scheduler.running
        await scheduler.task

    asyncio.run(run())
    assert tiers.table is not None and not router.open_admission
    assert all(g.state is GroupState.ACTIVE and g.effective == tiers.table.h for g in groups)
    assert scheduler.history[-1]["outcome"] == "published"


def test_pressure_aborts_experiment_and_canary_serves_again() -> None:
    first = asyncio.run(
        TierLocator(SurrogateBackend(seed=5), settings()).locate(PROMPT, [512, 2048])
    )
    groups, tiers, router, controller, scheduler = build(SurrogateBackend(seed=6), first)

    class SlowBackend(SurrogateBackend):
        async def open_window(self, *args, **kwargs):
            await asyncio.sleep(10)
            return await super().open_window(*args, **kwargs)

    scheduler.locator.backend = SlowBackend(seed=6)

    async def run():
        await controller.start()
        assert groups[0].state is GroupState.ACTIVE  # stored table: Canary serves
        assert scheduler.request("relocate", "test") == (True, "ok")
        await scheduler.step()  # drained (nothing in flight) -> exploring
        assert groups[0].state is GroupState.EXPLORING and scheduler.running
        await asyncio.sleep(0.05)
        router.rejections += 1  # production rejects, nothing is parked
        await controller.tick()
        await asyncio.gather(scheduler.task, return_exceptions=True)

    asyncio.run(run())
    assert scheduler.history[-1]["outcome"] == "aborted"
    assert groups[0].state is GroupState.ACTIVE and groups[0].tier is Tier.H
    assert tiers.table is first  # nothing published


def test_alpha_fit_ignores_one_slow_first_request() -> None:
    # K2 shape, plus a 3 s first request at 128 tokens (connector handshake).
    samples = [(128, 3036.0), (512, 60.0), (1024, 91.0), (128, 36.0), (512, 59.0)]
    samples += [(1024, 92.0), (128, 37.0), (512, 60.5), (1024, 90.0)]
    assert fit_alpha(samples) == pytest.approx(455, rel=0.15)


def test_failed_alpha_is_measured_again_on_retry() -> None:
    backend = SurrogateBackend(seed=8)
    calls = []
    original = backend.service_times

    async def flat_then_real(clock, prompts):
        calls.append(1)
        if len(calls) == 1:
            return [(p, 50.0, 300.0) for p in prompts]  # slope 0: fit fails
        return await original(clock, prompts)

    backend.service_times = flat_then_real
    locator = TierLocator(backend, settings())
    with pytest.raises(LocatorError):
        asyncio.run(locator.locate(PROMPT, [128, 512, 1024]))
    table = asyncio.run(locator.locate(PROMPT, [128, 512, 1024]))
    # the retry measures alpha again; then S(L) at every coarse clock
    assert len(calls) >= 2 and table.alpha_tokens > 0


def test_decode_runs_with_p_at_the_ceiling_and_needs_25pct_for_a_step() -> None:
    class StepBackend(SurrogateBackend):
        def __init__(self, seed, top_wall):
            super().__init__(seed=seed)
            self.top_wall = top_wall
            self.closed_clocks = []

        async def closed_window(self, clock, concurrency, seconds, rep=0, mix="all", **_):
            if mix == "all":  # the room step runs at H, not on this search
                self.closed_clocks.append(clock)
            w = await super().closed_window(clock, concurrency, seconds, rep)
            wall = self.top_wall if clock.decode_mhz == 1500 else 44
            if concurrency > wall:
                w.tpot_p95_ms, w.decode_preemptions, w.decode_waiting_max = 400.0, 5.0, 3.0
            else:
                w.tpot_p95_ms, w.decode_preemptions, w.decode_waiting_max = 90.0, 0.0, 0.0
            return w

    small = StepBackend(9, top_wall=48)  # +9 %: still a KV wall
    table = asyncio.run(TierLocator(small, settings()).locate(PROMPT, [128, 512, 1024]))
    assert table.decode_wall
    assert all(c.prefill_mhz == table.evidence["f_eff"] for c in small.closed_clocks)
    big = StepBackend(10, top_wall=96)  # D clock really buys capacity
    table = asyncio.run(TierLocator(big, settings()).locate(PROMPT, [128, 512, 1024]))
    assert not table.decode_wall


def test_no_feasible_fill_load_publishes_nothing() -> None:
    backend = SurrogateBackend(seed=11)
    locator = TierLocator(backend, settings())
    original = backend.open_window

    async def fill_fails(clock, eq_tps, alpha, seconds, abort_above, **kw):
        w = await original(clock, eq_tps, alpha, seconds, abort_above)
        if locator.run is not None and locator.run.phase == "fill":
            w.violations = w.requests
        return w

    async def joint_as_measured(hw, prefill_mhz, decode_mhz, load, alpha):
        return decode_mhz, {}  # keep D where decode chose it: fill windows are new

    backend.open_window = fill_fails
    locator.joint = joint_as_measured
    with pytest.raises(LocatorError, match="not published"):
        asyncio.run(locator.locate(PROMPT, [128, 512, 1024]))


def test_alpha_lengths_include_the_longest_prompt() -> None:
    _, _, _, _, scheduler = build(SurrogateBackend(seed=12))
    scheduler.lengths = LengthStats([(128, 8)] * 50 + [(4000, 8)], min_samples=1)
    assert 4000 in scheduler._prompts()


def test_joint_check_raises_d_when_slow_d_breaks_ttft_at_h() -> None:
    backend = SurrogateBackend(seed=13)
    original = backend.open_window

    async def slow_d_delays_first_token(clock, eq_tps, alpha, seconds, abort_above, **kw):
        w = await original(clock, eq_tps, alpha, seconds, abort_above)
        if clock.decode_mhz < 1000 and eq_tps > 0.6 * 4000:  # smoke r3: D 735 at 0.8 C0
            w.violations = w.requests // 4
        return w

    backend.open_window = slow_d_delays_first_token
    table = asyncio.run(TierLocator(backend, settings()).locate(PROMPT, [128, 512, 1024]))
    joint = table.evidence["joint"]
    assert joint["decode_alone"] < 1000 <= table.h.decode_mhz
    assert table.capacity_h >= table.evidence["target_load"]


def test_fill_bisects_capacity_between_fill_loads() -> None:
    backend = SurrogateBackend(seed=14)
    locator = TierLocator(backend, settings())
    original = backend.open_window

    async def wall_at_090(clock, eq_tps, alpha, seconds, abort_above, **kw):
        w = await original(clock, eq_tps, alpha, seconds, abort_above)
        c0 = (locator.run.evidence if locator.run else {}).get("capacity_c0")
        if locator.run is not None and locator.run.phase == "fill" and eq_tps > 0.9 * c0:
            w.violations = w.requests
        return w

    backend.open_window = wall_at_090
    table = asyncio.run(locator.locate(PROMPT, [128, 512, 1024]))
    c0 = table.evidence["capacity_c0"]
    assert 0.8 * c0 < table.capacity_h <= 0.9 * c0


def test_d_clock_choice_uses_the_median_of_repeated_windows() -> None:
    class NoisyBackend(SurrogateBackend):
        async def closed_window(self, clock, concurrency, seconds, rep=0, mix="all", **_):
            w = await super().closed_window(clock, concurrency, seconds, rep)
            if clock.decode_mhz == 300 and rep == 0 and concurrency < 44:
                w.decode_j_per_token = 0.001  # one absurd window must not decide
            return w

    table = asyncio.run(
        TierLocator(NoisyBackend(seed=15), settings()).locate(PROMPT, [128, 512, 1024])
    )
    ev = table.evidence["decode"]
    assert ev["decode_clock_repeats"] == 2
    assert table.h.decode_mhz != 300 or ev["decode_j_per_token"][300] > 0.001


def test_alpha_fit_reports_curvature() -> None:
    from canatune.controller.locator import curvature

    linear = [(t, 28 + 0.06 * t) for t in (128, 512, 1024, 2048)]
    bent = [(t, 28 + 0.06 * t + 2e-5 * t * t) for t in (128, 512, 1024, 2048)]
    assert abs(curvature(linear)["quad_share"]) < 1e-6
    assert curvature(bent)["quad_share"] == pytest.approx(
        2e-5 * 2048**2 / (28 + 0.06 * 2048 + 2e-5 * 2048**2), rel=1e-3
    )
    assert curvature(linear[:2]) is None


def test_contradicting_prefill_tables_are_measured_again() -> None:
    # The first table (the top clock, right after start) carries +25 ms of warm-up.
    backend = SurrogateBackend(seed=4)
    original = backend.service_times
    calls = []

    async def first_disturbed(clock, prompts):
        calls.append(clock)
        out = await original(clock, prompts)
        if len(calls) == 1:
            out = [(p, ms + 25.0, ttft + 25.0) for p, ms, ttft in out]
        return out

    backend.service_times = first_disturbed
    table = asyncio.run(TierLocator(backend, settings()).locate(PROMPT, [128, 512, 1024, 2048]))
    ev = table.evidence
    top = calls[0].prefill_mhz
    assert top in ev["prefill_consistency"]["remeasured"]
    assert not ev["prefill_consistency"]["envelope"]  # consistent after the re-measurement
    assert not TierLocator.table_conflicts(ev["prefill_ms_by_clock"], 0.02)
    assert ev["idle_ttft_ms"][128] < 180 + 28.1 + 0.0615 * 128 + 10  # refreshed too


def test_persistent_contradictions_take_the_lower_clocks_envelope() -> None:
    tables = {1080: {16: 44.0, 2048: 240.0}, 2520: {16: 67.0, 2048: 150.0}}
    assert TierLocator.table_conflicts(tables, 0.02) == {1080, 2520}
    assert not TierLocator.table_conflicts({1080: {16: 44.0}, 2520: {16: 44.5}}, 0.02)


def test_relocation_keeps_the_cluster_model() -> None:
    """A P-only relocation reuses park and D: their evidence comes along, so the
    published table still carries a model for the solver."""
    from canatune.domain.models import ClusterModel

    prompts = [128, 1024, 2048]
    first = asyncio.run(TierLocator(SurrogateBackend(seed=3), settings()).locate(PROMPT, prompts))
    backend = SurrogateBackend(seed=4)
    again = asyncio.run(TierLocator(backend, settings()).locate(PROMPT, prompts, previous=first))
    ev = again.evidence
    assert ev["idle_power_w"] == first.evidence["idle_power_w"]
    assert ev["decode"]["reused_from_previous"]
    assert ev["decode"]["kv_capacity_tokens"] == first.evidence["decode"]["kv_capacity_tokens"]
    model = ClusterModel.from_json(ev["model"])
    old = ClusterModel.from_json(first.evidence["model"])
    assert model.decode == old.decode and model.power_decode == old.power_decode
    assert model.power_prefill and model.b_star == old.b_star


def test_full_effort_reclaims_canary_only_after_probe_drain_and_locks_max():
    groups, tiers, router, controller, scheduler = build(SurrogateBackend(seed=6))
    import dataclasses

    router.settings = dataclasses.replace(router.settings, overload="best_effort")

    async def run():
        drained = asyncio.Event()
        entered = asyncio.Event()

        async def quiesce():
            entered.set()
            await drained.wait()

        scheduler.locator.backend.quiesce = quiesce
        await controller.start()
        controller.settings = dataclasses.replace(controller.settings, confirm_s=0)
        ticket = router.try_admit(PROMPT, True)
        ticket.admitted_at -= 3
        tick = asyncio.create_task(controller.tick())
        await entered.wait()
        assert groups[0].state is GroupState.EXPLORING
        assert all(g.tier is Tier.MAX and g.routable for g in groups if not g.canary)
        assert scheduler.may_start() == (False, "full_effort")
        drained.set()
        await tick
        assert all(g.routable and g.tier is Tier.MAX for g in groups)

    asyncio.run(run())


def test_failed_canary_drain_is_retried_without_releasing_group():
    import dataclasses

    groups, _, router, controller, scheduler = build(SurrogateBackend(seed=6))
    router.settings = dataclasses.replace(router.settings, overload="best_effort")
    attempts = []

    async def quiesce():
        attempts.append(1)
        if len(attempts) == 1:
            raise TimeoutError("still busy")

    scheduler.locator.backend.quiesce = quiesce

    async def run():
        await controller.start()
        controller.settings = dataclasses.replace(controller.settings, confirm_s=0)
        ticket = router.try_admit(PROMPT, True)
        ticket.admitted_at -= 3
        await controller.tick()
        assert router.full_effort and groups[0].state is GroupState.EXPLORING
        await controller.tick()
        assert len(attempts) == 2
        assert groups[0].routable and groups[0].tier is Tier.MAX

    asyncio.run(run())


def test_verify_uses_packed_peak_rate_and_failed_window_caps_the_point():
    from canatune.domain.models import Evaluation

    table = asyncio.run(
        TierLocator(SurrogateBackend(seed=5), settings()).locate(PROMPT, [512, 2048])
    )
    _, _, _, controller, scheduler = build(SurrogateBackend(seed=6), table, solver=True)
    controller.plan = Evaluation(2, table.h, True, 100, "decode", {}, (2.0, 0.5))
    calls = []
    prompt_limits = []
    scheduler.locator.backend.set_prompt_limit = lambda limit: prompt_limits.append(limit)

    async def failed(point, rate, alpha):
        calls.append(rate)
        window = await scheduler.locator.backend.open_window(point, 1, alpha, 1, 1)
        window.violations = window.requests
        return window

    scheduler.locator.verify = failed
    assert asyncio.run(scheduler._verify()) is False
    assert calls == [2.0] and prompt_limits == [None]
    assert table.evidence["model"]["caps"][table.h.key()] == 2.0
    assert not scheduler.verified
    assert not controller.solver().evaluate(2.0, [(PROMPT, 64)], 1, table.h).feasible


class LengthBackend(SurrogateBackend):
    """Surrogate D with a length model: TPOT = alpha(f) + beta X + gamma(f) K, a KV
    cache of 33648 tokens; probe prompts give 300 (short), 1000 (all) and 1700
    (long) context tokens per sequence."""

    KV = 33648
    CONTEXT = {"all": 1000.0, "short": 300.0, "long": 1700.0, "shortest": 150.0}

    async def kv_capacity(self):
        return self.KV

    def mix_context(self, mix="all"):
        return self.CONTEXT[mix]

    def shortest_prompt(self):
        return 128

    @staticmethod
    def tpot(f, x, k):
        return 55 * (1050 / f) ** 0.15 + 0.4 * x + 8e-4 * (1050 / f) ** 0.3 * k

    async def closed_window(self, clock, concurrency, seconds, rep=0, mix="all", **_):
        w = await super().closed_window(clock, concurrency, seconds, rep)
        f, context = clock.decode_mhz, self.CONTEXT[mix]
        full = concurrency * context > 0.95 * self.KV
        points = []
        for _ in range(max(6, 3 * concurrency)):
            x = concurrency * (1 + self.rng.uniform(-0.1, 0.1))
            k = x * context * (1 + self.rng.uniform(-0.2, 0.2))
            points.append((x, k, self.tpot(f, x, k) * (1 + self.rng.gauss(0, 0.01))))
        tpots = sorted(p[2] for p in points)
        return dataclasses.replace(
            w,
            tpot_p95_ms=tpots[int(0.95 * (len(tpots) - 1))] * (1.6 if full else 1.0),
            tpot_p50_ms=tpots[len(tpots) // 2],
            decode_preemptions=5.0 if full else 0.0,
            decode_waiting_max=3.0 if full else 0.0,
            decode_points=points,
        )


def test_length_model_is_fitted_at_the_top_and_the_chosen_d_clock() -> None:
    backend = LengthBackend(seed=3)
    table = asyncio.run(TierLocator(backend, settings()).locate(PROMPT, [128, 512, 1024, 2048]))
    d = table.evidence["decode"]
    assert d["decode_model"] == "length" and table.decode_kv_tokens == LengthBackend.KV
    f_d = next(iter(d["b_star_concurrency"]))  # the chosen D clock
    assert set(table.decode_length) == {"1500", f_d}
    alpha, beta, gamma, r95 = table.decode_length["1500"]
    assert alpha == pytest.approx(55 * (1050 / 1500) ** 0.15, rel=0.05)
    assert beta == pytest.approx(0.4, rel=0.25)
    assert gamma == pytest.approx(8e-4 * (1050 / 1500) ** 0.3, rel=0.15)
    # 200 ms TPOT: the KV wall binds at 1000 context tokens (0.9 x 33648 / 1000 = 30)
    assert table.decode_wall and table.decode_max_running == 30
    assert d["capacity"][f_d]["limit"] == "kv"


def test_tight_tpot_slo_makes_the_model_tpot_bound() -> None:
    backend = LengthBackend(seed=4)
    locator = TierLocator(backend, settings(tpot_slo_ms=80.0))
    table = asyncio.run(locator.locate(PROMPT, [128, 512, 1024, 2048]))
    d = table.evidence["decode"]
    f_d = next(iter(d["b_star_concurrency"]))
    assert not table.decode_wall and d["capacity"][f_d]["limit"] == "tpot"
    assert table.decode_max_running < 30


def test_infeasible_tpot_slo_is_reported_as_such() -> None:
    locator = TierLocator(LengthBackend(seed=5), settings(tpot_slo_ms=50.0))
    with pytest.raises(LocatorError, match="infeasible"):
        asyncio.run(locator.locate(PROMPT, [128, 512, 1024, 2048]))


def test_count_model_remains_selectable() -> None:
    backend = LengthBackend(seed=6)
    table = asyncio.run(
        TierLocator(backend, settings(decode_model="count")).locate(PROMPT, [128, 512, 1024])
    )
    assert table.decode_length is None and "decode_model" not in table.evidence["decode"]


def test_length_model_without_d_telemetry_falls_back_to_the_count_model() -> None:
    class NoTelemetry(LengthBackend):
        async def closed_window(self, clock, concurrency, seconds, rep=0, mix="all", **_):
            w = await super().closed_window(clock, concurrency, seconds, rep, mix)
            return dataclasses.replace(w, decode_points=[])

    table = asyncio.run(
        TierLocator(NoTelemetry(seed=7), settings()).locate(PROMPT, [128, 512, 1024, 2048])
    )
    assert table.decode_length is None and table.decode_max_running > 0


class NixlLengthBackend(LengthBackend):
    """NixlConnector on vLLM 0.15: requests pulling their KV from P sit in D's waiting
    queue, so num_requests_waiting > 0 in every window; KV use shows the real queue."""

    async def closed_window(self, clock, concurrency, seconds, rep=0, mix="all", **_):
        w = await super().closed_window(clock, concurrency, seconds, rep, mix)
        kv = min(1.0, concurrency * self.CONTEXT[mix] / self.KV)
        waiting = w.decode_waiting_max + 2.0
        return dataclasses.replace(w, decode_waiting_max=waiting, decode_kv_max=kv)


def test_nixl_kv_pulls_in_the_waiting_queue_do_not_make_d_unclean() -> None:
    nixl = settings(decode_waiting_includes_kv_pull=True)
    locator = TierLocator(SurrogateBackend(), nixl)
    window = WindowResult(
        ClockPoint(1410, 1410), "closed", 8, 60.0, requests=40, tpot_p95_ms=80.0,
        decode_waiting_max=3.0, decode_kv_max=0.4,
    )
    assert locator.decode_clean(window)
    assert not locator.decode_clean(dataclasses.replace(window, decode_kv_max=0.95))
    assert not locator.decode_clean(dataclasses.replace(window, decode_kv_max=None))
    assert not locator.decode_clean(dataclasses.replace(window, decode_preemptions=1.0))
    # P2pNccl (KV pushed before the request reaches D): any waiting request is a queue
    assert not TierLocator(SurrogateBackend(), settings()).decode_clean(window)

    table = asyncio.run(
        TierLocator(NixlLengthBackend(seed=3), nixl).locate(PROMPT, [128, 512, 1024, 2048])
    )
    assert table.evidence["decode"]["decode_model"] == "length"
    assert table.decode_wall and table.decode_max_running == 30
    # Without the setting the same D is never clean (job 10641935 on Vera)
    with pytest.raises(LocatorError, match="not clean"):
        asyncio.run(
            TierLocator(NixlLengthBackend(seed=3), settings()).locate(
                PROMPT, [128, 512, 1024, 2048]
            )
        )


def test_kv_pull_waiting_setting_follows_the_kv_connector() -> None:
    config = load_config()
    config.setdefault("kv_transfer", {})["connector"] = "NixlConnector"
    assert LocatorSettings.from_config(config).decode_waiting_includes_kv_pull
    config["kv_transfer"]["connector"] = "P2pNcclConnector"
    assert not LocatorSettings.from_config(config).decode_waiting_includes_kv_pull
    config.setdefault("canary", {}).setdefault("locator", {})[
        "decode_waiting_includes_kv_pull"
    ] = True
    assert LocatorSettings.from_config(config).decode_waiting_includes_kv_pull


def test_room_limits_measure_p_and_d_each_on_its_own():
    class RoomBackend:
        """P stops meeting the target above 4000 eq tokens/s; D holds 160 sequences."""

        def __init__(self):
            self.opens, self.fills = [], []

        def mix_context(self, mix="all"):
            return 256.0

        def shortest_prompt(self):
            return 128

        async def open_window(self, clock, load, alpha, seconds, abort, output_tokens=None,
                              prompt_lengths=None):
            self.opens.append((clock, output_tokens))
            bad = load > 4000
            return WindowResult(
                clock=clock, kind="open", load=load, duration_s=seconds, requests=100,
                violations=50 if bad else 0,
                samples=[ProbeSample(1500, 1000, 128, (), 0, 0, 100.0, bad, 0)
                         for _ in range(int(load / 228 * seconds))],
            )  # fmt: skip

        async def closed_window(self, clock, concurrency, seconds, **kw):
            self.fills.append((clock, concurrency, seconds, kw))
            return WindowResult(
                clock=clock, kind="closed", load=concurrency, duration_s=seconds,
                requests=concurrency,
                samples=[ProbeSample(1, 1, 128, (), 0, 0, 100.0, x >= 160, x)
                         for x in range(concurrency)],
            )  # fmt: skip

    backend = RoomBackend()
    locator = TierLocator(backend, settings())
    locator.run = LocatorRun(started_at=0.0)
    locator.run.evidence.update(
        prefill_ms_by_clock={"1500": {"128": 40.0, "2048": 200.0}},
        decode={"length_model": {"1000": [10.0, 0.05, 0.0, 1.0]}},
    )
    h = ClockPoint(1500, 1000)
    asyncio.run(locator.room_limits(h, 100.0, 1000.0, None))
    room = locator.run.evidence["room"]
    assert room["p_source"] == "measured" and room["p_rho"] > 0
    assert 3000 <= room["p_load"] <= 4000
    assert all(clock == h and output == 2 for clock, output in backend.opens)  # D stays idle
    assert (room["d_sequences"], room["d_source"]) == (160, "measured")
    clock, _, seconds, kw = backend.fills[0]
    assert clock == h and kw["mix"] == "shortest" and kw["stagger_s"] == seconds
    assert room["d_fill"]["rate_rps"] == pytest.approx(0.5 * room["p_rho"] / 0.04)
    # A relocation measures P again and keeps D's result.
    previous = TierTable(park=h, h=h, capacity_h=1.0, alpha_tokens=0.0, evidence={"room": room})
    opens = len(backend.opens)
    asyncio.run(locator.room_limits(h, 100.0, 1000.0, previous))
    assert len(backend.opens) > opens or locator.run.reused  # P searched (or served from cache)
    assert locator.run.evidence["room"]["d_sequences"] == 160 and len(backend.fills) == 1


def test_profile_calibration_searches_independent_limits_and_isolates_cached_windows():
    from canatune.domain.p_capacity import p_profiles

    profiles = p_profiles()
    caps = dict(zip(profiles, (1.5, .9, .85, .7, .65)))
    lut = {128: 20., 512: 40., 1024: 80., 2048: 160.}

    class Backend:
        def __init__(self):
            self.calls = []

        async def open_window(self, clock, load, alpha, seconds, abort, output_tokens=None,
                              prompt_lengths=None):
            self.calls.append((clock, output_tokens, tuple(prompt_lengths)))
            name = next(n for n, ls in profiles.items() if ls == tuple(prompt_lengths))
            rate = load / (sum(prompt_lengths) / len(prompt_lengths) + alpha)
            count = max(100, round(rate * seconds))
            lengths = [prompt_lengths[i % len(prompt_lengths)] for i in range(count)]
            rho = sum(lut[n] for n in lengths) / (1000 * seconds)
            bad = rho > caps[name]
            return WindowResult(
                clock, "open", load, seconds, requests=count,
                violations=count if bad else 0,
                samples=[ProbeSample(clock.prefill_mhz, clock.decode_mhz, n, (), 0, 0,
                                     2000 if bad else 100, bad) for n in lengths],
            )

    backend = Backend()
    locator = TierLocator(backend, settings())
    locator.run = LocatorRun(started_at=0, evidence={"prefill_ms_by_clock": {1500: lut}})
    costs = CostBook({1500: lut})
    h = ClockPoint(1500, 1000)
    result = asyncio.run(locator.room_p_profiles(h, 100, 5000, costs))
    assert set(result) == set(profiles)
    for name, row in result.items():
        assert row["p_source"] == "measured"
        assert row["prompt_lengths"] == list(profiles[name])
        assert .75 * caps[name] <= row["p_rho"] <= caps[name]
    assert all(clock == h and output == 2 for clock, output, _ in backend.calls)
    assert {ls for _, _, ls in backend.calls} == set(profiles.values())
    before = len(backend.calls)
    assert asyncio.run(locator.room_p_profiles(h, 100, 5000, costs)) == result
    assert len(backend.calls) == before  # own profile cache; never another distribution's window


def test_recheck_rebuilds_the_profile_bank_at_the_new_working_clock(monkeypatch):
    backend = SurrogateBackend(noise=0)
    monkeypatch.setattr(backend, "model", lambda f, r: (100, 0, 1 if f < 2040 else 10))
    locator = TierLocator(backend, settings())
    old_bank = {"short": {"prompt_lengths": [128], "p_rho": .8}}
    h = ClockPoint(2040, 1000)
    table = TierTable(h, h, 5000, 100, evidence={
        "prefill_ms_by_clock": {2040: {128: 20., 2048: 160.}},
        "room": {"p_profiles": old_bank, "d_sequences": 123},
    })
    calls = []

    async def rebuild(clock, alpha, capacity, costs):
        calls.append(clock)
        assert clock.prefill_mhz in locator.run.evidence["prefill_ms_by_clock"]
        return {"new": {"prompt_lengths": [128], "p_rho": .7}}

    monkeypatch.setattr(locator, "room_p_profiles", rebuild)
    moved = asyncio.run(locator.recheck(table, [128, 512, 1024, 2048]))
    assert moved.h.prefill_mhz < h.prefill_mhz and calls == [moved.h]
    assert moved.evidence["room"]["p_profiles"]["new"]["p_rho"] == .7
    assert moved.evidence["room"]["d_sequences"] == 123
    assert table.evidence["room"]["p_profiles"] == old_bank  # published old state is immutable


def test_room_p_limit_is_none_when_no_load_stops_meeting_the_target():
    class Easy:
        async def open_window(self, clock, load, alpha, seconds, abort, output_tokens=None):
            return WindowResult(clock=clock, kind="open", load=load, duration_s=seconds,
                                requests=100, violations=0)  # fmt: skip

    locator = TierLocator(Easy(), settings())
    locator.run = LocatorRun(started_at=0.0)
    costs = CostBook({"1500": {"128": 40.0, "2048": 200.0}})
    room = asyncio.run(locator.room_p_limit(ClockPoint(1500, 1000), 100.0, 1000.0, costs))
    assert room["p_rho"] is None and room["p_source"] == "none"
