"""Tier locator and Canary scheduler against a surrogate cluster.

The surrogate is the one used for the offline replay: P energy/latency
calibrated on K2 (job 267571: s(f) = 39.5 + 73428/f ms per 1024-token request,
power cap at ~2040 MHz, busy and idle power as measured), M/D/1 queueing plus a
180 ms P/D overhead for TTFT; D behaves like K3b (job 267654): a KV wall at
~44 sequences that no clock moves, TPOT barely depends on the clock.
"""

import asyncio
import math
import random

import pytest

from canatune.config import load_config
from canatune.controller.canary import CanaryScheduler, SchedulerSettings
from canatune.controller.locator import (
    Hardware,
    LocatorError,
    LocatorSettings,
    TierLocator,
    WindowResult,
    fit_alpha,
    spread,
    wilson_ucb,
)
from canatune.controller.router import CanaTuneRouter, RouterSettings
from canatune.controller.tier_controller import ControllerSettings, TierController
from canatune.domain.groups import ClockPoint, GroupState, Tier, TierState
from canatune.domain.load import LengthStats
from canatune.domain.risk import RiskTable
from canatune.infrastructure.clocks import GpuRef, NullClockActuator
from canatune.service import build_groups, identity

PROMPT = 1024
ALPHA = 28.1 / 0.0615  # K2 single requests: prefill = 28 ms + 0.0615 ms/token


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

    async def open_window(self, clock, eq_tps, alpha, seconds, abort_above):
        self._tick()
        f = clock.prefill_mhz
        rps = eq_tps / (PROMPT + alpha)
        ttft, viol, energy = self.model(f, rps)
        n = max(1, int(rps * seconds))
        k = sum(self.rng.random() < viol for _ in range(n))
        aborted = k / n > abort_above
        return WindowResult(
            clock=clock,
            kind="open",
            load=eq_tps,
            duration_s=seconds,
            requests=n,
            violations=k,
            ttft_p95_ms=ttft * (1 + self.rng.gauss(0, 0.03)),
            prefill_j_per_request=energy * (1 + self.rng.gauss(0, self.noise)),
            prefill_mhz_median=float(self.eff(f)),
            prefill_limited_fraction=self.limited(f),
            aborted=aborted,
        )

    async def closed_window(self, clock, concurrency, seconds):
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
    assert locator.run.windows <= 42


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


def test_two_tiers_when_low_load_band_excludes_h() -> None:
    backend = SurrogateBackend(seed=3)
    backend.pi = lambda f: 40 + 120 * (f / 2520) ** 3  # idle power grows steeply
    table = asyncio.run(TierLocator(backend, settings()).locate(PROMPT, [128, 1024, 2048]))
    assert table.l is not None and table.l.prefill_mhz < table.h.prefill_mhz
    assert table.tau_down < table.tau_up


def build(backend, table=None):
    config = load_config()
    config["controller"]["stagger_s"] = 0.0
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
    assert len(calls) == 2 and table.alpha_tokens > 0


def test_decode_runs_with_p_at_the_ceiling_and_needs_25pct_for_a_step() -> None:
    class StepBackend(SurrogateBackend):
        def __init__(self, seed, top_wall):
            super().__init__(seed=seed)
            self.top_wall = top_wall
            self.closed_clocks = []

        async def closed_window(self, clock, concurrency, seconds):
            self.closed_clocks.append(clock)
            w = await super().closed_window(clock, concurrency, seconds)
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

    async def fill_fails(clock, eq_tps, alpha, seconds, abort_above):
        w = await original(clock, eq_tps, alpha, seconds, abort_above)
        if locator.run is not None and locator.run.phase == "fill":
            w.violations = w.requests
        return w

    backend.open_window = fill_fails
    with pytest.raises(LocatorError, match="not published"):
        asyncio.run(locator.locate(PROMPT, [128, 512, 1024]))


def test_alpha_lengths_include_the_longest_prompt() -> None:
    _, _, _, _, scheduler = build(SurrogateBackend(seed=12))
    scheduler.lengths = LengthStats([(128, 8)] * 50 + [(4000, 8)], min_samples=1)
    assert 4000 in scheduler._prompts()
