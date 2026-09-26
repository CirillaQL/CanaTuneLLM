"""Canary tier locator: find park, H (and L), D clocks and capacities online.

Procedure (design v2 §6.2; validated offline by replaying K2/K3b/K4a data):

1. park     idle power at a few clocks, lowest power wins (noise -> lowest clock)
2. alpha    per-request fixed prefill cost in tokens: fit prefill time = a + b*L;
            prompt lengths whose TTFT misses the target even on an idle system are
            excluded from later probes (the Router rejects them; they would only
            make every load look infeasible)
3. ramp     at the highest clock, double the load (equivalent tokens/s) until TTFT
            p95 reaches the target, bisect twice -> C0; the clock actually reached
            under load is the search ceiling f_eff (power cap)
4. coarse   5 clocks from f_eff down at 0.8*C0; stop descending at the first
            infeasible clock (violation upper confidence bound > theta)
5. refine   golden section around the cheapest point, or bisection of the
            feasibility edge when the cheapest point sits next to an infeasible one
6. choose   H = highest clock whose energy is within (1+eps) of the minimum and
            that is not held down by power/thermal limits most of the time
7. tiers    repeat at 0.3*C0: publish L only if it saves more than 2*eps against
            H there (a second tier must pay for its clock switches)
8. decode   J/token over D clocks at a fixed concurrency; B* (clean concurrency)
            at the chosen and the highest D clock: equal -> KV wall
9. fill     windows at H over several loads so the risk table has samples before
            production switches to it; C_H = highest load meeting the target

Every window result is cached, so an aborted run resumes where it stopped.
The locator never touches production: the backend drives only the Canary pair.
"""

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from canatune.domain.groups import ClockPoint, TierTable
from canatune.infrastructure.records import JsonlLog


class LocatorError(RuntimeError):
    """The procedure cannot produce a usable tier table."""


# ---- measurement interface ------------------------------------------------------


@dataclass(frozen=True)
class Hardware:
    prefill_clocks: tuple[int, ...]  # supported SM clocks, ascending
    decode_clocks: tuple[int, ...]


@dataclass
class WindowResult:
    clock: ClockPoint
    kind: str  # "open" (Poisson, equivalent tokens/s) or "closed" (fixed concurrency)
    load: float
    duration_s: float
    requests: int = 0
    violations: int = 0
    ttft_p95_ms: float | None = None
    tpot_p95_ms: float | None = None
    prefill_j_per_request: float | None = None
    decode_j_per_token: float | None = None
    prefill_mhz_median: float | None = None
    prefill_limited_fraction: float = 0.0  # share of busy samples with power/thermal limits
    decode_preemptions: float = 0.0
    decode_waiting_max: float = 0.0
    decode_kv_max: float | None = None
    decode_running_max: float | None = None  # sequences D actually ran (telemetry)
    aborted: bool = False

    def summary(self) -> dict[str, Any]:
        return {
            "clock": self.clock.key(),
            "kind": self.kind,
            "load": round(self.load, 1),
            "requests": self.requests,
            "violations": self.violations,
            "ttft_p95_ms": self.ttft_p95_ms,
            "tpot_p95_ms": self.tpot_p95_ms,
            "prefill_j": self.prefill_j_per_request,
            "decode_j_tok": self.decode_j_per_token,
            "prefill_mhz": self.prefill_mhz_median,
            "limited": round(self.prefill_limited_fraction, 3),
            "preempt": self.decode_preemptions,
            "aborted": self.aborted,
        }


class ProbeBackend(Protocol):
    async def hardware(self) -> Hardware: ...

    async def idle_power(self, clock: ClockPoint, seconds: float) -> tuple[float, float]:
        """Mean (prefill W, decode W) with the Canary idle at `clock`."""
        ...

    async def service_times(
        self, clock: ClockPoint, prompts: Sequence[int]
    ) -> list[tuple[int, float, float | None]]:
        """(prompt tokens, prefill ms, TTFT ms) of sequential single requests."""
        ...

    def set_prompt_limit(self, max_prompt: int | None) -> float:
        """Probe only prompts <= max_prompt from now on; -> mean probe prompt length."""
        ...

    async def open_window(
        self, clock: ClockPoint, eq_tps: float, alpha: float, seconds: float, abort_above: float
    ) -> WindowResult:
        """Poisson probes at `eq_tps` equivalent tokens/s; may stop early once the
        observed violation rate is clearly above `abort_above`."""
        ...

    async def closed_window(
        self, clock: ClockPoint, concurrency: int, seconds: float
    ) -> WindowResult: ...


# ---- settings and helpers -----------------------------------------------------------


@dataclass(frozen=True)
class LocatorSettings:
    theta: float = 0.10
    eps: float = 0.02
    ttft_slo_ms: float = 500.0
    tpot_slo_ms: float = 200.0
    ttft_target_fraction: float = 0.8  # capacity: TTFT p95 <= 0.8 SLO
    target_load_fraction: float = 0.8  # search load = 0.8 C0
    low_load_fraction: float = 0.3
    l_tier_min_gain: float = 0.04  # add L only when it saves clearly more than noise (2 eps)
    fill_load_fractions: tuple[float, ...] = (0.3, 0.5, 0.8, 1.0)
    window_s: float = 20.0
    min_window_requests: int = 30  # below ~15 the violation bound cannot reach 10 %
    max_window_s: float = 60.0
    idle_window_s: float = 5.0
    park_candidates: int = 5
    coarse_points: int = 5
    refine_steps: int = 3
    limited_fraction_max: float = 0.3
    cap_gap: float = 0.05  # median busy clock this far below the lock -> power cap
    ucb_z: float = 1.28  # one-sided 90 %
    idle_noise_w: float = 0.5
    ramp_start_rps: float = 0.5
    ramp_max_steps: int = 10
    decode_probe_concurrency: int = 16
    decode_start_concurrency: int = 8
    decode_max_concurrency: int = 512
    decode_wall_tolerance: float = 0.10
    decode_kv_limit: float = 0.90
    service_repeats: int = 3
    cache_ttl_s: float = 3600.0

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "LocatorSettings":
        raw = dict(config.get("canary", {}).get("locator", {}))
        slo = config["experiment"]["slo"]
        raw.setdefault("theta", config.get("router", {}).get("theta", 0.1))
        raw["ttft_slo_ms"] = float(slo["ttft_ms"])
        raw["tpot_slo_ms"] = float(slo["tpot_ms"])
        if "fill_load_fractions" in raw:
            raw["fill_load_fractions"] = tuple(raw["fill_load_fractions"])
        known = set(cls.__dataclass_fields__)
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown canary.locator keys: {sorted(unknown)}")
        return cls(**raw)


def wilson_ucb(k: int, n: int, z: float) -> float:
    """One-sided Wilson upper confidence bound of a violation rate."""
    if n <= 0:
        return 1.0
    p = k / n
    den = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return min(1.0, (centre + half) / den)


def snap(grid: Sequence[int], mhz: float) -> int:
    return min(grid, key=lambda f: (abs(f - mhz), f))


def snap_down(grid: Sequence[int], mhz: float) -> int:
    below = [f for f in grid if f <= mhz]
    return max(below) if below else min(grid)


def spread(grid: Sequence[int], low: int, high: int, n: int) -> list[int]:
    """n clocks evenly spaced in [low, high], snapped to the grid, descending."""
    if n <= 1 or high <= low:
        return [snap(grid, high)]
    points = {snap(grid, low + (high - low) * i / (n - 1)) for i in range(n)}
    return sorted(points, reverse=True)


def fit_alpha(samples: Sequence[tuple[int, float]]) -> float:
    """prefill_ms = a + b * tokens on the per-length medians (one slow request,
    e.g. the first KV-connector handshake, cannot flip the slope); alpha = a / b."""
    by_length: dict[int, list[float]] = {}
    for tokens, ms in samples:
        by_length.setdefault(tokens, []).append(ms)
    if len(by_length) < 2:
        raise LocatorError("service-time fit needs at least two prompt lengths")
    points = [(t, sorted(v)[len(v) // 2]) for t, v in by_length.items()]
    n = len(points)
    mx = sum(t for t, _ in points) / n
    my = sum(ms for _, ms in points) / n
    sxx = sum((t - mx) ** 2 for t, _ in points)
    b = sum((t - mx) * (ms - my) for t, ms in points) / sxx
    a = my - b * mx
    if b <= 0:
        raise LocatorError(f"prefill time does not grow with prompt length: {sorted(points)}")
    return max(0.0, a / b)


@dataclass
class _Cached:
    result: WindowResult
    at: float


@dataclass
class LocatorRun:
    """Progress and evidence of one run, exposed through the control API."""

    started_at: float
    phase: str = "start"
    windows: int = 0
    reused: int = 0
    evidence: dict[str, Any] = field(default_factory=dict)


# ---- the locator ------------------------------------------------------------------------


class TierLocator:
    def __init__(
        self,
        backend: ProbeBackend,
        settings: LocatorSettings,
        *,
        log: JsonlLog | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.backend = backend
        self.s = settings
        self.log = log or JsonlLog(None)
        self._clock = clock
        self._cache: dict[tuple, _Cached] = {}
        self.run: LocatorRun | None = None
        self.mean_prompt = 512.0  # of the current probe length distribution

    # ---- cached measurements ------------------------------------------------------------

    def forget(self) -> None:
        """Drop cached windows (the workload changed; old windows are not comparable)."""
        self._cache.clear()

    def _phase(self, phase: str) -> None:
        assert self.run is not None
        self.run.phase = phase
        self.log.write({"event": "locator_phase", "phase": phase})

    async def _cached(self, key: tuple, measure: Callable[[], Any]) -> Any:
        assert self.run is not None
        hit = self._cache.get(key)
        if hit is not None and self._clock() - hit.at <= self.s.cache_ttl_s:
            self.run.reused += 1
            return hit.result
        result = await measure()
        self._cache[key] = _Cached(result, self._clock())
        self.run.windows += 1
        if isinstance(result, WindowResult):
            self.log.write({"event": "locator_window", **result.summary()})
        return result

    def open_seconds(self, eq_tps: float, alpha: float) -> float:
        """Long enough for `min_window_requests` at this load, within max_window_s."""
        rps = eq_tps / max(self.mean_prompt + alpha, 1.0)
        needed = self.s.min_window_requests / rps if rps > 0 else self.s.max_window_s
        return min(self.s.max_window_s, max(self.s.window_s, needed))

    async def open(self, clock: ClockPoint, eq_tps: float, alpha: float) -> WindowResult:
        key = ("open", clock, round(eq_tps, 1), round(alpha, 1))
        seconds = self.open_seconds(eq_tps, alpha)
        return await self._cached(
            key,
            lambda: self.backend.open_window(clock, eq_tps, alpha, seconds, 2 * self.s.theta),
        )

    async def closed(self, clock: ClockPoint, concurrency: int) -> WindowResult:
        key = ("closed", clock, concurrency)
        return await self._cached(
            key, lambda: self.backend.closed_window(clock, concurrency, self.s.window_s)
        )

    def feasible(self, w: WindowResult) -> bool:
        return w.requests > 0 and wilson_ucb(w.violations, w.requests, self.s.ucb_z) <= self.s.theta

    def meets_target(self, w: WindowResult) -> bool:
        target = self.s.ttft_target_fraction * self.s.ttft_slo_ms
        return (
            self.feasible(w)
            and not w.aborted
            and w.ttft_p95_ms is not None
            and w.ttft_p95_ms <= target
        )

    def decode_clean(self, w: WindowResult) -> bool:
        return (
            w.requests > 0
            and w.tpot_p95_ms is not None
            and w.tpot_p95_ms <= self.s.tpot_slo_ms
            and w.decode_preemptions == 0
            and w.decode_waiting_max == 0
        )

    # ---- steps ------------------------------------------------------------------------------

    async def park(self, hw: Hardware) -> ClockPoint:
        self._phase("park")
        p_points = spread(
            hw.prefill_clocks, hw.prefill_clocks[0], hw.prefill_clocks[-1], self.s.park_candidates
        )
        d_points = spread(
            hw.decode_clocks, hw.decode_clocks[0], hw.decode_clocks[-1], self.s.park_candidates
        )
        n = max(len(p_points), len(d_points))
        p_power: dict[int, float] = {}
        d_power: dict[int, float] = {}
        for i in range(n):  # P and D are separate GPUs: measure one pair per window
            point = ClockPoint(
                p_points[min(i, len(p_points) - 1)], d_points[min(i, len(d_points) - 1)]
            )
            p_w, d_w = await self._cached(
                ("idle", point),
                lambda point=point: self.backend.idle_power(point, self.s.idle_window_s),
            )
            p_power.setdefault(point.prefill_mhz, p_w)
            d_power.setdefault(point.decode_mhz, d_w)

        def lowest(power: dict[int, float]) -> int:
            best = min(power.values())
            return min(f for f, w in power.items() if w <= best + self.s.idle_noise_w)

        assert self.run is not None
        self.run.evidence["idle_power_w"] = {"prefill": p_power, "decode": d_power}
        return ClockPoint(lowest(p_power), lowest(d_power))

    async def alpha(self, top: ClockPoint, prompts: Sequence[int]) -> tuple[float, int | None]:
        """-> (alpha tokens, largest prompt that meets the TTFT target when idle;
        None when every tested length does)."""
        self._phase("alpha")
        lengths = sorted(set(prompts))
        samples = await self._cached(
            ("service", top, tuple(lengths)),
            lambda: self.backend.service_times(top, lengths * self.s.service_repeats),
        )
        self.log.write({"event": "locator_service", "clock": top.key(), "samples": samples})
        try:
            alpha = fit_alpha([(t, ms) for t, ms, _ in samples])
        except LocatorError:
            self._cache.pop(("service", top, tuple(lengths)), None)  # re-measure next time
            raise
        target = self.s.ttft_target_fraction * self.s.ttft_slo_ms
        idle_ttft: dict[int, float] = {}
        for tokens in lengths:
            values = sorted(ttft for t, _, ttft in samples if t == tokens and ttft is not None)
            if values:
                idle_ttft[tokens] = values[len(values) // 2]
        ok = [t for t in lengths if idle_ttft.get(t, float("inf")) <= target]
        if not ok:
            raise LocatorError("no prompt length meets the TTFT target even on an idle system")
        limit = None if len(ok) == len(lengths) else max(ok)
        assert self.run is not None
        self.run.evidence.update(
            {"alpha_tokens": alpha, "idle_ttft_ms": idle_ttft, "prompt_limit": limit}
        )
        return alpha, limit

    async def ramp(
        self, hw: Hardware, top: ClockPoint, alpha: float, mean_prompt: float
    ) -> tuple[float, int]:
        """-> (C0 in equivalent tokens/s, f_eff)."""
        self._phase("ramp")
        load = self.s.ramp_start_rps * (mean_prompt + alpha)
        good: WindowResult | None = None
        bad: WindowResult | None = None
        for _ in range(self.s.ramp_max_steps):
            w = await self.open(top, load, alpha)
            if self.meets_target(w):
                good, load = w, load * 2
            else:
                bad = w
                if good is not None:
                    break
                load /= 2  # even the start load is too much: go down
        if good is None:
            raise LocatorError("no load meets the TTFT target even at the highest clock")
        if bad is not None:
            lo, hi = good.load, bad.load
            for _ in range(2):
                w = await self.open(top, (lo + hi) / 2, alpha)
                if self.meets_target(w):
                    lo, good = w.load, w
                else:
                    hi = w.load
        f_eff = top.prefill_mhz
        busy = good.prefill_mhz_median
        if busy is not None and (
            busy < top.prefill_mhz * (1 - self.s.cap_gap)
            or good.prefill_limited_fraction > self.s.limited_fraction_max
        ):
            f_eff = snap_down(hw.prefill_clocks, busy)
        assert self.run is not None
        self.run.evidence.update({"capacity_c0": good.load, "f_eff": f_eff})
        return good.load, f_eff

    async def search_prefill(
        self, hw: Hardware, f_eff: int, decode_mhz: int, load: float, alpha: float, tag: str
    ) -> tuple[int, dict[int, WindowResult]]:
        """Coarse + refine at `load`; -> (chosen clock, all windows at this load)."""
        grid = [f for f in hw.prefill_clocks if f <= f_eff]
        step = grid[1] - grid[0] if len(grid) > 1 else 15
        meas: dict[int, WindowResult] = {}

        async def measure(f: int) -> WindowResult:
            f = snap(grid, f)
            if f not in meas:
                meas[f] = await self.open(ClockPoint(f, decode_mhz), load, alpha)
            return meas[f]

        self._phase(f"coarse_{tag}")
        for f in spread(grid, grid[0], f_eff, self.s.coarse_points):
            if not self.feasible(await measure(f)):
                break  # lower clocks only get slower

        def feasible_energy() -> dict[int, float]:
            return {
                f: w.prefill_j_per_request
                for f, w in meas.items()
                if self.feasible(w) and w.prefill_j_per_request is not None
            }

        energies = feasible_energy()
        if not energies:
            raise LocatorError(f"no feasible prefill clock at load {load:.0f}")
        best = min(energies, key=energies.get)
        xs = sorted(meas)
        i = xs.index(best)
        self._phase(f"refine_{tag}")
        if i > 0 and not self.feasible(meas[xs[i - 1]]):
            lo_f, hi_f = xs[i - 1], best  # optimum at the SLO edge: bisect it
            for _ in range(self.s.refine_steps + 2):
                if hi_f - lo_f <= 2 * step:
                    break
                mid = snap(grid, (lo_f + hi_f) / 2)
                w = await measure(mid)
                e = w.prefill_j_per_request
                if (
                    self.feasible(w)
                    and e is not None
                    and e <= energies[hi_f] * (1 + self.s.eps / 2)
                ):
                    hi_f = mid
                    energies = feasible_energy()
                else:
                    lo_f = mid
                    if self.feasible(w):
                        break  # feasible but costlier: the valley is above
        else:
            left, right = xs[max(i - 1, 0)], xs[min(i + 1, len(xs) - 1)]
            g = (math.sqrt(5) - 1) / 2
            for _ in range(self.s.refine_steps):
                if right - left <= 2 * step:
                    break
                c = snap(grid, right - g * (right - left))
                d = snap(grid, left + g * (right - left))
                wc, wd = await measure(c), await measure(d)
                ec, ed = wc.prefill_j_per_request, wd.prefill_j_per_request
                if not self.feasible(wc) or (
                    self.feasible(wd) and ed is not None and ec is not None and ed < ec
                ):
                    left = c
                else:
                    right = d
        energies = feasible_energy()
        return self.choose(meas, energies), meas

    def choose(self, meas: Mapping[int, WindowResult], energies: Mapping[int, float]) -> int:
        """Upper edge of the eps band, skipping clocks held down by limits."""
        emin = min(energies.values())
        band = [f for f, e in energies.items() if e <= (1 + self.s.eps) * emin]
        steady = [
            f for f in band if meas[f].prefill_limited_fraction <= self.s.limited_fraction_max
        ]
        return max(steady or band)

    async def decode(self, hw: Hardware, prefill_mhz: int) -> tuple[int, int, bool, dict]:
        """-> (decode clock, B*, KV wall?, evidence)."""
        self._phase("decode_clock")
        target = self.s.ttft_target_fraction * self.s.tpot_slo_ms
        per_clock: dict[int, WindowResult] = {}
        for f in spread(
            hw.decode_clocks, hw.decode_clocks[0], hw.decode_clocks[-1], self.s.coarse_points
        ):
            w = await self.closed(ClockPoint(prefill_mhz, f), self.s.decode_probe_concurrency)
            per_clock[f] = w
            if w.tpot_p95_ms is None or w.tpot_p95_ms > target:
                break
        ok = {
            f: w.decode_j_per_token
            for f, w in per_clock.items()
            if w.decode_j_per_token is not None
            and w.tpot_p95_ms is not None
            and w.tpot_p95_ms <= target
        }
        if not ok:
            raise LocatorError("no decode clock meets the TPOT target")
        emin = min(ok.values())
        f_d = max(f for f, e in ok.items() if e <= (1 + self.s.eps) * emin)

        async def b_star(f: int) -> tuple[int, WindowResult | None]:
            """Largest clean client concurrency and its window."""
            point = ClockPoint(prefill_mhz, f)
            c, good, bad = self.s.decode_start_concurrency, 0, None
            good_w: WindowResult | None = None
            while c <= self.s.decode_max_concurrency:
                w = await self.closed(point, c)
                if self.decode_clean(w):
                    good, good_w, c = c, w, c * 2
                else:
                    bad = c
                    break
            if bad is not None and good > 0:
                lo, hi = good, bad
                for _ in range(2):
                    mid = (lo + hi) // 2
                    if mid in (lo, hi):
                        break
                    w = await self.closed(point, mid)
                    if self.decode_clean(w):
                        lo, good_w = mid, w
                    else:
                        hi = mid
                good = lo
            return good, good_w

        self._phase("decode_wall")
        b_low, b_window = await b_star(f_d)
        if b_low <= 0:
            raise LocatorError("decode is not clean even at the start concurrency")
        # Wall check with one window: is the highest D clock clean clearly beyond B*?
        top_d = hw.decode_clocks[-1]
        beyond = math.ceil(b_low * (1 + self.s.decode_wall_tolerance)) + 1
        if f_d == top_d:
            wall, b_high = True, b_low
        else:
            clean = self.decode_clean(await self.closed(ClockPoint(prefill_mhz, top_d), beyond))
            wall, b_high = not clean, (beyond if clean else b_low)
        # Admission compares D running+waiting sequences, so publish what D actually
        # ran in the clean B* window (client concurrency also waits at P).
        running = None if b_window is None else b_window.decode_running_max
        b_published = int(running) if running else b_low
        evidence = {
            "decode_j_per_token": ok,
            "b_star_concurrency": {str(f_d): b_low, f"{top_d}_at_least": b_high},
            "b_star_running": b_published,
            "kv_wall": wall,
        }
        return f_d, b_published, wall, evidence

    async def fill(self, point: ClockPoint, c0: float, alpha: float) -> float:
        """Windows at H over several loads (risk-table samples); -> C_H."""
        self._phase("fill")
        capacity = 0.0
        for fraction in self.s.fill_load_fractions:
            w = await self.open(point, fraction * c0, alpha)
            if self.meets_target(w):
                capacity = max(capacity, w.load)
        if capacity <= 0:
            capacity = min(self.s.fill_load_fractions) * c0 * 0.5
        return capacity

    # ---- full and partial runs ---------------------------------------------------------

    async def locate(
        self, mean_prompt: float, prompts: Sequence[int], previous: TierTable | None = None
    ) -> TierTable:
        """Full calibration. With `previous`, park and D results are reused
        (a P-only relocation after a load or length shift)."""
        self.run = LocatorRun(started_at=self._clock())
        self.mean_prompt = mean_prompt
        hw = await self.backend.hardware()
        top = ClockPoint(hw.prefill_clocks[-1], hw.decode_clocks[-1])
        park = previous.park if previous is not None else await self.park(hw)
        self.backend.set_prompt_limit(None)
        alpha, limit = await self.alpha(top, prompts)
        mean_prompt = self.mean_prompt = self.backend.set_prompt_limit(limit)
        c0, f_eff = await self.ramp(hw, top, alpha, mean_prompt)
        f_h, meas_hi = await self.search_prefill(
            hw, f_eff, top.decode_mhz, self.s.target_load_fraction * c0, alpha, "target"
        )

        # One working tier or two: does the low-load band contain H?
        self._phase("tiers")
        low_load = self.s.low_load_fraction * c0
        low: dict[int, WindowResult] = {}
        for f in sorted({f for f in meas_hi if f <= f_h}, reverse=True)[:4]:
            low[f] = await self.open(ClockPoint(f, top.decode_mhz), low_load, alpha)
        low_e = {
            f: w.prefill_j_per_request
            for f, w in low.items()
            if self.feasible(w) and w.prefill_j_per_request is not None
        }
        f_l = None
        if low_e and f_h in low_e:
            if low_e[f_h] > (1 + self.s.l_tier_min_gain) * min(low_e.values()):
                f_l = self.choose(low, low_e)

        if previous is not None:
            f_d, b_star, wall = (
                previous.h.decode_mhz,
                previous.decode_max_running,
                previous.decode_wall,
            )
            d_evidence = {"reused_from_previous": True}
        else:
            f_d, b_star, wall, d_evidence = await self.decode(hw, f_h)
        h = ClockPoint(f_h, f_d)
        capacity = await self.fill(h, c0, alpha)

        assert self.run is not None
        self.run.evidence.update(
            {
                "f_h": f_h,
                "f_l": f_l,
                "target_load": self.s.target_load_fraction * c0,
                "low_load": low_load,
                "prefill_energy_target": {
                    str(f): w.prefill_j_per_request for f, w in sorted(meas_hi.items())
                },
                "prefill_energy_low": {str(f): e for f, e in sorted(low_e.items())},
                "decode": d_evidence,
                "windows": self.run.windows,
                "reused_windows": self.run.reused,
                "duration_s": self._clock() - self.run.started_at,
            }
        )
        table = TierTable(
            park=park,
            h=h,
            capacity_h=capacity,
            alpha_tokens=alpha,
            l=None if f_l is None else ClockPoint(f_l, f_d),
            tau_up=None if f_l is None else 0.65 * c0,
            tau_down=None if f_l is None else 0.45 * c0,
            decode_max_running=b_star,
            decode_kv_limit=self.s.decode_kv_limit,
            decode_wall=wall,
            published_at=time.time(),
            evidence=dict(self.run.evidence),
        )
        self._phase("done")
        return table

    async def recheck(self, table: TierTable, prompts: Sequence[int]) -> TierTable:
        """Periodic neighbour check of H at the target load: move H one step when a
        neighbour is feasible and cheaper beyond eps (hill climbing)."""
        self.run = LocatorRun(started_at=self._clock())
        self._phase("recheck")
        hw = await self.backend.hardware()
        grid = list(hw.prefill_clocks)
        i = grid.index(table.h.prefill_mhz) if table.h.prefill_mhz in grid else None
        if i is None:
            return table
        load = float(
            table.evidence.get("target_load") or self.s.target_load_fraction * table.capacity_h
        )
        stride = max(1, len(grid) // 50)
        meas: dict[int, WindowResult] = {}
        for j in (i - stride, i, i + stride):
            if 0 <= j < len(grid):
                meas[grid[j]] = await self.open(
                    ClockPoint(grid[j], table.h.decode_mhz), load, table.alpha_tokens
                )
        energies = {
            f: w.prefill_j_per_request
            for f, w in meas.items()
            if self.feasible(w) and w.prefill_j_per_request is not None
        }
        self._phase("done")
        if table.h.prefill_mhz not in energies:
            return table
        best = min(energies, key=energies.get)
        if best != table.h.prefill_mhz and energies[best] < energies[table.h.prefill_mhz] * (
            1 - self.s.eps
        ):
            moved = TierTable.from_json(table.to_json())
            moved.h = ClockPoint(best, table.h.decode_mhz)
            moved.published_at = time.time()
            moved.evidence = {**table.evidence, "recheck": {str(f): e for f, e in energies.items()}}
            return moved
        return table
