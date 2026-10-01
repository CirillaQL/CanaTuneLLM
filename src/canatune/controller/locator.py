"""Canary tier locator: find park, H (and L), D clocks and capacities online.

Procedure (design v2 §6.2):

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
7. tables   single-request S(L) at every coarse P clock (the solver's P choices)
8. decode   J/token over D clocks at a fixed concurrency, plus half that
            concurrency (D iteration time against running sequences per clock);
            B* (clean concurrency) at the chosen and the highest D clock
9. joint    P at H together with D at the chosen clock at 0.8*C0 (and L at
            0.3*C0): raise D a step until it holds; P and D are each measured with
            the other at its ceiling, and a slow D also delays the first token
10. fill    windows at H over several loads so the risk table has samples before
            production switches to it; C_H = highest load meeting the target,
            bisected between the last feasible and the first infeasible load
11. model   from every probe and window of the run: the TTFT predictor, the slack
            risk seed and the KV-in-flight gate (`domain.calibration`), and the
            cluster model for the solver (`domain.models`: S(L) per P clock, KV
            residence, D iteration per D clock, power per clock vs load, the
            SLO-limited P utilization at C_H, the D limits)
H is the working point at the search load; production picks its configuration
per load and length mix with the solver and the Canary verifies it (`verify`).

Every window result is cached, so an aborted run resumes where it stopped.
The locator never touches production: the backend drives only the Canary pair.
"""

import math
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from canatune.domain.admission import CostBook
from canatune.domain.calibration import calibrate_admission, fit_plateau_slope
from canatune.domain.groups import ClockPoint, TierTable
from canatune.domain.models import ClusterModel, fit_decode, fit_power, interpolate
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
    samples: list = field(default_factory=list, repr=False)  # ProbeSample per request
    # For the cluster model (power vs clock and load, D iteration vs running):
    prefill_avg_w: float | None = None
    decode_avg_w: float | None = None
    decode_busy_fraction: float | None = None  # share of samples with D running > 0
    decode_running_mean: float | None = None
    tpot_p50_ms: float | None = None

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
            "waiting_max": self.decode_waiting_max,
            "kv_max": self.decode_kv_max,
            "running_max": self.decode_running_max,
            "duration_s": round(self.duration_s, 2),
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
        self, clock: ClockPoint, concurrency: int, seconds: float, rep: int = 0
    ) -> WindowResult:
        """Fixed concurrency; `rep` selects another fixed prompt trace."""
        ...


# ---- settings and helpers -----------------------------------------------------------


@dataclass(frozen=True)
class LocatorSettings:
    theta: float = 0.10
    eps: float = 0.02
    ttft_slo_ms: float = 500.0
    tpot_slo_ms: float = 200.0
    target_load_fraction: float = 0.8  # search load = 0.8 C0
    fill_load_fractions: tuple[float, ...] = (0.3, 0.5, 0.8, 1.0)
    fill_bisect_steps: int = 2  # C_H between the fill loads
    window_s: float = 20.0
    min_window_requests: int = 60  # 0 violations -> bound ~3 %: tells 5 % from 10 %
    max_window_s: float = 120.0
    idle_window_s: float = 5.0
    park_candidates: int = 5
    coarse_points: int = 5
    refine_steps: int = 3
    limited_fraction_max: float = 0.3
    cap_gap: float = 0.05  # median busy clock this far below the lock -> power cap
    ucb_z: float = 1.28  # one-sided 90 %
    idle_noise_w: float = 0.5
    ramp_start_rps: float = 1.0
    ramp_max_steps: int = 10
    decode_probe_concurrency: int = 16
    decode_start_concurrency: int = 8
    decode_max_concurrency: int = 512
    decode_wall_tolerance: float = 0.25  # frequency step only if B* grows >= 25 %
    decode_load_fraction: float = 0.7  # choose the D clock at 0.7 B* (near full load)
    decode_kv_limit: float = 0.90
    decode_window_s: float = 60.0  # closed windows: several requests per worker
    decode_clock_repeats: int = 2  # D clock choice: median J/token of this many windows
    service_repeats: int = 3
    service_lengths: tuple[int, ...] = ()  # extra S(L) lengths; () = 16, 64, 256, ... up
    # to the longest probe length (the plateau and its knee need short lengths)
    cache_ttl_s: float = 3600.0

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "LocatorSettings":
        raw = dict(config.get("canary", {}).get("locator", {}))
        slo = config["experiment"]["slo"]
        raw.setdefault("theta", config.get("router", {}).get("theta", 0.1))
        raw["ttft_slo_ms"] = float(slo["ttft_ms"])
        raw["tpot_slo_ms"] = float(slo["tpot_ms"])
        for key in ("fill_load_fractions", "service_lengths"):
            if key in raw:
                raw[key] = tuple(raw[key])
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


def _medians(samples: Sequence[tuple[int, float]]) -> list[tuple[int, float]]:
    by_length: dict[int, list[float]] = {}
    for tokens, ms in samples:
        by_length.setdefault(tokens, []).append(ms)
    return sorted((t, sorted(v)[len(v) // 2]) for t, v in by_length.items())


def curvature(samples: Sequence[tuple[int, float]]) -> dict[str, float] | None:
    """Least-squares a + b*L + c*L^2 on the per-length medians; `quad_share` is the
    c*L^2 part of the prefill time at the longest length (how far from linear)."""
    points = _medians(samples)
    if len(points) < 3:
        return None
    # normal equations for [a, b, c]
    s = [sum(t**k for t, _ in points) for k in range(5)]
    r = [sum(ms * t**k for t, ms in points) for k in range(3)]
    m = [[s[0], s[1], s[2]], [s[1], s[2], s[3]], [s[2], s[3], s[4]]]

    def det(x):
        return (
            x[0][0] * (x[1][1] * x[2][2] - x[1][2] * x[2][1])
            - x[0][1] * (x[1][0] * x[2][2] - x[1][2] * x[2][0])
            + x[0][2] * (x[1][0] * x[2][1] - x[1][1] * x[2][0])
        )

    d = det(m)
    if d == 0:
        return None
    coef = []
    for i in range(3):
        mi = [row[:] for row in m]
        for j in range(3):
            mi[j][i] = r[j]
        coef.append(det(mi) / d)
    a, b, c = coef
    top = points[-1][0]
    total = a + b * top + c * top * top
    return {
        "a_ms": a,
        "b_ms": b,
        "c_ms": c,
        "quad_share": (c * top * top / total) if total > 0 else 0.0,
    }


def fit_alpha(samples: Sequence[tuple[int, float]]) -> float:
    """prefill_ms = a + b * tokens on the per-length medians (one slow request,
    e.g. the first KV-connector handshake, cannot flip the slope); alpha = a / b."""
    points = _medians(samples)
    if len(points) < 2:
        raise LocatorError("service-time fit needs at least two prompt lengths")
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
    samples: list = field(default_factory=list)  # ProbeSample of every window
    results: list = field(default_factory=list)  # every WindowResult of the run


@dataclass(frozen=True)
class AdmissionInputs:
    """What the admission calibration needs besides the windows (config + priors)."""

    kv_bytes_per_token: float
    kv_buffer_bytes: float
    predictor_prior: tuple[float, ...] | None = None  # None: plain least squares


def service_lengths(prompts: Sequence[int], extra: Sequence[int] = ()) -> list[int]:
    """Probe lengths plus a geometric ladder 16, 64, 256, ... below the longest."""
    top = max(prompts)
    ladder = list(extra)
    if not ladder:
        n = 16
        while n < top:
            ladder.append(n)
            n *= 4
    return sorted(set(prompts) | {n for n in ladder if n <= top})


# ---- the locator ------------------------------------------------------------------------


class TierLocator:
    def __init__(
        self,
        backend: ProbeBackend,
        settings: LocatorSettings,
        *,
        log: JsonlLog | None = None,
        clock: Callable[[], float] = time.monotonic,
        admission: AdmissionInputs | None = None,
    ) -> None:
        self.backend = backend
        self.s = settings
        self.admission = admission  # None: no admission calibration (KV geometry unknown)
        self.log = log or JsonlLog(None)
        self._clock = clock
        self._cache: dict[tuple, _Cached] = {}
        self.run: LocatorRun | None = None
        self._previous: TierTable | None = None  # the table a relocation builds on
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
            if isinstance(hit.result, WindowResult):
                self.run.samples.extend(hit.result.samples)
                self.run.results.append(hit.result)
            return hit.result
        result = await measure()
        self._cache[key] = _Cached(result, self._clock())
        self.run.windows += 1
        if isinstance(result, WindowResult):
            self.run.samples.extend(result.samples)
            self.run.results.append(result)
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

    async def closed(self, clock: ClockPoint, concurrency: int, rep: int = 0) -> WindowResult:
        key = ("closed", clock, concurrency, rep)
        return await self._cached(
            key,
            lambda: self.backend.closed_window(clock, concurrency, self.s.decode_window_s, rep=rep),
        )

    async def closed_median(self, clock: ClockPoint, concurrency: int) -> WindowResult:
        """decode_clock_repeats windows with different fixed traces: clean only if every
        one is clean, J/token their median (one window alone is noisy)."""
        ws = [
            await self.closed(clock, concurrency, rep) for rep in range(self.s.decode_clock_repeats)
        ]
        energies = sorted(w.decode_j_per_token for w in ws if w.decode_j_per_token is not None)
        return replace(
            ws[0],
            requests=sum(w.requests for w in ws),
            violations=sum(w.violations for w in ws),
            tpot_p95_ms=max((w.tpot_p95_ms for w in ws if w.tpot_p95_ms is not None), default=None),
            decode_j_per_token=statistics.median(energies) if energies else None,
            decode_preemptions=sum(w.decode_preemptions for w in ws),
            decode_waiting_max=max(w.decode_waiting_max for w in ws),
            decode_kv_max=max(
                (w.decode_kv_max for w in ws if w.decode_kv_max is not None), default=None
            ),
            decode_running_max=max(
                (w.decode_running_max for w in ws if w.decode_running_max is not None), default=None
            ),
        )

    def feasible(self, w: WindowResult) -> bool:
        return w.requests > 0 and wilson_ucb(w.violations, w.requests, self.s.ucb_z) <= self.s.theta

    def meets_target(self, w: WindowResult) -> bool:
        """Capacity uses the admission criterion itself (violation bound <= theta);
        TTFT p95 is recorded but not a separate, stricter target."""
        return self.feasible(w) and not w.aborted

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
        lengths = service_lengths(prompts, self.s.service_lengths)
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
        target = self.s.ttft_slo_ms
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
            {
                "alpha_tokens": alpha,
                "alpha_source": getattr(self.backend, "service_source", "unknown"),
                "alpha_fit": {
                    "prefill_ms_by_length": dict(_medians([(t, ms) for t, ms, _ in samples])),
                    "quadratic": curvature([(t, ms) for t, ms, _ in samples]),
                },
                "idle_ttft_ms": idle_ttft,
                "prompt_limit": limit,
                "prefill_ms_by_clock": {
                    top.prefill_mhz: dict(_medians([(t, ms) for t, ms, _ in samples]))
                },
                # KV transfer + D's first token on an idle pair, per prompt length
                "residence_ms_by_length": dict(
                    _medians([(t, ttft - ms) for t, ms, ttft in samples if ttft is not None])
                ),
            }
        )
        return alpha, limit

    async def prefill_table(
        self, clock: ClockPoint, prompts: Sequence[int], *, fresh: bool = False, idle: bool = False
    ) -> dict[int, float]:
        """Single-request S(L) at `clock` (the clocks production runs at). `fresh`
        re-measures; `idle` also refreshes the idle TTFT and residence (the top clock)."""
        lengths = service_lengths(prompts, self.s.service_lengths)
        key = ("service", clock, tuple(lengths))
        if fresh:
            self._cache.pop(key, None)
        samples = await self._cached(
            key, lambda: self.backend.service_times(clock, lengths * self.s.service_repeats)
        )
        table = dict(_medians([(t, ms) for t, ms, _ in samples]))
        assert self.run is not None
        ev = self.run.evidence
        ev.setdefault("prefill_ms_by_clock", {})[clock.prefill_mhz] = table
        if idle:
            timed = [(t, ttft) for t, _, ttft in samples if ttft is not None]
            ev["idle_ttft_ms"] = dict(_medians(timed)) or ev.get("idle_ttft_ms", {})
            ev["residence_ms_by_length"] = dict(
                _medians([(t, ttft - ms) for t, ms, ttft in samples if ttft is not None])
            ) or ev.get("residence_ms_by_length", {})
        return table

    @staticmethod
    def table_conflicts(tables: Mapping[int, Mapping[int, float]], eps: float) -> set[int]:
        """Clocks whose S(L) contradict each other: a higher P clock slower than a
        lower one (beyond eps) at some length."""
        out: set[int] = set()
        clocks = sorted(tables)
        for i, lo in enumerate(clocks):
            for hi in clocks[i + 1 :]:
                if any(
                    n in tables[lo] and ms > (1 + eps) * tables[lo][n]
                    for n, ms in tables[hi].items()
                ):
                    out |= {lo, hi}
        return out

    async def consistent_tables(self, top: ClockPoint, prompts: Sequence[int]) -> None:
        """A higher P clock is never slower for one request, so a contradiction is a
        disturbed measurement (warm-up, a connection being set up). The clocks
        involved are measured once more; what still contradicts takes the envelope of
        the lower clocks (a higher clock at least as fast)."""
        assert self.run is not None
        ev = self.run.evidence
        tables = ev.get("prefill_ms_by_clock", {})
        bad = self.table_conflicts(tables, self.s.eps)
        record: dict[str, Any] = {"remeasured": sorted(bad), "envelope": []}
        for f in sorted(bad):
            clock = top if f == top.prefill_mhz else ClockPoint(f, top.decode_mhz)
            await self.prefill_table(clock, prompts, fresh=True, idle=clock == top)
        left = self.table_conflicts(tables, self.s.eps)
        if left:
            clocks = sorted(tables)
            for i, f in enumerate(clocks):
                for lo in clocks[:i]:
                    for n in tables[f]:
                        if n in tables[lo]:
                            tables[f][n] = min(tables[f][n], tables[lo][n])
            record["envelope"] = sorted(left)
        ev["prefill_consistency"] = record
        if bad:
            self.log.write({"event": "locator_tables", **record})

    def calibrate(
        self,
        *,
        park: ClockPoint | None = None,
        capacity_window: WindowResult | None = None,
        b_star: int | None = None,
    ) -> None:
        """From every probe and window of this run: plateau + slope per clock, the
        admission parameters (evidence["admission"]) and the cluster model for the
        solver (evidence["model"])."""
        assert self.run is not None
        ev = self.run.evidence
        tables = ev.get("prefill_ms_by_clock", {})
        ev["prefill_fit"] = {f: fit_plateau_slope(t) for f, t in tables.items()}
        a = self.admission
        costs = CostBook(tables)
        result = None
        if a is not None:
            result = calibrate_admission(
                self.run.samples,
                costs,
                prior=a.predictor_prior,
                ttft_slo_ms=self.s.ttft_slo_ms,
                theta=self.s.theta,
                kv_bytes_per_token=a.kv_bytes_per_token,
                buffer_bytes=a.kv_buffer_bytes,
            )
        if result is not None:
            ev["admission"] = result
            self.log.write({"event": "locator_admission", **result})
        if park is None or capacity_window is None:
            return
        model = self.build_model(costs, park, capacity_window, b_star)
        if model is not None:
            ev["model"] = model.to_json()
            self.log.write({"event": "locator_model", **ev["model"]})

    def _utilization(self, w: WindowResult, costs: CostBook) -> float:
        """Share of the window P spent prefilling: sum S_f(L) / duration (single-request
        service times; batching makes the real busy share lower)."""
        cost = costs.for_clock(w.clock.prefill_mhz)
        return sum(cost(s.prompt_tokens) for s in w.samples) / 1000.0 / max(w.duration_s, 1e-6)

    def build_model(
        self, costs: CostBook, park: ClockPoint, capacity_window: WindowResult, b_star: int | None
    ) -> ClusterModel | None:
        assert self.run is not None
        ev = self.run.evidence
        idle = ev.get("idle_power_w") or {}
        idle_p = {int(f): float(w) for f, w in (idle.get("prefill") or {}).items()}
        idle_d = {int(f): float(w) for f, w in (idle.get("decode") or {}).items()}
        results = self.run.results
        p_windows = [
            (w.clock.prefill_mhz, w.prefill_avg_w, self._utilization(w, costs))
            for w in results
            if w.prefill_avg_w is not None and w.samples
        ]
        d_windows = [
            (w.clock.decode_mhz, w.decode_avg_w, w.decode_busy_fraction)
            for w in results
            if w.decode_avg_w is not None and w.decode_busy_fraction
        ]
        d_points: dict[int, list[tuple[float, float]]] = {}
        for w in results:
            if w.decode_running_mean and w.tpot_p50_ms and w.decode_running_mean >= 1:
                d_points.setdefault(w.clock.decode_mhz, []).append(
                    (w.decode_running_mean, w.tpot_p50_ms)
                )
        decode = fit_decode(d_points)
        power_p = fit_power(idle_p, p_windows)
        power_d = fit_power(idle_d, d_windows)
        previous = self._previous
        old = (previous.evidence.get("model") if previous is not None else None) or None
        if old is not None:
            # A relocation reuses D: its iteration and power fits come from the run
            # that measured D (this run has no D windows of its own).
            reused = ClusterModel.from_json(old)
            decode = decode or reused.decode
            power_d = power_d or reused.power_decode
        # SLO-limited P utilization: the highest P utilization among windows that met
        # the target (a measured lower bound; at C_H the limit may be D or KV, the
        # low-clock windows of the search push P itself further).
        rho = max(
            [self._utilization(w, costs) for w in results if w.samples and self.meets_target(w)]
            + [self._utilization(capacity_window, costs)]
        )
        if not decode or not power_p or not power_d or rho <= 0:
            self.log.write(
                {
                    "event": "locator_model_incomplete",
                    "decode": bool(decode),
                    "power_prefill": bool(power_p),
                    "power_decode": bool(power_d),
                    "rho": rho,
                }
            )
            return None
        adm = ev.get("admission") or {}
        gate = adm.get("kv_gate_fraction")
        d_ev = ev.get("decode") or {}
        b_by_clock = {int(f): int(c) for f, c in (d_ev.get("b_star_concurrency") or {}).items()}
        park_w = (interpolate(idle_p, park.prefill_mhz) or 0.0) + (
            interpolate(idle_d, park.decode_mhz) or 0.0
        )
        return ClusterModel(
            prefill={
                int(f): {int(k): float(v) for k, v in t.items()}
                for f, t in ev.get("prefill_ms_by_clock", {}).items()
            },
            residence={
                int(k): float(v) for k, v in (ev.get("residence_ms_by_length") or {}).items()
            },
            decode=decode,
            power_prefill=power_p,
            power_decode=power_d,
            park_power_w=park_w,
            rho_prefill=rho,
            ttft_slo_ms=self.s.ttft_slo_ms,
            tpot_slo_ms=self.s.tpot_slo_ms,
            kv_bytes_per_token=0.0 if self.admission is None else self.admission.kv_bytes_per_token,
            kv_gate_bytes=None
            if gate is None or self.admission is None
            else gate * self.admission.kv_buffer_bytes,
            kv_capacity_tokens=d_ev.get("kv_capacity_tokens"),
            b_star=b_by_clock or ({} if b_star is None else {park.decode_mhz: b_star}),
            prompt_limit=ev.get("prompt_limit"),
            predictor_coef=adm.get("predictor_coef"),
            slack_counts=adm.get("slack_counts"),
            theta=self.s.theta,
        )

    async def verify(self, point: ClockPoint, rate_rps: float, alpha: float) -> WindowResult:
        """Layer 3: one window at a configuration the solver chose, at the per-group
        rate with the current length mix (the probe samples production's lengths)."""
        self.run = LocatorRun(started_at=self._clock())
        self._phase("verify")
        load = rate_rps * (self.mean_prompt + alpha)
        w = await self.open(point, load, alpha)
        self._phase("done")
        return w

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
            raise LocatorError("no load meets the SLO even at the highest clock")
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
        """-> (decode clock, B*, KV wall?, evidence).

        P runs at its ceiling so it does not limit D concurrency. B* is searched at
        the highest D clock first (the most D can take), the D clock is chosen by
        J/token at 0.7 B* (near full load, where it matters), and B* is re-checked
        at the chosen clock: a >= 25 % larger B* at the top clock means a frequency
        step, otherwise a KV wall.
        """
        top_d = hw.decode_clocks[-1]

        async def b_star(f: int, hint: int | None = None) -> tuple[int, WindowResult | None]:
            """Largest clean client concurrency at D clock f and its window."""
            point = ClockPoint(prefill_mhz, f)
            good, good_w, bad = 0, None, None
            if hint is not None:  # usually B*(f) <= B*(top): test the hint first
                w = await self.closed(point, hint)
                if self.decode_clean(w):
                    return hint, w
                bad = hint
                c = max(self.s.decode_start_concurrency, hint // 2)
            else:
                c = self.s.decode_start_concurrency
            while c <= self.s.decode_max_concurrency and (bad is None or c < bad):
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
        b_top, _ = await b_star(top_d)
        if b_top <= 0:
            raise LocatorError("decode is not clean even at the start concurrency")

        self._phase("decode_clock")
        load = max(1, round(self.s.decode_load_fraction * b_top))
        per_clock: dict[int, WindowResult] = {}
        for f in spread(hw.decode_clocks, hw.decode_clocks[0], top_d, self.s.coarse_points):
            w = await self.closed_median(ClockPoint(prefill_mhz, f), load)
            per_clock[f] = w
            if not self.decode_clean(w):
                break  # lower D clocks only get slower
            if load >= 2:  # a second load level: D iteration time vs running sequences
                await self.closed(ClockPoint(prefill_mhz, f), max(1, load // 2))
        ok = {
            f: w.decode_j_per_token
            for f, w in per_clock.items()
            if w.decode_j_per_token is not None and self.decode_clean(w)
        }
        if not ok:
            raise LocatorError("no decode clock is clean at 0.7 B*")
        emin = min(ok.values())
        f_d = max(f for f, e in ok.items() if e <= (1 + self.s.eps) * emin)

        self._phase("decode_verify")
        if f_d == top_d:
            b_low, b_window = await b_star(top_d)  # cached
        else:
            b_low, b_window = await b_star(f_d, hint=b_top)
        if b_low <= 0:
            raise LocatorError("decode is not clean at the chosen clock")
        if f_d == top_d:
            # Only the top clock may be clean at 0.7 B*: lower clocks lose capacity,
            # i.e. a frequency step; with every tested clock clean it is a KV wall.
            wall = all(self.decode_clean(w) for w in per_clock.values())
        else:
            wall = b_top < b_low * (1 + self.s.decode_wall_tolerance)
        # Admission compares D running+waiting sequences, so publish what D actually
        # ran in the clean B* window (client concurrency also waits at P).
        running = None if b_window is None else b_window.decode_running_max
        b_published = int(running) if running else b_low
        # K* candidate (KV tokens D held in the clean B* window); recorded, not used yet.
        capacity = None
        if hasattr(self.backend, "kv_capacity"):
            capacity = await self.backend.kv_capacity()
        kv_peak = None if b_window is None else b_window.decode_kv_max
        evidence = {
            "decode_load_concurrency": load,
            "decode_j_per_token": ok,
            "b_star_concurrency": {str(f_d): b_low, str(top_d): b_top},
            "b_star_running": b_published,
            "kv_wall": wall,
            "kv_capacity_tokens": capacity,
            "kv_peak_clean": kv_peak,
            "k_star_tokens": None if capacity is None or kv_peak is None else kv_peak * capacity,
            "decode_window_s": self.s.decode_window_s,
            "decode_clock_repeats": self.s.decode_clock_repeats,
        }
        return f_d, b_published, wall, evidence

    async def joint(
        self, hw: Hardware, prefill_mhz: int, decode_mhz: int, load: float, alpha: float
    ) -> tuple[int, dict[str, Any]]:
        """-> (lowest D clock from `decode_mhz` up that holds with P at `prefill_mhz`
        under `load`, evidence). The top D clock is where P was searched, so its
        window is usually cached."""
        top_d = hw.decode_clocks[-1]
        ladder = sorted(set(spread(hw.decode_clocks, decode_mhz, top_d, self.s.coarse_points - 1)))
        tried: dict[str, Any] = {}
        for f in ladder:
            w = await self.open(ClockPoint(prefill_mhz, f), load, alpha)
            tried[str(f)] = {
                "requests": w.requests,
                "violations": w.violations,
                "ttft_p95_ms": w.ttft_p95_ms,
            }
            if self.meets_target(w):
                return f, tried
        raise LocatorError(
            f"P {prefill_mhz} MHz misses the SLO at load {load:.0f} with every D clock"
        )

    async def fill(self, point: ClockPoint, c0: float, alpha: float) -> tuple[float, WindowResult]:
        """Windows at H over several loads (risk-table samples); -> (C_H, its window)."""
        self._phase("fill")
        good: list[WindowResult] = []
        bad: list[float] = []
        for fraction in self.s.fill_load_fractions:
            w = await self.open(point, fraction * c0, alpha)
            if self.meets_target(w):
                good.append(w)
            else:
                bad.append(w.load)
        if not good:
            raise LocatorError("no fill load is feasible at H: table not published")
        best = max(good, key=lambda w: w.load)
        above = [load for load in bad if load > best.load]
        if above:
            hi = min(above)
            for _ in range(self.s.fill_bisect_steps):
                w = await self.open(point, (best.load + hi) / 2, alpha)
                if self.meets_target(w):
                    best = w
                else:
                    hi = w.load
        return best.load, best

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
        self._previous = previous
        if previous is not None:
            # Reused park point: its idle power readings (the power model's base) too.
            park = previous.park
            idle = previous.evidence.get("idle_power_w")
            if idle:
                self.run.evidence["idle_power_w"] = idle
        else:
            park = await self.park(hw)
        self.backend.set_prompt_limit(None)
        alpha, limit = await self.alpha(top, prompts)
        mean_prompt = self.mean_prompt = self.backend.set_prompt_limit(limit)
        c0, f_eff = await self.ramp(hw, top, alpha, mean_prompt)
        f_h, meas_hi = await self.search_prefill(
            hw, f_eff, top.decode_mhz, self.s.target_load_fraction * c0, alpha, "target"
        )
        self._phase("tables")  # S(L) at every P clock the solver may choose
        grid = [f for f in hw.prefill_clocks if f <= f_eff]
        for f in sorted(set(spread(grid, grid[0], f_eff, self.s.coarse_points)) | {f_h}):
            if f != top.prefill_mhz:
                await self.prefill_table(ClockPoint(f, top.decode_mhz), prompts)
        await self.consistent_tables(top, prompts)

        if previous is not None:
            f_d, b_star, wall = (
                previous.h.decode_mhz,
                previous.decode_max_running,
                previous.decode_wall,
            )
            d_evidence = {**(previous.evidence.get("decode") or {}), "reused_from_previous": True}
        else:
            f_d, b_star, wall, d_evidence = await self.decode(hw, f_eff)

        self._phase("joint")
        target_load = self.s.target_load_fraction * c0
        f_d_alone = f_d
        f_d, joint_h = await self.joint(hw, f_h, f_d, target_load, alpha)
        joint_evidence: dict[str, Any] = {"h": joint_h, "decode_alone": f_d_alone}
        h = ClockPoint(f_h, f_d)
        capacity, capacity_window = await self.fill(h, c0, alpha)

        assert self.run is not None
        self.run.evidence.update(
            {
                "f_h": f_h,
                "target_load": target_load,
                "capacity_rps": capacity / max(self.mean_prompt + alpha, 1.0),
                "prefill_energy_target": {
                    str(f): w.prefill_j_per_request for f, w in sorted(meas_hi.items())
                },
                "decode": d_evidence,
                "joint": joint_evidence,
                "capacity_h_fraction": capacity / c0,
                "windows": self.run.windows,
                "reused_windows": self.run.reused,
                "duration_s": self._clock() - self.run.started_at,
            }
        )
        self._phase("model")  # after the evidence above: the model reads the D results
        self.calibrate(park=park, capacity_window=capacity_window, b_star=b_star)
        table = TierTable(
            park=park,
            h=h,
            capacity_h=capacity,
            alpha_tokens=alpha,
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
