"""HALO-style online bandit: a comparison system without the Canary.

Follows HALO (Esfandiari et al., "Safety-Constrained Contextual Bandit for Dynamic Power
Management", SuRE 2026), its inference controller, adapted to P/D disaggregation:

* One controller per phase: prefill uses TTFT against the TTFT SLO, decode TPOT against
  the TPOT SLO. All groups of a phase share its clock (HALO controls one GPU per process).
* Windows of `window_s` (30 s). Each window is classified by the p95 latency of the
  phase over the SLO: GREEN below 0.40 (downshifts only), AMBER up to 0.95 (all
  actions), RED above it (upshifts only). On the first entry into RED a deterministic
  emergency upshift of `emergency_rungs` rungs replaces the bandit's choice.
* Actions: {-2, -1, 0, +1, +2} rungs on a `step_mhz` (150 MHz) ladder from the phase's
  MAX down to its configured floor (clock_control.min_mhz).
* LinUCB, one ridge model per action (lambda = 1), context
  [1, mean latency / SLO (clipped at 3), SM utilisation / 100, power / power limit,
  clock / MAX], exploration alpha(t) = max(alpha_min, alpha0 / sqrt(1 + t / H)) with
  alpha0 1.0, alpha_min 0.1, H 40 windows.
* Cost: Energy-squared-delay, (E / 1000)^2 * (mean latency / 1000), times
  1 + lambda ((p95 / L0)^gamma - 1) once p95 exceeds L0 = 0.65 SLO (lambda 1, gamma 2);
  reward = -cost / beta, beta the first window's cost (a constant scale).

Deviations from HALO, all in its favour or forced by P/D: the latency signal comes from
every production request finished in the window instead of a separate probe stream (more
samples, no probe load), energy from the NVML energy counters of the phase's GPUs
instead of 50 ms power samples, and the routing is the baseline's round robin (HALO
does not route); `halo.routing: watermark` uses the feedback heuristic's watermarks.
"""

import asyncio
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from canatune.controller.feedback import FeedbackRouter, ladder, lock_all
from canatune.controller.tier_controller import TierController
from canatune.domain.groups import ClockPoint, Group
from canatune.infrastructure.records import JsonlLog

ACTIONS = (-2, -1, 0, 1, 2)  # rungs; positive = a higher clock


class HaloConfigError(ValueError):
    """HALO comparison settings are invalid."""


@dataclass(frozen=True)
class HaloSettings:
    enabled: bool = False
    window_s: float = 30.0
    step_mhz: int = 150
    emergency_rungs: int = 4
    green: float = 0.40
    red: float = 0.95
    penalty_start: float = 0.65
    penalty_lambda: float = 1.0
    penalty_gamma: float = 2.0
    alpha0: float = 1.0
    alpha_min: float = 0.1
    horizon: float = 40.0
    ridge: float = 1.0
    routing: str = "round_robin"
    prefill_min_mhz: int = 600
    decode_min_mhz: int = 300

    def __post_init__(self) -> None:
        if (not (math.isfinite(self.window_s) and self.window_s > 0) or self.step_mhz < 1
                or self.emergency_rungs < 1 or not 0 < self.green < self.red
                or not 0 < self.penalty_start or self.alpha_min < 0 or self.alpha0 < 0
                or not self.horizon > 0 or not self.ridge > 0
                or self.routing not in ("round_robin", "watermark")):
            raise HaloConfigError("invalid halo settings")

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "HaloSettings":
        raw = config.get("halo") or {}
        floors = (config.get("clock_control") or {}).get("min_mhz") or {}
        defaults = cls()
        kwargs: dict[str, Any] = {}
        for name in defaults.__dataclass_fields__:
            if name in raw:
                kind = type(getattr(defaults, name))
                kwargs[name] = kind(raw[name])
        kwargs.setdefault("prefill_min_mhz", int(floors.get("prefill", 600)))
        kwargs.setdefault("decode_min_mhz", int(floors.get("decode", 300)))
        return cls(**kwargs)


def solve(a: list[list[float]], b: list[float]) -> list[float]:
    """x with a x = b (Gaussian elimination with partial pivoting; a is small, SPD)."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        m[col], m[pivot] = m[pivot], m[col]
        for r in range(n):
            if r != col and m[r][col] != 0:
                f = m[r][col] / m[col][col]
                m[r] = [x - f * y for x, y in zip(m[r], m[col])]
    return [m[i][n] / m[i][i] for i in range(n)]


@dataclass
class LinUcbArm:
    a: list[list[float]]
    b: list[float]

    @classmethod
    def new(cls, d: int, ridge: float) -> "LinUcbArm":
        return cls([[ridge if i == j else 0.0 for j in range(d)] for i in range(d)], [0.0] * d)

    def update(self, x: Sequence[float], reward: float) -> None:
        for i, xi in enumerate(x):
            self.b[i] += xi * reward
            for j, xj in enumerate(x):
                self.a[i][j] += xi * xj

    def ucb(self, x: Sequence[float], alpha: float) -> float:
        theta = solve(self.a, self.b)
        ainv_x = solve(self.a, list(x))
        width = math.sqrt(max(0.0, sum(xi * yi for xi, yi in zip(x, ainv_x))))
        return sum(t * xi for t, xi in zip(theta, x)) + alpha * width


def p95(values: Sequence[float]) -> float:
    v = sorted(values)
    if not v:
        return 0.0
    pos = 0.95 * (len(v) - 1)
    lo = int(pos)
    return v[lo] + (pos - lo) * (v[min(lo + 1, len(v) - 1)] - v[lo])


@dataclass
class PhaseBandit:
    phase: str
    slo_ms: float
    settings: HaloSettings
    ladder: list[int] = field(default_factory=list)  # MAX first
    level: int = 0  # index into the ladder
    arms: dict[int, LinUcbArm] = field(default_factory=dict)
    t: int = 0
    beta: float | None = None
    zone: str = "green"
    last: tuple[list[float], int] | None = None  # (context, action) of the running window

    def __post_init__(self) -> None:
        self.arms = {a: LinUcbArm.new(5, self.settings.ridge) for a in ACTIONS}

    @property
    def mhz(self) -> int:
        return self.ladder[self.level]

    def move(self, rungs: int) -> None:
        """Positive rungs raise the clock (towards index 0)."""
        self.level = min(max(self.level - rungs, 0), len(self.ladder) - 1)

    def classify(self, p95_ms: float) -> str:
        ratio = p95_ms / self.slo_ms
        s = self.settings
        return "green" if ratio < s.green else ("red" if ratio > s.red else "amber")

    def cost(self, energy_j: float, mean_ms: float, p95_ms: float) -> float:
        s = self.settings
        base = (energy_j / 1000.0) ** 2 * (mean_ms / 1000.0)
        start = s.penalty_start * self.slo_ms
        if p95_ms > start:
            base *= 1 + s.penalty_lambda * ((p95_ms / start) ** s.penalty_gamma - 1)
        return base

    def alpha(self) -> float:
        s = self.settings
        return max(s.alpha_min, s.alpha0 / math.sqrt(1 + self.t / s.horizon))

    def step(self, latencies: Sequence[float], energy_j: float, util: float, power_ratio: float,
             top: int) -> dict[str, Any]:
        """One window's end: score the running action, then choose the next one."""
        mean = sum(latencies) / len(latencies) if latencies else 0.0
        tail = p95(latencies)
        cost = self.cost(energy_j, mean, tail)
        if self.beta is None and cost > 0:
            self.beta = cost
        reward = -cost / (self.beta or 1.0)
        # A window without finished requests carries no latency evidence (HALO's probes
        # always do): its zero cost would read as the best reward, so it trains nothing.
        if self.last is not None and latencies:
            x_prev, a_prev = self.last
            self.arms[a_prev].update(x_prev, reward)
        previous, self.zone = self.zone, self.classify(tail)
        x = [1.0, min(mean / self.slo_ms, 3.0), util / 100.0, power_ratio, self.mhz / top]
        self.t += 1
        if self.zone == "red" and previous != "red":
            action, how = self.settings.emergency_rungs, "emergency"
            self.last = None  # not a bandit decision
        else:
            allowed = {"green": (-2, -1), "amber": ACTIONS, "red": (1, 2)}[self.zone]
            alpha = self.alpha()
            action = max(allowed, key=lambda a: (self.arms[a].ucb(x, alpha), -abs(a)))
            how = "bandit"
            self.last = (x, action)
        before = self.mhz
        self.move(action)
        return {"phase": self.phase, "zone": self.zone, "samples": len(latencies),
                "mean_ms": round(mean, 1), "p95_ms": round(tail, 1),
                "energy_j": round(energy_j, 1), "cost": cost, "reward": reward,
                "action": action, "how": how, "from_mhz": before, "to_mhz": self.mhz}


class HaloController:
    def __init__(
        self,
        groups: Sequence[Group],
        router: FeedbackRouter,
        controller: TierController,
        settings: HaloSettings,
        *,
        log: JsonlLog | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.groups = list(groups)
        self.router = router
        self.controller = controller
        self.settings = settings
        self.log = log or JsonlLog(None)
        self._clock = clock
        rs = router.settings
        self.bandits = {
            "prefill": PhaseBandit("prefill", rs.ttft_slo_ms, settings),
            "decode": PhaseBandit("decode", rs.tpot_slo_ms, settings),
        }
        self._applied: ClockPoint | None = None
        self._energy: dict[str, float] = {}  # endpoint -> last energy counter (mJ)
        self._samples: dict[str, list[tuple[float, float]]] = {"prefill": [], "decode": []}

    def endpoints(self, phase: str) -> list[str]:
        return [g.prefill if phase == "prefill" else g.decode for g in self.groups]

    async def setup(self) -> None:
        top = self.controller.tiers.max_point
        refs = self.controller.gpu_refs
        g = self.groups[0]
        for phase, endpoint, floor, mhz in (
                ("prefill", g.prefill, self.settings.prefill_min_mhz, top.prefill_mhz),
                ("decode", g.decode, self.settings.decode_min_mhz, top.decode_mhz)):
            supported: list[int] = []
            if endpoint in refs:
                try:
                    supported = await self.controller.actuator.supported_clocks(refs[endpoint])
                except Exception as error:
                    self.log.write({"event": "halo_clock_error", "phase": phase,
                                    "error": repr(error)})
            steps = max(1, round((mhz - floor) / self.settings.step_mhz))
            self.bandits[phase].ladder = ladder(supported or [mhz], floor, mhz, steps)
        self.log.write({"event": "halo_setup", "settings": self.settings.__dict__,
                        "ladders": {p: b.ladder for p, b in self.bandits.items()}})
        await self.apply("start")
        await self.sample()  # energy counters at the start of the first window

    def point(self) -> ClockPoint:
        return ClockPoint(self.bandits["prefill"].mhz, self.bandits["decode"].mhz)

    async def apply(self, reason: str) -> None:
        target = self.point()
        if target == self._applied:
            return
        ok = await lock_all(self.groups, self.router, self.controller, target)
        self._applied = target
        self.log.write({"event": "halo_clock", "clock": target.key(), "reason": reason, "ok": ok})

    async def sample(self) -> dict[str, float]:
        """Read every agent once: utilisation and power ratio samples per phase (kept for
        the window), -> energy in J per phase since the previous sample."""
        refs = self.controller.gpu_refs
        by_gpu = {(ref.agent_url, ref.gpu): name for name, ref in refs.items()}
        phase_of = {e: p for p in ("prefill", "decode") for e in self.endpoints(p)}
        energy = {"prefill": 0.0, "decode": 0.0}
        for agent in sorted({ref.agent_url for ref in refs.values()}):
            try:
                readings = await self.controller.actuator.readings(agent)
            except Exception as error:
                self.log.write({"event": "halo_reading_error", "agent": agent,
                                "error": repr(error)})
                continue
            for reading in readings:
                endpoint = by_gpu.get((agent, reading.get("index")))
                phase = phase_of.get(endpoint or "")
                if phase is None:
                    continue
                counter = reading.get("energy_mj")
                if counter is not None:
                    old = self._energy.get(endpoint)
                    if old is not None and counter >= old:
                        energy[phase] += (counter - old) / 1000.0
                    self._energy[endpoint] = float(counter)
                limit = reading.get("power_limit_w") or 0.0
                util = reading.get("util_pct")
                power = reading.get("power_w")
                if util is not None and power is not None and limit > 0:
                    self._samples[phase].append((float(util), float(power) / limit))
        return energy

    def latencies(self, now: float) -> dict[str, list[float]]:
        outcomes = self.router.finished_outcomes
        while outcomes and outcomes[0][0] < now - self.settings.window_s:
            outcomes.popleft()
        return {"prefill": [o[3] for o in outcomes if o[3] is not None],
                "decode": [o[4] for o in outcomes if o[4] is not None]}

    async def window(self, energy: Mapping[str, float]) -> None:
        now = self._clock()
        lat = self.latencies(now)
        top = self.controller.tiers.max_point
        decisions = []
        for phase, bandit in self.bandits.items():
            samples = self._samples[phase]
            util = sum(u for u, _ in samples) / len(samples) if samples else 0.0
            ratio = sum(r for _, r in samples) / len(samples) if samples else 0.0
            peak = top.prefill_mhz if phase == "prefill" else top.decode_mhz
            decisions.append(bandit.step(lat[phase], energy[phase], util, ratio, peak))
            samples.clear()
        self.log.write({"event": "halo_window", "decisions": decisions})
        moved = [f"{d['phase']}_{d['how']}" for d in decisions if d["from_mhz"] != d["to_mhz"]]
        if moved:
            await self.apply(",".join(moved))

    async def run(self, stop: asyncio.Event) -> None:
        await self.setup()
        next_window = self._clock() + self.settings.window_s
        energy = {"prefill": 0.0, "decode": 0.0}
        while not stop.is_set():
            try:
                got = await self.sample()
                for phase in energy:
                    energy[phase] += got[phase]
                await self.controller._log_energy(self._clock())
                if self._clock() >= next_window:
                    next_window = self._clock() + self.settings.window_s
                    await self.window(energy)
                    energy = {"prefill": 0.0, "decode": 0.0}
            except Exception as error:  # a bad tick must not stop control
                self.log.write({"event": "halo_error", "error": repr(error)})
            try:
                await asyncio.wait_for(stop.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                pass

    def state(self) -> dict[str, Any]:
        return {"clock": self.point().key(),
                "zones": {p: b.zone for p, b in self.bandits.items()},
                "windows": {p: b.t for p, b in self.bandits.items()}}
