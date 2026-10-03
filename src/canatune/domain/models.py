"""Cluster model from the Canary's measurements, and the solver over it.

Everything here comes from the tier locator's windows on this cluster (no
constants from other runs):

  prefill   S_f(L) single-request tables per P clock (plateau + slope)
  residence idle TTFT - prefill per prompt length: KV transfer + D's first token
  decode    T_iter(f_d, X) = alpha + delta * X per D clock (window-level TPOT
            medians against the mean running sequences, two or more loads per clock)
  power     P: idle(f) + k(f) * utilization; D: idle(f_d) + c(f_d) * busy share
            (idle from the park step, k and c from the windows' average power)
  ttft      the admission predictor fitted on the Canary's probes (TTFT from own
            S(L), pending P work, KV in flight, decoding) and the slack-risk counts
            of the calibration (violation risk by predicted slack)
  rho_p     the highest P utilization (sum S_f(L) / s) among windows that met the
            SLO (a lower bound; used only without the predictor)
  limits    D KV capacity (vLLM cache blocks), clean D concurrency B*, the
            KV-in-flight gate (calibration), the prompt limit

The solver evaluates every (groups n, P clock, D clock) for a request rate and a
length distribution, packing each group up to its feasible per-group capacity
before opening another decode group. Empty active groups pay idle power.
TTFT: the predictor at the steady-state means of its features
(pending P work = M/G/1 mean unfinished work lambda E[S^2] / 2(1 - rho); KV in
flight = lambda E[L kv residence]; decoding = Little's law on T_iter) for every
length of the mix, its violation risk from the slack-risk counts, averaged over the
mix, against theta (the same risk measure admission uses). D:
running sequences against the KV / B* / TPOT limits. KV bytes in flight against
the gate. Among the feasible configurations the lowest power (active groups at
their clocks, the rest parked).
Canary verification windows that failed cap a point's per-group rate.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from canatune.domain.admission import CostBook, PrefillCost, SlackRisk, _solve, slack_edges
from canatune.domain.groups import ClockPoint


def interpolate(points: Mapping[int, float], x: float) -> float | None:
    """Piecewise-linear over measured points, constant beyond the ends."""
    if not points:
        return None
    xs = sorted(points)
    if x <= xs[0]:
        return float(points[xs[0]])
    if x >= xs[-1]:
        return float(points[xs[-1]])
    for a, b in zip(xs, xs[1:]):
        if a <= x <= b:
            return points[a] + (points[b] - points[a]) * (x - a) / (b - a)
    return None


def fit_line(points: Sequence[tuple[float, float]]) -> tuple[float, float] | None:
    """Least squares y = a + b x; None without two distinct x."""
    if len(points) < 2:
        return None
    n = len(points)
    mx = sum(x for x, _ in points) / n
    my = sum(y for _, y in points) / n
    sxx = sum((x - mx) ** 2 for x, _ in points)
    if sxx <= 0:
        return None
    b = sum((x - mx) * (y - my) for x, y in points) / sxx
    return my - b * mx, b


def fit_decode(points: Mapping[int, Sequence[tuple[float, float]]]) -> dict[int, list[float]]:
    """Per D clock (running sequences, TPOT ms) window points -> [alpha, delta]. A
    clock with one load level shares the slope of the clocks that have two."""
    fits: dict[int, list[float]] = {}
    for f, pts in points.items():
        line = fit_line(pts)
        if line is not None and line[1] >= 0:
            fits[f] = [line[0], line[1]]
    slopes = [d for _, d in fits.values()]
    for f, pts in points.items():
        if f not in fits and pts and slopes:
            delta = sorted(slopes)[len(slopes) // 2]
            alpha = sum(y - delta * x for x, y in pts) / len(pts)
            fits[f] = [alpha, delta]
    return fits


def fit_decode_length(points: Sequence[tuple[float, float, float]]) -> list[float] | None:
    """(running sequences X, KV tokens they hold K, TPOT ms) per request -> [alpha,
    beta, gamma, r95]: a decode step costs alpha plus beta per sequence plus gamma
    per context token it reads (Ramani & Tantawi, arXiv 2609.20957, summed over the
    batch instead of averaged over one mean request). r95 is the 95th percentile of
    the residuals, so alpha + beta X + gamma K + r95 bounds a request's TPOT the way
    the SLO's p95 does. A negative slope (too little spread in X or K to separate
    them) is fixed at 0 and the other one refitted."""
    if len(points) < 4:
        return None
    for cols in ((0, 1), (0,), (1,)):
        rows = [[1.0, *(p[c] for c in cols)] for p in points]
        n = len(rows[0])
        ata = [[sum(r[i] * r[j] for r in rows) for j in range(n)] for i in range(n)]
        aty = [sum(r[i] * p[2] for r, p in zip(rows, points)) for i in range(n)]
        solution = _solve(ata, aty)
        if solution is None or any(c < 0 for c in solution[1:]):
            continue
        slopes = dict(zip(cols, solution[1:]))
        alpha, beta, gamma = solution[0], slopes.get(0, 0.0), slopes.get(1, 0.0)
        residuals = sorted(p[2] - (alpha + beta * p[0] + gamma * p[1]) for p in points)
        r95 = residuals[min(len(residuals) - 1, int(0.95 * (len(residuals) - 1) + 0.5))]
        return [alpha, beta, gamma, max(0.0, r95)]
    return None


def decode_tpot(coef: Sequence[float], running: float, kv_tokens: float) -> float:
    """TPOT bound (ms) of a request decoding among `running` sequences holding
    `kv_tokens` context tokens."""
    alpha, beta, gamma, r95 = coef
    return alpha + beta * running + gamma * kv_tokens + r95


def decode_capacity(
    coef: Sequence[float],
    context: float,
    tpot_slo_ms: float,
    kv_tokens: float | None = None,
    kv_limit: float = 1.0,
) -> dict[str, Any]:
    """Sequences of `context` tokens each that one D holds within the TPOT SLO and
    within kv_limit of its KV cache; `limit` names the binding one."""
    alpha, beta, gamma, r95 = coef
    per_sequence = beta + gamma * context
    tpot = (tpot_slo_ms - alpha - r95) / per_sequence if per_sequence > 0 else math.inf
    kv = kv_limit * kv_tokens / context if kv_tokens and context > 0 else math.inf
    sequences = max(0.0, min(tpot, kv))
    return {
        "sequences": None if math.isinf(sequences) else sequences,
        "tpot_sequences": None if math.isinf(tpot) else max(0.0, tpot),
        "kv_sequences": None if math.isinf(kv) else kv,
        "limit": "kv" if kv < tpot else "tpot",
    }


def decode_coef(model: Mapping[str, Sequence[float]], mhz: float) -> list[float] | None:
    """Length-model coefficients at D clock `mhz`: those of the nearest measured
    clock at or below it (a lower clock is never faster), else the lowest one."""
    if not model:
        return None
    clocks = sorted(int(f) for f in model)
    below = [f for f in clocks if f <= mhz]
    return [float(c) for c in model[str(max(below) if below else clocks[0])]]


def fit_power(
    idle: Mapping[int, float], windows: Sequence[tuple[int, float, float]]
) -> dict[int, list[float]]:
    """(clock, average W, utilization) windows -> per clock [idle W, dynamic W at
    full utilization]; idle interpolated from the idle measurements."""
    by_clock: dict[int, list[float]] = {}
    for f, watts, util in windows:
        base = interpolate(idle, f)
        if base is None or util < 0.05:
            continue
        by_clock.setdefault(f, []).append(max(0.0, (watts - base) / util))
    out = {}
    for f, ks in by_clock.items():
        ks.sort()
        out[f] = [float(interpolate(idle, f)), ks[len(ks) // 2]]
    return out


@dataclass
class ClusterModel:
    prefill: dict[int, dict[int, float]]  # P clock -> {prompt tokens: ms}
    residence: dict[int, float]  # prompt tokens -> idle TTFT - prefill (ms)
    decode: dict[int, list[float]]  # D clock -> [alpha ms, delta ms per sequence]
    power_prefill: dict[int, list[float]]  # P clock -> [idle W, dynamic W]
    power_decode: dict[int, list[float]]  # D clock -> [idle W, dynamic W]
    park_power_w: float  # one parked pair (P + D) at the park clocks
    rho_prefill: float  # SLO-limited P utilization (P-work seconds per second)
    ttft_slo_ms: float
    tpot_slo_ms: float
    kv_bytes_per_token: float
    kv_gate_bytes: float | None = None
    kv_capacity_tokens: int | None = None
    b_star: dict[int, int] = field(default_factory=dict)  # D clock -> clean concurrency
    prompt_limit: int | None = None
    caps: dict[str, float] = field(default_factory=dict)  # "p/d" -> verified rate cap
    predictor_coef: list[float] | None = None  # the admission predictor (calibration)
    slack_counts: dict[str, list[int]] | None = None  # calibration's slack-risk counts
    theta: float = 0.1

    def to_json(self) -> dict[str, Any]:
        return {
            "prefill": {str(f): {str(k): v for k, v in t.items()} for f, t in self.prefill.items()},
            "residence": {str(k): v for k, v in self.residence.items()},
            "decode": {str(f): v for f, v in self.decode.items()},
            "power_prefill": {str(f): v for f, v in self.power_prefill.items()},
            "power_decode": {str(f): v for f, v in self.power_decode.items()},
            "park_power_w": self.park_power_w,
            "rho_prefill": self.rho_prefill,
            "ttft_slo_ms": self.ttft_slo_ms,
            "tpot_slo_ms": self.tpot_slo_ms,
            "kv_bytes_per_token": self.kv_bytes_per_token,
            "kv_gate_bytes": self.kv_gate_bytes,
            "kv_capacity_tokens": self.kv_capacity_tokens,
            "b_star": {str(f): v for f, v in self.b_star.items()},
            "prompt_limit": self.prompt_limit,
            "caps": dict(self.caps),
            "predictor_coef": self.predictor_coef,
            "slack_counts": self.slack_counts,
            "theta": self.theta,
        }

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> "ClusterModel":
        def ints(d):
            return {int(k): v for k, v in (d or {}).items()}

        return cls(
            prefill={int(f): ints(t) for f, t in raw["prefill"].items()},
            residence=ints(raw["residence"]),
            decode=ints(raw["decode"]),
            power_prefill=ints(raw["power_prefill"]),
            power_decode=ints(raw["power_decode"]),
            park_power_w=float(raw["park_power_w"]),
            rho_prefill=float(raw["rho_prefill"]),
            ttft_slo_ms=float(raw["ttft_slo_ms"]),
            tpot_slo_ms=float(raw["tpot_slo_ms"]),
            kv_bytes_per_token=float(raw["kv_bytes_per_token"]),
            kv_gate_bytes=raw.get("kv_gate_bytes"),
            kv_capacity_tokens=raw.get("kv_capacity_tokens"),
            b_star=ints(raw.get("b_star")),
            prompt_limit=raw.get("prompt_limit"),
            caps=dict(raw.get("caps") or {}),
            predictor_coef=raw.get("predictor_coef"),
            slack_counts=raw.get("slack_counts"),
            theta=float(raw.get("theta", 0.1)),
        )

    @property
    def points(self) -> list[ClockPoint]:
        p = [f for f in self.prefill if f in self.power_prefill]
        d = [f for f in self.decode if f in self.power_decode]
        return [ClockPoint(fp, fd) for fp in sorted(p) for fd in sorted(d)]

    def costs(self) -> CostBook:
        return CostBook(self.prefill)


@dataclass(frozen=True)
class Evaluation:
    n: int
    point: ClockPoint
    feasible: bool
    power_w: float
    binding: str  # constraint with the highest use: prefill, decode, kv, ttft, cap
    use: dict[str, float]  # each constraint's load / limit
    rates: tuple[float, ...] = ()  # concentrated per-group request rates


class Solver:
    def __init__(self, model: ClusterModel, groups: int) -> None:
        self.model = model
        self.groups = groups
        self._costs = model.costs()
        self._slack = None
        self._capacities: dict[tuple, float] = {}
        if model.slack_counts:
            self._slack = SlackRisk(slack_edges(model.ttft_slo_ms))
            self._slack.set_seed(model.slack_counts)

    MIX_POINTS = 16  # quantile representatives of the recent length mix

    def _mix(self, mix: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
        kept = sorted(mix)  # offered long requests must not disappear from demand
        if len(kept) <= self.MIX_POINTS:
            return kept
        k = self.MIX_POINTS
        representatives = [kept[int((i + 0.5) * len(kept) / k)] for i in range(k)]
        representatives[-1] = kept[-1]  # never lose a rare longest prompt
        return representatives

    def evaluate(
        self, rate_rps: float, mix: Sequence[tuple[int, int]], n: int, point: ClockPoint
    ) -> Evaluation:
        """Fill groups to measured/modelled capacity before opening another D.

        Each active D pays its busy-power cost once; more tokens in its batch
        share that cost. Keep n physical active groups, including any empty ones.
        """
        if n < 1 or rate_rps < 0:
            raise ValueError("positive group count and non-negative rate required")
        capacity = self.capacity(mix, point)
        remaining = rate_rps
        rates = []
        for i in range(n):
            rate = remaining if i == n - 1 else min(remaining, capacity)
            rates.append(rate)
            remaining -= rate
        evaluations = [self._uniform(r, mix, 1, point) for r in rates]
        use = {k: max(e.use.get(k, 0) for e in evaluations) for k in evaluations[0].use}
        park = self.model.park_power_w
        power = sum(e.power_w - (self.groups - 1) * park for e in evaluations)
        power += (self.groups - n) * park
        return Evaluation(
            n, point, all(e.feasible for e in evaluations), power,
            max(use, key=use.get), use, tuple(rates),
        )

    def capacity(self, mix: Sequence[tuple[int, int]], point: ClockPoint) -> float:
        """Largest feasible per-group rate; shared by packing and Canary verify."""
        key = (point, tuple(self._mix(mix)))
        if key not in self._capacities:
            lo, hi = 0.0, 1.0
            if self._uniform(0, mix, 1, point).feasible:
                for _ in range(30):
                    if not self._uniform(hi, mix, 1, point).feasible:
                        break
                    hi *= 2
                for _ in range(24):
                    mid = (lo + hi) / 2
                    if self._uniform(mid, mix, 1, point).feasible:
                        lo = mid
                    else:
                        hi = mid
            self._capacities[key] = lo
        return self._capacities[key]

    def _uniform(
        self, rate_rps: float, mix: Sequence[tuple[int, int]], n: int, point: ClockPoint
    ) -> Evaluation:
        m = self.model
        mix = self._mix(mix)
        if not mix:
            raise ValueError("the solver needs a length mix")
        r = rate_rps / max(n, 1)
        cost: PrefillCost = self._costs.for_clock(point.prefill_mhz)
        mean_s = sum(cost(p) for p, _ in mix) / len(mix)
        mean_out = sum(o for _, o in mix) / len(mix)
        mean_ctx = sum(p + o for p, o in mix) / len(mix)
        use: dict[str, float] = {}
        rho = r * mean_s / 1000.0  # P busy share (single-request service times)
        # D: running sequences by Little's law with T_iter(X) = alpha + delta X
        alpha, delta = self._decode(point.decode_mhz)
        load = r * mean_out / 1000.0  # tokens per ms of iteration
        running = math.inf if load * delta >= 1 else load * alpha / (1 - load * delta)
        res = sum(
            p * m.kv_bytes_per_token * (interpolate(m.residence, p) or 0.0) for p, _ in mix
        ) / len(mix)
        inflight = r * res / 1000.0  # KV bytes in flight (mean)
        top = max(p for p, _ in mix)
        if m.predictor_coef is not None and self._slack is not None:
            # Violation risk at the steady-state means of the predictor's features.
            if rho >= 1 or not math.isfinite(running):
                use["prefill"] = math.inf
            else:
                mean_s2 = sum(cost(p) ** 2 for p, _ in mix) / len(mix)
                pending = (r / 1000.0) * mean_s2 / (2 * (1 - rho))  # M/G/1 unfinished work
                risk = 0.0
                for p, _ in mix:
                    x = (1.0, cost(p), pending, inflight / 1e9, running)
                    predicted = sum(c * v for c, v in zip(m.predictor_coef, x))
                    risk += self._slack.estimate(m.ttft_slo_ms - predicted).risk
                use["prefill"] = risk / len(mix) / m.theta if m.theta > 0 else math.inf
        else:
            use["prefill"] = rho / m.rho_prefill if m.rho_prefill > 0 else math.inf
        limits = []
        if m.kv_capacity_tokens:
            limits.append(m.kv_capacity_tokens / mean_ctx)
        b = self._b_star(point.decode_mhz)
        if b:
            limits.append(float(b))
        if delta > 0:
            limits.append((m.tpot_slo_ms - alpha) / delta)
        x_max = min(limits) if limits else math.inf
        use["decode"] = running / x_max if x_max > 0 else math.inf
        # KV bytes in flight (transfer + D's first token) against the gate
        if m.kv_gate_bytes:
            use["kv"] = inflight / m.kv_gate_bytes
        # idle TTFT of the longest prompt
        idle_ttft = cost(top) + (interpolate(m.residence, top) or 0.0)
        use["ttft"] = idle_ttft / m.ttft_slo_ms
        cap = m.caps.get(point.key())
        if cap is not None:
            use["cap"] = r / cap if cap > 0 else math.inf
        # A verified cap is strict: the point failed at that rate.
        feasible = all(v <= 1.0 for k, v in use.items() if k != "cap") and use.get("cap", 0) < 1

        busy = min(1.0, running) if math.isfinite(running) else 1.0
        power = (
            n
            * (
                self._power(m.power_prefill, point.prefill_mhz, min(rho, 1.0))
                + self._power(m.power_decode, point.decode_mhz, busy)
            )
            + (self.groups - n) * m.park_power_w
        )
        binding = max(use, key=use.get)
        return Evaluation(n, point, feasible, power, binding, use)

    def solve(
        self,
        rate_rps: float,
        mix: Sequence[tuple[int, int]],
        *,
        min_groups: int = 1,
        groups: int | None = None,
    ) -> Evaluation:
        """Lowest-power feasible configuration; when nothing is feasible, the one
        with the most headroom (all groups at the fastest point)."""
        top = groups or self.groups
        evals = [
            self.evaluate(rate_rps, mix, n, point)
            for n in range(min_groups, top + 1)
            for point in self.model.points
        ]
        if not evals:
            raise ValueError("the model has no clock points")
        ok = [e for e in evals if e.feasible]
        if ok:
            return min(ok, key=lambda e: (e.power_w, e.n))
        # Nothing feasible: the most headroom; ties (one constraint binds whatever the
        # other clock) go to more groups and higher clocks, so the answer is stable.
        return min(
            evals,
            key=lambda e: (
                round(max(e.use.values()), 3),
                -e.n,
                -e.point.prefill_mhz,
                -e.point.decode_mhz,
            ),
        )

    def _decode(self, f: int) -> tuple[float, float]:
        fits = self.model.decode
        alpha = interpolate({k: v[0] for k, v in fits.items()}, f)
        delta = interpolate({k: v[1] for k, v in fits.items()}, f)
        return float(alpha or 0.0), float(delta or 0.0)

    def _b_star(self, f: int) -> int | None:
        if not self.model.b_star:
            return None
        below = [k for k in self.model.b_star if k <= f]
        return self.model.b_star[max(below) if below else min(self.model.b_star)]

    @staticmethod
    def _power(table: Mapping[int, list[float]], f: int, util: float) -> float:
        idle = interpolate({k: v[0] for k, v in table.items()}, f) or 0.0
        dyn = interpolate({k: v[1] for k, v in table.items()}, f) or 0.0
        return idle + dyn * util
