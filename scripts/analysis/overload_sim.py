"""Discrete-event simulation of P/D pairs driving the real CanaTune Router and
Controller on a simulated clock. The hardware is a `Physics` instance; REFERENCE
is calibrated on r6b / r7 (L40S P, L4 D, Mistral-7B).

Compares, on the same arrival trace (the smoke-2 load profile):
  baseline  round robin over all pairs at MAX (default system: no admission)
  reject    CanaTune, state-based admission, overload: reject (503)
  serve     CanaTune, state-based admission, overload: serve (rescue, backfill doomed)
  serve_dispatch   the same, doomed requests dispatched at once (naive serve-all)

Physics per pair (REFERENCE values, sources in brackets):
  P   batches everything queued (<= 8192 tokens): t0 + beta_f * max(sum L, L*);
      beta_f = beta_top x f_eff / min(f, f_eff): above f_eff the power cap holds
      the clock [r6b: t0 ~30 ms, L* ~200, beta 68 / 75 / 113 us/token at 2520 /
      1545 / 1080 MHz, 2520 capped near 1550-1700]
  KV  L x 131072 bytes; P hands each request's KV to D's receive buffer (1 GB);
      a full buffer makes P retry the send every 50 ms and start nothing else
      [r7: "Peer Out Of Memory" retries, P stalls, congestion collapse]; the KV
      crosses a per-pair link (6 Gb/s)
  D   iterations of alpha_D(f) + delta_D(f) x running ms [IBM fit, r6b: 1170 MHz
      57 + 1.38 X, 2040 MHz 55 + 1.41 X, 735 MHz 60 + 2.16 X]; a request joins
      at an iteration start once its KV arrived and D's KV cache has room for
      prompt + output (26k tokens: then the long mix fails near 2 req/s, the
      default mix near 5 and the short mix runs past 6, as in r7), which frees its
      buffer; first token one iteration later plus a fixed D-first overhead
  fixed proxy / HTTP overhead before P [smoke 2 timing: ~40 ms]
  power  idle + utilization x dynamic(f / f_top) per GPU (synthetic: L40S-like P
      up to ~325 W, L4-like D up to 72 W); parked groups idle at the lowest clock
Clock changes are instantaneous.

  python scripts/analysis/overload_sim.py [--calibrate] [--profile ...] [--out DIR]
"""

import argparse
import asyncio
import heapq
import itertools
import json
import statistics
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from canatune import loadgen
from canatune.config import load_config
from canatune.controller.locator import (
    AdmissionInputs,
    Hardware,
    LocatorSettings,
    TierLocator,
    WindowResult,
)
from canatune.controller.router import CanaTuneRouter, RouterSettings, Ticket
from canatune.controller.tier_controller import ControllerSettings, TierController
from canatune.domain.calibration import ProbeSample
from canatune.domain.groups import ClockPoint, Group, GroupState, Tier, TierState, TierTable
from canatune.domain.load import LengthStats
from canatune.domain.risk import RiskTable
from canatune.infrastructure.clocks import NullClockActuator

TTFT_SLO_MS, TPOT_SLO_MS = 1000.0, 200.0


@dataclass(frozen=True)
class Physics:
    """One kind of P/D pair (what the simulated cluster really is)."""

    name: str = "reference"
    kv_bytes_per_token: float = 131072
    kv_buffer_bytes: float = 1e9
    link_bytes_s: float = 6e9 / 8
    send_retry_s: float = 0.05
    p_t0_ms: float = 30.0
    p_lstar: int = 200
    p_beta_top_ms: float = 0.068  # ms per token at f_eff and above
    p_f_eff: int = 1700  # the power cap holds P near this clock
    p_batch_tokens: int = 8192
    d_alpha_ms: float = 55.0
    d_delta_ms: float = 1.41
    d_f_knee: int = 1150  # below it D iterations stretch
    d_kv_tokens: int = 26000
    d_first_extra_ms: float = 60.0
    overhead_ms: float = 40.0
    p_clocks: tuple[int, ...] = (1080, 1290, 1545, 1755, 2010, 2265, 2520)
    d_clocks: tuple[int, ...] = (735, 960, 1170, 1395, 1605, 2040)
    p_idle_w: float = 35.0
    p_dyn_w: float = 290.0
    d_idle_w: float = 16.0
    d_dyn_w: float = 56.0

    @property
    def max_point(self) -> ClockPoint:
        return ClockPoint(self.p_clocks[-1], self.d_clocks[-1])

    def p_beta(self, f: int) -> float:
        return self.p_beta_top_ms * self.p_f_eff / min(f, self.p_f_eff)

    def prefill_ms(self, f: int, tokens: int) -> float:
        return self.p_t0_ms + self.p_beta(f) * max(tokens, self.p_lstar)

    def d_iter_ms(self, f: int, running: int) -> float:
        stretch = max(1.0, self.d_f_knee / f)
        return self.d_alpha_ms * stretch**0.2 + self.d_delta_ms * stretch * running

    def p_power(self, f: int, util: float) -> float:
        x = min(f, self.p_clocks[-1]) / self.p_clocks[-1]
        return self.p_idle_w * (0.75 + 0.25 * x) + util * self.p_dyn_w * x**2.4

    def d_power(self, f: int, util: float) -> float:
        x = f / self.d_clocks[-1]
        return self.d_idle_w * (0.75 + 0.25 * x) + util * self.d_dyn_w * x**2.0


REFERENCE = Physics()  # an example environment (what the simulated hardware is)
MAX = REFERENCE.max_point
LENGTHS = [(128, 64), (512, 64), (1024, 64)]  # the request length mix


@dataclass
class Req:
    id: int
    at: float
    phase: str
    prompt: int
    output: int
    waiter: int = 0
    ticket: Ticket | None = None
    pair: "Pair | None" = None
    first: float | None = None
    done: float | None = None
    emitted: int = 0
    last_token: float | None = None  # time of the last emitted token
    status: str = "pending"  # ok / rejected
    kv: float = 0.0
    state: tuple = ()  # (at P prompts, in-flight tokens, decoding) at dispatch
    worker: int | None = None  # closed loop: the worker that sends the next one


@dataclass
class Pair:
    name: str
    clock: ClockPoint = MAX
    p_queue: deque = field(default_factory=deque)
    p_busy: bool = False
    unsent: deque = field(default_factory=deque)  # P done, KV not yet in D's buffer
    link_free: float = 0.0
    buffer: float = 0.0
    d_ready: deque = field(default_factory=deque)  # KV arrived, waiting to join
    d_running: list = field(default_factory=list)
    d_looping: bool = False
    send_retries: int = 0
    at_p: dict = field(default_factory=dict)  # request id -> prompt (dispatch .. P done)
    inflight_tokens: int = 0  # P done .. first token
    decoding: int = 0
    p_busy_s: float = 0.0
    d_busy_s: float = 0.0
    d_running_integral: float = 0.0  # running sequences x seconds (mean while busy)
    d_waiting_max: int = 0
    d_running_max: int = 0
    d_kv_max: float = 0.0


class Sim:
    def __init__(
        self,
        n_pairs: int,
        policy: str,
        clocks: ClockPoint = MAX,
        capacity_rps=4.0,
        router_overrides: dict | None = None,
        *,
        phys: Physics = REFERENCE,
        table: TierTable | None = None,
    ):
        self.phys = phys
        self.router_overrides = router_overrides or {}
        self.now = 0.0
        self.events: list = []
        self.seq = itertools.count()
        self.pairs = [Pair(f"G{i}", clocks) for i in range(n_pairs)]
        self.policy = policy
        self.rr = itertools.cycle(self.pairs)
        self.requests: list[Req] = []
        self.timeline: list[dict] = []
        self.energy_j = 0.0
        self.router = self.controller = None
        self.on_done = None  # closed loop: callback(req)
        self._last_busy = {p.name: (0.0, 0.0) for p in self.pairs}
        if policy != "baseline":
            if table is None:
                raise ValueError("CanaTune policies need the Canary's table (run_canary)")
            self._cantune(table)

    # ---- CanaTune --------------------------------------------------------------------

    def _cantune(self, table: TierTable) -> None:
        config = load_config()
        config["router"]["admission"] = "slack"
        config["router"]["overload"] = {"reject": "reject", "best_effort": "best_effort"}.get(
            self.policy, "serve"
        )
        config["router"]["doomed"] = "dispatch" if self.policy == "serve_dispatch" else "backfill"
        config["controller"]["stagger_s"] = 0.0
        config["kv_transfer"]["kv_buffer_bytes"] = self.phys.kv_buffer_bytes
        config["kv_transfer"]["kv_bytes_per_token"] = self.phys.kv_bytes_per_token
        config["router"].update(self.router_overrides)
        groups = [Group(p.name, f"P{i}", f"D{i}") for i, p in enumerate(self.pairs)]
        tiers = TierState(max_point=self.phys.max_point, table=table)
        risk = RiskTable.from_config(config["risk"], {"sim": True})
        clock = lambda: self.now  # noqa: E731
        lengths = LengthStats(LENGTHS, min_samples=50)
        self.router = CanaTuneRouter(
            groups, risk, RouterSettings.from_config(config), tiers, clock=clock, lengths=lengths
        )
        self.controller = TierController(
            groups,
            self.router,
            ControllerSettings.from_config(config),
            NullClockActuator(),
            {},
            tiers,
            clock=clock,
        )
        self.groups = {g.name: g for g in groups}
        for g, p in zip(groups, self.pairs):
            g.state, g.tier, g.effective = GroupState.ACTIVE, Tier.H, table.h
            p.clock = table.h

    # ---- event loop ------------------------------------------------------------------

    def at(self, t: float, fn, *args) -> None:
        heapq.heappush(self.events, (t, next(self.seq), fn, args))

    def run(self, arrivals, end_s: float) -> None:
        for a in arrivals:
            self.at(
                a.at_s, self.arrive, Req(a.index, a.at_s, a.phase, a.prompt_tokens, a.output_tokens)
            )
        self.loop(end_s)

    def loop(self, end_s: float) -> None:
        if self.controller is not None:
            self.at(0.0, self.tick)
        self.at(0.0, self.sample)
        while self.events:
            t, _, fn, args = heapq.heappop(self.events)
            if t > end_s + 120 and fn in (self.tick, self.sample):
                continue
            self.now = t
            fn(*args)

    def tick(self) -> None:
        asyncio.run(self.controller.tick())
        for p in self.pairs:
            g = self.groups[p.name]
            if g.effective is not None:
                p.clock = g.effective
        self.at(self.now + self.controller.settings.period_s, self.tick)

    def sample(self) -> None:
        row = {"t": round(self.now, 1)}
        for p in self.pairs:
            g = self.groups.get(p.name) if self.router else None
            state = "active" if g is None else g.state.value
            row[p.name] = {
                "state": state,
                "tier": "max" if g is None else g.tier.value,
                "buffer_mb": round(p.buffer / 1e6),
                "running": len(p.d_running),
                "p_queue": len(p.p_queue),
            }
            # Energy of the last second: utilization at the current clocks; parked
            # groups idle at the lowest clocks.
            p_last, d_last = self._last_busy[p.name]
            p_util = min(1.0, p.p_busy_s - p_last)
            d_util = min(1.0, p.d_busy_s - d_last)
            self._last_busy[p.name] = (p.p_busy_s, p.d_busy_s)
            if state == "park":
                f_p, f_d = self.phys.p_clocks[0], self.phys.d_clocks[0]
            else:
                f_p, f_d = p.clock.prefill_mhz, p.clock.decode_mhz
            self.energy_j += self.phys.p_power(f_p, p_util) + self.phys.d_power(f_d, d_util)
        if self.router is not None:
            row["holding"] = len(self.router._holding)
        self.timeline.append(row)
        self.at(self.now + 1.0, self.sample)

    # ---- admission -------------------------------------------------------------------

    def arrive(self, r: Req) -> None:
        r.kv = r.prompt * self.phys.kv_bytes_per_token
        self.requests.append(r)
        if self.router is None:
            self.dispatch(r, next(self.rr))
            return
        r.waiter = self.router.new_waiter()
        self.retry(r)

    def retry(self, r: Req) -> None:
        result = self.router.step(r.waiter, r.prompt, True, (self.now - r.at) * 1000.0)
        if isinstance(result, Ticket):
            r.ticket = result
            self.dispatch(r, next(p for p in self.pairs if p.name == result.group.name))
        elif result == "reject":
            r.status = "rejected"
        else:
            self.at(self.now + self.router.settings.retry_period_ms / 1000.0, self.retry, r)

    def dispatch(self, r: Req, pair: Pair) -> None:
        r.pair = pair
        r.state = (tuple(pair.at_p.values()), pair.inflight_tokens, pair.decoding)
        pair.at_p[r.id] = r.prompt
        self.at(self.now + self.phys.overhead_ms / 1000.0, self.p_enqueue, r)

    # ---- P ---------------------------------------------------------------------------

    def p_enqueue(self, r: Req) -> None:
        r.pair.p_queue.append(r)
        self.p_start(r.pair)

    def p_start(self, pair: Pair) -> None:
        if pair.p_busy or pair.unsent or not pair.p_queue:
            return
        batch, tokens = [], 0
        limit = self.phys.p_batch_tokens
        while pair.p_queue and (not batch or tokens + pair.p_queue[0].prompt <= limit):
            r = pair.p_queue.popleft()
            batch.append(r)
            tokens += r.prompt
        pair.p_busy = True
        dur = self.phys.prefill_ms(pair.clock.prefill_mhz, tokens) / 1000.0
        pair.p_busy_s += dur
        self.at(self.now + dur, self.p_done, pair, batch)

    def p_done(self, pair: Pair, batch: list) -> None:
        pair.p_busy = False
        pair.unsent.extend(batch)
        for r in batch:
            pair.at_p.pop(r.id, None)
            pair.inflight_tokens += r.prompt
            if r.ticket is not None:
                self.router.prefill_done(r.ticket)
        self.send(pair)

    def send(self, pair: Pair) -> None:
        while pair.unsent and pair.buffer + pair.unsent[0].kv <= self.phys.kv_buffer_bytes:
            r = pair.unsent.popleft()
            pair.buffer += r.kv
            start = max(self.now, pair.link_free)
            pair.link_free = start + r.kv / self.phys.link_bytes_s
            self.at(pair.link_free, self.kv_arrived, r)
        if pair.unsent:  # buffer full: P retries the send and starts nothing else
            pair.send_retries += 1
            self.at(self.now + self.phys.send_retry_s, self.send, pair)
        else:
            self.p_start(pair)

    # ---- D ---------------------------------------------------------------------------

    def kv_arrived(self, r: Req) -> None:
        r.pair.d_ready.append(r)
        if not r.pair.d_looping:
            r.pair.d_looping = True
            self.d_iter(r.pair)

    def d_iter(self, pair: Pair) -> None:
        joined = []
        used = sum(r.prompt + r.output for r in pair.d_running)
        cap = self.phys.d_kv_tokens
        while pair.d_ready and used + pair.d_ready[0].prompt + pair.d_ready[0].output <= cap:
            r = pair.d_ready.popleft()
            used += r.prompt + r.output
            pair.buffer -= r.kv  # pulled into D's KV cache
            joined.append(r)
        pair.d_running.extend(joined)
        pair.d_waiting_max = max(pair.d_waiting_max, len(pair.d_ready))
        pair.d_running_max = max(pair.d_running_max, len(pair.d_running))
        pair.d_kv_max = max(pair.d_kv_max, used / cap)
        if not pair.d_running:
            pair.d_looping = False
            return
        if joined:
            self.send(pair)  # buffer space for P's pending sends
        dur = self.phys.d_iter_ms(pair.clock.decode_mhz, len(pair.d_running)) / 1000.0
        pair.d_busy_s += dur
        pair.d_running_integral += dur * len(pair.d_running)
        self.at(self.now + dur, self.d_end, pair, joined)

    def d_end(self, pair: Pair, joined: list) -> None:
        for r in list(pair.d_running):
            if r in joined:
                r.first = self.now + self.phys.d_first_extra_ms / 1000.0
                pair.inflight_tokens -= r.prompt
                pair.decoding += 1
                if r.ticket is not None:
                    self.router.first_token(r.ticket)
            elif r.ticket is not None and r.last_token is not None:
                # As the proxy does per streamed chunk: the latest token spacing.
                self.router.token_progress(r.ticket, (self.now - r.last_token) * 1000.0)
            r.last_token = self.now
            r.emitted += 1
            if r.emitted >= r.output:
                pair.d_running.remove(r)
                pair.decoding -= 1
                r.done = max(self.now, r.first)
                r.status = "ok"
                if r.ticket is not None:
                    self.router.finish(
                        r.ticket,
                        status="ok",
                        ttft_ms=(r.first - r.at) * 1000.0,
                        tpot_ms=self.tpot(r),
                        output_tokens=r.output,
                    )
                if self.on_done is not None:
                    self.on_done(r)
        self.d_iter(pair)

    @staticmethod
    def tpot(r: Req) -> float | None:
        return None if r.output <= 1 else (r.done - r.first) * 1000.0 / (r.output - 1)

    def probe_sample(self, r: Req) -> ProbeSample:
        at_p, inflight, decoding = r.state
        ttft = None if r.first is None else (r.first - r.at) * 1000.0
        return ProbeSample(
            r.pair.clock.prefill_mhz,
            r.pair.clock.decode_mhz,
            r.prompt,
            at_p,
            inflight,
            decoding,
            ttft,
            None if r.status != "ok" else violated(r),
        )


# ---- summaries ---------------------------------------------------------------------------


def violated(r: Req) -> bool:
    if r.status != "ok":
        return True
    return (r.first - r.at) * 1000 > TTFT_SLO_MS or (Sim.tpot(r) or 0) > TPOT_SLO_MS


def pct(values, q):
    if not values:
        return None
    s = sorted(values)
    return round(s[min(len(s) - 1, int(q * len(s)))])


def summarize(sim: Sim, phases: list[dict]) -> dict:
    out = {}
    for ph in phases + [{"name": "total"}]:
        rows = [r for r in sim.requests if ph["name"] in ("total", r.phase)]
        served = [r for r in rows if r.status == "ok"]
        ttft = [(r.first - r.at) * 1000 for r in served]
        good = [r for r in rows if not violated(r)]
        over = [r for r in served if r.ticket is not None and r.ticket.overflow]
        out[ph["name"]] = {
            "offered": len(rows),
            "served": len(served),
            "rejected": sum(r.status == "rejected" for r in rows),
            "served_late": len(served) - len(good),
            "good": len(good),
            "goodput": round(len(good) / max(len(rows), 1), 3),
            "ttft_p50": pct(ttft, 0.5),
            "ttft_p95": pct(ttft, 0.95),
            "ttft_p99": pct(ttft, 0.99),
            "overflow": len(over),
            "overflow_good": sum(not violated(r) for r in over),
            "rescue": sum(r.ticket.overflow == "rescue" for r in over),
            "late_ttft_p50": pct([(r.first - r.at) * 1000 for r in served if violated(r)], 0.5),
        }
        if ph["name"] != "total":
            t = [row for row in sim.timeline if ph["start_s"] <= row["t"] < ph["end_s"]]
        else:
            t = sim.timeline
        names = [p.name for p in sim.pairs]
        out[ph["name"]]["active_group_s"] = sum(
            row[n]["state"] in ("active", "draining") for row in t for n in names
        )
        out[ph["name"]]["max_group_s"] = sum(row[n]["tier"] == "max" for row in t for n in names)
    out["send_retries"] = sum(p.send_retries for p in sim.pairs)
    if sim.router is not None:
        out["router"] = {k: v for k, v in sim.router.state().items() if k != "groups"}
    return out


class SimBackend:
    """`ProbeBackend` over one simulated pair: the real tier locator measures the
    simulated hardware through it exactly as through the Canary probe (windows,
    per-request state samples, average power, D running sequences)."""

    service_source = "metrics"

    def __init__(self, phys: Physics, lengths=LENGTHS, seed: int = 7) -> None:
        import random

        self.phys = phys
        self.lengths = list(lengths)
        self.rng = random.Random(seed)
        self.limit: int | None = None
        self.windows = 0
        self.sim_seconds = 0.0

    def _pairs(self) -> list[tuple[int, int]]:
        pairs = [p for p in self.lengths if self.limit is None or p[0] <= self.limit]
        return pairs or self.lengths[:1]

    async def hardware(self) -> Hardware:
        return Hardware(self.phys.p_clocks, self.phys.d_clocks)

    async def idle_power(self, clock: ClockPoint, seconds: float) -> tuple[float, float]:
        noise = 1 + self.rng.gauss(0, 0.005)
        return (
            self.phys.p_power(clock.prefill_mhz, 0.0) * noise,
            self.phys.d_power(clock.decode_mhz, 0.0) * noise,
        )

    async def service_times(self, clock, prompts):
        out = []
        for n in prompts:
            pre = self.phys.prefill_ms(clock.prefill_mhz, n) * (1 + self.rng.gauss(0, 0.01))
            ttft = (
                self.phys.overhead_ms
                + pre
                + n * self.phys.kv_bytes_per_token / self.phys.link_bytes_s * 1000
                + self.phys.d_iter_ms(clock.decode_mhz, 1)
                + self.phys.d_first_extra_ms
            )
            out.append((n, pre, ttft))
        return out

    def set_prompt_limit(self, max_prompt: int | None) -> float:
        self.limit = max_prompt
        pairs = self._pairs()
        return sum(p for p, _ in pairs) / len(pairs)

    def _result(self, s: "Sim", clock: ClockPoint, kind: str, load: float) -> WindowResult:
        pair = s.pairs[0]
        reqs = [r for r in s.requests if r.status == "ok"]
        end = max((r.done for r in reqs), default=1.0)
        start = min((r.at for r in s.requests), default=0.0)
        duration = max(end - start, 1e-3)
        f_p, f_d = clock.prefill_mhz, clock.decode_mhz
        p_j = self.phys.p_power(f_p, 0) * duration + pair.p_busy_s * (
            self.phys.p_power(f_p, 1) - self.phys.p_power(f_p, 0)
        )
        d_j = self.phys.d_power(f_d, 0) * duration + pair.d_busy_s * (
            self.phys.d_power(f_d, 1) - self.phys.d_power(f_d, 0)
        )
        tokens = sum(r.output for r in reqs)
        tpots = sorted(t for t in (s.tpot(r) for r in reqs) if t is not None)
        self.windows += 1
        self.sim_seconds += duration
        busy = min(1.0, pair.d_busy_s / duration)
        running = pair.d_running_integral / duration if duration else 0.0
        return WindowResult(
            clock=clock,
            kind=kind,
            load=load,
            duration_s=duration,
            requests=len(reqs),
            violations=sum(violated(r) for r in reqs),
            ttft_p95_ms=pct([(r.first - r.at) * 1000 for r in reqs], 0.95),
            tpot_p95_ms=pct(tpots, 0.95),
            prefill_j_per_request=p_j / len(reqs) if reqs else None,
            decode_j_per_token=d_j / tokens if tokens else None,
            prefill_mhz_median=float(min(f_p, self.phys.p_f_eff)),
            prefill_limited_fraction=1.0 if f_p > self.phys.p_f_eff else 0.0,
            decode_waiting_max=float(pair.d_waiting_max),
            decode_kv_max=pair.d_kv_max,
            decode_running_max=float(pair.d_running_max),
            samples=[s.probe_sample(r) for r in reqs],
            prefill_avg_w=p_j / duration,
            decode_avg_w=d_j / duration,
            decode_busy_fraction=busy,
            decode_running_mean=running / busy if busy > 0 else None,
            tpot_p50_ms=tpots[len(tpots) // 2] if tpots else None,
        )

    async def open_window(self, clock, eq_tps, alpha, seconds, abort_above):
        import random

        trace = random.Random(int(eq_tps * 10) + 17)
        pairs = self._pairs()
        mean = sum(p for p, _ in pairs) / len(pairs)
        count = max(1, round(eq_tps / (mean + alpha) * seconds))
        arrivals = [
            loadgen.Arrival(i, t, "w", *pairs[trace.randrange(len(pairs))])
            for i, t in enumerate(sorted(trace.uniform(0, seconds) for _ in range(count)))
        ]
        s = Sim(1, "baseline", clock, phys=self.phys)
        s.run(arrivals, seconds)
        return self._result(s, clock, "open", eq_tps)

    async def closed_window(self, clock, concurrency, seconds, rep=0):
        import random

        trace = random.Random(concurrency * 101 + rep)
        pairs = self._pairs()
        s = Sim(1, "baseline", clock, phys=self.phys)
        ids = iter(range(10**9))
        stagger = min(15.0, seconds / 3)

        def send(worker: int, at: float) -> None:
            if at >= seconds:
                return
            prompt = pairs[trace.randrange(len(pairs))][0]
            s.at(at, s.arrive, Req(next(ids), at, "w", prompt, 256, worker=worker))

        s.on_done = lambda r: send(r.worker, s.now)
        for w in range(concurrency):
            send(w, stagger * w / max(concurrency, 1))
        s.loop(seconds)
        return self._result(s, clock, "closed", concurrency)

    async def kv_capacity(self) -> int:
        return self.phys.d_kv_tokens


def run_canary(phys: Physics = REFERENCE, lengths=LENGTHS) -> tuple[TierTable, SimBackend]:
    """The real tier locator on the simulated pair; deployment inputs only: the KV
    geometry and kv_buffer_size (what config.yaml and the model's config.json give)."""
    backend = SimBackend(phys, lengths)
    locator = TierLocator(
        backend,
        LocatorSettings.from_config(load_config()),
        admission=AdmissionInputs(
            kv_bytes_per_token=phys.kv_bytes_per_token, kv_buffer_bytes=phys.kv_buffer_bytes
        ),
    )
    prompts = sorted({p for p, _ in lengths})
    table = asyncio.run(locator.locate(sum(prompts) / len(prompts), prompts))
    return table, backend


def trace(table: TierTable, profile: str, seed: int, lengths=LENGTHS):
    """The load profile in units of the Canary's measured per-group capacity."""
    return loadgen.build_trace(
        loadgen.parse_profile(profile),
        lengths,
        capacity_h=table.capacity_h,
        alpha=table.alpha_tokens,
        seed=seed,
    )


def calibrate(clock: ClockPoint, rates, seconds: float, seed: int) -> list[dict]:
    """Single-pair rate sweep of the simulated environment (what the hardware is)."""
    rows = []
    for rate in rates:
        phases = loadgen.parse_profile(f"run:{seconds}:{rate}")
        meta, arrivals = loadgen.build_trace(
            phases, LENGTHS, capacity_h=sum(p for p, _ in LENGTHS) / 3, alpha=0.0, seed=seed
        )
        sim = Sim(1, "baseline", clock)
        sim.run(arrivals, seconds)
        reqs = [r for r in sim.requests if r.at > 10]  # skip the fill-up
        ttft = [(r.first - r.at) * 1000 for r in reqs if r.status == "ok"]
        rows.append(
            {
                "rps": rate,
                "violated": round(sum(map(violated, reqs)) / max(len(reqs), 1), 3),
                "ttft_p50": pct(ttft, 0.5),
                "ttft_p95": pct(ttft, 0.95),
                "send_retries": sim.pairs[0].send_retries,
            }
        )
    return rows


def run_policy(policy: str, table: TierTable, arrivals, meta, overrides, groups: int = 2):
    clocks = MAX if policy == "baseline" else table.h
    sim = Sim(groups, policy, clocks, router_overrides=overrides, table=table)
    sim.run(arrivals, meta["duration_s"])
    res = summarize(sim, meta["phases"])
    res["energy_kj"] = sim.energy_j / 1000
    return sim, res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--calibrate", action="store_true", help="single-pair rate sweep")
    ap.add_argument("--profile", default=loadgen.DEFAULT_PROFILE)
    ap.add_argument("--load-scale", type=float, default=1.0, help="x the profile's loads")
    ap.add_argument("--seed", type=int, default=20261001)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--policies", default="baseline,reject,serve_dispatch,serve")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--set", action="append", default=[], help="router key=value (JSON value)")
    args = ap.parse_args()
    if args.calibrate:
        for clock in (ClockPoint(REFERENCE.p_clocks[2], REFERENCE.d_clocks[2]), MAX):
            print(f"single pair at {clock.key()}:")
            for row in calibrate(clock, [1, 2, 3, 3.5, 4, 4.5, 5, 6], 120, args.seed):
                print("  ", row)
        return 0
    table, backend = run_canary()
    print(
        f"Canary: {backend.windows} windows; H {table.h.key()},"
        f" C_H {table.evidence.get('capacity_rps', 0):.2f} req/s"
    )
    profile = ",".join(
        ":".join([f[0], f[1], str(float(f[2]) * args.load_scale)] + f[3:])
        for f in (item.split(":") for item in args.profile.split(","))
    )
    overrides = {k: json.loads(v) for k, v in (item.split("=", 1) for item in args.set)}
    keys = ("goodput", "served", "rejected", "served_late", "ttft_p95", "energy_kj")
    rows: dict = {}
    for seed in range(args.seed, args.seed + args.seeds):
        meta, arrivals = trace(table, profile, seed)
        for policy in args.policies.split(","):
            sim, res = run_policy(policy, table, arrivals, meta, overrides)
            for phase in [p["name"] for p in meta["phases"]] + ["total"]:
                row = dict(res[phase])
                row["served"] = row["served"] / max(row["offered"], 1)
                row["energy_kj"] = res["energy_kj"] if phase == "total" else None
                rows.setdefault((phase, policy), []).append(row)
            if args.out is not None:
                args.out.mkdir(parents=True, exist_ok=True)
                (args.out / f"{policy}_{seed}_timeline.json").write_text(json.dumps(sim.timeline))
    print(f"mean of {args.seeds} seed(s), load x {args.load_scale} (served = share of offered)")
    print(f"{'phase':>9} {'policy':>15} " + " ".join(f"{k:>11s}" for k in keys))
    for (phase, policy), values in rows.items():
        cells = []
        for k in keys:
            vals = [v[k] for v in values if v.get(k) is not None]
            cells.append(f"{statistics.mean(vals):>11.3f}" if vals else f"{'':>11s}")
        print(f"{phase:>9} {policy:>15} " + " ".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
