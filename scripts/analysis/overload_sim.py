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

L4_E is the hardware of jobs E-G (3 x L40S P, 3 x L4 D, Mistral-7B, vLLM 0.15.1),
from what was measured there (simulator ground truth only; the simulated CanaTune
still measures everything through its own Canary):
  P   S(L) = plateau(f) + slope(f) x max(0, L - knee(f)) per P clock, P power
      idle(f) + utilization x dynamic(f) [job E Canary: prefill_fit, power model]
  KV  the residence (transfer + D's first step + overhead) is ~82-86 ms at every
      prompt length [job E Canary]: a fast link and a fixed D-first overhead
  D   iterations of alpha(f) + beta X + gamma(f) K (X running sequences, K the
      context tokens they hold: prompt + tokens generated) [r6b decode/load/mix
      windows, beta shared: 735 MHz 59.4 + 0.407 X + 1.30e-3 K, 1170 MHz 58.1 +
      0.407 X + 8.4e-4 K, 2040 MHz 55.1 + 0.407 X + 7.7e-4 K, job E production
      within 1.5 ms; 300 MHz from job E's Canary count model], per-request TPOT
      jitter 1 ms (residual spread), KV cache 33000 tokens (vLLM) admitted on
      current usage with preemption and recompute, D power from the job E Canary
  proxy 92 ms + exponential(11 ms) before P in production (admission, HTTP, gap;
      job E's TTFT p50 / p95 at 0.5 req/s); probes go direct

  python scripts/analysis/overload_sim.py [--calibrate] [--profile ...] [--out DIR]
  python scripts/analysis/overload_sim.py --replay JOB/results/ID --physics l4e \
      [--slo 1000,200] [--policies baseline,best_effort]
  (replays the job's own `vllm bench serve` arrivals and lengths per stage and
  prints the simulated stages next to the measured ones)
"""

import argparse
import asyncio
import heapq
import itertools
import json
import random
import statistics
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path

from canatune import loadgen
from canatune.config import load_config
from canatune.controller.locator import (
    AdmissionInputs,
    Hardware,
    LocatorError,
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
from canatune.infrastructure.telemetry import EndpointSnapshot, Telemetry

TTFT_SLO_MS, TPOT_SLO_MS = 1000.0, 200.0


def configure_slo(ttft_ms: float, tpot_ms: float) -> None:
    """SLO of the simulated deployment (summaries and every CanaTune config)."""
    global TTFT_SLO_MS, TPOT_SLO_MS
    TTFT_SLO_MS, TPOT_SLO_MS = float(ttft_ms), float(tpot_ms)


CONFIG_OVERRIDES: dict[str, object] = {}  # dotted key -> value, for every config


def sim_config() -> dict:
    config = load_config()
    config["experiment"]["slo"] = {"ttft_ms": TTFT_SLO_MS, "tpot_ms": TPOT_SLO_MS}
    for key, value in CONFIG_OVERRIDES.items():
        node = config
        *path, last = key.split(".")
        for part in path:
            node = node.setdefault(part, {})
        node[last] = value
    return config


def _interp(table: tuple, f: float, index: int) -> float:
    """Column `index` of rows (f, ...) at clock f: linear, constant beyond the ends."""
    rows = sorted(table)
    if f <= rows[0][0]:
        return rows[0][index]
    for a, b in zip(rows, rows[1:]):
        if f <= b[0]:
            return a[index] + (b[index] - a[index]) * (f - a[0]) / (b[0] - a[0])
    return rows[-1][index]


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
    # Table-driven variants (empty: the formulas above). Rows per clock:
    d_model: tuple = ()  # (f, alpha ms, beta ms/sequence, gamma ms/context token)
    p_model: tuple = ()  # (f, plateau ms, slope ms/token, knee tokens)
    p_power_model: tuple = ()  # (f, idle W, dynamic W at full utilization)
    d_power_model: tuple = ()
    tpot_jitter_ms: float = 0.0  # per-request TPOT noise (sd)
    # D admission: "reserve" (prompt + whole output must fit, the original model) or
    # "vllm" (the prompt fits now; when growth fills the cache the latest sequence is
    # preempted and recomputed on D when it rejoins)
    d_admit: str = "reserve"
    d_recompute_ms_per_token: float = 0.2
    probe_overhead_ms: float | None = None  # Canary probes skip the proxy (None: same)
    overhead_jitter_ms: float = 0.0  # production: + exponential(mean) before P

    @property
    def max_point(self) -> ClockPoint:
        return ClockPoint(self.p_clocks[-1], self.d_clocks[-1])

    def p_beta(self, f: int) -> float:
        return self.p_beta_top_ms * self.p_f_eff / min(f, self.p_f_eff)

    def prefill_ms(self, f: int, tokens: int) -> float:
        if self.p_model:
            plateau, slope, knee = (_interp(self.p_model, f, i) for i in (1, 2, 3))
            return plateau + slope * max(0.0, tokens - knee)
        return self.p_t0_ms + self.p_beta(f) * max(tokens, self.p_lstar)

    def d_iter_ms(self, f: int, running: int, kv_tokens: float = 0.0) -> float:
        if self.d_model:
            alpha, beta, gamma = (_interp(self.d_model, f, i) for i in (1, 2, 3))
            return alpha + beta * running + gamma * kv_tokens
        stretch = max(1.0, self.d_f_knee / f)
        return self.d_alpha_ms * stretch**0.2 + self.d_delta_ms * stretch * running

    def p_power(self, f: int, util: float) -> float:
        if self.p_power_model:
            return _interp(self.p_power_model, f, 1) + util * _interp(self.p_power_model, f, 2)
        x = min(f, self.p_clocks[-1]) / self.p_clocks[-1]
        return self.p_idle_w * (0.75 + 0.25 * x) + util * self.p_dyn_w * x**2.4

    def d_power(self, f: int, util: float) -> float:
        if self.d_power_model:
            return _interp(self.d_power_model, f, 1) + util * _interp(self.d_power_model, f, 2)
        x = f / self.d_clocks[-1]
        return self.d_idle_w * (0.75 + 0.25 * x) + util * self.d_dyn_w * x**2.0


REFERENCE = Physics()  # an example environment (what the simulated hardware is)
MAX = REFERENCE.max_point
LENGTHS = [(128, 64), (512, 64), (1024, 64)]  # the request length mix

# The hardware of jobs E-G (sources in the module docstring).
L4_E = Physics(
    name="l4e",
    link_bytes_s=60e9,  # residence barely grows with the prompt (82 -> 86 ms)
    d_first_extra_ms=27.0,  # residence ~82 ms = transfer + one D step (~55 ms) + this
    d_kv_tokens=33000,
    overhead_ms=92.0,  # with the jitter: job E's TTFT p50 / p95 at 0.5 req/s
    overhead_jitter_ms=11.0,
    probe_overhead_ms=0.0,
    p_clocks=(600, 840, 1080, 1320, 1560, 1800, 2040, 2280, 2520),
    p_f_eff=2520,  # no power cap under load [job E/F Canary: f_eff 2520]
    d_clocks=(300, 735, 1170, 1605, 2040),  # the Canary's coarse D points on the L4
    p_model=(
        (600, 50.27, 0.19526, 156),
        (1080, 45.19, 0.09723, 38),
        (1560, 47.99, 0.06802, 0),
        (2040, 47.99, 0.0546, 0),
        (2520, 47.19, 0.0546, 0),
    ),
    # 300 MHz: job E's Canary count model there (82.1 + 5.37 X, 3.85 x the 1170
    # slope), split like 1170; the other rows from r6b
    d_model=((300, 82.1, 1.57, 3.2e-3), (735, 59.39, 0.407, 1.30e-3),
             (1170, 58.10, 0.407, 8.39e-4), (2040, 55.09, 0.407, 7.73e-4)),  # fmt: skip
    p_power_model=((600, 58.6, 111.7), (1080, 60.3, 161.2), (1560, 61.9, 183.8),
                   (2040, 64.2, 205.8), (2520, 75.1, 218.5)),  # fmt: skip
    d_power_model=((300, 20.8, 24.5), (735, 21.2, 38.0), (1170, 21.9, 42.8), (1605, 24.8, 47.5),
                   (2040, 28.5, 43.8)),  # fmt: skip
    tpot_jitter_ms=1.0,
    d_admit="vllm",
)
# The same cluster with D measured in situ: the Canary fits of job G (and, nearly
# identical, H) at the two clocks production uses, from 336 / 219 probe requests
# over 1.5-44 sequences and 1.8k-26k context tokens (corr 0.51 / 0.77).
L4_G = replace(
    L4_E,
    name="l4g",
    d_model=((300, 82.1, 1.57, 3.2e-3), (735, 59.39, 0.407, 1.30e-3),
             (1170, 57.73, 0.192, 1.007e-3), (2040, 56.16, 0.272, 7.11e-4)),  # fmt: skip
)
PHYSICS = {"reference": REFERENCE, "l4e": L4_E, "l4g": L4_G}


@dataclass
class Req:
    id: int
    at: float
    phase: str
    prompt: int
    output: int
    waiter: int = 0
    ticket: Ticket | None = None
    pair: "Pair | None" = None  # its P (and D with fixed pairing)
    d_pair: "Pair | None" = None  # its D
    first: float | None = None
    done: float | None = None
    emitted: int = 0
    last_token: float | None = None  # time of the last emitted token
    status: str = "pending"  # ok / rejected
    kv: float = 0.0
    state: tuple = ()  # (at P prompts, in-flight tokens, decoding) at dispatch
    worker: int | None = None  # closed loop: the worker that sends the next one
    jitter: float = 0.0  # TPOT noise of this request (ms)
    x_sum: float = 0.0  # running sequences x seconds while it decoded
    k_sum: float = 0.0  # context tokens on D x seconds while it decoded
    t_sum: float = 0.0
    pulled: bool = False  # KV moved from D's receive buffer into its cache
    recompute: bool = False  # preempted: D recomputes its context when it rejoins


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
    retry_scheduled: bool = False
    at_p: dict = field(default_factory=dict)  # request id -> prompt (dispatch .. P done)
    inflight_tokens: int = 0  # P done .. first token
    decoding: int = 0
    p_busy_s: float = 0.0
    d_busy_s: float = 0.0
    d_running_integral: float = 0.0  # running sequences x seconds (mean while busy)
    d_waiting_max: int = 0
    d_running_max: int = 0
    d_kv_max: float = 0.0
    preemptions: int = 0


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
        overhead_ms: float | None = None,
        solver: bool = True,
        lengths=LENGTHS,
        seed: int = 0,
    ):
        self.phys = phys
        self.overhead_ms = phys.overhead_ms if overhead_ms is None else overhead_ms
        self.solver = solver
        self.lengths = list(lengths)
        self.rng = random.Random(seed)
        self.telemetry: Telemetry | None = None
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
        config = sim_config()
        config["router"]["admission"] = "slack"
        config["router"]["overload"] = {"reject": "reject", "best_effort": "best_effort"}.get(
            self.policy, "serve"
        )
        config["router"]["doomed"] = "dispatch" if self.policy == "serve_dispatch" else "backfill"
        config["controller"]["stagger_s"] = 0.0
        # The solver runs whenever the table carries a cluster model; the static
        # comparison passes a table without one (selfcal_sim.static).
        config["controller"]["solver"] = self.solver
        config["kv_transfer"]["kv_buffer_bytes"] = self.phys.kv_buffer_bytes
        config["kv_transfer"]["kv_bytes_per_token"] = self.phys.kv_bytes_per_token
        config["router"].update(self.router_overrides)
        groups = [Group(p.name, f"P{i}", f"D{i}") for i, p in enumerate(self.pairs)]
        tiers = TierState(max_point=self.phys.max_point, table=table)
        risk = RiskTable.from_config(config["risk"], {"sim": True})
        clock = lambda: self.now  # noqa: E731
        lengths = LengthStats(self.lengths, min_samples=50)
        # D's /metrics as the proxy scrapes them (running, waiting, KV usage).
        self.telemetry = Telemetry(
            {}, period_s=0.25, client_factory=lambda: None, clock=clock
        )
        self.router = CanaTuneRouter(
            groups, risk, RouterSettings.from_config(config), tiers, clock=clock,
            lengths=lengths, telemetry=self.telemetry,
        )  # fmt: skip
        self.control_log: list[dict] = []
        sim = self

        class _Log:  # the Controller's mode changes, with simulated time
            def write(self, record: dict) -> None:
                if record.get("event") == "control_mode":
                    sim.control_log.append({"t": round(sim.now, 1), **record})

        self.controller = TierController(
            groups,
            self.router,
            ControllerSettings.from_config(config),
            NullClockActuator(),
            {},
            tiers,
            clock=clock,
            log=_Log(),
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
            self.at(0.0, self.scrape)
        self.at(0.0, self.sample)
        while self.events:
            t, _, fn, args = heapq.heappop(self.events)
            if t > end_s + 120 and fn in (self.tick, self.sample, self.scrape):
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

    @staticmethod
    def kv_used(pair: Pair) -> float:
        """Context tokens D holds: prompt plus the tokens generated so far."""
        return sum(r.prompt + r.emitted for r in pair.d_running)

    def scrape(self) -> None:
        for i, p in enumerate(self.pairs):
            snapshot = EndpointSnapshot(
                self.now, float(len(p.d_running)), float(len(p.d_ready)),
                self.kv_used(p) / self.phys.d_kv_tokens, float(p.preemptions), True,
            )  # fmt: skip
            self.telemetry.update(f"D{i}", snapshot)
        self.at(self.now + self.telemetry.period_s, self.scrape)

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
                "kv": round(self.kv_used(p) / self.phys.d_kv_tokens, 3),
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
            p_w, d_w = self.phys.p_power(f_p, p_util), self.phys.d_power(f_d, d_util)
            self.energy_j += p_w + d_w
            row["w"] = row.get("w", 0.0) + p_w + d_w
            row["wp"] = row.get("wp", 0.0) + p_w
            row["wd"] = row.get("wd", 0.0) + d_w
        if self.router is not None:
            row["holding"] = len(self.router._holding)
        self.timeline.append(row)
        self.at(self.now + 1.0, self.sample)

    # ---- admission -------------------------------------------------------------------

    def arrive(self, r: Req) -> None:
        r.kv = r.prompt * self.phys.kv_bytes_per_token
        if self.phys.tpot_jitter_ms:
            r.jitter = self.rng.gauss(0.0, self.phys.tpot_jitter_ms)
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
            by_name = {p.name: p for p in self.pairs}
            self.dispatch(r, by_name[result.group.name], by_name[result.dgroup.name])
        elif result == "reject":
            r.status = "rejected"
        else:
            self.at(self.now + self.router.settings.retry_period_ms / 1000.0, self.retry, r)

    def dispatch(self, r: Req, pair: Pair, d_pair: Pair | None = None) -> None:
        r.pair = pair
        r.d_pair = pair if d_pair is None else d_pair
        r.state = (tuple(pair.at_p.values()), pair.inflight_tokens, r.d_pair.decoding)
        pair.at_p[r.id] = r.prompt
        delay = self.overhead_ms
        if self.phys.overhead_jitter_ms and self.overhead_ms > 0:
            delay += self.rng.expovariate(1.0 / self.phys.overhead_jitter_ms)
        self.at(self.now + delay / 1000.0, self.p_enqueue, r)

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

    def send(self, pair: Pair, retry: bool = False) -> None:
        """P sends finished KV in order over its link into each request's D receive
        buffer; a full buffer at the head blocks P (it retries and starts nothing)."""
        if retry:
            pair.retry_scheduled = False
        while pair.unsent:
            dest = pair.unsent[0].d_pair
            if dest.buffer + pair.unsent[0].kv > self.phys.kv_buffer_bytes:
                break
            r = pair.unsent.popleft()
            dest.buffer += r.kv
            start = max(self.now, pair.link_free)
            pair.link_free = start + r.kv / self.phys.link_bytes_s
            self.at(pair.link_free, self.kv_arrived, r)
        if pair.unsent:  # buffer full: P retries the send and starts nothing else
            if not pair.retry_scheduled:
                pair.send_retries += 1
                pair.retry_scheduled = True
                self.at(self.now + self.phys.send_retry_s, self.send, pair, True)
        else:
            self.p_start(pair)

    # ---- D ---------------------------------------------------------------------------

    def kv_arrived(self, r: Req) -> None:
        d = r.d_pair
        d.d_ready.append(r)
        if not d.d_looping:
            d.d_looping = True
            self.d_iter(d)

    def d_iter(self, pair: Pair) -> None:
        joined = []
        cap = self.phys.d_kv_tokens
        vllm = self.phys.d_admit == "vllm"
        if vllm:  # the context it holds now, within vLLM's 1 % watermark
            used, limit = self.kv_used(pair), 0.99 * cap

            def need(r):
                return r.prompt + r.emitted
        else:
            used, limit = sum(r.prompt + r.output for r in pair.d_running), cap

            def need(r):
                return r.prompt + r.output

        while pair.d_ready and used + need(pair.d_ready[0]) <= limit:
            r = pair.d_ready.popleft()
            used += need(r)
            if not r.pulled:
                pair.buffer -= r.kv  # pulled into D's KV cache
                r.pulled = True
            joined.append(r)
        pair.d_running.extend(joined)
        recompute_s = 0.0
        if vllm:
            # The step appends a token per sequence: preempt the latest until it fits.
            while pair.d_running and self.kv_used(pair) + len(pair.d_running) > cap:
                r = pair.d_running.pop()
                if r in joined:
                    joined.remove(r)
                r.recompute = True
                pair.d_ready.appendleft(r)
                pair.preemptions += 1
            redo = [r for r in joined if r.recompute]
            recompute_s = sum(r.prompt + r.emitted for r in redo) * (
                self.phys.d_recompute_ms_per_token / 1000.0
            )
            for r in redo:
                r.recompute = False
            used = self.kv_used(pair)
        pair.d_waiting_max = max(pair.d_waiting_max, len(pair.d_ready))
        pair.d_running_max = max(pair.d_running_max, len(pair.d_running))
        pair.d_kv_max = max(pair.d_kv_max, used / cap)
        if not pair.d_running:
            pair.d_looping = False
            return
        if joined:
            for p in self.pairs:  # buffer space for the P's pending sends to this D
                if p.unsent and p.unsent[0].d_pair is pair:
                    self.send(p)
        held = self.kv_used(pair)
        n = len(pair.d_running)
        dur = self.phys.d_iter_ms(pair.clock.decode_mhz, n, held) / 1000.0 + recompute_s
        pair.d_busy_s += dur
        pair.d_running_integral += dur * n
        for r in pair.d_running:
            if r not in joined:  # decode steps only (the join step yields the first token)
                r.x_sum += n * dur
                r.k_sum += held * dur
                r.t_sum += dur
        self.at(self.now + dur, self.d_end, pair, joined)

    def d_end(self, pair: Pair, joined: list) -> None:
        for r in list(pair.d_running):
            if r in joined and r.first is None:
                r.first = self.now + self.phys.d_first_extra_ms / 1000.0
                r.pair.inflight_tokens -= r.prompt
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
        if r.output <= 1:
            return None
        return (r.done - r.first) * 1000.0 / (r.output - 1) + r.jitter

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
        kvs = [row[n].get("kv", 0.0) for row in t for n in names]
        out[ph["name"]]["kv_max"] = max(kvs, default=0.0)
        out[ph["name"]]["kv_full_s"] = sum(k >= 0.98 for k in kvs)  # D-seconds near full
        tpots = [t for t in (sim.tpot(r) for r in served) if t is not None]
        out[ph["name"]].update(
            {
                "tpot_p50": pct(tpots, 0.5),
                "tpot_p95": pct(tpots, 0.95),
                "ttft_late": sum((r.first - r.at) * 1000 > TTFT_SLO_MS for r in served),
                "tpot_late": sum((sim.tpot(r) or 0) > TPOT_SLO_MS for r in served),
                "best_effort": sum(
                    r.ticket is not None and r.ticket.overflow == "best_effort" for r in served
                ),
            }
        )
        if ph["name"] != "total" and served:
            # Energy over the stage's own span (first arrival to last completion), as
            # the benchmark driver measures it.
            start, end = min(r.at for r in rows), max(r.done for r in served)
            span = [row for row in sim.timeline if start <= row["t"] < end]
            for key, name in (("w", "energy_kj"), ("wp", "p_energy_kj"), ("wd", "d_energy_kj")):
                out[ph["name"]][name] = sum(row.get(key, 0.0) for row in span) / 1000
    out["send_retries"] = sum(p.send_retries for p in sim.pairs)
    out["preemptions"] = sum(p.preemptions for p in sim.pairs)
    out["control"] = getattr(sim, "control_log", [])
    if sim.router is not None:
        out["router"] = {k: v for k, v in sim.router.state().items() if k != "groups"}
    return out


class SimBackend:
    """`ProbeBackend` over one simulated pair: the real tier locator measures the
    simulated hardware through it exactly as through the Canary probe (windows,
    per-request state samples, average power, D running sequences)."""

    service_source = "metrics"

    decode_output_tokens = 256  # closed windows, as the probe

    def __init__(self, phys: Physics, lengths=LENGTHS, seed: int = 7, salt: int = 0) -> None:
        self.phys = phys
        self.salt = salt  # another attempt draws other window traces
        self.lengths = list(lengths)
        self.rng = random.Random(seed)
        self.limit: int | None = None
        self.windows = 0
        self.sim_seconds = 0.0

    def _pairs(self) -> list[tuple[int, int]]:
        pairs = [p for p in self.lengths if self.limit is None or p[0] <= self.limit]
        return pairs or self.lengths[:1]

    def _mix(self, mix: str) -> list[int]:
        """Prompts of a closed-window mix, as the probe: lower / upper half."""
        prompts = sorted(p for p, _ in self._pairs())
        if mix == "short":
            return prompts[: (len(prompts) + 1) // 2]
        if mix == "long":
            return prompts[len(prompts) // 2 :]
        return prompts

    def mix_context(self, mix: str = "all") -> float:
        prompts = self._mix(mix)
        return sum(prompts) / len(prompts) + self.decode_output_tokens / 2

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
                + self.phys.d_iter_ms(clock.decode_mhz, 1, n)
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
            decode_preemptions=float(pair.preemptions),
            decode_kv_max=pair.d_kv_max,
            decode_running_max=float(pair.d_running_max),
            samples=[s.probe_sample(r) for r in reqs],
            prefill_avg_w=p_j / duration,
            decode_avg_w=d_j / duration,
            decode_busy_fraction=busy,
            decode_running_mean=running / busy if busy > 0 else None,
            tpot_p50_ms=tpots[len(tpots) // 2] if tpots else None,
            decode_points=[
                (r.x_sum / r.t_sum, r.k_sum / r.t_sum, s.tpot(r))
                for r in reqs
                if r.t_sum > 0 and s.tpot(r) is not None
            ],
        )

    def _window_sim(self, clock: ClockPoint) -> "Sim":
        return Sim(1, "baseline", clock, phys=self.phys, overhead_ms=self.phys.probe_overhead_ms,
                   seed=self.rng.randrange(10**9))  # fmt: skip

    async def open_window(self, clock, eq_tps, alpha, seconds, abort_above):
        trace = random.Random(int(eq_tps * 10) + 17 + 1_000_003 * self.salt)
        pairs = self._pairs()
        mean = sum(p for p, _ in pairs) / len(pairs)
        count = max(1, round(eq_tps / (mean + alpha) * seconds))
        arrivals = [
            loadgen.Arrival(i, t, "w", *pairs[trace.randrange(len(pairs))])
            for i, t in enumerate(sorted(trace.uniform(0, seconds) for _ in range(count)))
        ]
        s = self._window_sim(clock)
        s.run(arrivals, seconds)
        return self._result(s, clock, "open", eq_tps)

    async def closed_window(self, clock, concurrency, seconds, rep=0, mix="all"):
        salt = {"all": 0, "short": 1, "long": 2}[mix]
        trace = random.Random(concurrency * 101 + rep + 1_000_003 * salt + 7_919 * self.salt)
        prompts = self._mix(mix)
        s = self._window_sim(clock)
        ids = iter(range(10**9))
        stagger = min(15.0, seconds / 3)

        def send(worker: int, at: float) -> None:
            if at >= seconds:
                return
            prompt = prompts[trace.randrange(len(prompts))]
            out = self.decode_output_tokens
            s.at(at, s.arrive, Req(next(ids), at, "w", prompt, out, worker=worker))

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
    prompts = sorted({p for p, _ in lengths})
    for attempt in range(3):  # as the service: a failed calibration runs again
        backend = SimBackend(phys, lengths, seed=7 + attempt, salt=attempt)
        locator = TierLocator(
            backend,
            LocatorSettings.from_config(sim_config()),
            admission=AdmissionInputs(
                kv_bytes_per_token=phys.kv_bytes_per_token, kv_buffer_bytes=phys.kv_buffer_bytes
            ),
        )
        try:
            table = asyncio.run(locator.locate(sum(prompts) / len(prompts), prompts))
        except LocatorError as error:
            print(f"Canary attempt {attempt + 1} failed: {error}")
            continue
        return table, backend
    raise LocatorError("calibration failed (3 attempts)")


def trace(table: TierTable, profile: str, seed: int, lengths=LENGTHS):
    """The load profile in units of the Canary's measured per-group capacity."""
    return loadgen.build_trace(
        loadgen.parse_profile(profile),
        lengths,
        capacity_h=table.capacity_h,
        alpha=table.alpha_tokens,
        seed=seed,
    )


def replay_trace(run_dir: Path) -> tuple[dict, list]:
    """Arrivals and lengths of a run's `vllm bench serve` stages (bench_plan.json,
    stages/<name>/bench.json: start_times, input_lens, output_lens), on the run's own
    timeline (gaps included)."""
    plan = json.loads((run_dir / "bench_plan.json").read_text())
    stages = []
    for st in plan["stages"]:
        path = run_dir / "stages" / st["name"] / "bench.json"
        if path.exists():
            stages.append((st["name"], json.loads(path.read_text())))
    t0 = min(min(b["start_times"]) for _, b in stages)
    rows, phases = [], []
    for name, b in stages:
        starts = [t - t0 for t in b["start_times"]]
        for at, prompt, output in zip(starts, b["input_lens"], b["output_lens"]):
            if output >= 1:
                rows.append((at, name, int(prompt), int(output)))
        phases.append({"name": name, "start_s": min(starts), "end_s": max(starts)})
    rows.sort()
    arrivals = [loadgen.Arrival(i, *row) for i, row in enumerate(rows)]
    return {"phases": phases, "duration_s": phases[-1]["end_s"]}, arrivals


def run_pairs(run_dir: Path) -> list[tuple[int, int]]:
    """(prompt, output) tokens of a run's benchmark requests."""
    pairs = []
    for st in json.loads((run_dir / "bench_plan.json").read_text())["stages"]:
        path = run_dir / "stages" / st["name"] / "bench.json"
        if path.exists():
            b = json.loads(path.read_text())
            pairs += [(int(i), int(o)) for i, o in zip(b["input_lens"], b["output_lens"]) if o >= 1]
    return pairs


def synthetic_trace(run_dir: Path, rates, stage_s: float, gap_s: float, seed: int):
    """Poisson stages at other rates with the run's own (prompt, output) pairs."""
    return pairs_trace(run_pairs(run_dir), rates, stage_s, gap_s, seed)


def pairs_trace(pairs, rates, stage_s: float, gap_s: float, seed: int):
    """Poisson stages at `rates`, (prompt, output) drawn from `pairs`."""
    rng = random.Random(seed)
    rows, phases, t = [], [], 0.0
    for k, rate in enumerate(rates):
        name, at, start = f"s{k}_{rate:g}rps", t, t
        while True:
            at += rng.expovariate(rate)
            if at >= start + stage_s:
                break
            rows.append((at, name, *pairs[rng.randrange(len(pairs))]))
        phases.append({"name": name, "start_s": start, "end_s": start + stage_s})
        t = start + stage_s + gap_s + 30.0  # the stage drains before the gap
    arrivals = [loadgen.Arrival(i, *row) for i, row in enumerate(rows)]
    return {"phases": phases, "duration_s": phases[-1]["end_s"]}, arrivals


# ---- BurstGPT (Azure OpenAI traces: timestamp, model, request / response tokens) ----


def burstgpt_rows(paths, model: str, max_len: int):
    """-> (timestamps s, prompt tokens, output tokens) of the requests that returned
    tokens and fit max_len, for model "GPT-4", "ChatGPT" or "all"; also the dropped
    share (too long). Parts 1, 2, ... share one clock (part 2 continues part 1)."""
    import csv

    import numpy as np

    t, prompt, output = [], [], []
    long = kept = 0
    for path in paths:
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                ts = float(row["Timestamp"])
                if model != "all" and row["Model"] != model:
                    continue
                n_in, n_out = int(row["Request tokens"]), int(row["Response tokens"])
                if n_out <= 0:
                    continue
                if n_in + n_out > max_len:
                    long += 1
                    continue
                kept += 1
                t.append(ts)
                prompt.append(n_in)
                output.append(n_out)
    order = np.argsort(np.array(t), kind="stable")
    return (np.array(t)[order], np.array(prompt)[order], np.array(output)[order],
            long / max(long + kept, 1))  # fmt: skip


def burstgpt_window_trace(t, prompt, output, hours: float, rate: float | None, segment_s: float):
    """The busiest `hours` of the trace, replayed with its own arrival times, time
    compressed to a mean `rate` (None: real time); phases of `segment_s` simulated
    seconds."""
    import numpy as np

    span = hours * 3600.0
    starts = np.arange(t.min(), t.max() - span + 1, 600.0)  # 10-minute steps
    counts = np.searchsorted(t, starts + span) - np.searchsorted(t, starts)
    s0 = float(starts[int(np.argmax(counts))])
    sel = (t >= s0) & (t < s0 + span)
    natural = sel.sum() / span
    speed = 1.0 if rate is None else rate / natural
    at = (t[sel] - s0) / speed
    dur = span / speed
    n_seg = max(1, int(np.ceil(dur / segment_s)))
    phases = [{"name": f"w{k:02d}", "start_s": k * segment_s,
               "end_s": min(dur, (k + 1) * segment_s)} for k in range(n_seg)]  # fmt: skip
    rows = [(float(a), phases[min(int(a // segment_s), n_seg - 1)]["name"], int(p), int(o))
            for a, p, o in zip(at, prompt[sel], output[sel])]  # fmt: skip
    arrivals = [loadgen.Arrival(i, *row) for i, row in enumerate(rows)]
    info = {"window_start_s": s0, "requests": int(sel.sum()), "natural_rps": float(natural),
            "speed": float(speed), "duration_s": float(dur)}  # fmt: skip
    return {"phases": phases, "duration_s": dur, "window": info}, arrivals


def measured_stages(run_dir: Path) -> dict:
    path = run_dir / "bench_summary.json"
    return json.loads(path.read_text())["stages"] if path.exists() else {}


def run_replay(args) -> int:
    """The job's benchmark on the simulated hardware: Canary (cold start lengths, as
    the job), then CanaTune best effort and the baseline on the same arrivals."""
    phys = PHYSICS[args.physics]
    job = args.replay
    if args.burstgpt:
        t, prompt, output, dropped = burstgpt_rows(
            [Path(p) for p in args.burstgpt.split(",")], args.bgpt_model, args.max_len
        )
        print(f"BurstGPT {args.bgpt_model}: {len(t)} requests (dropped {dropped:.2%} longer"
              f" than {args.max_len}); prompt mean {prompt.mean():.0f},"
              f" output mean {output.mean():.0f}")  # fmt: skip
        if args.bgpt_hours:
            meta, arrivals = burstgpt_window_trace(
                t, prompt, output, args.bgpt_hours, args.bgpt_rate, args.segment_s
            )
            print(f"window: {meta['window']}")
        else:
            rates = [float(x) for x in (args.rates or "1,2,4").split(",")]
            meta, arrivals = pairs_trace(list(zip(prompt.tolist(), output.tolist())), rates,
                                         180.0, 30.0, args.seed)  # fmt: skip
    elif args.rates:
        rates = [float(x) for x in args.rates.split(",")]
        meta, arrivals = synthetic_trace(job / "cantune", rates, 180.0, 30.0, args.seed)
    else:
        meta, arrivals = replay_trace(job / "cantune")
    cold = [(p, args.output_cap) for p in (128, 512, 1024, 2048)]
    table, backend = run_canary(phys, cold)
    d = table.evidence.get("decode") or {}
    print(f"Canary: {backend.windows} windows; H {table.h.key()}, B* {table.decode_max_running},"
          f" decode model {d.get('decode_model', 'count')}")  # fmt: skip
    for f, coef in (table.decode_length or {}).items():
        cap = (d.get("capacity") or {}).get(f) or {}
        seqs = cap.get("sequences") or 0.0
        print(f"  D {f} MHz: alpha {coef[0]:.2f} beta {coef[1]:.3f} gamma {coef[2]:.2e}"
              f" r95 {coef[3]:.2f}; capacity {seqs:.1f} ({cap.get('limit')})")  # fmt: skip
    results = {}
    for policy in args.policies.split(","):
        clocks = phys.max_point if policy == "baseline" else table.h
        sim = Sim(args.groups, policy, clocks, phys=phys, table=None if policy == "baseline"
                  else table, solver=args.solver, lengths=cold, seed=args.seed)  # fmt: skip
        sim.run(arrivals, meta["duration_s"])
        results[policy] = summarize(sim, meta["phases"])
        if args.out is not None:
            args.out.mkdir(parents=True, exist_ok=True)
            (args.out / f"replay_{policy}.json").write_text(json.dumps(results[policy], indent=1))
    measured = {} if job is None else {"best_effort": measured_stages(job / "cantune"),
                                       "baseline": measured_stages(job / "baseline")}  # fmt: skip
    keys = ("goodput", "ttft_p50", "ttft_p95", "tpot_p50", "tpot_p95", "energy_kj")
    real_keys = ("goodput", "ttft_p50_ms", "ttft_p95_ms", "tpot_p50_ms", "tpot_p95_ms", "energy_j")
    print(f"SLO {TTFT_SLO_MS:.0f} / {TPOT_SLO_MS:.0f} ms; sim | measured")
    print(f"{'stage':>11} {'policy':>11} " + " ".join(f"{k:>17s}" for k in keys) + "  warn")
    for ph in [p["name"] for p in meta["phases"]]:
        for policy, res in results.items():
            row, real = res[ph], measured.get(policy, {}).get(ph, {})
            cells = []
            for k, rk in zip(keys, real_keys):
                sim_v, real_v = row.get(k), real.get(rk)
                if rk == "energy_j" and real_v is not None:
                    real_v /= 1000
                cells.append(f"{_num(sim_v):>8}|{_num(real_v):<8}")
            print(f"{ph:>11} {policy:>11} " + " ".join(cells) + f"  {row.get('best_effort', 0)}")
    if {"baseline", "best_effort"} <= set(results):
        print("energy vs baseline (sim | measured):")
        for ph in [p["name"] for p in meta["phases"]]:
            sim_e = [results[k][ph].get("energy_kj") for k in ("best_effort", "baseline")]
            real_e = [measured.get(k, {}).get(ph, {}).get("energy_j")
                      for k in ("best_effort", "baseline")]  # fmt: skip
            print(f"  {ph:>11}  {_ratio(*sim_e)} | {_ratio(*real_e)}")
        totals = {k: sum(results[k][p["name"]].get("energy_kj") or 0 for p in meta["phases"])
                  for k in ("best_effort", "baseline")}  # fmt: skip
        good = {k: results[k]["total"]["goodput"] for k in ("best_effort", "baseline")}
        modes = [(c["t"], c["mode"], c.get("reason"))
                 for c in results["best_effort"].get("control", [])]  # fmt: skip
        print(f"total: energy {_ratio(totals['best_effort'], totals['baseline'])}, goodput"
              f" {good['best_effort']:.3f} vs {good['baseline']:.3f}; Controller: {modes[:12]}")
    return 0


def _num(v) -> str:
    if v is None:
        return "-"
    return f"{v:.3f}" if isinstance(v, float) and v <= 1.0 else f"{v:.0f}"


def _ratio(a, b) -> str:
    return "-" if not a or not b else f"{(a / b - 1) * 100:+.1f}%"


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
    ap.add_argument("--replay", type=Path, default=None, help="job results/<id> dir to replay")
    ap.add_argument("--physics", choices=sorted(PHYSICS), default="reference")
    ap.add_argument("--slo", default=None, help="TTFT,TPOT ms (default 1000,200)")
    ap.add_argument("--groups", type=int, default=None, help="pairs (replay default 3)")
    ap.add_argument("--solver", action="store_true", help="replay: controller.solver on")
    ap.add_argument("--output-cap", type=int, default=256, help="replay: canary.max_output_tokens")
    ap.add_argument("--config", action="append", default=[], help="dotted.key=value (JSON value)")
    ap.add_argument("--rates", default=None, help="replay: Poisson stages at these req/s instead")
    ap.add_argument("--burstgpt", default=None, help="BurstGPT CSV(s), comma separated, in order")
    ap.add_argument("--bgpt-model", default="GPT-4", choices=("GPT-4", "ChatGPT", "all"))
    ap.add_argument("--bgpt-hours", type=float, default=None,
                    help="replay the busiest window of this many hours (else Poisson --rates)")
    ap.add_argument("--bgpt-rate", type=float, default=None,
                    help="window: compress time to this mean req/s (default: real time)")
    ap.add_argument("--segment-s", type=float, default=120.0, help="window: phase length")
    ap.add_argument("--max-len", type=int, default=4096, help="model max length (prompt+output)")
    args = ap.parse_args()
    for item in args.config:
        key, value = item.split("=", 1)
        CONFIG_OVERRIDES[key] = json.loads(value)
    if args.slo:
        configure_slo(*(float(x) for x in args.slo.split(",")))
    if args.replay is not None or args.burstgpt:
        args.groups = args.groups or 3
        if args.policies == "baseline,reject,serve_dispatch,serve":
            args.policies = "baseline,best_effort"
        return run_replay(args)
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
