"""Discrete-event simulation of the P/D pairs, calibrated on r6b / r7, driving the
real CanaTune Router and Controller on a simulated clock.

Compares, on the same arrival trace (the smoke-2 load profile):
  baseline  round robin over all pairs at MAX (default system: no admission)
  reject    CanaTune, state-based admission, overload: reject (503)
  serve     CanaTune, state-based admission, overload: serve (rescue, backfill doomed)
  serve_dispatch   the same, doomed requests dispatched at once (naive serve-all)

Physics per pair (sources in brackets):
  P   batches everything queued (<= 8192 tokens): t0 + beta_f * max(sum L, L*)
      [r6b plateau + slope: t0 ~30 ms, L* ~200, beta 68 / 75 / 113 us/token at
      2520 / 1545 / 1080 MHz]
  KV  L x 131072 bytes; P hands each request's KV to D's receive buffer (1 GB);
      a full buffer makes P retry the send every 50 ms and start nothing else
      [r7: "Peer Out Of Memory" retries, P stalls, congestion collapse]; the KV
      crosses a per-pair link of 3.8 Gb/s [r7 single-pair peak]
  D   iterations of alpha_D + delta_D * running ms [IBM fit, r6b: 1170 MHz
      57 + 1.38 X, 2040 MHz 55 + 1.41 X, 735 MHz 60 + 2.16 X]; a request joins
      at an iteration start once its KV arrived and D's KV cache has room for
      prompt + output (26k tokens: the 22-24 sequence wall of ~1k-token requests;
      then the long mix fails near 2 req/s, the default mix near 5 and the short
      mix runs past 6, as in r7), which frees its buffer; first token one
      iteration later plus a fixed D-first overhead
  fixed proxy / HTTP overhead before P [smoke 2 timing: ~40 ms]
Clock changes are instantaneous and energy is not modelled: group-seconds active
and at MAX are the energy proxies (park and lower clocks save, MAX costs).

  python scripts/analysis/overload_sim.py [--calibrate] [--profile ...] [--out DIR]
"""

import argparse
import asyncio
import heapq
import itertools
import json
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from canatune import loadgen
from canatune.config import load_config
from canatune.controller.router import CanaTuneRouter, RouterSettings, Ticket
from canatune.controller.tier_controller import ControllerSettings, TierController
from canatune.domain.groups import ClockPoint, Group, GroupState, Tier, TierState, TierTable
from canatune.domain.risk import RiskTable
from canatune.infrastructure.clocks import NullClockActuator

KV_BYTES_PER_TOKEN = 131072
KV_BUFFER_BYTES = 1e9
LINK_BYTES_S = 6e9 / 8
SEND_RETRY_S = 0.05
P_T0_MS, P_LSTAR = 30.0, 200
P_BETA_MS = {2520: 0.068, 1545: 0.075, 1080: 0.113}
P_BATCH_TOKENS = 8192
D_ITER = {2040: (55.0, 1.41), 1170: (57.0, 1.38), 735: (60.0, 2.16)}
D_KV_TOKENS = 26000
B_STAR = 40  # Canary's clean D concurrency for the default mix (published B*)
D_FIRST_EXTRA_MS = 60.0
OVERHEAD_MS = 40.0
TTFT_SLO_MS, TPOT_SLO_MS = 1000.0, 200.0

MAX = ClockPoint(2520, 2040)
H = ClockPoint(1545, 1170)
PARK = ClockPoint(1080, 735)
ALPHA_TOKENS = 650.0
LENGTHS = [(128, 64), (512, 64), (1024, 64)]  # canary.default_lengths
# Canary single-request length table at H (r6b/r7 medians at 1545 MHz)
LUT_H = {16: 45, 128: 52, 256: 58, 512: 75, 1024: 98, 2048: 177, 3072: 264}


def nearest(table: dict, mhz: int):
    return table[min(table, key=lambda f: abs(f - mhz))]


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
    status: str = "pending"  # ok / rejected

    @property
    def kv(self) -> float:
        return self.prompt * KV_BYTES_PER_TOKEN


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


class Sim:
    def __init__(
        self,
        n_pairs: int,
        policy: str,
        clocks: ClockPoint = MAX,
        capacity_rps=4.0,
        router_overrides: dict | None = None,
    ):
        self.router_overrides = router_overrides or {}
        self.now = 0.0
        self.events: list = []
        self.seq = itertools.count()
        self.pairs = [Pair(f"G{i}", clocks) for i in range(n_pairs)]
        self.policy = policy
        self.rr = itertools.cycle(self.pairs)
        self.requests: list[Req] = []
        self.timeline: list[dict] = []
        self.router = self.controller = None
        if policy != "baseline":
            self._cantune(capacity_rps)

    # ---- CanaTune --------------------------------------------------------------------

    def _cantune(self, capacity_rps: float) -> None:
        config = load_config()
        config["router"]["admission"] = "slack"
        config["router"]["overload"] = "reject" if self.policy == "reject" else "serve"
        config["router"]["doomed"] = "dispatch" if self.policy == "serve_dispatch" else "backfill"
        config["controller"]["stagger_s"] = 0.0
        config["router"].update(self.router_overrides)
        groups = [Group(p.name, f"P{i}", f"D{i}") for i, p in enumerate(self.pairs)]
        mean_prompt = sum(p for p, _ in LENGTHS) / len(LENGTHS)
        table = TierTable(
            park=PARK,
            h=H,
            capacity_h=capacity_rps * (mean_prompt + ALPHA_TOKENS),
            alpha_tokens=ALPHA_TOKENS,
            decode_max_running=B_STAR,
            published_at=1.0,
            evidence={"alpha_fit": {"prefill_ms_by_length": LUT_H}},
        )
        tiers = TierState(max_point=MAX, table=table)
        risk = RiskTable.from_config(config["risk"], {"sim": True})
        clock = lambda: self.now  # noqa: E731
        self.router = CanaTuneRouter(
            groups, risk, RouterSettings.from_config(config), tiers, clock=clock
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
        for g in groups:
            g.state, g.tier, g.effective = GroupState.ACTIVE, Tier.H, H

    # ---- event loop ------------------------------------------------------------------

    def at(self, t: float, fn, *args) -> None:
        heapq.heappush(self.events, (t, next(self.seq), fn, args))

    def run(self, arrivals, end_s: float) -> None:
        for a in arrivals:
            self.at(
                a.at_s, self.arrive, Req(a.index, a.at_s, a.phase, a.prompt_tokens, a.output_tokens)
            )
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
            row[p.name] = {
                "state": "active" if g is None else g.state.value,
                "tier": "max" if g is None else g.tier.value,
                "buffer_mb": round(p.buffer / 1e6),
                "running": len(p.d_running),
                "p_queue": len(p.p_queue),
            }
        if self.router is not None:
            row["holding"] = len(self.router._holding)
        self.timeline.append(row)
        self.at(self.now + 1.0, self.sample)

    # ---- admission -------------------------------------------------------------------

    def arrive(self, r: Req) -> None:
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
        self.at(self.now + OVERHEAD_MS / 1000.0, self.p_enqueue, r)

    # ---- P ---------------------------------------------------------------------------

    def p_enqueue(self, r: Req) -> None:
        r.pair.p_queue.append(r)
        self.p_start(r.pair)

    def p_start(self, pair: Pair) -> None:
        if pair.p_busy or pair.unsent or not pair.p_queue:
            return
        batch, tokens = [], 0
        while pair.p_queue and (not batch or tokens + pair.p_queue[0].prompt <= P_BATCH_TOKENS):
            r = pair.p_queue.popleft()
            batch.append(r)
            tokens += r.prompt
        beta = nearest(P_BETA_MS, pair.clock.prefill_mhz)
        pair.p_busy = True
        self.at(
            self.now + (P_T0_MS + beta * max(tokens, P_LSTAR)) / 1000.0, self.p_done, pair, batch
        )

    def p_done(self, pair: Pair, batch: list) -> None:
        pair.p_busy = False
        pair.unsent.extend(batch)
        for r in batch:
            if r.ticket is not None:
                self.router.prefill_done(r.ticket)
        self.send(pair)

    def send(self, pair: Pair) -> None:
        while pair.unsent and pair.buffer + pair.unsent[0].kv <= KV_BUFFER_BYTES:
            r = pair.unsent.popleft()
            pair.buffer += r.kv
            start = max(self.now, pair.link_free)
            pair.link_free = start + r.kv / LINK_BYTES_S
            self.at(pair.link_free, self.kv_arrived, r)
        if pair.unsent:  # buffer full: P retries the send and starts nothing else
            pair.send_retries += 1
            self.at(self.now + SEND_RETRY_S, self.send, pair)
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
        while (
            pair.d_ready and used + pair.d_ready[0].prompt + pair.d_ready[0].output <= D_KV_TOKENS
        ):
            r = pair.d_ready.popleft()
            used += r.prompt + r.output
            pair.buffer -= r.kv  # pulled into D's KV cache
            joined.append(r)
        pair.d_running.extend(joined)
        if not pair.d_running:
            pair.d_looping = False
            return
        if joined:
            self.send(pair)  # buffer space for P's pending sends
        a, d = nearest(D_ITER, pair.clock.decode_mhz)
        dur = (a + d * len(pair.d_running)) / 1000.0
        self.at(self.now + dur, self.d_end, pair, joined)

    def d_end(self, pair: Pair, joined: list) -> None:
        for r in list(pair.d_running):
            if r in joined:
                r.first = self.now + D_FIRST_EXTRA_MS / 1000.0
                if r.ticket is not None:
                    self.router.first_token(r.ticket)
            r.emitted += 1
            if r.emitted >= r.output:
                pair.d_running.remove(r)
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
        self.d_iter(pair)

    @staticmethod
    def tpot(r: Req) -> float | None:
        return None if r.output <= 1 else (r.done - r.first) * 1000.0 / (r.output - 1)


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


def calibrate(clock: ClockPoint, rates, seconds: float, seed: int) -> list[dict]:
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
                "running_mean": round(statistics.mean(r["G0"]["running"] for r in sim.timeline)),
            }
        )
    return rows


def multi_seed(args) -> int:
    overrides = {k: json.loads(v) for k, v in (item.split("=", 1) for item in args.set)}
    mean_prompt = sum(p for p, _ in LENGTHS) / len(LENGTHS)
    keys = ("goodput", "served", "rejected", "served_late", "max_group_s", "active_group_s")
    rows: dict = {}
    for seed in range(args.seed, args.seed + args.seeds):
        meta, arrivals = loadgen.build_trace(
            loadgen.parse_profile(args.profile),
            LENGTHS,
            capacity_h=args.capacity_rps * (mean_prompt + ALPHA_TOKENS),
            alpha=ALPHA_TOKENS,
            seed=seed,
        )
        for policy in args.policies.split(","):
            sim = Sim(2, policy, MAX if policy == "baseline" else H, args.capacity_rps, overrides)
            sim.run(arrivals, meta["duration_s"])
            res = summarize(sim, meta["phases"])
            for phase in [p["name"] for p in meta["phases"]] + ["total"]:
                row = dict(res[phase])
                row["served"] = row["served"] / max(row["offered"], 1)
                rows.setdefault((phase, policy), []).append([row[k] for k in keys])
    print(f"C_H {args.capacity_rps} rps, mean of {args.seeds} seeds (served = share of offered)")
    print(f"{'phase':>9} {'policy':>15} " + " ".join(f"{k:>14s}" for k in keys))
    for (phase, policy), values in rows.items():
        means = [statistics.mean(v[i] for v in values) for i in range(len(keys))]
        print(f"{phase:>9} {policy:>15} " + " ".join(f"{m:>14.3f}" for m in means))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--calibrate", action="store_true", help="single-pair rate sweep")
    ap.add_argument("--profile", default=loadgen.DEFAULT_PROFILE)
    ap.add_argument("--capacity-rps", type=float, default=4.0, help="C_H of one group (req/s)")
    ap.add_argument("--seed", type=int, default=20261001)
    ap.add_argument("--policies", default="baseline,reject,serve_dispatch,serve")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--set", action="append", default=[], help="router key=value (JSON value)")
    ap.add_argument("--seeds", type=int, default=1, help="> 1: mean over seeds, short table")
    args = ap.parse_args()
    if args.calibrate:
        for clock in (H, MAX):
            print(f"single pair at {clock.key()}:")
            for row in calibrate(clock, [1, 2, 3, 3.5, 4, 4.5, 5, 6], 120, args.seed):
                print("  ", row)
        return 0
    if args.seeds > 1:
        return multi_seed(args)
    mean_prompt = sum(p for p, _ in LENGTHS) / len(LENGTHS)
    meta, arrivals = loadgen.build_trace(
        loadgen.parse_profile(args.profile),
        LENGTHS,
        capacity_h=args.capacity_rps * (mean_prompt + ALPHA_TOKENS),
        alpha=ALPHA_TOKENS,
        seed=args.seed,
    )
    print(
        f"trace: {len(arrivals)} requests over {meta['duration_s']:.0f} s,"
        f" C_H {args.capacity_rps} rps"
    )
    results = {}
    for policy in args.policies.split(","):
        started = time.monotonic()
        overrides = {k: json.loads(v) for k, v in (item.split("=", 1) for item in args.set)}
        sim = Sim(2, policy, MAX if policy == "baseline" else H, args.capacity_rps, overrides)
        sim.run(arrivals, meta["duration_s"])
        results[policy] = summarize(sim, meta["phases"])
        print(f"{policy}: simulated in {time.monotonic() - started:.0f} s")
        if args.out is not None:
            args.out.mkdir(parents=True, exist_ok=True)
            with (args.out / f"{policy}_requests.jsonl").open("w") as f:
                for r in sim.requests:
                    f.write(
                        json.dumps(
                            {
                                "id": r.id,
                                "at": round(r.at, 3),
                                "phase": r.phase,
                                "prompt": r.prompt,
                                "status": r.status,
                                "group": None if r.pair is None else r.pair.name,
                                "ttft_ms": None
                                if r.first is None
                                else round((r.first - r.at) * 1000),
                                "violated": violated(r),
                                "overflow": None if r.ticket is None else r.ticket.overflow,
                                "predicted_ms": None if r.ticket is None else r.ticket.predicted_ms,
                            }
                        )
                        + "\n"
                    )
            (args.out / f"{policy}_timeline.json").write_text(json.dumps(sim.timeline))
    if args.out is not None:
        (args.out / "summary.json").write_text(
            json.dumps({"meta": meta, "results": results}, indent=1)
        )
    phases = [p["name"] for p in meta["phases"]] + ["total"]
    keys = (
        "offered",
        "served",
        "rejected",
        "served_late",
        "goodput",
        "ttft_p95",
        "overflow",
        "overflow_good",
        "late_ttft_p50",
        "active_group_s",
        "max_group_s",
    )
    for name in phases:
        print(f"\n[{name}]")
        print("  " + " ".join(f"{k:>14s}" for k in ("policy",) + keys))
        for policy, res in results.items():
            row = res[name]
            print("  " + " ".join(f"{str(v):>14s}" for v in [policy] + [row[k] for k in keys]))
    for policy, res in results.items():
        print(
            f"{policy}: send retries {res['send_retries']}",
            res.get("router", {}).get("overflows", ""),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
