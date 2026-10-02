"""Load generator for production-path runs (smoke test 2, end-to-end comparisons).

A profile is a sequence of phases; each phase offers Poisson arrivals at a load
given in units of one group's C_H (the published H capacity), so the same profile
exercises consolidation, waking and boosting on any cluster. The rate follows
the tier table's load unit:

    rate (req/s) = load x C_H / (mean prompt + alpha)

The trace (arrival times and lengths) is generated once and saved; a baseline
run replays the same file, so both runs see identical requests. Prompts are
random token ids regenerated from (seed, index), with `ignore_eos` so every run
does the same work.

While the trace plays, the generator records
* requests.jsonl  one line per request as the client saw it (status, group, TTFT, TPOT),
* energy.jsonl    NVML readings of every GPU from the node agents (1 s period),
* timeline.jsonl  the CanaTune state (groups, tiers, loads) when the policy has one,
* actions.jsonl   profile actions, e.g. asking the Canary for a recheck,
and `summarize` turns these into per-phase outcomes and energy.
"""

import asyncio
import json
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

from canatune.infrastructure.records import JsonlLog
from canatune.proxy.proxy import StreamTimer

ACTIONS = ("canary_recheck",)

# Two groups (Canary G0 + production G1): park -> steady H -> wake -> boost to MAX ->
# unboost and park again -> Canary recheck under light load -> abort it under pressure.
DEFAULT_PROFILE = (
    "low:180:0.3,mid:120:0.6,high:150:1.3,overload:90:2.2,cool:180:0.3,"
    "canary:40:0.4:canary_recheck,pressure:150:1.2,tail:60:0.3"
)


class ProfileError(ValueError):
    """The load profile is malformed."""


@dataclass(frozen=True)
class Phase:
    name: str
    seconds: float
    load: float  # offered load in units of one group's C_H
    action: str | None = None


def parse_profile(text: str) -> list[Phase]:
    """`name:seconds:load[:action],...` -> phases."""
    phases = []
    for item in (part.strip() for part in text.split(",")):
        if not item:
            continue
        fields = item.split(":")
        if len(fields) not in (3, 4):
            raise ProfileError(f"phase must be name:seconds:load[:action]: {item!r}")
        name, seconds, load = fields[0], float(fields[1]), float(fields[2])
        action = fields[3] if len(fields) == 4 else None
        if not name or seconds <= 0 or load < 0:
            raise ProfileError(f"phase needs a name, seconds > 0 and load >= 0: {item!r}")
        if action is not None and action not in ACTIONS:
            raise ProfileError(f"unknown action {action!r}; known: {ACTIONS}")
        phases.append(Phase(name, seconds, load, action))
    if not phases:
        raise ProfileError("empty profile")
    if len({p.name for p in phases}) != len(phases):
        raise ProfileError("phase names must be unique")
    return phases


@dataclass(frozen=True)
class Arrival:
    index: int
    at_s: float  # offset from the trace start
    phase: str
    prompt_tokens: int
    output_tokens: int


def rate_rps(load: float, capacity_h: float, alpha: float, mean_prompt: float) -> float:
    return load * capacity_h / (mean_prompt + alpha)


def build_trace(
    phases: Sequence[Phase],
    pairs: Sequence[tuple[int, int]],
    *,
    capacity_h: float,
    alpha: float,
    seed: int,
) -> tuple[dict[str, Any], list[Arrival]]:
    """Poisson arrivals per phase, (prompt, output) drawn uniformly from `pairs`."""
    if not pairs or capacity_h <= 0:
        raise ProfileError("need length pairs and a positive C_H")
    rng = random.Random(seed)
    mean_prompt = sum(p for p, _ in pairs) / len(pairs)
    arrivals: list[Arrival] = []
    meta_phases = []
    start = 0.0
    for phase in phases:
        rate = rate_rps(phase.load, capacity_h, alpha, mean_prompt)
        end = start + phase.seconds
        t = start
        while rate > 0:
            t += rng.expovariate(rate)
            if t >= end:
                break
            prompt, output = pairs[rng.randrange(len(pairs))]
            arrivals.append(Arrival(len(arrivals), t, phase.name, prompt, output))
        meta_phases.append({**asdict(phase), "start_s": start, "end_s": end, "rate_rps": rate})
        start = end
    meta = {
        "seed": seed,
        "capacity_h": capacity_h,
        "alpha": alpha,
        "mean_prompt": mean_prompt,
        "pairs": [list(p) for p in pairs],
        "phases": meta_phases,
        "duration_s": start,
        "requests": len(arrivals),
    }
    return meta, arrivals


def save_trace(path: str | Path, meta: Mapping[str, Any], arrivals: Sequence[Arrival]) -> None:
    payload = {"meta": dict(meta), "arrivals": [asdict(a) for a in arrivals]}
    Path(path).write_text(json.dumps(payload) + "\n", encoding="utf-8")


def load_trace(path: str | Path) -> tuple[dict[str, Any], list[Arrival]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload["meta"], [Arrival(**a) for a in payload["arrivals"]]


def prompt_ids(seed: int, index: int, n: int, low: int = 1000, high: int = 31000) -> list[int]:
    rng = random.Random(seed * 1_000_003 + index)
    return [rng.randrange(low, high) for _ in range(n)]


def _compact_state(state: Mapping[str, Any]) -> dict[str, Any]:
    router = state.get("router") or {}
    canary = state.get("canary") or {}
    return {
        "groups": [
            {k: g.get(k) for k in ("name", "state", "tier", "effective", "n_inflight", "n_await")}
            for g in router.get("groups", [])
        ],
        "loads": (state.get("controller") or {}).get("loads"),
        "mode": (state.get("controller") or {}).get("mode"),
        "admitted": router.get("admitted"),
        "rejections": router.get("rejections"),
        "canary_running": canary.get("running"),
        "canary_phase": canary.get("phase"),
    }


class LoadRun:
    """Plays one trace against the proxy and records what happens."""

    def __init__(
        self,
        meta: Mapping[str, Any],
        arrivals: Sequence[Arrival],
        *,
        base_url: str,
        model: str,
        out_dir: str | Path,
        agents: Sequence[str] = (),
        endpoint: str = "/v1/completions",
        state_poll: bool = True,
        actions: bool = True,
        poll_period_s: float = 1.0,
        request_timeout_s: float = 300.0,
        drain_timeout_s: float = 180.0,
        client_factory: Callable[[], httpx.AsyncClient] | None = None,
    ) -> None:
        self.meta = dict(meta)
        self.arrivals = list(arrivals)
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.out = Path(out_dir)
        self.agents = list(agents)
        self.endpoint = endpoint
        self.state_poll = state_poll
        self.actions = actions
        self.poll_period_s = poll_period_s
        self.drain_timeout_s = drain_timeout_s
        self.client_factory = client_factory or (
            lambda: httpx.AsyncClient(
                timeout=httpx.Timeout(request_timeout_s, connect=10),
                limits=httpx.Limits(max_connections=None, max_keepalive_connections=64),
                trust_env=False,
            )
        )
        self.out.mkdir(parents=True, exist_ok=True)
        self.requests = JsonlLog(self.out / "requests.jsonl")
        self.energy = JsonlLog(self.out / "energy.jsonl")
        self.timeline = JsonlLog(self.out / "timeline.jsonl")
        self.action_log = JsonlLog(self.out / "actions.jsonl")
        self._t0 = 0.0

    def now_s(self) -> float:
        return time.monotonic() - self._t0

    # ---- one request -------------------------------------------------------------------

    async def send(self, client: httpx.AsyncClient, arrival: Arrival) -> None:
        body = {
            "model": self.model,
            "prompt": prompt_ids(self.meta["seed"], arrival.index, arrival.prompt_tokens),
            "max_tokens": arrival.output_tokens,
            "ignore_eos": True,
            "temperature": 0.0,
            "stream": True,
        }
        sent = self.now_s()
        timer = StreamTimer(clock=self.now_s)
        status, headers = "ok", {}
        try:
            async with client.stream("POST", self.base_url + self.endpoint, json=body) as response:
                headers = response.headers
                if response.status_code == 503:
                    status = "rejected"
                elif response.status_code != 200:
                    status = f"http_{response.status_code}"
                async for chunk in response.aiter_bytes():
                    if status == "ok":
                        timer.feed(chunk)
        except Exception as error:  # a failed request is data
            status = f"error: {type(error).__name__}"
        done = self.now_s()
        times = timer.token_times
        self.requests.write(
            {
                "index": arrival.index,
                "phase": arrival.phase,
                "scheduled_s": arrival.at_s,
                "sent_s": sent,
                "done_s": done,
                "prompt_tokens": arrival.prompt_tokens,
                "output_target": arrival.output_tokens,
                "status": status,
                "group": headers.get("X-CanaTune-Group"),
                "prefill": headers.get("X-CanaTune-Prefill-Endpoint"),
                "decode": headers.get("X-CanaTune-Decode-Endpoint"),
                "ttft_ms": (times[0] - sent) * 1000.0 if times else None,
                "tpot_ms": timer.tpot_ms(),
                "events": len(times),
                "e2e_ms": (done - sent) * 1000.0,
            }
        )

    # ---- background pollers ------------------------------------------------------------

    async def _poll(self, client: httpx.AsyncClient, stop: asyncio.Event) -> None:
        while not stop.is_set():
            t = self.now_s()
            if self.state_poll:
                try:
                    response = await client.get(self.base_url + "/canatune/state", timeout=5)
                    response.raise_for_status()
                    self.timeline.write({"t_s": t, **_compact_state(response.json())})
                except Exception as error:
                    self.timeline.write({"t_s": t, "error": type(error).__name__})
            for agent in self.agents:
                try:
                    response = await client.get(agent.rstrip("/") + "/gpus", timeout=5)
                    response.raise_for_status()
                    gpus = response.json()
                    self.energy.write({"t_s": self.now_s(), "agent": agent, "gpus": gpus})
                except Exception as error:
                    self.energy.write({"t_s": self.now_s(), "agent": agent, "error": repr(error)})
            try:
                await asyncio.wait_for(stop.wait(), timeout=self.poll_period_s)
            except asyncio.TimeoutError:
                pass

    async def _action(self, client: httpx.AsyncClient, phase: Mapping[str, Any]) -> None:
        """Ask the Canary for a recheck; retry every 5 s until it starts or the phase ends."""
        if phase.get("action") != "canary_recheck":
            return
        while self.now_s() < phase["end_s"]:
            try:
                response = await client.post(
                    self.base_url + "/canatune/canary", json={"action": "recheck"}, timeout=5
                )
                result = response.json()
            except Exception as error:
                result = {"ok": False, "reason": repr(error)}
            self.action_log.write(
                {"t_s": self.now_s(), "phase": phase["name"], "action": "canary_recheck", **result}
            )
            if result.get("ok"):
                return
            await asyncio.sleep(5)

    # ---- the run -----------------------------------------------------------------------

    async def run(self) -> dict[str, Any]:
        stop = asyncio.Event()
        pending: set[asyncio.Task] = set()
        started_wall = time.time()
        self._t0 = time.monotonic()
        async with self.client_factory() as client:
            poller = asyncio.create_task(self._poll(client, stop))
            actions = [
                asyncio.create_task(self._at(phase["start_s"], self._action(client, phase)))
                for phase in self.meta["phases"]
                if self.actions and phase.get("action")
            ]
            for arrival in self.arrivals:
                delay = arrival.at_s - self.now_s()
                if delay > 0:
                    await asyncio.sleep(delay)
                task = asyncio.create_task(self.send(client, arrival))
                pending.add(task)
                task.add_done_callback(pending.discard)
            tail = self.meta["duration_s"] - self.now_s()
            if tail > 0:
                await asyncio.sleep(tail)
            played_s = self.now_s()
            if pending:
                await asyncio.wait(set(pending), timeout=self.drain_timeout_s)
            unfinished = len(pending)
            for task in list(pending) + actions:
                task.cancel()
            await asyncio.gather(*pending, *actions, return_exceptions=True)
            stop.set()
            await poller
        run = {
            "started_wall": started_wall,
            "played_s": played_s,
            "finished_s": self.now_s(),
            "unfinished": unfinished,
            "requests": len(self.arrivals),
        }
        (self.out / "run.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
        return run

    async def _at(self, at_s: float, coroutine: Any) -> None:
        delay = at_s - self.now_s()
        if delay > 0:
            await asyncio.sleep(delay)
        await coroutine


# ---- analysis ----------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def _quantile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1) + 0.5))]


def energy_series(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[tuple[float, float]]]:
    """(agent|gpu index) -> [(t_s, energy_mj)] from energy.jsonl rows."""
    series: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        for gpu in row.get("gpus") or []:
            if gpu.get("energy_mj") is None:
                continue
            key = f"{row['agent']}|{gpu.get('index')}"
            series.setdefault(key, []).append((float(row["t_s"]), float(gpu["energy_mj"])))
    for points in series.values():
        points.sort()
    return series


def energy_between(
    points: Sequence[tuple[float, float]], a: float, b: float, slack_s: float = 2.0
) -> float | None:
    """Joules between offsets a and b, interpolating the cumulative counter. Bounds up
    to `slack_s` outside the readings are clamped (the first poll lands just after 0)."""
    if len(points) < 2 or a < points[0][0] - slack_s or b > points[-1][0] + slack_s:
        return None

    def at(t: float) -> float:
        if t <= points[0][0]:
            return points[0][1]
        for (t0, e0), (t1, e1) in zip(points, points[1:]):
            if t0 <= t <= t1:
                return e0 if t1 == t0 else e0 + (e1 - e0) * (t - t0) / (t1 - t0)
        return points[-1][1]

    return (at(b) - at(a)) / 1000.0


def summarize(
    out_dir: str | Path,
    meta: Mapping[str, Any],
    *,
    ttft_slo_ms: float,
    tpot_slo_ms: float,
    gpu_names: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Per-phase outcomes and energy of one run; `gpu_names` maps agent|index -> endpoint."""
    out = Path(out_dir)
    requests = _read_jsonl(out / "requests.jsonl")
    series = energy_series(_read_jsonl(out / "energy.jsonl"))
    names = dict(gpu_names or {})
    windows = [(p["name"], p["start_s"], p["end_s"]) for p in meta["phases"]]
    windows.append(("all", 0.0, float(meta["duration_s"])))
    phases = {}
    for name, start, end in windows:
        rows = [r for r in requests if name == "all" or r["phase"] == name]
        ok = [r for r in rows if r["status"] == "ok"]
        violated = [
            r
            for r in ok
            if r["ttft_ms"] is None
            or r["ttft_ms"] > ttft_slo_ms
            or (r["tpot_ms"] is not None and r["tpot_ms"] > tpot_slo_ms)
        ]
        ttfts = [r["ttft_ms"] for r in ok if r["ttft_ms"] is not None]
        per_gpu = {names.get(k, k): energy_between(v, start, end) for k, v in series.items()}
        known = [e for e in per_gpu.values() if e is not None]
        energy = sum(known) if known and len(known) == len(per_gpu) else None
        groups: dict[str, int] = {}
        for r in ok:
            groups[str(r.get("group"))] = groups.get(str(r.get("group")), 0) + 1
        phases[name] = {
            "seconds": end - start,
            "offered": len(rows),
            "ok": len(ok),
            "rejected": sum(r["status"] == "rejected" for r in rows),
            "errors": sum(r["status"] not in ("ok", "rejected") for r in rows),
            "violated": len(violated),
            "violation_rate": len(violated) / len(ok) if ok else None,
            "ttft_p50_ms": _quantile(ttfts, 0.5),
            "ttft_p95_ms": _quantile(ttfts, 0.95),
            "energy_j": energy,
            "power_w": None if energy is None else energy / (end - start),
            "j_per_ok_request": None if energy is None or not ok else energy / len(ok),
            "energy_by_gpu_j": per_gpu,
            "ok_by_group": groups,
        }
    return {"phases": phases, "meta": {k: v for k, v in meta.items() if k != "pairs"}}
