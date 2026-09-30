"""Model measurements on the P/D pairs (experiments E1 + E2).

Outside the CanaTune control loop (no Router, no Controller): the script locks the
pairs' clocks through the node agents and sends requests straight to vLLM, timing
every stage itself, so that the parameters of the TTFT and energy models can be
fitted:

  TTFT = fixed + P queue(load) + S_f(L) + KV transfer(L, bytes in flight) + D first token

Stages, each a list of windows (one window = one clock setting and one workload):
  warmup  open-loop traffic after a (re)start, never recorded as done: the first
          ~60 s after start-up are slower (r6b)
  lut     single sequential requests over a length grid per P clock  -> S_f(L)
  power   idle power at each P and each D clock                      -> P_idle(f)
  decode  closed loop at several concurrencies per D clock           -> T_iter(X), D power
  load    open-loop rate scans per P clock, default mix, one pair    -> queueing, stages
  mix     open-loop rate scans of other length mixes                 -> cross-mix capacity
  scans   explicit scans: {name, p, d, mix, rates, pairs}            -> P x D grid, cliff,
          two pairs at once (shared P->D link)

Windows record vLLM counter deltas, energy of every GPU of the pairs in use,
network byte counters of both nodes (link load of all groups) and sampled queue
state. A window with a stall (a P round trip above `stall_ms`, after which the last
requests ran normally again) is repeated once and never ends a rate scan: in r6b a
42 s stall was taken for saturation.

Every request and window is appended to JSONL when it ends; a rerun with the same
output directory skips finished windows. `tag` (e.g. the KV send type) is part of
every window key, so runs of different configurations never mix.
"""

import asyncio
import json
import random
import statistics
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx

from canatune.infrastructure.records import JsonlLog
from canatune.infrastructure.telemetry import prom_value
from canatune.loadgen import prompt_ids
from canatune.proxy.proxy import Endpoint, StreamTimer, pd_transport_id

DEFAULT_MIX = [[128, 64], [512, 64], [1024, 64], [2048, 64]]

# Counters bracketed around every window (differences; missing ones stay None).
COUNTERS = {
    "prefill_time_sum": "vllm:request_prefill_time_seconds_sum",
    "prefill_time_count": "vllm:request_prefill_time_seconds_count",
    "queue_time_sum": "vllm:request_queue_time_seconds_sum",
    "ttft_sum": "vllm:time_to_first_token_seconds_sum",
    "ttft_count": "vllm:time_to_first_token_seconds_count",
    "itl_sum": "vllm:inter_token_latency_seconds_sum",
    "itl_count": "vllm:inter_token_latency_seconds_count",
    "generation_tokens": "vllm:generation_tokens_total",
    "prompt_tokens": "vllm:prompt_tokens_total",
    "iteration_tokens_sum": "vllm:iteration_tokens_total_sum",
    "iteration_tokens_count": "vllm:iteration_tokens_total_count",
    "preemptions": "vllm:num_preemptions_total",
}
GAUGES = {
    "running": "vllm:num_requests_running",
    "waiting": "vllm:num_requests_waiting",
    "kv_usage": "vllm:kv_cache_usage_perc",
}


@dataclass(frozen=True)
class Pair:
    """One fixed P/D pair and the agents (URL, GPU index) of its two GPUs."""

    name: str
    prefill: Endpoint
    decode: Endpoint
    prefill_agent: tuple[str, int]
    decode_agent: tuple[str, int]


@dataclass
class Plan:
    """What to measure; every field can be overridden from a JSON file."""

    tag: str = "async"
    prefill_clocks: list[int] = field(default_factory=lambda: [2520, 1545, 1080])
    decode_clocks: list[int] = field(default_factory=lambda: [2040, 1170, 735])
    lut_clocks: list[int] | None = None  # default: prefill_clocks
    base_prefill_clock: int = 2520  # P clock during decode windows
    base_decode_clock: int = 1170  # D clock during lut, load and mix windows
    stages: list[str] = field(default_factory=lambda: ["lut", "power", "decode", "load", "mix"])
    warmup_s: float = 90.0
    warmup_rate: float = 1.0
    lut_lengths: list[int] = field(
        default_factory=lambda: [16, 32, 64, 128, 256, 512, 1024, 2048, 3072]
    )
    lut_repeats: int = 7
    warmup_requests: int = 3
    power_s: float = 10.0
    settle_s: float = 2.0
    decode_concurrency: list[int] = field(default_factory=lambda: [1, 4, 8, 16, 24, 32])
    decode_output: int = 256
    decode_window_s: float = 60.0
    decode_stagger_s: float = 15.0
    load_rates: list[float] = field(default_factory=lambda: [0.5, 1, 2, 3, 4, 5, 6, 7])
    load_window_s: float = 90.0
    load_output: int = 64
    mixes: dict[str, list[list[int]]] = field(
        default_factory=lambda: {
            "default": DEFAULT_MIX,
            "short": [[64, 64], [128, 64], [256, 64]],
            "long": [[1024, 64], [2048, 64], [3072, 64]],
        }
    )
    mix_clock: int = 2520
    mix_names: list[str] = field(default_factory=lambda: ["short", "long"])
    scans: list[dict[str, Any]] = field(default_factory=list)
    saturation_ttft_ms: float = 3000.0  # stop a rate scan once TTFT p95 exceeds this
    stall_ms: float = 10000.0  # a P round trip this long is a stall, not queueing
    drain_timeout_s: float = 120.0
    seed: int = 20261002

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> "Plan":
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown plan keys: {sorted(unknown)}")
        plan = cls(**raw)
        for scan in plan.scans:
            missing = {"name", "p", "d", "rates"} - set(scan)
            if missing:
                raise ValueError(f"scan {scan} lacks {sorted(missing)}")
        return plan


@dataclass
class Outcome:
    status: str
    pair: str
    prompt_tokens: int
    output_tokens: int
    sent_s: float
    prefill_ms: float | None  # P HTTP round trip (P queue + prefill + P front end)
    p_engine_ms: float | None  # P's own prefill time (counter delta, sequential only)
    gap_ms: float | None  # P done -> decode request sent
    decode_first_ms: float | None  # decode sent -> first token (D front end + KV + step)
    ttft_ms: float | None
    tpot_ms: float | None
    done_s: float


def p95(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(0.95 * (len(ordered) - 1) + 0.5))]


class Measure:
    def __init__(
        self,
        plan: Plan,
        pairs: Sequence[Pair],
        model: str,
        out_dir: str | Path,
        *,
        deadline: float | None = None,  # monotonic time after which no window starts
        client_factory: Callable[[], httpx.AsyncClient] | None = None,
        sleep: Callable[[float], Any] = asyncio.sleep,
        sample_period_s: float = 0.5,
    ) -> None:
        if not pairs:
            raise ValueError("need at least one pair")
        self.plan = plan
        self.pairs = list(pairs)
        self.primary = self.pairs[0]
        self.model = model
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.deadline = deadline
        self.client_factory = client_factory or (
            lambda: httpx.AsyncClient(
                timeout=httpx.Timeout(300, connect=10),
                limits=httpx.Limits(max_connections=None, max_keepalive_connections=64),
                trust_env=False,
            )
        )
        self.sleep = sleep
        self.sample_period_s = sample_period_s
        self.requests = JsonlLog(self.out / "requests.jsonl")
        self.windows = JsonlLog(self.out / "windows.jsonl")
        self.done = self._done_keys()
        self._t0 = time.monotonic()
        self._index = 0

    # compatibility with single-pair callers
    @property
    def prefill(self) -> Endpoint:
        return self.primary.prefill

    @property
    def decode(self) -> Endpoint:
        return self.primary.decode

    def _done_keys(self) -> set[str]:
        path = self.out / "windows.jsonl"
        if not path.exists():
            return set()
        keys = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                keys.add(json.loads(line)["key"])
            except (ValueError, KeyError):
                continue
        return keys

    def now_s(self) -> float:
        return time.monotonic() - self._t0

    def time_left(self) -> float:
        return float("inf") if self.deadline is None else self.deadline - time.monotonic()

    def _pairs(self, n: int) -> list[Pair]:
        if n > len(self.pairs):
            raise ValueError(f"scan needs {n} pairs, have {len(self.pairs)}")
        return self.pairs[:n]

    def _agents(self) -> list[str]:
        urls = []
        for pair in self.pairs:
            for url, _ in (pair.prefill_agent, pair.decode_agent):
                if url not in urls:
                    urls.append(url)
        return urls

    # ---- agents and counters ------------------------------------------------------------

    async def lock(
        self,
        client: httpx.AsyncClient,
        prefill_mhz: int,
        decode_mhz: int,
        pairs: Sequence[Pair] | None = None,
    ) -> None:
        for pair in pairs or [self.primary]:
            for (url, gpu), mhz, role in (
                (pair.prefill_agent, prefill_mhz, "prefill"),
                (pair.decode_agent, decode_mhz, "decode"),
            ):
                response = await client.post(
                    f"{url}/gpus/{gpu}/lock", json={"mhz": mhz}, timeout=30
                )
                response.raise_for_status()
                if not response.json().get("ok"):
                    raise RuntimeError(f"lock {pair.name} {role} {mhz} MHz: {response.json()}")
        await self.sleep(self.plan.settle_s)

    async def energy(self, client: httpx.AsyncClient) -> dict[str, dict[str, Any]]:
        """Readings of every GPU of every pair, keyed by endpoint name."""
        by_agent: dict[str, list[dict[str, Any]]] = {}
        out: dict[str, dict[str, Any]] = {}
        for pair in self.pairs:
            for endpoint, (url, gpu) in (
                (pair.prefill, pair.prefill_agent),
                (pair.decode, pair.decode_agent),
            ):
                try:
                    if url not in by_agent:
                        by_agent[url] = (await client.get(f"{url}/gpus", timeout=10)).json()
                    out[endpoint.name] = next(r for r in by_agent[url] if r.get("index") == gpu)
                except Exception as error:
                    out[endpoint.name] = {"error": repr(error)}
        return out

    async def net(self, client: httpx.AsyncClient) -> dict[str, Any]:
        out = {}
        for url in self._agents():
            try:
                response = await client.get(f"{url}/net", timeout=10)
                response.raise_for_status()
                out[url] = response.json()
            except Exception as error:
                out[url] = {"error": repr(error)}
        return out

    async def metrics(self, client: httpx.AsyncClient, endpoint: Endpoint) -> str | None:
        try:
            response = await client.get(
                f"http://{endpoint.http_host}:{endpoint.http_port}/metrics", timeout=10
            )
            response.raise_for_status()
            return response.text
        except Exception:
            return None

    async def counters(
        self, client: httpx.AsyncClient, pairs: Sequence[Pair] | None = None
    ) -> dict[str, dict[str, float | None]]:
        """Counters keyed by endpoint name (primary pair also as prefill/decode)."""
        out = {}
        for pair in pairs or [self.primary]:
            for endpoint in (pair.prefill, pair.decode):
                text = await self.metrics(client, endpoint)
                out[endpoint.name] = {
                    k: (None if text is None else prom_value(text, (name,)))
                    for k, name in COUNTERS.items()
                }
        out["prefill"] = out.get(self.primary.prefill.name, {})
        out["decode"] = out.get(self.primary.decode.name, {})
        return out

    async def _sample(
        self, client: httpx.AsyncClient, stop: asyncio.Event, into: list, pairs: Sequence[Pair]
    ) -> None:
        while not stop.is_set():
            row = {"t_s": self.now_s()}
            for i, pair in enumerate(pairs):
                prefix = "" if i == 0 else f"{pair.name}_"
                for role, endpoint in (("prefill", pair.prefill), ("decode", pair.decode)):
                    text = await self.metrics(client, endpoint)
                    for k, name in GAUGES.items():
                        value = None if text is None else prom_value(text, (name,))
                        row[f"{prefix}{role}_{k}"] = value
            into.append(row)
            try:
                await asyncio.wait_for(stop.wait(), timeout=self.sample_period_s)
            except asyncio.TimeoutError:
                pass

    # ---- one request ----------------------------------------------------------------------

    async def request(
        self,
        client: httpx.AsyncClient,
        prompt: int,
        output: int,
        *,
        engine_time: bool = False,
        pair: Pair | None = None,
    ) -> Outcome:
        pair = pair or self.primary
        self._index += 1
        body = {
            "model": self.model,
            "prompt": prompt_ids(self.plan.seed, self._index, prompt),
            "max_tokens": output,
            "ignore_eos": True,
            "temperature": 0.0,
            "stream": True,
        }
        headers = {
            "X-Request-Id": pd_transport_id(f"m-{uuid.uuid4().hex}", pair.prefill, pair.decode)
        }
        before = await self.metrics(client, pair.prefill) if engine_time else None
        sent = self.now_s()
        timer = StreamTimer(clock=self.now_s)
        status = "ok"
        prefill_ms = gap_ms = decode_first_ms = p_engine_ms = None
        try:
            response = await client.post(
                pair.prefill.completions_url,
                json={**body, "stream": False, "max_tokens": 1},
                headers=headers,
            )
            p_done = self.now_s()
            prefill_ms = (p_done - sent) * 1000.0
            if response.status_code != 200:
                raise RuntimeError(f"prefill HTTP {response.status_code}")
            d_sent = self.now_s()
            gap_ms = (d_sent - p_done) * 1000.0
            async with client.stream(
                "POST", pair.decode.completions_url, json=body, headers=headers
            ) as stream:
                if stream.status_code != 200:
                    raise RuntimeError(f"decode HTTP {stream.status_code}")
                async for chunk in stream.aiter_bytes():
                    timer.feed(chunk)
            if timer.token_times:
                decode_first_ms = (timer.token_times[0] - d_sent) * 1000.0
        except Exception as error:  # a failed request is data
            status = f"error: {type(error).__name__}: {error}"[:200]
        if engine_time and before is not None:
            after = await self.metrics(client, pair.prefill)
            name = (COUNTERS["prefill_time_sum"],)
            s0, s1 = prom_value(before, name), None if after is None else prom_value(after, name)
            if s0 is not None and s1 is not None:
                p_engine_ms = (s1 - s0) * 1000.0
        times = timer.token_times
        return Outcome(
            status=status,
            pair=pair.name,
            prompt_tokens=prompt,
            output_tokens=len(times),
            sent_s=sent,
            prefill_ms=prefill_ms,
            p_engine_ms=p_engine_ms,
            gap_ms=gap_ms,
            decode_first_ms=decode_first_ms,
            ttft_ms=(times[0] - sent) * 1000.0 if times else None,
            tpot_ms=timer.tpot_ms(),
            done_s=self.now_s(),
        )

    # ---- windows ------------------------------------------------------------------------

    async def window(
        self,
        client: httpx.AsyncClient,
        key: str,
        info: Mapping[str, Any],
        clocks: tuple[int, int],
        body: Callable[[], Any],
        estimate_s: float,
        *,
        pairs: Sequence[Pair] | None = None,
        record: bool = True,
    ) -> dict[str, Any] | None:
        """Lock, bracket counters, energy and network, run `body`, log; skip if done."""
        if key in self.done:
            return None
        if self.time_left() < estimate_s + 30:
            raise TimeoutError(f"no time left for {key}")
        pairs = list(pairs or [self.primary])
        await self.lock(client, *clocks, pairs=pairs)
        c0, e0, n0 = await self.counters(client, pairs), await self.energy(client), None
        n0 = await self.net(client)
        t0 = self.now_s()
        samples: list[dict] = []
        stop = asyncio.Event()
        sampler = asyncio.create_task(self._sample(client, stop, samples, pairs))
        try:
            outcomes: list[Outcome] = await body()
        finally:
            stop.set()
            await sampler
        c1, e1, n1, t1 = (
            await self.counters(client, pairs),
            await self.energy(client),
            await self.net(client),
            self.now_s(),
        )
        summary = self._summary(
            key, info, clocks, pairs, outcomes, (c0, c1), (e0, e1), (n0, n1), t1 - t0, samples
        )
        if record:
            for o in outcomes:
                self.requests.write({"key": key, **info, **asdict(o)})
            self.windows.write(summary)
            self.done.add(key)
        return summary

    def _summary(
        self, key, info, clocks, pairs, outcomes, counters, energy, net, duration, samples
    ):
        ok = [o for o in outcomes if o.status == "ok"]
        names = [e.name for p in pairs for e in (p.prefill, p.decode)]

        def delta(name_: str, counter: str) -> float | None:
            a, b = counters[0].get(name_, {}).get(counter), counters[1].get(name_, {}).get(counter)
            return None if a is None or b is None else b - a

        joules = {}
        for name_ in sorted(set(energy[0]) & set(energy[1])):
            a, b = energy[0][name_], energy[1][name_]
            if "energy_mj" in a and "energy_mj" in b:
                joules[name_] = (b["energy_mj"] - a["energy_mj"]) / 1000.0
        # primary pair also under the role names (single-pair analyses)
        if self.primary.prefill.name in joules:
            joules["prefill"] = joules[self.primary.prefill.name]
        if self.primary.decode.name in joules:
            joules["decode"] = joules[self.primary.decode.name]

        network = {}
        for url, before in net[0].items():
            after = net[1].get(url, {})
            a, b = before.get("interfaces") or {}, after.get("interfaces") or {}
            network[url] = {
                iface: {k: b[iface][k] - a[iface][k] for k in ("rx_bytes", "tx_bytes")}
                for iface in a
                if iface in b
            }

        def mean_of(name_: str) -> float | None:
            values = [s[name_] for s in samples if s.get(name_) is not None]
            return statistics.fmean(values) if values else None

        def stat(attr: str) -> dict[str, float | None]:
            values = [getattr(o, attr) for o in ok if getattr(o, attr) is not None]
            return {
                "mean": statistics.fmean(values) if values else None,
                "p50": statistics.median(values) if values else None,
                "p95": p95(values),
            }

        gauges = sorted({k for s in samples for k in s if k != "t_s"})
        return {
            "key": key,
            **info,
            "tag": self.plan.tag,
            "prefill_mhz": clocks[0],
            "decode_mhz": clocks[1],
            "pairs": [p.name for p in pairs],
            "duration_s": duration,
            "requests": len(outcomes),
            "ok": len(ok),
            "failed": len(outcomes) - len(ok),
            "stalls": self._stalls(ok),
            "ttft_ms": stat("ttft_ms"),
            "prefill_ms": stat("prefill_ms"),
            "decode_first_ms": stat("decode_first_ms"),
            "gap_ms": stat("gap_ms"),
            "tpot_ms": stat("tpot_ms"),
            "energy_j": joules,
            "power_w": {k: v / duration for k, v in joules.items()} if duration > 0 else {},
            "net_bytes": network,
            "counters": {
                role: {c: delta(role, c) for c in COUNTERS}
                for role in ["prefill", "decode", *names]
            },
            "sampled": {g: mean_of(g) for g in gauges},
            "sampled_max": {
                g: max((s[g] for s in samples if s.get(g) is not None), default=None)
                for g in gauges
            },
        }

    def _stalls(self, ok: Sequence[Outcome]) -> int:
        """Requests caught in a stall: a P round trip above `stall_ms` in a window whose
        last tenth ran normally again. Overload looks different: the queue only grows,
        so the last requests are the slowest (r6b: a 42 s stall at 0.5 req/s recovered
        within the window; 5 req/s never did)."""
        slow = [o for o in ok if (o.prefill_ms or 0) > self.plan.stall_ms]
        if not slow:
            return 0
        ordered = sorted(ok, key=lambda o: o.sent_s)
        tail = [o.prefill_ms or 0 for o in ordered[-max(3, len(ordered) // 10) :]]
        return len(slow) if statistics.median(tail) < self.plan.stall_ms / 10 else 0

    # ---- stages -----------------------------------------------------------------------------

    def _mix(self, mix: str) -> list[tuple[int, int]]:
        return [(int(a), int(b)) for a, b in self.plan.mixes[mix]]

    async def stage_warmup(self, client: httpx.AsyncClient) -> None:
        """Traffic after every (re)start, on all pairs; recorded but never 'done'."""
        p = self.plan
        clocks = (p.base_prefill_clock, p.base_decode_clock)
        key = f"warmup|{p.tag}|{int(time.time())}"
        old_window = p.load_window_s
        p.load_window_s = p.warmup_s
        try:
            await self.window(
                client,
                key,
                {"stage": "warmup"},
                clocks,
                lambda: self._open(client, "default", p.warmup_rate, self.pairs),
                estimate_s=p.warmup_s,
                pairs=self.pairs,
                record=False,
            )
        finally:
            p.load_window_s = old_window

    async def stage_lut(self, client: httpx.AsyncClient) -> None:
        p = self.plan
        for mhz in p.lut_clocks or p.prefill_clocks:
            lengths = [length for length in p.lut_lengths for _ in range(p.lut_repeats)]
            random.Random(p.seed + mhz).shuffle(lengths)

            async def body(lengths=lengths) -> list[Outcome]:
                for _ in range(p.warmup_requests):
                    await self.request(client, 128, 2)
                return [await self.request(client, n, 2, engine_time=True) for n in lengths]

            await self.window(
                client,
                f"lut|{p.tag}|{mhz}",
                {"stage": "lut"},
                (mhz, p.base_decode_clock),
                body,
                estimate_s=len(lengths) * 0.6 + 10,
            )

    async def stage_power(self, client: httpx.AsyncClient) -> None:
        p = self.plan
        points = [(f, p.base_decode_clock) for f in p.prefill_clocks]
        points += [(p.base_prefill_clock, f) for f in p.decode_clocks]
        for clocks in dict.fromkeys(points):

            async def body() -> list[Outcome]:
                await self.sleep(p.power_s)
                return []

            await self.window(
                client,
                f"power|{p.tag}|{clocks[0]}|{clocks[1]}",
                {"stage": "power"},
                clocks,
                body,
                estimate_s=p.power_s + p.settle_s,
            )

    async def stage_decode(self, client: httpx.AsyncClient) -> None:
        p = self.plan
        pairs = self._mix("default")
        for mhz in p.decode_clocks:
            for concurrency in p.decode_concurrency:
                rng = random.Random(p.seed + mhz * 101 + concurrency)
                end_at = 0.0

                async def worker(i: int, concurrency=concurrency, rng=rng) -> list[Outcome]:
                    stagger = min(p.decode_stagger_s, p.decode_window_s / 3)
                    await self.sleep(stagger * i / concurrency)
                    results = []
                    while self.now_s() < end_at:
                        prompt, _ = pairs[rng.randrange(len(pairs))]
                        results.append(await self.request(client, prompt, p.decode_output))
                    return results

                async def body(concurrency=concurrency) -> list[Outcome]:
                    nonlocal end_at
                    end_at = self.now_s() + p.decode_window_s
                    parts = await asyncio.gather(*(worker(i) for i in range(concurrency)))
                    return [o for part in parts for o in part]

                await self.window(
                    client,
                    f"decode|{p.tag}|{mhz}|{concurrency}",
                    {"stage": "decode", "concurrency": concurrency},
                    (p.base_prefill_clock, mhz),
                    body,
                    estimate_s=p.decode_window_s + 40,
                )

    async def _open(
        self, client: httpx.AsyncClient, mix: str, rate: float, pairs: Sequence[Pair]
    ) -> list[Outcome]:
        """Open loop: `rate` req/s on each pair, fixed-count uniform arrivals."""
        p = self.plan
        lengths = self._mix(mix)
        start = self.now_s()
        schedule = []
        for i, pair in enumerate(pairs):
            rng = random.Random(p.seed + int(rate * 1000) + len(mix) + 7919 * i)
            count = max(1, round(rate * p.load_window_s))
            for at in sorted(rng.uniform(0, p.load_window_s) for _ in range(count)):
                schedule.append((at, pair, lengths[rng.randrange(len(lengths))][0]))
        schedule.sort(key=lambda item: item[0])
        tasks = []
        for at, pair, prompt in schedule:
            delay = start + at - self.now_s()
            if delay > 0:
                await self.sleep(delay)
            tasks.append(
                asyncio.create_task(self.request(client, prompt, p.load_output, pair=pair))
            )
        done, pending = await asyncio.wait(tasks, timeout=p.drain_timeout_s)
        for task in pending:
            task.cancel()
        return [t.result() for t in done if not t.cancelled() and t.exception() is None]

    def _read_window(self, key: str) -> dict[str, Any] | None:
        path = self.out / "windows.jsonl"
        if not path.exists():
            return None
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("key") == key:
                return row
        return None

    async def _scan(
        self,
        client: httpx.AsyncClient,
        stage: str,
        name: str,
        clocks: tuple[int, int],
        mix: str,
        rates: Sequence[float],
        n_pairs: int = 1,
    ) -> None:
        p = self.plan
        pairs = self._pairs(n_pairs)
        for rate in rates:
            base = f"{stage}|{p.tag}|{name}|{clocks[0]}|{clocks[1]}|{mix}|{n_pairs}|{rate}"
            info = {
                "stage": stage,
                "scan": name,
                "mix": mix,
                "rate_rps": rate,
                "rate_per_pair": rate,
                "n_pairs": n_pairs,
            }
            summary = None
            for attempt in range(2):  # a stalled window is repeated once
                key = base if attempt == 0 else f"{base}#retry"
                result = await self.window(
                    client,
                    key,
                    {**info, "attempt": attempt},
                    clocks,
                    lambda rate=rate: self._open(client, mix, rate, pairs),
                    estimate_s=p.load_window_s + 30,
                    pairs=pairs,
                )
                summary = result if result is not None else self._read_window(key)
                if not summary or not summary.get("stalls"):
                    break
            if summary and summary.get("stalls"):
                continue  # stalled twice: no verdict, try the next rate
            ttft = (summary or {}).get("ttft_ms", {}).get("p95")
            if ttft is not None and ttft > p.saturation_ttft_ms:
                break  # saturated: higher rates only add queueing

    async def stage_load(self, client: httpx.AsyncClient) -> None:
        for mhz in self.plan.prefill_clocks:
            await self._scan(
                client,
                "load",
                "load",
                (mhz, self.plan.base_decode_clock),
                "default",
                self.plan.load_rates,
            )

    async def stage_mix(self, client: httpx.AsyncClient) -> None:
        for mix in self.plan.mix_names:
            clocks = (self.plan.mix_clock, self.plan.base_decode_clock)
            await self._scan(client, "mix", "mix", clocks, mix, self.plan.load_rates)

    async def stage_scans(self, client: httpx.AsyncClient) -> None:
        for scan in self.plan.scans:
            await self._scan(
                client,
                "scan",
                str(scan["name"]),
                (int(scan["p"]), int(scan["d"])),
                str(scan.get("mix", "default")),
                [float(r) for r in scan["rates"]],
                int(scan.get("pairs", 1)),
            )

    async def run(self) -> dict[str, Any]:
        stages = {
            "warmup": self.stage_warmup,
            "lut": self.stage_lut,
            "power": self.stage_power,
            "decode": self.stage_decode,
            "load": self.stage_load,
            "mix": self.stage_mix,
            "scans": self.stage_scans,
        }
        unknown = set(self.plan.stages) - set(stages)
        if unknown:
            raise ValueError(f"unknown stages: {sorted(unknown)}")
        status = "done"
        async with self.client_factory() as client:
            for name in self.plan.stages:
                try:
                    await stages[name](client)
                except TimeoutError as error:
                    status = f"out_of_time: {error}"
                    break
        return {"status": status, "windows_done": len(self.done)}
