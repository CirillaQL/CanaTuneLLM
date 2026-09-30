"""Model measurements on one P/D pair (experiments E1 + E2).

Outside the CanaTune control loop (no Router, no Controller): the script locks the
pair's clocks through the node agents and sends requests straight to vLLM, timing
every stage itself, so that the parameters of the TTFT and energy models can be
fitted:

  TTFT = fixed + P queue(load) + S_f(L) + KV transfer(L, bytes in flight) + D first token

Stages, each a list of windows (one window = one clock setting and one workload):
  lut     single sequential requests over a length grid at each P clock  -> S_f(L)
  power   idle power at each P and each D clock                          -> P_idle(f)
  decode  closed loop at several concurrencies per D clock               -> T_iter(X), D power
  load    open-loop Poisson rates per P clock (default mix)              -> queueing, TTFT stages
  mix     open-loop rates for other length mixes at one P clock          -> cross-mix capacity

Every request (stage timestamps) and every window (vLLM counter deltas, energy per
GPU, sampled queue state) is appended to JSONL as soon as it ends; a rerun with the
same output directory skips finished windows, so a crash costs one window.
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


@dataclass
class Plan:
    """What to measure; every field can be overridden from a JSON file."""

    prefill_clocks: list[int] = field(default_factory=lambda: [2520, 1545, 1080])
    decode_clocks: list[int] = field(default_factory=lambda: [2040, 1170, 735])
    base_prefill_clock: int = 2520  # P clock during decode windows
    base_decode_clock: int = 1170  # D clock during lut and load windows
    stages: list[str] = field(default_factory=lambda: ["lut", "power", "decode", "load", "mix"])
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
    saturation_ttft_ms: float = 3000.0  # stop a rate scan once TTFT p95 exceeds this
    drain_timeout_s: float = 120.0
    seed: int = 20261002

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> "Plan":
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown plan keys: {sorted(unknown)}")
        return cls(**raw)


@dataclass
class Outcome:
    status: str
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
        prefill: Endpoint,
        decode: Endpoint,
        agents: Mapping[str, tuple[str, int]],  # "prefill"/"decode" -> (agent URL, GPU index)
        model: str,
        out_dir: str | Path,
        *,
        deadline: float | None = None,  # monotonic time after which no window starts
        client_factory: Callable[[], httpx.AsyncClient] | None = None,
        sleep: Callable[[float], Any] = asyncio.sleep,
        sample_period_s: float = 0.5,
    ) -> None:
        self.plan = plan
        self.prefill = prefill
        self.decode = decode
        self.agents = dict(agents)
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

    # ---- agents and counters ------------------------------------------------------------

    async def lock(self, client: httpx.AsyncClient, prefill_mhz: int, decode_mhz: int) -> None:
        for role, mhz in (("prefill", prefill_mhz), ("decode", decode_mhz)):
            url, gpu = self.agents[role]
            response = await client.post(f"{url}/gpus/{gpu}/lock", json={"mhz": mhz}, timeout=30)
            response.raise_for_status()
            if not response.json().get("ok"):
                raise RuntimeError(f"lock {role} {mhz} MHz failed: {response.json()}")
        await self.sleep(self.plan.settle_s)

    async def energy(self, client: httpx.AsyncClient) -> dict[str, dict[str, Any]]:
        out = {}
        for role, (url, gpu) in self.agents.items():
            try:
                response = await client.get(f"{url}/gpus", timeout=10)
                reading = next(r for r in response.json() if r.get("index") == gpu)
                out[role] = reading
            except Exception as error:
                out[role] = {"error": repr(error)}
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

    async def counters(self, client: httpx.AsyncClient) -> dict[str, dict[str, float | None]]:
        out = {}
        for role, endpoint in (("prefill", self.prefill), ("decode", self.decode)):
            text = await self.metrics(client, endpoint)
            out[role] = {
                k: (None if text is None else prom_value(text, (name,)))
                for k, name in COUNTERS.items()
            }
        return out

    async def _sample(self, client: httpx.AsyncClient, stop: asyncio.Event, into: list) -> None:
        while not stop.is_set():
            row = {"t_s": self.now_s()}
            for role, endpoint in (("prefill", self.prefill), ("decode", self.decode)):
                text = await self.metrics(client, endpoint)
                for k, name in GAUGES.items():
                    row[f"{role}_{k}"] = None if text is None else prom_value(text, (name,))
            into.append(row)
            try:
                await asyncio.wait_for(stop.wait(), timeout=self.sample_period_s)
            except asyncio.TimeoutError:
                pass

    # ---- one request ----------------------------------------------------------------------

    async def request(
        self, client: httpx.AsyncClient, prompt: int, output: int, *, engine_time: bool = False
    ) -> Outcome:
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
            "X-Request-Id": pd_transport_id(f"m-{uuid.uuid4().hex}", self.prefill, self.decode)
        }
        before = await self.counters(client) if engine_time else None
        sent = self.now_s()
        timer = StreamTimer(clock=self.now_s)
        status = "ok"
        prefill_ms = gap_ms = decode_first_ms = p_engine_ms = None
        try:
            response = await client.post(
                self.prefill.completions_url,
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
                "POST", self.decode.completions_url, json=body, headers=headers
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
            after = await self.counters(client)
            s0, s1 = before["prefill"]["prefill_time_sum"], after["prefill"]["prefill_time_sum"]
            if s0 is not None and s1 is not None:
                p_engine_ms = (s1 - s0) * 1000.0
        times = timer.token_times
        return Outcome(
            status=status,
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
    ) -> dict[str, Any] | None:
        """Lock, bracket counters and energy, run `body`, log; skip if done or no time."""
        if key in self.done:
            return None
        if self.time_left() < estimate_s + 30:
            raise TimeoutError(f"no time left for {key}")
        await self.lock(client, *clocks)
        c0, e0, t0 = await self.counters(client), await self.energy(client), self.now_s()
        samples: list[dict] = []
        stop = asyncio.Event()
        sampler = asyncio.create_task(self._sample(client, stop, samples))
        try:
            outcomes: list[Outcome] = await body()
        finally:
            stop.set()
            await sampler
        c1, e1, t1 = await self.counters(client), await self.energy(client), self.now_s()
        for o in outcomes:
            self.requests.write({"key": key, **info, **asdict(o)})
        summary = self._summary(key, info, clocks, outcomes, (c0, c1), (e0, e1), t1 - t0, samples)
        self.windows.write(summary)
        self.done.add(key)
        return summary

    def _summary(self, key, info, clocks, outcomes, counters, energy, duration, samples):
        ok = [o for o in outcomes if o.status == "ok"]

        def delta(role: str, name: str) -> float | None:
            a, b = counters[0][role].get(name), counters[1][role].get(name)
            return None if a is None or b is None else b - a

        joules = {}
        for role in ("prefill", "decode"):
            a, b = energy[0].get(role, {}), energy[1].get(role, {})
            if "energy_mj" in a and "energy_mj" in b:
                joules[role] = (b["energy_mj"] - a["energy_mj"]) / 1000.0

        def mean_of(name: str) -> float | None:
            values = [s[name] for s in samples if s.get(name) is not None]
            return statistics.fmean(values) if values else None

        def stat(attr: str) -> dict[str, float | None]:
            values = [getattr(o, attr) for o in ok if getattr(o, attr) is not None]
            return {
                "mean": statistics.fmean(values) if values else None,
                "p50": statistics.median(values) if values else None,
                "p95": p95(values),
            }

        return {
            "key": key,
            **info,
            "prefill_mhz": clocks[0],
            "decode_mhz": clocks[1],
            "duration_s": duration,
            "requests": len(outcomes),
            "ok": len(ok),
            "ttft_ms": stat("ttft_ms"),
            "prefill_ms": stat("prefill_ms"),
            "decode_first_ms": stat("decode_first_ms"),
            "gap_ms": stat("gap_ms"),
            "tpot_ms": stat("tpot_ms"),
            "energy_j": joules,
            "power_w": {k: v / duration for k, v in joules.items()} if duration > 0 else {},
            "counters": {
                role: {name: delta(role, name) for name in COUNTERS}
                for role in ("prefill", "decode")
            },
            "sampled": {
                name: mean_of(name)
                for name in (
                    "prefill_running",
                    "prefill_waiting",
                    "decode_running",
                    "decode_waiting",
                    "decode_kv_usage",
                )
            },
            "sampled_max": {
                name: max((s[name] for s in samples if s.get(name) is not None), default=None)
                for name in ("prefill_waiting", "decode_running", "decode_kv_usage")
            },
        }

    # ---- stages -----------------------------------------------------------------------------

    async def stage_lut(self, client: httpx.AsyncClient) -> None:
        p = self.plan
        for mhz in p.prefill_clocks:
            lengths = [length for length in p.lut_lengths for _ in range(p.lut_repeats)]
            random.Random(p.seed + mhz).shuffle(lengths)

            async def body(lengths=lengths) -> list[Outcome]:
                for _ in range(p.warmup_requests):
                    await self.request(client, 128, 2)
                return [await self.request(client, n, 2, engine_time=True) for n in lengths]

            await self.window(
                client,
                f"lut|{mhz}",
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
                f"power|{clocks[0]}|{clocks[1]}",
                {"stage": "power"},
                clocks,
                body,
                estimate_s=p.power_s + p.settle_s,
            )

    def _pairs(self, mix: str) -> list[tuple[int, int]]:
        return [(int(a), int(b)) for a, b in self.plan.mixes[mix]]

    async def stage_decode(self, client: httpx.AsyncClient) -> None:
        p = self.plan
        pairs = self._pairs("default")
        for mhz in p.decode_clocks:
            for concurrency in p.decode_concurrency:
                rng = random.Random(p.seed + mhz * 101 + concurrency)
                end_at = None

                async def worker(i: int) -> list[Outcome]:
                    await self.sleep(
                        min(p.decode_stagger_s, p.decode_window_s / 3) * i / concurrency
                    )
                    results = []
                    while self.now_s() < end_at:
                        prompt, _ = pairs[rng.randrange(len(pairs))]
                        results.append(await self.request(client, prompt, p.decode_output))
                    return results

                async def body() -> list[Outcome]:
                    nonlocal end_at
                    end_at = self.now_s() + p.decode_window_s
                    parts = await asyncio.gather(*(worker(i) for i in range(concurrency)))
                    return [o for part in parts for o in part]

                await self.window(
                    client,
                    f"decode|{mhz}|{concurrency}",
                    {"stage": "decode", "concurrency": concurrency},
                    (p.base_prefill_clock, mhz),
                    body,
                    estimate_s=p.decode_window_s + 40,
                )

    async def _open(self, client: httpx.AsyncClient, mix: str, rate: float) -> list[Outcome]:
        p = self.plan
        pairs = self._pairs(mix)
        rng = random.Random(p.seed + int(rate * 1000) + len(mix))
        count = max(1, round(rate * p.load_window_s))
        times = sorted(rng.uniform(0, p.load_window_s) for _ in range(count))
        start = self.now_s()
        tasks = []
        for at in times:
            delay = start + at - self.now_s()
            if delay > 0:
                await self.sleep(delay)
            prompt, _ = pairs[rng.randrange(len(pairs))]
            tasks.append(asyncio.create_task(self.request(client, prompt, p.load_output)))
        done, pending = await asyncio.wait(tasks, timeout=p.drain_timeout_s)
        for task in pending:
            task.cancel()
        return [t.result() for t in done if not t.cancelled() and t.exception() is None]

    async def _scan(self, client: httpx.AsyncClient, stage: str, mhz: int, mix: str) -> None:
        p = self.plan
        for rate in p.load_rates:
            key = f"{stage}|{mhz}|{mix}|{rate}"
            summary = await self.window(
                client,
                key,
                {"stage": stage, "mix": mix, "rate_rps": rate},
                (mhz, p.base_decode_clock),
                lambda rate=rate: self._open(client, mix, rate),
                estimate_s=p.load_window_s + 30,
            )
            if summary is None:  # done in an earlier run: read it back for the stop rule
                summary = self._read_window(key)
            ttft = (summary or {}).get("ttft_ms", {}).get("p95")
            if ttft is not None and ttft > p.saturation_ttft_ms:
                break  # saturated: higher rates only add queueing

    def _read_window(self, key: str) -> dict[str, Any] | None:
        for line in (self.out / "windows.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("key") == key:
                return row
        return None

    async def stage_load(self, client: httpx.AsyncClient) -> None:
        for mhz in self.plan.prefill_clocks:
            await self._scan(client, "load", mhz, "default")

    async def stage_mix(self, client: httpx.AsyncClient) -> None:
        for mix in self.plan.mix_names:
            await self._scan(client, "mix", self.plan.mix_clock, mix)

    async def run(self) -> dict[str, Any]:
        stages = {
            "lut": self.stage_lut,
            "power": self.stage_power,
            "decode": self.stage_decode,
            "load": self.stage_load,
            "mix": self.stage_mix,
        }
        status = "done"
        async with self.client_factory() as client:
            for name in self.plan.stages:
                try:
                    await stages[name](client)
                except TimeoutError as error:
                    status = f"out_of_time: {error}"
                    break
        return {"status": status, "windows_done": len(self.done)}
