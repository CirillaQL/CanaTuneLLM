"""Probe backend: synthetic requests on the Canary pair and window measurements.

Probes are random token-id prompts whose (prompt, output) lengths are drawn from
the recent production length distribution; outputs use `ignore_eos` so every
clock under test does exactly the same work. Random tokens never hit the prefix
cache. Probes go straight to the Canary P/D pair (never through admission), and
their outcomes are recorded in the risk table under the Canary's clock point.

Energy comes from the NVML cumulative counter read through the node agents;
the same polling gives the median busy SM clock and the share of samples in
which power or thermal limits held the clock down.
"""

import asyncio
import random
import statistics
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import httpx

from canatune.controller.locator import Hardware, WindowResult
from canatune.domain.groups import ClockPoint, Group
from canatune.domain.load import LengthStats
from canatune.domain.risk import RiskTable, is_violation
from canatune.infrastructure.clocks import ClockActuator, GpuRef
from canatune.infrastructure.records import JsonlLog
from canatune.infrastructure.telemetry import Telemetry
from canatune.proxy.proxy import Endpoint, StreamTimer, pd_transport_id

# NVML clocks-event reasons that mean "held below the requested clock".
LIMIT_MASK = 0x4 | 0x8 | 0x20 | 0x40 | 0x80  # SW power cap, HW slowdown, thermal, power brake


def p95(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(0.95 * (len(ordered) - 1) + 0.5))]


@dataclass
class ProbeOutcome:
    status: str
    prompt_tokens: int
    output_tokens: int
    prefill_ms: float | None
    ttft_ms: float | None
    tpot_ms: float | None
    n_await_at_send: int
    d_busy_at_send: bool


@dataclass(frozen=True)
class ProbeSettings:
    settle_s: float = 2.0
    sample_period_s: float = 0.25
    drain_timeout_s: float = 60.0
    abort_min_requests: int = 10
    vocab_low: int = 1000
    vocab_high: int = 31000
    seed: int = 20260925
    prefill_min_mhz: int = 0  # never probe below these clocks
    decode_min_mhz: int = 0

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "ProbeSettings":
        raw = dict(config.get("canary", {}).get("probe", {}))
        known = set(cls.__dataclass_fields__)
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown canary.probe keys: {sorted(unknown)}")
        return cls(**raw)


class CanaryProbe:
    def __init__(
        self,
        canary: Group,
        endpoints: Mapping[str, Endpoint],
        actuator: ClockActuator,
        refs: Mapping[str, GpuRef],
        lengths: LengthStats,
        table: RiskTable,
        settings: ProbeSettings,
        *,
        client_factory: Callable[[], httpx.AsyncClient],
        telemetry: Telemetry | None = None,
        ttft_slo_ms: float = 500.0,
        tpot_slo_ms: float = 200.0,
        log: JsonlLog | None = None,
    ) -> None:
        self.canary = canary
        self.prefill = endpoints[canary.prefill]
        self.decode = endpoints[canary.decode]
        self.actuator = actuator
        self.refs = refs
        self.lengths = lengths
        self.table = table
        self.s = settings
        self.client_factory = client_factory
        self.telemetry = telemetry
        self.ttft_slo_ms = ttft_slo_ms
        self.tpot_slo_ms = tpot_slo_ms
        self.log = log or JsonlLog(None)
        self.rng = random.Random(settings.seed)
        self._model: str | None = None
        self._n_await = 0
        self._decoding = 0

    # ---- clocks and readings -------------------------------------------------------------

    async def lock(self, point: ClockPoint) -> None:
        results = await asyncio.gather(
            self.actuator.lock(self.refs[self.canary.prefill], point.prefill_mhz),
            self.actuator.lock(self.refs[self.canary.decode], point.decode_mhz),
        )
        if not all(r.get("ok") for r in results):
            raise RuntimeError(f"clock lock failed: {results}")
        await asyncio.sleep(self.s.settle_s)

    async def _reading(self, endpoint: str) -> dict[str, Any]:
        ref = self.refs[endpoint]
        for reading in await self.actuator.readings(ref.agent_url):
            if reading.get("index") == ref.gpu:
                return reading
        raise RuntimeError(f"agent has no reading for {endpoint}")

    async def _readings(self) -> tuple[dict[str, Any], dict[str, Any]]:
        return await asyncio.gather(
            self._reading(self.canary.prefill), self._reading(self.canary.decode)
        )

    async def hardware(self) -> Hardware:
        p = await self.actuator.supported_clocks(self.refs[self.canary.prefill])
        d = await self.actuator.supported_clocks(self.refs[self.canary.decode])
        p = tuple(sorted(f for f in p if f >= self.s.prefill_min_mhz))
        d = tuple(sorted(f for f in d if f >= self.s.decode_min_mhz))
        if len(p) < 2 or len(d) < 2:
            raise RuntimeError("agents report too few supported clocks")
        return Hardware(p, d)

    async def idle_power(self, clock: ClockPoint, seconds: float) -> tuple[float, float]:
        await self.lock(clock)
        (p0, d0), t0 = await self._readings(), time.monotonic()
        await asyncio.sleep(seconds)
        (p1, d1), t1 = await self._readings(), time.monotonic()
        dt = t1 - t0
        return (
            (p1["energy_mj"] - p0["energy_mj"]) / 1000.0 / dt,
            (d1["energy_mj"] - d0["energy_mj"]) / 1000.0 / dt,
        )

    # ---- one probe request ------------------------------------------------------------------

    async def _model_name(self, client: httpx.AsyncClient) -> str:
        if self._model is None:
            response = await client.get(
                f"http://{self.prefill.http_host}:{self.prefill.http_port}/v1/models"
            )
            response.raise_for_status()
            self._model = response.json()["data"][0]["id"]
        return self._model

    async def request(
        self, client: httpx.AsyncClient, prompt_tokens: int, output_tokens: int
    ) -> ProbeOutcome:
        prompt = [
            self.rng.randrange(self.s.vocab_low, self.s.vocab_high) for _ in range(prompt_tokens)
        ]
        body = {
            "model": await self._model_name(client),
            "prompt": prompt,
            "max_tokens": output_tokens,
            "ignore_eos": True,
            "temperature": 0.0,
            "stream": True,
        }
        transport_id = pd_transport_id(f"canary-{uuid.uuid4().hex}", self.prefill, self.decode)
        headers = {"X-Request-Id": transport_id}
        n_await, d_busy = self._n_await, self._decoding > 0
        self._n_await += 1
        sent = time.monotonic()
        prefill_ms = ttft_ms = tpot_ms = None
        timer = StreamTimer()
        status = "ok"
        decoding = False
        try:
            prefill_body = {**body, "stream": False, "max_tokens": 1}
            response = await client.post(
                self.prefill.completions_url, json=prefill_body, headers=headers
            )
            prefill_ms = (time.monotonic() - sent) * 1000.0
            if response.status_code != 200:
                raise httpx.HTTPError(f"prefill HTTP {response.status_code}")
            async with client.stream(
                "POST", self.decode.completions_url, json=body, headers=headers
            ) as stream:
                if stream.status_code != 200:
                    raise httpx.HTTPError(f"decode HTTP {stream.status_code}")
                async for chunk in stream.aiter_bytes():
                    if timer.feed(chunk) and ttft_ms is None:
                        ttft_ms = (timer.token_times[0] - sent) * 1000.0
                        self._n_await -= 1
                        self._decoding += 1
                        decoding = True
            tpot_ms = timer.tpot_ms()
        except Exception as error:  # a failed probe is data, never a crash (cancel passes)
            status = f"error: {error!r}"
        finally:
            if decoding:
                self._decoding -= 1
            else:
                self._n_await -= 1
        outcome = ProbeOutcome(
            status,
            prompt_tokens,
            len(timer.token_times),
            prefill_ms,
            ttft_ms,
            tpot_ms,
            n_await,
            d_busy,
        )
        return outcome

    def _record(self, clock: ClockPoint, outcome: ProbeOutcome) -> bool | None:
        violated = is_violation(
            outcome.ttft_ms,
            outcome.tpot_ms,
            ttft_slo_ms=self.ttft_slo_ms,
            tpot_slo_ms=self.tpot_slo_ms,
        )
        if outcome.status == "ok" and violated is not None:
            cell = self.table.cell(
                clock, outcome.n_await_at_send, outcome.prompt_tokens, outcome.d_busy_at_send
            )
            self.table.record(cell, violated)
        return violated

    async def service_times(
        self, clock: ClockPoint, prompts: Sequence[int]
    ) -> list[tuple[int, float]]:
        await self.lock(clock)
        out = []
        async with self.client_factory() as client:
            for tokens in prompts:
                outcome = await self.request(client, tokens, 1)
                if outcome.status == "ok" and outcome.prefill_ms is not None:
                    out.append((tokens, outcome.prefill_ms))
        return out

    # ---- windows -------------------------------------------------------------------------------

    async def _append_sample(self, samples: list) -> None:
        p, d = await self._readings()
        decode = None if self.telemetry is None else self.telemetry.latest.get(self.decode.name)
        samples.append((time.monotonic(), p, d, decode))

    async def _sample(self, stop: asyncio.Event, samples: list) -> None:
        """Background readings for clock/limit statistics; the energy counters are
        bracketed by explicit readings before the first and after the last probe."""
        while not stop.is_set():
            try:
                await self._append_sample(samples)
            except Exception as error:  # a missed sample is not fatal
                self.log.write({"event": "probe_sample_error", "error": repr(error)})
            try:
                await asyncio.wait_for(stop.wait(), timeout=self.s.sample_period_s)
            except asyncio.TimeoutError:
                pass

    def _result(
        self,
        clock: ClockPoint,
        kind: str,
        load: float,
        outcomes: list[ProbeOutcome],
        samples: list,
        violations: int,
        aborted: bool,
    ) -> WindowResult:
        ok = [o for o in outcomes if o.status == "ok"]
        first, last = samples[0], samples[-1]  # explicit readings around the probes
        duration = last[0] - first[0]
        p_j = (last[1]["energy_mj"] - first[1]["energy_mj"]) / 1000.0
        d_j = (last[2]["energy_mj"] - first[2]["energy_mj"]) / 1000.0
        busy = [s[1] for s in samples if s[1].get("power_w", 0) > 0]
        mhz = [r["sm_mhz"] for r in busy]
        limited = [
            r for r in busy if r.get("throttle_reasons") and int(r["throttle_reasons"]) & LIMIT_MASK
        ]
        decode_snaps = [s[3] for s in samples if s[3] is not None]
        preempt = 0.0
        if len(decode_snaps) >= 2 and decode_snaps[0].preemptions_total is not None:
            preempt = max(
                0.0, (decode_snaps[-1].preemptions_total or 0.0) - decode_snaps[0].preemptions_total
            )
        tokens = sum(o.output_tokens for o in ok)
        return WindowResult(
            clock=clock,
            kind=kind,
            load=load,
            duration_s=duration,
            requests=len(ok),
            violations=violations,
            ttft_p95_ms=p95([o.ttft_ms for o in ok if o.ttft_ms is not None]),
            tpot_p95_ms=p95([o.tpot_ms for o in ok if o.tpot_ms is not None]),
            prefill_j_per_request=p_j / len(ok) if ok else None,
            decode_j_per_token=d_j / tokens if tokens else None,
            prefill_mhz_median=statistics.median(mhz) if mhz else None,
            prefill_limited_fraction=len(limited) / len(busy) if busy else 0.0,
            decode_preemptions=preempt,
            decode_waiting_max=max((s.waiting or 0.0 for s in decode_snaps), default=0.0),
            decode_kv_max=max((s.kv_usage or 0.0 for s in decode_snaps), default=None),
            aborted=aborted,
        )

    async def open_window(
        self, clock: ClockPoint, eq_tps: float, alpha: float, seconds: float, abort_above: float
    ) -> WindowResult:
        await self.lock(clock)
        pairs = self.lengths.sample(self.rng, max(1, int(eq_tps * seconds / 64) + 64))
        mean_eq = sum(p for p, _ in pairs) / len(pairs) + alpha
        rate = eq_tps / mean_eq  # requests/s
        outcomes: list[ProbeOutcome] = []
        tasks: set[asyncio.Task] = set()
        stop_sampling = asyncio.Event()
        samples: list = []
        violations = 0
        aborted = False
        async with self.client_factory() as client:
            await self._append_sample(samples)
            sampler = asyncio.create_task(self._sample(stop_sampling, samples))
            start = time.monotonic()
            next_at, i = start, 0

            async def one(prompt_tokens: int, output_tokens: int) -> None:
                nonlocal violations
                outcome = await self.request(client, prompt_tokens, output_tokens)
                outcomes.append(outcome)
                if self._record(clock, outcome):
                    violations += 1

            while time.monotonic() - start < seconds:
                done = len(outcomes)
                if done >= self.s.abort_min_requests and violations / done > abort_above:
                    aborted = True
                    break
                delay = next_at - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(min(delay, 0.05))
                    continue
                prompt_tokens, output_tokens = pairs[i % len(pairs)]
                i += 1
                task = asyncio.create_task(one(prompt_tokens, output_tokens))
                tasks.add(task)
                task.add_done_callback(tasks.discard)
                next_at += self.rng.expovariate(rate)
            if tasks:
                await asyncio.wait(tasks, timeout=self.s.drain_timeout_s)
            for task in tasks:
                task.cancel()
            stop_sampling.set()
            await sampler
            await self._append_sample(samples)
        return self._result(clock, "open", eq_tps, outcomes, samples, violations, aborted)

    async def closed_window(
        self, clock: ClockPoint, concurrency: int, seconds: float
    ) -> WindowResult:
        await self.lock(clock)
        outcomes: list[ProbeOutcome] = []
        stop_sampling = asyncio.Event()
        samples: list = []
        violations = 0
        async with self.client_factory() as client:
            await self._append_sample(samples)
            sampler = asyncio.create_task(self._sample(stop_sampling, samples))
            deadline = time.monotonic() + seconds

            async def worker() -> None:
                nonlocal violations
                while time.monotonic() < deadline:
                    prompt_tokens, output_tokens = self.lengths.sample(self.rng, 1)[0]
                    outcome = await self.request(client, prompt_tokens, output_tokens)
                    outcomes.append(outcome)
                    if self._record(clock, outcome):
                        violations += 1

            workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
            done, pending = await asyncio.wait(workers, timeout=seconds + self.s.drain_timeout_s)
            for task in pending:
                task.cancel()
            stop_sampling.set()
            await sampler
            await self._append_sample(samples)
        return self._result(clock, "closed", concurrency, outcomes, samples, violations, False)
