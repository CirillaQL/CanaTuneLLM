"""CanaryProbe against mocked vLLM endpoints and a fake agent."""

import asyncio
import dataclasses
import json
import time

import httpx
import pytest

from canatune.config import load_config
from canatune.controller.probe import CanaryProbe, ProbeSettings
from canatune.domain.groups import ClockPoint
from canatune.domain.load import LengthStats
from canatune.domain.risk import RiskTable
from canatune.infrastructure.clocks import GpuRef
from canatune.proxy.proxy import parse_endpoints
from canatune.service import build_groups, identity

# NixlConnector: P names the blocks D reads.
KV_PARAMS = {"do_remote_prefill": True, "remote_block_ids": [1], "remote_engine_id": "p"}


class FakeAgent:
    """Energy counters that grow 100 W (P) / 30 W (D); P is power capped at 2040."""

    def __init__(self) -> None:
        self.locked: list[tuple[int, int]] = []
        self.clock = {0: 2520, 1: 1500}
        self._t0 = time.monotonic()

    async def lock(self, ref, mhz):
        self.locked.append((ref.gpu, mhz))
        self.clock[ref.gpu] = mhz
        return {"ok": True, "gpu": ref.gpu, "sm_mhz": mhz}

    async def readings(self, agent_url):
        t = time.monotonic() - self._t0  # counters integrate constant power
        sm = min(self.clock[0], 2040)
        return [
            {
                "index": 0,
                "sm_mhz": sm,
                "power_w": 100.0,
                "energy_mj": t * 100_000.0,
                "throttle_reasons": 0x4 if self.clock[0] > 2040 else 0,
            },
            {"index": 1, "sm_mhz": self.clock[1], "power_w": 30.0, "energy_mj": t * 30_000.0},
        ]

    async def supported_clocks(self, ref):
        return list(range(210, 2521, 15)) if ref.gpu == 0 else list(range(210, 1501, 15))


class FakeMetrics:
    """vLLM counters: P prefill time grows 0.05 ms/token + 20 ms; D counts every
    generated token although the stream packs two tokens per SSE event."""

    def __init__(self) -> None:
        self.prefill_sum = 0.0
        self.prefill_count = 0
        self.generated = 0

    def text(self, port: int) -> str:
        if port == 8100:
            return (
                f'vllm:request_prefill_time_seconds_sum{{model_name="m"}} {self.prefill_sum}\n'
                f'vllm:request_prefill_time_seconds_count{{model_name="m"}} {self.prefill_count}\n'
            )
        return (
            f'vllm:generation_tokens_total{{model_name="m"}} {self.generated}\n'
            'vllm:cache_config_info{block_size="16",cache_dtype="auto",'
            'num_gpu_blocks="2103",gpu_memory_utilization="0.82"} 1.0\n'
        )


def vllm_handler(requests, metrics=None):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "/models/mistral"}]})
        if request.url.path == "/metrics":
            if metrics is None:
                return httpx.Response(404)
            return httpx.Response(200, text=metrics.text(request.url.port))
        body = json.loads(request.content)
        requests.append((request.url.port, body, request.headers["X-Request-Id"]))
        if request.url.port == 8100:  # prefill
            if metrics is not None:
                metrics.prefill_sum += (20 + 0.05 * len(body["prompt"])) / 1000
                metrics.prefill_count += 1
            return httpx.Response(
                200, json={"choices": [{"text": "x"}], "kv_transfer_params": KV_PARAMS}
            )
        n = body["max_tokens"]
        if metrics is not None:
            metrics.generated += n
            sse = b"".join(b'data: {"choices":[{"text":"tt"}]}\n\n' for _ in range((n + 1) // 2))
            return httpx.Response(
                200,
                content=sse + b"data: [DONE]\n\n",
                headers={"content-type": "text/event-stream"},
            )
        sse = b"".join(b'data: {"choices":[{"text":"t"}]}\n\n' for _ in range(n))
        return httpx.Response(
            200, content=sse + b"data: [DONE]\n\n", headers={"content-type": "text/event-stream"}
        )

    return handler


def make_probe(requests, agent, metrics=None):
    config = load_config()
    config["topology"]["prefill_nodegroup"]["node"] = "127.0.0.1"
    config["topology"]["decode_nodegroup"]["node"] = "127.0.0.2"
    canary = build_groups(config)[0]
    table = RiskTable.from_config(config["risk"], identity(config))
    transport = httpx.MockTransport(vllm_handler(requests, metrics))
    probe = CanaryProbe(
        canary,
        parse_endpoints(config),
        agent,
        {"P0": GpuRef("http://p", 0), "D0": GpuRef("http://d", 1)},
        LengthStats([(64, 3), (256, 5)], min_samples=1),
        table,
        ProbeSettings(settle_s=0.0, sample_period_s=0.01, abort_min_requests=5),
        client_factory=lambda: httpx.AsyncClient(transport=transport),
    )
    return probe, table


def test_probe_requests_are_random_tokens_with_ignore_eos_on_the_canary_pair() -> None:
    requests: list = []
    probe, _ = make_probe(requests, FakeAgent())

    async def run():
        async with probe.client_factory() as client:
            return await probe.request(client, 100, 5)

    outcome = asyncio.run(run())
    assert outcome.status == "ok" and outcome.output_tokens == 5
    assert outcome.ttft_ms is not None and outcome.prefill_ms is not None
    (p_port, p_body, p_id), (d_port, d_body, d_id) = requests
    assert (p_port, d_port) == (8100, 8200)  # Canary pair P0/D0 only
    assert p_id == d_id and "canary-" in p_id
    assert len(d_body["prompt"]) == 100 and all(1000 <= t < 31000 for t in d_body["prompt"])
    assert d_body["ignore_eos"] is True and d_body["max_tokens"] == 5
    assert p_body["max_tokens"] == 1 and p_body["stream"] is False
    assert d_body["model"] == "/models/mistral"
    assert probe._n_await == 0 and probe._decoding == 0


def test_open_window_measures_energy_clock_limits_and_records_risk() -> None:
    requests: list = []
    agent = FakeAgent()
    probe, table = make_probe(requests, agent)
    result = asyncio.run(probe.open_window(ClockPoint(2520, 1500), 2000.0, 0.0, 0.3, 0.2))
    assert agent.locked[:2] == [(0, 2520), (1, 1500)]
    assert result.requests > 0 and result.violations == 0
    assert result.prefill_j_per_request == pytest.approx(
        100.0 * result.duration_s / result.requests, rel=0.35
    )
    assert result.prefill_mhz_median == 2040  # capped below the lock
    assert result.prefill_limited_fraction == 1.0
    assert result.decode_j_per_token is not None
    cells = table.to_json()["cells"]
    assert cells and all(key.startswith("2520/1500|") for key in cells)


def test_closed_window_idle_power_service_times_and_hardware() -> None:
    requests: list = []
    agent = FakeAgent()
    probe, _ = make_probe(requests, agent)
    closed = asyncio.run(probe.closed_window(ClockPoint(1815, 1050), 3, 0.2))
    assert closed.kind == "closed" and closed.requests >= 3
    assert closed.tpot_p95_ms is not None
    p_w, d_w = asyncio.run(probe.idle_power(ClockPoint(900, 450), 0.05))
    assert p_w > 0 and d_w > 0
    samples = asyncio.run(probe.service_times(ClockPoint(2520, 1500), [128, 1024]))
    assert [t for t, _, _ in samples] == [128, 1024]
    assert all(ttft is not None and prefill is not None for _, prefill, ttft in samples)
    assert probe.set_prompt_limit(100) == 64  # only the (64, 3) pair remains
    assert probe.set_prompt_limit(None) == 160
    hw = asyncio.run(probe.hardware())
    assert hw.prefill_clocks[-1] == 2520 and hw.decode_clocks[-1] == 1500


def test_same_load_replays_the_same_trace_at_every_clock() -> None:
    requests: list = []
    probe, _ = make_probe(requests, FakeAgent())
    asyncio.run(probe.open_window(ClockPoint(2520, 1500), 3000.0, 0.0, 0.3, 1.0))
    first = [len(body["prompt"]) for port, body, _ in requests if port == 8100]
    requests.clear()
    asyncio.run(probe.open_window(ClockPoint(1305, 1500), 3000.0, 0.0, 0.3, 1.0))
    second = [len(body["prompt"]) for port, body, _ in requests if port == 8100]
    assert first and first == second  # identical arrivals and lengths


def test_closed_windows_use_long_outputs() -> None:
    requests: list = []
    probe, _ = make_probe(requests, FakeAgent())
    asyncio.run(probe.closed_window(ClockPoint(1815, 1050), 2, 0.1))
    decode_bodies = [body for port, body, _ in requests if port == 8200]
    assert decode_bodies and all(b["max_tokens"] == 256 for b in decode_bodies)


def test_open_window_sends_exactly_rate_times_seconds_requests() -> None:
    requests: list = []
    probe, _ = make_probe(requests, FakeAgent())
    # lengths (64, 3) and (256, 5): mean prompt 160; alpha 40 -> 200 eq tokens/request
    asyncio.run(probe.open_window(ClockPoint(2520, 1500), 4000.0, 40.0, 0.5, 1.0))
    assert len([1 for port, _, _ in requests if port == 8100]) == 10  # 20 req/s x 0.5 s


def test_counters_give_prefill_time_decode_tokens_and_kv_capacity() -> None:
    requests: list = []
    metrics = FakeMetrics()
    probe, _ = make_probe(requests, FakeAgent(), metrics)
    samples = asyncio.run(probe.service_times(ClockPoint(2520, 1500), [128, 1024]))
    assert probe.service_source == "metrics"
    assert [round(ms, 3) for _, ms, _ in samples] == [26.4, 71.2]  # 20 ms + 0.05 ms/token
    closed = asyncio.run(probe.closed_window(ClockPoint(1815, 1050), 2, 0.1))
    decoded = sum(
        body["max_tokens"]
        for port, body, _ in requests
        if port == 8200 and body["max_tokens"] == 256
    )
    # D counted every token although only half as many SSE events arrived
    energy = closed.decode_j_per_token * decoded
    assert energy == pytest.approx(30.0 * closed.duration_s, rel=0.35)
    assert asyncio.run(probe.kv_capacity()) == 2103 * 16


def test_closed_window_trace_is_fixed_by_concurrency_and_rep() -> None:
    def prompts(rep):
        requests: list = []
        probe, _ = make_probe(requests, FakeAgent())
        probe.s = ProbeSettings(settle_s=0.0, sample_period_s=0.01, decode_stagger_s=0.0)
        asyncio.run(probe.closed_window(ClockPoint(1815, 1050), 2, 0.05, rep=rep))
        return sorted(len(b["prompt"]) for port, b, _ in requests if port == 8200)[:2]

    assert prompts(0) == prompts(0)


@pytest.mark.parametrize("kind", ["open", "closed"])
def test_cancelled_window_joins_all_probe_requests_and_sampler(kind):
    probe, _ = make_probe([], FakeAgent())

    async def run():
        started = asyncio.Event()
        live = set()

        async def blocking_request(*args):
            task = asyncio.current_task()
            live.add(task)
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                live.remove(task)

        probe.request = blocking_request
        window = (
            probe.open_window(ClockPoint(2520, 1500), 2000, 0, 10, 1)
            if kind == "open" else probe.closed_window(ClockPoint(2520, 1500), 3, 10)
        )
        task = asyncio.create_task(window)
        await asyncio.wait_for(started.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not live
        assert asyncio.all_tasks() == {asyncio.current_task()}
    asyncio.run(run())


def test_quiesce_requires_both_vllm_nodes_idle_and_available():
    from dataclasses import replace

    probe, _ = make_probe([], FakeAgent())
    probe.s = replace(probe.s, drain_timeout_s=0.04, sample_period_s=0.001)
    calls = []

    async def metrics(client, endpoint):
        calls.append(endpoint)
        busy = len(calls) <= 2
        return f"vllm:num_requests_running {int(busy)}\nvllm:num_requests_waiting 0\n"

    probe._metrics = metrics
    asyncio.run(probe.quiesce())
    assert calls == [probe.prefill, probe.decode] * 2

    async def unavailable(*args):
        return None

    probe._metrics = unavailable
    with pytest.raises(TimeoutError, match="has not drained"):
        asyncio.run(probe.quiesce())


def test_mixes_split_the_prompt_distribution_and_closed_windows_use_them() -> None:
    requests: list = []
    probe, _ = make_probe(requests, FakeAgent())  # lengths (64, 3) and (256, 5)
    assert probe._mix_prompts("short") == [64] and probe._mix_prompts("long") == [256]
    assert probe.mix_context("all") == pytest.approx(160 + 128)  # prompt + half of 256
    assert probe.mix_context("long") == pytest.approx(256 + 128)
    asyncio.run(probe.closed_window(ClockPoint(1815, 1050), 2, 0.1, mix="long"))
    prompts = {len(body["prompt"]) for port, body, _ in requests if port == 8100}
    assert prompts == {256}


def test_decode_points_average_d_state_over_each_request() -> None:
    from types import SimpleNamespace

    from canatune.controller.probe import ProbeOutcome, decode_points

    def snap(t, running, kv):
        return SimpleNamespace(taken_at=t, running=running, kv_usage=kv)

    snaps = [snap(0.5, 1, 0.1), snap(1.5, 3, 0.3), snap(1.5, 3, 0.3), snap(2.5, 5, 0.5)]
    outcome = ProbeOutcome("ok", 100, 20, 10.0, 50.0, 60.0, 0, False,
                           first_token_at=1.0, last_token_at=3.0)  # fmt: skip
    # snapshots at 1.5 and 2.5 (the duplicate counts once): X 4, K 0.4 x 1000
    assert decode_points([outcome], snaps, 1000) == [(4.0, 400.0, 60.0)]
    assert decode_points([outcome], snaps, None) == []  # no KV size: no samples
    alone = dataclasses.replace(outcome, first_token_at=2.0)
    assert decode_points([alone], snaps, 1000) == []  # one snapshot is too few
