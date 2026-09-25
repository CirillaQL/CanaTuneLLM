"""CanaryProbe against mocked vLLM endpoints and a fake agent."""

import asyncio
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


def vllm_handler(requests):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "/models/mistral"}]})
        body = json.loads(request.content)
        requests.append((request.url.port, body, request.headers["X-Request-Id"]))
        if request.url.port == 8100:  # prefill
            return httpx.Response(200, json={"choices": [{"text": "x"}]})
        n = body["max_tokens"]
        sse = b"".join(b'data: {"choices":[{"text":"t"}]}\n\n' for _ in range(n))
        return httpx.Response(
            200, content=sse + b"data: [DONE]\n\n", headers={"content-type": "text/event-stream"}
        )

    return handler


def make_probe(requests, agent):
    config = load_config()
    config["topology"]["prefill_nodegroup"]["node"] = "127.0.0.1"
    config["topology"]["decode_nodegroup"]["node"] = "127.0.0.2"
    canary = build_groups(config)[0]
    table = RiskTable.from_config(config["risk"], identity(config))
    transport = httpx.MockTransport(vllm_handler(requests))
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
