"""Telemetry parsing and the per-node GPU agent (dry-run backend)."""

import asyncio

import httpx
import pytest

from canatune.infrastructure.gpu_agent import ClockService, DryRunBackend, create_agent_app
from canatune.infrastructure.telemetry import EndpointSnapshot, Telemetry, parse_snapshot

METRICS = """# HELP vllm:num_requests_running x
vllm:num_requests_running{engine="0",model_name="m"} 3.0
vllm:num_requests_running_other 99
vllm:num_requests_waiting{engine="0",model_name="m"} 1.0
vllm:kv_cache_usage_perc{engine="0",model_name="m"} 0.42
vllm:num_preemptions_total{engine="0",model_name="m"} 7.0
"""


def test_parse_snapshot() -> None:
    s = parse_snapshot(METRICS, 5.0)
    assert (s.running, s.waiting, s.kv_usage, s.preemptions_total) == (3.0, 1.0, 0.42, 7.0)


def test_preemption_delta_and_freshness() -> None:
    now = [0.0]
    t = Telemetry({"D0": "http://d/metrics"}, period_s=1, client_factory=None, clock=lambda: now[0])
    t.update("D0", EndpointSnapshot(0.0, 1, 0, 0.1, 5, True))
    t.update("D0", EndpointSnapshot(0.5, 1, 0, 0.1, 8, True))
    assert t.preemption_delta["D0"] == 3
    now[0] = 0.9
    assert t.fresh("D0", 1.0) is not None
    now[0] = 2.0
    assert t.fresh("D0", 1.0) is None


def test_scrape_once_with_mock_transport() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(200, text=METRICS))
    t = Telemetry(
        {"D0": "http://d/metrics"},
        period_s=1,
        client_factory=lambda: httpx.AsyncClient(transport=transport),
    )

    async def run():
        async with httpx.AsyncClient(transport=transport) as client:
            await t.scrape_once(client)

    asyncio.run(run())
    assert t.latest["D0"].running == 3.0


def test_clock_service_guards_and_confirms() -> None:
    service = ClockService(DryRunBackend([0, 1]), [0, 1], [1050, 1500])
    result = service.lock(1, 1050)
    assert result.ok and result.sm_mhz == 1050 and result.reach_ms is not None
    with pytest.raises(ValueError):
        service.lock(1, 2100)
    with pytest.raises(KeyError):
        service.lock(5, 1050)


def test_clock_service_reports_backend_failure() -> None:
    class Failing(DryRunBackend):
        def lock(self, index, mhz):
            raise RuntimeError("sudo: a password is required")

    result = ClockService(Failing([0]), [0], [1050]).lock(0, 1050)
    assert not result.ok and "password" in result.error


def test_agent_http_api() -> None:
    app = create_agent_app(ClockService(DryRunBackend([0]), [0], [1305, 1815]))

    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://agent") as client:
            locked = await client.post("/gpus/0/lock", json={"mhz": 1815})
            assert locked.status_code == 200 and locked.json()["sm_mhz"] == 1815
            assert (await client.post("/gpus/0/lock", json={"mhz": 2520})).status_code == 400
            assert (await client.post("/gpus/3/lock", json={"mhz": 1305})).status_code == 404
            readings = (await client.get("/gpus")).json()
            assert readings[0]["index"] == 0 and readings[0]["sm_mhz"] == 1815

    asyncio.run(run())


def test_agent_allows_supported_clocks_above_min_and_lists_them() -> None:
    backend = DryRunBackend([0], supported=range(210, 2521, 15))
    service = ClockService(backend, [0], min_mhz=600)
    assert service.lock(0, 1815).ok
    with pytest.raises(ValueError):
        service.lock(0, 450)  # below the configured floor
    with pytest.raises(ValueError):
        service.lock(0, 1816)  # not a supported clock
    app = create_agent_app(service)

    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://agent") as client:
            clocks = (await client.get("/gpus/0/clocks")).json()["sm_mhz"]
            assert clocks[0] == 600 and clocks[-1] == 2520
            assert (await client.get("/gpus/3/clocks")).status_code == 404

    asyncio.run(run())
