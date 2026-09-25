"""CPU-only request flow tests for the P/D proxy."""

import asyncio

import httpx
import pytest

from canatune.app import create_app
from canatune.config import load_config


def make_client(handler):
    config = load_config()
    config["topology"]["prefill_nodegroup"]["node"] = "127.0.0.1"
    config["topology"]["decode_nodegroup"]["node"] = "127.0.0.2"
    transport = httpx.MockTransport(handler)
    app = create_app(
        config,
        client_factory=lambda: httpx.AsyncClient(transport=transport),
    )
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy")


def test_production_round_robin_and_shared_kv_request_id() -> None:
    upstream = []

    def handler(request):
        upstream.append(request)
        if request.url.port in (8101, 8102, 8103):
            return httpx.Response(200, json={"choices": []})
        return httpx.Response(200, json={"choices": [{"text": "done"}]})

    async def run():
        async with make_client(handler) as client:
            for expected in ("P1", "P2", "P3", "P1"):
                response = await client.post(
                    "/v1/completions",
                    json={"model": "example", "prompt": "hello", "max_tokens": 8},
                )
                assert response.status_code == 200
                assert response.headers["X-CanaTune-Prefill-Endpoint"] == expected
                assert response.json() == {"choices": [{"text": "done"}]}

    asyncio.run(run())
    assert len(upstream) == 8
    for prefill, decode in zip(upstream[::2], upstream[1::2], strict=True):
        assert prefill.headers["X-Request-Id"] == decode.headers["X-Request-Id"]
        assert "___prefill_addr_127.0.0.1:145" in prefill.headers["X-Request-Id"]
        assert "___decode_addr_127.0.0.2:145" in prefill.headers["X-Request-Id"]
        assert prefill.url.port in (8101, 8102, 8103)
        assert decode.url.port == prefill.url.port + 100
        assert prefill.read() != decode.read()


def test_canary_stream_is_forwarded_from_decode() -> None:
    upstream = []
    sse = b'data: {"choices":[{"text":"hello"}]}\n\ndata: [DONE]\n\n'

    class SSEStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield sse[:18]
            yield sse[18:]

    def handler(request):
        upstream.append(request)
        if request.url.port == 8100:
            return httpx.Response(200, json={"choices": []})
        return httpx.Response(
            200, stream=SSEStream(), headers={"content-type": "text/event-stream"}
        )

    async def run():
        async with make_client(handler) as client:
            response = await client.post(
                "/v1/completions",
                headers={"X-CanaTune-Route": "canary", "X-Request-Id": "my-request"},
                json={"model": "example", "prompt": "hello", "stream": True, "max_tokens": 8},
            )
            assert response.status_code == 200
            assert response.content == sse
            assert response.headers["content-type"].startswith("text/event-stream")
            assert response.headers["X-CanaTune-Prefill-Endpoint"] == "P0"
            assert response.headers["X-CanaTune-Decode-Endpoint"] == "D0"

    asyncio.run(run())
    assert [request.url.port for request in upstream] == [8100, 8200]
    assert upstream[0].headers["X-Request-Id"].endswith("_my-request")
    assert upstream[0].headers["X-Request-Id"] == upstream[1].headers["X-Request-Id"]
    assert upstream[0].read().find(b'"max_tokens":1') != -1
    assert upstream[0].read().find(b'"stream":false') != -1
    assert upstream[1].read().find(b'"max_tokens":8') != -1
    assert upstream[1].read().find(b'"stream":true') != -1


def test_prefill_failure_does_not_call_decode() -> None:
    upstream = []

    def handler(request):
        upstream.append(request)
        return httpx.Response(503)

    async def run():
        async with make_client(handler) as client:
            response = await client.post(
                "/v1/completions", json={"model": "example", "prompt": "hello"}
            )
            assert response.status_code == 502
            assert "prefill returned HTTP 503" in response.json()["error"]

    asyncio.run(run())
    assert len(upstream) == 1


def test_stream_requires_real_decode_sse() -> None:
    def handler(request):
        return httpx.Response(200, json={"choices": []})

    async def run():
        async with make_client(handler) as client:
            response = await client.post(
                "/v1/completions",
                json={"model": "example", "prompt": "hello", "stream": True},
            )
            assert response.status_code == 502
            assert response.json()["error"] == "decode did not return an SSE stream"

    asyncio.run(run())


def make_cantune(handler):
    from canatune.service import build_runtime

    config = load_config()
    config["routing"]["policy"] = "cantune"
    config["telemetry"]["enabled"] = False
    config["topology"]["prefill_nodegroup"]["node"] = "127.0.0.1"
    config["topology"]["decode_nodegroup"]["node"] = "127.0.0.2"
    transport = httpx.MockTransport(handler)

    def factory():
        return httpx.AsyncClient(transport=transport)

    holder = {}

    def runtime_factory(settings, metrics_urls, client_factory):
        holder["runtime"] = build_runtime(settings, metrics_urls, client_factory)
        return holder["runtime"]

    app = create_app(config, client_factory=factory, runtime_factory=runtime_factory)
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy")
    return client, holder["runtime"]


def test_stream_timer_handles_split_events() -> None:
    from canatune.proxy.proxy import StreamTimer

    ticks = iter([1.0, 1.1, 1.3])
    timer = StreamTimer(clock=lambda: next(ticks))
    data = (
        b'data: {"choices":[{"text":"a"}]}\n\ndata: {"choices":[{"text":""}]}\n\n'
        b'data: {"choices":[{"text":"c"}]}\n\ndata: [DONE]\n\n'
    )
    assert timer.feed(data[:10]) == 0
    assert timer.feed(data[10:50]) == 1
    assert timer.feed(data[50:]) == 2
    assert timer.tpot_ms() == pytest.approx(150.0)


def test_cantune_cold_start_admits_all_then_risk_admission() -> None:
    from canatune.domain.groups import ClockPoint, TierTable

    tokens = (
        b"".join(b'data: {"choices":[{"text":"t%d"}]}\n\n' % i for i in range(4))
        + b"data: [DONE]\n\n"
    )

    class SSEStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield tokens

    def handler(request):
        if request.url.port < 8200:
            return httpx.Response(200, json={"choices": []})
        return httpx.Response(
            200, stream=SSEStream(), headers={"content-type": "text/event-stream"}
        )

    async def run():
        client, runtime = make_cantune(handler)
        await runtime.start()
        runtime.stop.set()
        async with client:
            # Cold start: no tier table, production at MAX, every request admitted.
            response = await client.post(
                "/v1/completions",
                json={"model": "m", "prompt": [5] * 100, "max_tokens": 4, "stream": True},
            )
            assert response.status_code == 200
            assert response.headers["X-CanaTune-Group"] == "G1"
            assert response.content == tokens
            record = runtime.router.log.recent[-1]
            assert record["status"] == "ok" and record["output_tokens"] == 4
            assert record["table_skip"] is None and record["violated"] is False
            assert record["cell"].startswith("2520/1500|")  # sample at the MAX point
            long = await client.post(
                "/v1/completions", json={"model": "m", "prompt": [5] * 4000, "max_tokens": 4}
            )
            assert long.status_code == 200
            state = (await client.get("/canatune/state")).json()
            assert state["router"]["open_admission"] and state["router"]["admitted"] == 2
            assert all(g["n_inflight"] == 0 for g in state["router"]["groups"])
            assert list(runtime.lengths._pairs) == [(100, 4)]  # non-stream has no count

            # A published table switches to risk admission: unknown cells at H reject.
            table = TierTable(ClockPoint(900, 450), ClockPoint(1815, 1050), 3000.0, 460.0)
            runtime.controller.settings = type(runtime.controller.settings)(stagger_s=0.0)
            await runtime.controller.publish(table, "test")
            assert (await client.get("/canatune/tiers")).json()["table"]["h"] == "1815/1050"
            rejected = await client.post(
                "/v1/completions", json={"model": "m", "prompt": [5] * 128, "max_tokens": 4}
            )
            assert rejected.status_code == 503
            assert rejected.headers["X-CanaTune-Rejected"] == "1"
            risk = (await client.get("/canatune/risk")).json()
            assert risk["exploration_queue"]
        await asyncio.gather(*runtime.tasks, return_exceptions=True)

    asyncio.run(run())


def test_cantune_lifespan_starts_and_stops_control_loops(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CANATUNE_RECORD_DIR", str(tmp_path))
    monkeypatch.setenv("CANATUNE_RISK_TABLE", str(tmp_path / "risk.json"))
    metrics = 'vllm:num_requests_running{m="x"} 0\nvllm:kv_cache_usage_perc{m="x"} 0.1\n'

    def handler(request):
        return httpx.Response(200, text=metrics)

    config = load_config()
    config["routing"]["policy"] = "cantune"
    config["telemetry"]["period_s"] = 0.01
    config["controller"]["period_s"] = 0.01
    transport = httpx.MockTransport(handler)
    app = create_app(config, client_factory=lambda: httpx.AsyncClient(transport=transport))
    runtime = app.state.runtime

    async def run():
        async with app.router.lifespan_context(app):
            await asyncio.sleep(0.1)
            # No clock control: no locator, so the Canary serves at MAX as well.
            assert all(
                g.state.value == "active" and g.effective is not None for g in runtime.groups
            )
            assert runtime.telemetry.fresh("D1", 1.0).kv_usage == 0.1

    asyncio.run(run())
    assert (tmp_path / "risk.json").exists()
    assert (tmp_path / "events.jsonl").read_text().count('"event": "tier"') >= 4
