"""CPU-only request flow tests for the P/D proxy."""

import asyncio

import httpx

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
