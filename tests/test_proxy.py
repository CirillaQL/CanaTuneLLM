"""CPU-only request flow tests for the P/D proxy."""

import asyncio
import json

import httpx
import pytest

from canatune.app import create_app
from canatune.config import load_config

# NixlConnector: P names the blocks D reads (vLLM's kv_transfer_params).
KV_PARAMS = {
    "do_remote_prefill": True,
    "do_remote_decode": False,
    "remote_block_ids": [1, 2],
    "remote_engine_id": "p-engine",
    "remote_request_id": "p-request",
    "remote_host": "127.0.0.1",
    "remote_port": 14579,
}
PREFILL_OK = {"choices": [], "kv_transfer_params": KV_PARAMS}


def make_client(handler, connector=None):
    config = load_config()
    if connector is not None:
        config["kv_transfer"]["connector"] = connector
    config["topology"]["prefill_nodegroup"]["node"] = "127.0.0.1"
    config["topology"]["decode_nodegroup"]["node"] = "127.0.0.2"
    transport = httpx.MockTransport(handler)
    app = create_app(
        config,
        client_factory=lambda: httpx.AsyncClient(transport=transport),
    )
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy")


def test_production_round_robin_and_shared_kv_request_id() -> None:
    """P2pNcclConnector: P pushes to the D address carried in the request id."""
    upstream = []

    def handler(request):
        upstream.append(request)
        if request.url.port in (8101, 8102, 8103):
            return httpx.Response(200, json={"choices": []})
        return httpx.Response(200, json={"choices": [{"text": "done"}]})

    async def run():
        async with make_client(handler, connector="P2pNcclConnector") as client:
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


def test_nixl_decode_reads_the_blocks_prefill_names() -> None:
    """NixlConnector: P keeps the KV (do_remote_decode) and D gets P's
    kv_transfer_params; the request id stays as the client gave it."""
    upstream = []

    def handler(request):
        upstream.append(request)
        if request.url.port < 8200:
            return httpx.Response(200, json=PREFILL_OK)
        return httpx.Response(200, json={"choices": [{"text": "done"}]})

    async def run():
        async with make_client(handler) as client:
            response = await client.post(
                "/v1/completions",
                headers={"X-Request-Id": "my-request"},
                json={"model": "example", "prompt": "hello", "max_tokens": 8},
            )
            assert response.status_code == 200

    asyncio.run(run())
    prefill, decode = upstream
    assert prefill.headers["X-Request-Id"] == decode.headers["X-Request-Id"] == "my-request"
    sent = json.loads(prefill.read())
    assert sent["kv_transfer_params"]["do_remote_decode"] is True
    assert sent["max_tokens"] == 1 and sent["stream"] is False
    assert json.loads(decode.read()) == {
        "model": "example", "prompt": "hello", "max_tokens": 8, "kv_transfer_params": KV_PARAMS
    }


def test_nixl_prefill_always_stops_at_its_one_token() -> None:
    """NIXL hands KV over only when P stops at max_tokens: EOS and stop strings are
    off for P; D keeps the client's conditions."""
    upstream = []

    def handler(request):
        upstream.append(request)
        if request.url.port < 8200:
            return httpx.Response(200, json=PREFILL_OK)
        return httpx.Response(200, json={"choices": [{"text": "done"}]})

    body = {"model": "m", "prompt": "hi", "max_tokens": 8, "stop": ["\n"], "min_tokens": 2,
            "stop_token_ids": [2]}

    async def run():
        async with make_client(handler) as client:
            assert (await client.post("/v1/completions", json=body)).status_code == 200

    asyncio.run(run())
    prefill, decode = (json.loads(r.read()) for r in upstream)
    assert prefill["ignore_eos"] is True and prefill["max_tokens"] == 1
    assert not {"stop", "stop_token_ids", "min_tokens"} & prefill.keys()
    assert decode == {**body, "kv_transfer_params": KV_PARAMS}


def test_nixl_prefill_without_kv_params_never_reaches_decode() -> None:
    """Without kv_transfer_params D would compute the prompt itself: an error instead."""
    upstream = []

    def handler(request):
        upstream.append(request)
        return httpx.Response(200, json={"choices": []})

    async def run():
        async with make_client(handler) as client:
            response = await client.post(
                "/v1/completions", json={"model": "example", "prompt": "hello", "max_tokens": 8}
            )
            assert response.status_code == 502
            assert "kv_transfer_params" in response.json()["error"]

    asyncio.run(run())
    assert [request.url.port for request in upstream] == [8101]


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
            return httpx.Response(200, json=PREFILL_OK)
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
    assert upstream[0].headers["X-Request-Id"] == "my-request"
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
        return httpx.Response(200, json=PREFILL_OK)

    async def run():
        async with make_client(handler) as client:
            response = await client.post(
                "/v1/completions",
                json={"model": "example", "prompt": "hello", "stream": True},
            )
            assert response.status_code == 502
            assert response.json()["error"] == "decode did not return an SSE stream"

    asyncio.run(run())


def make_cantune(handler, admission="cells", overload="reject"):
    from canatune.service import build_runtime

    config = load_config()
    config["routing"]["policy"] = "cantune"
    config["router"]["admission"] = admission
    config["router"]["overload"] = overload
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
            return httpx.Response(200, json=PREFILL_OK)
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


def test_cantune_slack_admission_tracks_stages_through_the_proxy() -> None:
    from canatune.domain.groups import ClockPoint, TierTable

    tokens = b'data: {"choices":[{"text":"t"}]}\n\n' * 3 + b"data: [DONE]\n\n"
    seen = {}

    class SSEStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            # The decode call starts after P returned: the KV is in flight now.
            group = runtime_ref["rt"].router.groups[1]
            seen["inflight"] = group.inflight_bytes
            seen["at_p"] = group.n_at_p
            yield tokens

    def handler(request):
        if request.url.port < 8200:
            return httpx.Response(200, json=PREFILL_OK)
        return httpx.Response(
            200, stream=SSEStream(), headers={"content-type": "text/event-stream"}
        )

    runtime_ref = {}

    async def run():
        client, runtime = make_cantune(handler, admission="slack")
        runtime_ref["rt"] = runtime
        table = TierTable(ClockPoint(900, 450), ClockPoint(1815, 1050), 3000.0, 460.0)
        table.evidence = {
            "prefill_ms_by_clock": {"1815": {"128": 40, "2048": 150}},
            "admission": {  # as the Canary's calibration publishes it
                "predictor_coef": [150.0, 1.0, 1.0, 800.0, 2.0],
                "slack_counts": {str(b): [50, 0] for b in range(9, 14)},
                "kv_gate_fraction": 0.5,
            },
        }
        runtime.controller.tiers.table = table
        await runtime.start()
        runtime.stop.set()
        async with client:
            response = await client.post(
                "/v1/completions",
                json={"model": "m", "prompt": [5] * 1000, "max_tokens": 3, "stream": True},
            )
            assert response.status_code == 200
            record = runtime.router.log.recent[-1]
            assert record["status"] == "ok" and record["predicted_ms"] is not None
            assert record["timing"]["prefill_ms"] is not None
            state = (await client.get("/canatune/state")).json()["router"]
            assert state["admission"] == "slack"
            g1 = next(g for g in state["groups"] if g["name"] == "G1")
            assert (g1["n_at_p"], g1["inflight_mb"], g1["n_decoding"]) == (0, 0, 0)
        await asyncio.gather(*runtime.tasks, return_exceptions=True)

    asyncio.run(run())
    assert seen == {"inflight": 1000 * 131072, "at_p": 0}
    assert runtime_ref["rt"].router.predictor.state()["samples"] == 1


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


def test_stage_ms_breaks_ttft_into_stages() -> None:
    from canatune.proxy.proxy import stage_ms

    stamps = {
        "prefill_sent": 10.01,
        "prefill_done": 10.16,
        "decode_sent": 10.161,
        "first_token": 10.4,
    }
    timing = stage_ms(10.0, stamps)
    assert timing["admit_ms"] is None
    assert timing["to_prefill_ms"] == pytest.approx(10.0)
    assert timing["prefill_ms"] == pytest.approx(150.0)
    assert timing["gap_ms"] == pytest.approx(1.0)
    assert timing["decode_first_ms"] == pytest.approx(239.0)
    assert stage_ms(10.0, {})["prefill_ms"] is None


@pytest.mark.parametrize("stage", ["prefill", "decode"])
def test_cancelled_request_releases_its_reservation(stage) -> None:
    """A request cancelled after admission and before streaming (client gone,
    timeout, shutdown) leaves no count behind in the Router."""
    from canatune.domain.groups import ClockPoint, TierTable

    entered = asyncio.Event()

    async def handler(request):
        is_prefill = request.url.port < 8200
        if (stage == "prefill") == is_prefill:
            entered.set()
            await asyncio.Event().wait()  # never answers
        return httpx.Response(200, json=PREFILL_OK)

    async def run():
        client, runtime = make_cantune(handler, admission="slack")
        table = TierTable(ClockPoint(900, 450), ClockPoint(1815, 1050), 3000.0, 460.0)
        table.evidence = {
            "prefill_ms_by_clock": {"1815": {"128": 40, "2048": 150}},
            "admission": {
                "predictor_coef": [150.0, 1.0, 1.0, 800.0, 2.0],
                "slack_counts": {str(b): [50, 0] for b in range(9, 14)},
                "kv_gate_fraction": 0.5,
            },
        }
        runtime.controller.tiers.table = table
        await runtime.start()
        runtime.stop.set()
        async with client:
            task = asyncio.create_task(
                client.post(
                    "/v1/completions",
                    json={"model": "m", "prompt": [5] * 500, "max_tokens": 3, "stream": True},
                )
            )
            await asyncio.wait_for(entered.wait(), 5)
            busy = [g for g in runtime.router.groups if g.n_inflight]
            assert len(busy) == 1
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        group = busy[0]
        assert (group.n_inflight, group.n_at_p, group.n_await) == (0, 0, 0)
        assert group.pending_ms == 0 and group.inflight_bytes == 0
        assert runtime.router.log.recent[-1]["status"] == "client_disconnected"
        await asyncio.gather(*runtime.tasks, return_exceptions=True)

    asyncio.run(run())


def test_proxy_marks_non_streaming_requests_for_production_feedback() -> None:
    seen = []
    runtime_ref = {}

    class SSEStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"choices":[{"text":"t"}]}\n\ndata: [DONE]\n\n'

    def handler(request):
        if request.url.port >= 8200:  # decode call: the request is admitted and live
            live = list(runtime_ref["rt"].router._live.values())
            seen.extend(t.streaming for t in live)
            if live and live[0].streaming:
                return httpx.Response(
                    200, stream=SSEStream(), headers={"content-type": "text/event-stream"}
                )
        return httpx.Response(200, json={**PREFILL_OK, "choices": [{"text": "t"}]})

    async def run():
        client, runtime = make_cantune(handler, admission="slack", overload="best_effort")
        runtime_ref["rt"] = runtime
        await runtime.start()
        runtime.stop.set()
        async with client:
            for stream in (False, True):
                response = await client.post(
                    "/v1/completions",
                    json={"model": "m", "prompt": [5] * 64, "max_tokens": 1, "stream": stream},
                )
                assert response.status_code == 200
        await asyncio.gather(*runtime.tasks, return_exceptions=True)

    asyncio.run(run())
    assert seen == [False, True]


CHAT_SSE = (
    b'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}\n\n'
    b'data: {"choices":[{"index":0,"delta":{"content":"Hel"}}]}\n\n'
    b'data: {"choices":[{"index":0,"delta":{"content":"lo"}}]}\n\n'
    b'data: {"choices":[{"index":0,"delta":{"content":"!"},"finish_reason":"stop"}]}\n\n'
    b'data: {"choices":[],"usage":{"prompt_tokens":9,"completion_tokens":3}}\n\n'
    b"data: [DONE]\n\n"
)


def test_chat_token_chunks_skip_the_role_announcement_and_empty_deltas() -> None:
    from canatune.proxy.proxy import StreamTimer, is_token_chunk

    assert is_token_chunk({"text": ""})  # completions: an empty text is still a token
    assert not is_token_chunk({"delta": {"role": "assistant", "content": ""}})
    assert is_token_chunk({"delta": {"role": "assistant", "content": "Hi"}})
    assert is_token_chunk({"delta": {"content": ""}})
    assert not is_token_chunk({"delta": {}, "finish_reason": "stop"})
    assert is_token_chunk({"delta": {"tool_calls": [{"index": 0}]}})
    timer = StreamTimer(clock=iter([1.0, 1.1, 1.2]).__next__)
    assert timer.feed(CHAT_SSE) == 3  # role chunk and usage chunk are no tokens


def test_chat_prompt_length_uses_the_chat_template() -> None:
    from canatune.proxy.proxy import chat_prompt_tokens

    class Tokenizer:
        chat_template = "{{ messages }}"

        def apply_chat_template(self, messages, chat_template=None, **kwargs):
            self.seen = (chat_template, kwargs["add_generation_prompt"], kwargs["tokenize"])
            return [1] + [7] * sum(len(m["content"].split()) for m in messages) + [2]

    messages = [{"role": "user", "content": "one two three"}]
    tokenizer = Tokenizer()
    assert chat_prompt_tokens({"messages": messages}, tokenizer, None, 4.0) == (5, True)
    assert tokenizer.seen == (None, True, True)
    # The template vLLM serves with is the one rendered; a request's own wins.
    chat_prompt_tokens({"messages": messages}, tokenizer, "served", 4.0)
    assert tokenizer.seen[0] == "served"
    chat_prompt_tokens({"messages": messages, "chat_template": "own"}, tokenizer, "served", 4.0)
    assert tokenizer.seen[0] == "own"
    # No tokenizer (or no template anywhere): estimated, not exact; parts count as text.
    parts = [{"role": "user", "content": [{"type": "text", "text": "x" * 40}]}]
    assert chat_prompt_tokens({"messages": parts}, None, None, 4.0) == (10, False)
    tokenizer.chat_template = None
    assert chat_prompt_tokens({"messages": messages}, tokenizer, None, 4.0)[1] is False
    with pytest.raises(ValueError):
        chat_prompt_tokens({"prompt": "hi"}, tokenizer, None, 4.0)


def test_chat_completions_go_through_p_and_d_chat_endpoints() -> None:
    upstream = []
    runtime_ref = {}

    class SSEStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield CHAT_SSE

    def handler(request):
        upstream.append(request)
        if request.url.port < 8200:
            return httpx.Response(
                200, json={**PREFILL_OK, "choices": [{"message": {"content": "H"}}]}
            )
        return httpx.Response(
            200, stream=SSEStream(), headers={"content-type": "text/event-stream"}
        )

    body = {
        "model": "m",
        "messages": [{"role": "user", "content": "hello there"}],
        "max_tokens": 3,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    async def run():
        client, runtime = make_cantune(handler, admission="slack", overload="best_effort")
        runtime_ref["rt"] = runtime
        await runtime.start()
        runtime.stop.set()
        async with client:
            response = await client.post("/v1/chat/completions", json=body)
            assert response.status_code == 200
            assert response.content == CHAT_SSE
            record = runtime.router.log.recent[-1]
            assert record["status"] == "ok" and record["output_tokens"] == 3
            assert record["ttft_ms"] is not None and record["tpot_ms"] is not None
            bad = await client.post("/v1/chat/completions", json={"model": "m", "prompt": "x"})
            assert bad.status_code == 400
        await asyncio.gather(*runtime.tasks, return_exceptions=True)

    asyncio.run(run())
    prefill, decode = upstream
    assert prefill.url.path == decode.url.path == "/v1/chat/completions"
    assert prefill.headers["X-Request-Id"] == decode.headers["X-Request-Id"]
    sent = json.loads(prefill.read())
    assert sent["max_tokens"] == 1 and sent["stream"] is False and "stream_options" not in sent
    assert sent["messages"] == body["messages"]
    assert json.loads(decode.read()) == {**body, "kv_transfer_params": KV_PARAMS}


def test_baseline_serves_chat_and_models() -> None:
    upstream = []

    def handler(request):
        upstream.append(request)
        if request.url.path == "/v1/models":
            if request.url.port == 8201:  # the first production D is down
                return httpx.Response(503, json={"error": "down"})
            return httpx.Response(200, json={"data": [{"id": "mistral"}]})
        if request.url.port < 8200:
            return httpx.Response(200, json=PREFILL_OK)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    async def run():
        async with make_client(handler) as client:
            chat = await client.post(
                "/v1/chat/completions",
                json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            )
            assert chat.status_code == 200
            assert chat.json() == {"choices": [{"message": {"content": "ok"}}]}
            models = await client.get("/v1/models")
            assert models.status_code == 200 and models.json()["data"][0]["id"] == "mistral"

    asyncio.run(run())
    assert [r.url.path for r in upstream[:2]] == ["/v1/chat/completions"] * 2
    assert [r.url.port for r in upstream[2:]] == [8201, 8202]


def test_an_error_event_in_the_decode_stream_is_a_decode_error() -> None:
    """vLLM ends a failed request (a KV read under load_failure_policy fail) with an
    error event in an otherwise normal stream: the proxy records it as decode_error."""
    from canatune.proxy.proxy import StreamTimer

    timer = StreamTimer(clock=lambda: 1.0)
    timer.feed(b'data: {"choices":[{"text":"a"}]}\n\n')
    timer.feed(b'data: {"error": {"message": "Internal server error"}}\n\ndata: [DONE]\n\n')
    assert len(timer.token_times) == 1 and timer.error == "Internal server error"

    class SSEStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"error": {"message": "Internal server error"}}\n\ndata: [DONE]\n\n'

    def handler(request):
        if request.url.port < 8200:
            return httpx.Response(200, json=PREFILL_OK)
        return httpx.Response(
            200, stream=SSEStream(), headers={"content-type": "text/event-stream"}
        )

    async def run():
        client, runtime = make_cantune(handler, admission="slack", overload="best_effort")
        await runtime.start()
        runtime.stop.set()
        async with client:
            response = await client.post(
                "/v1/completions",
                json={"model": "m", "prompt": "hello", "max_tokens": 4, "stream": True},
            )
            assert response.status_code == 200
            assert runtime.router.log.recent[-1]["status"] == "decode_error"
        await asyncio.gather(*runtime.tasks, return_exceptions=True)

    asyncio.run(run())
