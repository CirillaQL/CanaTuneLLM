"""Multi-process proxy (canatune.proxy.workers): the decision process and its workers,
here in one event loop over a real Unix socket."""

import asyncio
import json
import os
import shutil
import tempfile
from contextlib import AsyncExitStack, asynccontextmanager

import httpx
import pytest

from canatune.config import load_config
from canatune.controller.router import RouterSettings
from canatune.infrastructure.records import JsonlLog
from canatune.proxy.proxy import PairSelector
from canatune.proxy.workers import (
    DecisionServer,
    RemoteRouter,
    create_decision_app,
    create_worker_app,
    proxy_workers,
)

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
TOKENS = 5
SSE = b"".join(b'data: {"choices":[{"text":"t"}]}\n\n' for _ in range(TOKENS)) + b"data: [DONE]\n\n"


def make_config(policy="cantune"):
    config = load_config()
    config["routing"]["policy"] = policy
    config["router"]["admission"] = "slack"
    config["router"]["overload"] = "best_effort"
    config["telemetry"]["enabled"] = False
    config["topology"]["prefill_nodegroup"]["node"] = "127.0.0.1"
    config["topology"]["decode_nodegroup"]["node"] = "127.0.0.2"
    return config


class Stream(httpx.AsyncByteStream):
    async def __aiter__(self):
        for i in range(0, len(SSE), 40):  # events split across chunks
            yield SSE[i:i + 40]
            await asyncio.sleep(0.001)


def handler(request):
    if request.url.port < 8200:
        return httpx.Response(200, json=PREFILL_OK)
    if json.loads(request.read()).get("stream"):
        return httpx.Response(200, stream=Stream(), headers={"content-type": "text/event-stream"})
    return httpx.Response(200, json={"choices": [{"text": "done"}]})


def upstream():
    transport = httpx.MockTransport(handler)
    return lambda: httpx.AsyncClient(transport=transport)


@pytest.fixture
def ipc_path():
    # Unix socket paths are short: not pytest's (long) tmp_path
    workdir = tempfile.mkdtemp(prefix="ct-", dir="/tmp")
    yield os.path.join(workdir, "ipc.sock")
    shutil.rmtree(workdir, ignore_errors=True)


@asynccontextmanager
async def service(config, ipc_path, workers=2):
    """The decision app (lifespan: runtime, worker connections) and `workers` worker
    apps, each with its own client."""
    holder = {}

    def runtime_factory(settings, metrics_urls, client_factory):
        from canatune.service import build_runtime

        holder["runtime"] = build_runtime(settings, metrics_urls, client_factory)
        return holder["runtime"]

    decision = create_decision_app(config, ipc_path=ipc_path, client_factory=upstream(),
                                   runtime_factory=runtime_factory)  # fmt: skip
    control = httpx.ASGITransport(app=decision)
    async with AsyncExitStack() as stack:
        await stack.enter_async_context(decision.router.lifespan_context(decision))
        clients = []
        for _ in range(workers):
            remote = RemoteRouter(RouterSettings.from_config(config), ipc_path)
            app = create_worker_app(config, remote, control_transport=control,
                                    client_factory=upstream())  # fmt: skip
            await stack.enter_async_context(app.router.lifespan_context(app))
            clients.append(await stack.enter_async_context(httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://proxy")))  # fmt: skip
        yield decision, clients


async def settle(server):
    for _ in range(50):  # the workers' last messages reach the decision process
        await asyncio.sleep(0.01)
        if not any(w.tickets or w.admitting for w in server.workers):
            return


def test_workers_serve_through_the_decision_processs_router(ipc_path):
    async def run():
        async with service(make_config(), ipc_path) as (decision, clients):
            runtime, server = decision.state.runtime, decision.state.decision
            await asyncio.sleep(0.02)  # the connections reach the decision process
            assert len(server.workers) == 2
            for i in range(6):
                response = await clients[i % 2].post(
                    "/v1/completions",
                    json={"model": "m", "prompt": [5] * 64, "max_tokens": TOKENS,
                          "stream": i % 3 != 0},  # fmt: skip
                )
                assert response.status_code == 200
                if i % 3:
                    assert response.content == SSE
            await settle(server)
            records = [r for r in runtime.router.log.recent if r.get("event") == "request"]
            assert len(records) == 6 and all(r["status"] == "ok" for r in records)
            streamed = [r for r in records if r["ttft_ms"] is not None]
            assert len(streamed) == 4 and all(r["output_tokens"] == TOKENS for r in streamed)
            assert all(r["timing"]["prefill_ms"] is not None for r in records)
            groups = runtime.router.groups
            assert not any(g.n_inflight or g.n_at_p or g.n_decoding for g in groups)
            assert not runtime.router._live
            # every streamed request's gaps reached its D's TPOT window
            assert sum(len(g) for g in runtime.router._gaps.values()) == 4 * (TOKENS - 1)
            health = await clients[0].get("/health")
            assert health.json()["policy"] == "cantune"
            state = await clients[1].get("/canatune/state")  # forwarded to the decision
            assert state.status_code == 200 and "router" in state.json()

    asyncio.run(run())


def test_a_lost_worker_releases_its_requests(ipc_path):
    async def run():
        async with service(make_config(), ipc_path, workers=0) as (decision, _):
            runtime, server = decision.state.runtime, decision.state.decision
            remote = RemoteRouter(RouterSettings.from_config(make_config()), ipc_path)
            await remote.connect()
            ticket = await remote.admit(64, True)
            remote.prefill_done(ticket)
            remote.first_token(ticket)
            await asyncio.sleep(0.05)
            assert sum(g.n_inflight for g in runtime.router.groups) == 1
            await remote.close()  # the worker goes away mid-request
            await settle(server)
            await asyncio.sleep(0.05)
            assert not any(g.n_inflight or g.n_decoding for g in runtime.router.groups)
            record = runtime.router.log.recent[-1]
            assert record["status"] == "client_disconnected"
            assert any(e.get("event") == "proxy_worker_lost" for e in runtime.events.recent)

    asyncio.run(run())


def test_baseline_round_robin_and_records_stay_global(ipc_path, tmp_path, monkeypatch):
    monkeypatch.setenv("CANATUNE_RECORD_DIR", str(tmp_path))

    async def run():
        async with service(make_config("round_robin"), ipc_path) as (decision, clients):
            assert decision.state.runtime is None
            for i in range(6):
                response = await clients[i % 2].post(
                    "/v1/completions", json={"model": "m", "prompt": "hi", "stream": True}
                )
                assert response.status_code == 200
            await asyncio.sleep(0.05)

    asyncio.run(run())
    records = [json.loads(line) for line in (tmp_path / "requests.jsonl").read_text().splitlines()]
    assert [r["prefill"] for r in records] == ["P1", "P2", "P3"] * 2  # one round robin
    assert all(r["status"] == "ok" and r["output_tokens"] == TOKENS for r in records)


class HoldingRouter:
    """admit waits until cancelled (a request held by best_effort)."""

    def __init__(self):
        self.cancelled = asyncio.Event()

    async def admit(self, tokens, exact, *, streaming=True):
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


def test_a_client_gone_while_held_cancels_its_admission(ipc_path):
    async def run():
        config = make_config()
        router = HoldingRouter()
        runtime = type("R", (), {"router": router, "events": JsonlLog(None)})()
        server = DecisionServer(runtime, PairSelector(config), JsonlLog(None))
        await server.start(ipc_path)
        remote = RemoteRouter(RouterSettings.from_config(config), ipc_path)
        await remote.connect()
        held = asyncio.create_task(remote.admit(64, True))
        await asyncio.sleep(0.05)
        held.cancel()  # the client disconnects while its request waits
        await asyncio.wait_for(router.cancelled.wait(), 1.0)
        await remote.close()
        await server.close()

    asyncio.run(run())


def test_admission_answers_rejected_once_the_decision_process_is_gone(ipc_path):
    async def run():
        config = make_config()
        runtime = type("R", (), {"router": HoldingRouter(), "events": JsonlLog(None)})()
        server = DecisionServer(runtime, PairSelector(config), JsonlLog(None))
        await server.start(ipc_path)
        lost = asyncio.Event()
        remote = RemoteRouter(RouterSettings.from_config(config), ipc_path, on_lost=lost.set)
        await remote.connect()
        held = asyncio.create_task(remote.admit(64, True))
        await asyncio.sleep(0.05)
        await server.close()
        assert await asyncio.wait_for(held, 1.0) is None  # -> 503, not a crash
        await asyncio.wait_for(lost.wait(), 1.0)
        assert await remote.admit(64, True) is None
        await remote.close()

    asyncio.run(run())


def test_proxy_workers_setting():
    assert proxy_workers({"proxy": {}}, {}) == 0
    assert proxy_workers({"proxy": {"workers": 3}}, {}) == 3
    assert proxy_workers({"proxy": {"workers": 3}}, {"CANATUNE_PROXY_WORKERS": "5"}) == 5
    with pytest.raises(ValueError):
        proxy_workers({"proxy": {"workers": -1}}, {})
