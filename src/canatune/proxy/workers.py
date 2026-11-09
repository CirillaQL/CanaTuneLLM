"""Multi-process proxy: one decision process and N proxy worker processes.

One event loop relaying every streamed token (httpx/h11 in, uvicorn out, SSE timing)
was the service's throughput limit (~28 req/s ShareGPT on Vera, the loop at ~97 % of a
core). With `proxy.workers` (or CANATUNE_PROXY_WORKERS) N > 0, `python -m canatune`
runs:

- the decision process (this process): Router, Controller, Canary, telemetry and the
  records; the control API (/canatune/*) and /health on a Unix socket. It binds the
  proxy's TCP socket and hands it to the workers, which all accept on it (an idle
  worker takes the next connection).
- N proxy workers (`python -m canatune.proxy.workers`): the proxy routes as in one
  process (prompt token counts, P then D, the SSE relay, stream timing), with a
  RemoteRouter in place of the Router; /canatune/* is forwarded to the decision
  process.

Each worker holds one Unix-socket connection to the decision process (frames: a 4-byte
length and a pickle; both ends are this program, run by the same user). Per request:
admit (answered: the ticket, or rejected), prefill_done, first_token and finish; the
token gaps of all of a worker's streams go in one message per router.stream_report_s
(a stall gap, tpot_stall_factor x the SLO, at once), as the one-process proxy reports
them. The baseline (round_robin) asks the decision process for the next pair, so the
round robin and the request record stay as in one process. A worker that goes away
releases its requests (client_disconnected); a worker that exits is restarted.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import os
import pickle
import shutil
import signal
import socket
import struct
import subprocess
import sys
import tempfile
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import Response

from canatune.config import load_config
from canatune.controller.router import RouterSettings
from canatune.infrastructure.records import JsonlLog
from canatune.proxy.proxy import PairSelector, create_proxy_router

_LENGTH = struct.Struct("!I")


def encode(message: Any) -> bytes:
    data = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
    return _LENGTH.pack(len(data)) + data


async def read_message(reader: asyncio.StreamReader) -> Any:
    (size,) = _LENGTH.unpack(await reader.readexactly(_LENGTH.size))
    return pickle.loads(await reader.readexactly(size))


def proxy_workers(config: Mapping[str, Any], env: Mapping[str, str] | None = None) -> int:
    """Number of proxy worker processes: CANATUNE_PROXY_WORKERS, else proxy.workers;
    0 (default): one process serves everything."""
    env = os.environ if env is None else env
    raw = env.get("CANATUNE_PROXY_WORKERS") or config.get("proxy", {}).get("workers", 0)
    try:
        workers = int(raw)
    except (TypeError, ValueError):
        raise ValueError(f"proxy workers must be an integer >= 0, got {raw!r}") from None
    if workers < 0:
        raise ValueError(f"proxy workers must be >= 0, got {workers}")
    return workers


# ---- decision process -------------------------------------------------------------


@dataclass(eq=False)  # identity: a set member
class _Worker:
    """One worker connection: its admitted tickets and its admissions in progress."""

    writer: asyncio.StreamWriter
    tickets: dict[int, Any] = field(default_factory=dict)
    admitting: dict[int, asyncio.Task] = field(default_factory=dict)

    def send(self, message: Any) -> None:
        if not self.writer.is_closing():
            self.writer.write(encode(message))


class DecisionServer:
    """The decision process's side of the worker connections."""

    def __init__(self, runtime: Any, selector: PairSelector, request_log: JsonlLog) -> None:
        self.router = None if runtime is None else runtime.router
        self.events = None if runtime is None else runtime.events
        self.selector = selector
        self.request_log = request_log
        self.workers: set[_Worker] = set()
        self._next_pair = 0
        self._server: asyncio.AbstractServer | None = None
        self._tasks: set[asyncio.Task] = set()

    def event(self, record: dict[str, Any]) -> None:
        if self.events is not None:
            self.events.write(record)
        else:
            print(f"CanaTune: {record}", file=sys.stderr, flush=True)

    async def start(self, path: str) -> None:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(path)
        self._server = await asyncio.start_unix_server(self._serve, path=path)

    async def close(self) -> None:
        if self._server is not None:
            self._server.close()
        for worker in list(self.workers):
            worker.writer.close()
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        worker = _Worker(writer)
        self.workers.add(worker)
        task = asyncio.current_task()
        if task is not None:
            self._tasks.add(task)
        try:
            while True:
                message = await read_message(reader)
                try:
                    self._handle(worker, message)
                except Exception as error:  # one bad message never drops the worker
                    self.event({"event": "proxy_ipc_error", "op": message[0],
                                "error": repr(error)})  # fmt: skip
        except (asyncio.IncompleteReadError, ConnectionError, asyncio.CancelledError):
            pass
        finally:
            self.workers.discard(worker)
            self._release(worker)
            writer.close()
            if task is not None:
                self._tasks.discard(task)

    def _release(self, worker: _Worker) -> None:
        """The worker is gone: its admissions stop waiting, its requests end."""
        for admitting in list(worker.admitting.values()):
            admitting.cancel()
        tickets = list(worker.tickets.values())
        worker.tickets.clear()
        for ticket in tickets:
            self.router.finish(ticket, status="client_disconnected", ttft_ms=None,
                               tpot_ms=None, output_tokens=0)  # fmt: skip
        if tickets:
            self.event({"event": "proxy_worker_lost", "requests": len(tickets)})

    def _handle(self, worker: _Worker, message: tuple) -> None:
        op = message[0]
        if op == "reports":
            for ticket_id, gaps in message[1]:
                ticket = worker.tickets.get(ticket_id)
                if ticket is not None:
                    self.router.stream_report(ticket, gaps)
        elif op == "admit":
            _, rid, tokens, exact, streaming = message
            task = asyncio.create_task(self._admit(worker, rid, tokens, exact, streaming))
            worker.admitting[rid] = task
        elif op == "prefill_done":
            ticket = worker.tickets.get(message[1])
            if ticket is not None:
                self.router.prefill_done(ticket)
        elif op == "first_token":
            ticket = worker.tickets.get(message[1])
            if ticket is not None:
                self.router.first_token(ticket)
                self.router.stream_report(ticket)  # TTFT feedback at once
        elif op == "finish":
            _, ticket_id, gaps, status, ttft_ms, tpot_ms, output_tokens, timing = message
            ticket = worker.tickets.pop(ticket_id, None)
            if ticket is not None:
                if gaps:
                    self.router.stream_report(ticket, gaps)
                self.router.finish(ticket, status=status, ttft_ms=ttft_ms, tpot_ms=tpot_ms,
                                   output_tokens=output_tokens, timing=timing)  # fmt: skip
        elif op == "cancel":
            admitting = worker.admitting.get(message[1])
            if admitting is not None:
                admitting.cancel()
        elif op == "choose":
            pairs = self.selector.production_pairs
            prefill, decode = pairs[self._next_pair % len(pairs)]
            self._next_pair += 1
            worker.send(("chosen", message[1], prefill, decode))
        elif op == "log":
            self.request_log.write(message[1])
        else:
            raise ValueError(f"unknown message {op!r}")

    async def _admit(self, worker: _Worker, rid: int, tokens: int, exact: bool,
                     streaming: bool) -> None:  # fmt: skip
        try:
            ticket = await self.router.admit(tokens, exact, streaming=streaming)
        finally:
            worker.admitting.pop(rid, None)
        if ticket is None:
            worker.send(("rejected", rid))
            return
        if worker not in self.workers:  # the worker went away while this waited
            self.router.finish(ticket, status="client_disconnected", ttft_ms=None,
                               tpot_ms=None, output_tokens=0)  # fmt: skip
            return
        worker.tickets[ticket.id] = ticket
        group, dgroup = ticket.group, ticket.dgroup
        worker.send(("admitted", rid, ticket.id, group.name, group.prefill,
                     dgroup.name, dgroup.decode))  # fmt: skip


class WorkerPool:
    """The proxy worker processes: started with the listening socket's fd, restarted
    when one exits while the service runs (at most `max_restarts` times in all)."""

    def __init__(self, count: int, listen_fd: int, env: Mapping[str, str],
                 event: Callable[[dict[str, Any]], None], *,
                 command: Sequence[str] | None = None, max_restarts: int = 20) -> None:  # fmt: skip
        self.count = count
        self.listen_fd = listen_fd
        self.env = dict(env)
        self.event = event
        self.command = list(command or (sys.executable, "-m", "canatune.proxy.workers"))
        self.max_restarts = max_restarts
        self.restarts = 0
        self.processes: dict[int, subprocess.Popen[bytes]] = {}
        self._stopping = False
        self._monitor: asyncio.Task | None = None

    def _spawn(self, index: int) -> None:
        env = {**self.env, "CANATUNE_PROXY_FD": str(self.listen_fd),
               "CANATUNE_PROXY_WORKER": str(index)}  # fmt: skip
        self.processes[index] = subprocess.Popen(self.command, env=env,
                                                 pass_fds=(self.listen_fd,))  # fmt: skip

    def start(self) -> None:
        for index in range(self.count):
            self._spawn(index)
        self._monitor = asyncio.create_task(self._watch())

    async def _watch(self) -> None:
        while not self._stopping:
            await asyncio.sleep(1.0)
            for index, process in list(self.processes.items()):
                code = process.poll()
                if code is None or self._stopping:
                    continue
                restart = self.restarts < self.max_restarts
                self.event({"event": "proxy_worker_exit", "worker": index, "code": code,
                            "restart": restart})  # fmt: skip
                if restart:
                    self.restarts += 1
                    self._spawn(index)
                else:
                    del self.processes[index]

    async def stop(self, timeout_s: float = 10.0) -> None:
        self._stopping = True
        if self._monitor is not None:
            self._monitor.cancel()
            await asyncio.gather(self._monitor, return_exceptions=True)
        for process in self.processes.values():
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
        loop = asyncio.get_running_loop()
        for process in self.processes.values():
            try:
                await asyncio.wait_for(loop.run_in_executor(None, process.wait), timeout_s)
            except asyncio.TimeoutError:
                process.kill()


def create_decision_app(
    config: Mapping[str, Any],
    *,
    ipc_path: str,
    pool: WorkerPool | None = None,
    client_factory: Callable[[], httpx.AsyncClient] | None = None,
    runtime_factory: Callable[..., Any] | None = None,
) -> FastAPI:
    """The decision process's app: the control API and /health (served on a Unix
    socket), the worker connections at `ipc_path`, the worker processes (`pool`)."""
    from canatune.api.control import create_control_router
    from canatune.app import build_service_runtime

    runtime = build_service_runtime(config, client_factory, runtime_factory)
    selector = PairSelector(config)
    record_dir = os.environ.get("CANATUNE_RECORD_DIR")
    request_log = JsonlLog(
        os.path.join(record_dir, "requests.jsonl")
        if runtime is None and record_dir else None
    )  # fmt: skip
    server = DecisionServer(runtime, selector, request_log)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        if runtime is not None:
            await runtime.start()
        await server.start(ipc_path)
        if pool is not None:
            pool.event = server.event
            pool.start()
        try:
            yield
        finally:
            if pool is not None:
                await pool.stop()
            await server.close()
            if runtime is not None:
                await runtime.shutdown()

    app = FastAPI(title="CanaTune decision process", lifespan=lifespan)
    app.state.runtime = runtime
    app.state.decision = server

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "policy": "cantune" if runtime is not None else "round_robin",
            "production_pairs": selector.production_pairs,
            "proxy_workers": len(server.workers),
        }

    if runtime is not None:
        app.include_router(create_control_router(runtime))
    return app


def run_decision(config: Mapping[str, Any], workers: int) -> None:
    """`python -m canatune` with N > 0 proxy workers (see the module docstring)."""
    import uvicorn

    proxy = config["proxy"]
    listener = socket.create_server((str(proxy["host"]), int(proxy["port"])), backlog=2048)
    listener.set_inheritable(True)
    # Unix socket paths are short (108 bytes): node-local temp, not the job directory.
    workdir = tempfile.mkdtemp(prefix="canatune-")
    ipc_path = os.path.join(workdir, "ipc.sock")
    control_path = os.path.join(workdir, "control.sock")
    env = {**os.environ, "CANATUNE_IPC": ipc_path, "CANATUNE_CONTROL_UDS": control_path}
    env.pop("CANATUNE_PROFILE_OUT", None)  # the stack sampler profiles this process only
    pool = WorkerPool(workers, listener.fileno(), env, event=lambda r: None)
    app = create_decision_app(config, ipc_path=ipc_path, pool=pool)
    print(f"CanaTune: decision process, {workers} proxy workers on "
          f"{proxy['host']}:{proxy['port']}", flush=True)  # fmt: skip
    try:
        uvicorn.run(app, uds=control_path, access_log=False)
    finally:
        listener.close()
        shutil.rmtree(workdir, ignore_errors=True)


# ---- proxy worker -----------------------------------------------------------------


@dataclass(frozen=True)
class _GroupRef:
    name: str
    prefill: str | None = None
    decode: str | None = None


@dataclass
class RemoteTicket:
    """A worker's handle of a ticket the decision process holds."""

    id: int
    group: _GroupRef
    dgroup: _GroupRef


class DecisionUnavailable(ConnectionError):
    """The decision process went away."""


class RemoteRouter:
    """The Router as a proxy worker sees it: the calls the proxy makes, sent to the
    decision process. The proxy's StreamReporter hands every chunk's gaps here
    (stream_report_period_s 0); they are batched over all of this worker's streams."""

    stream_report_period_s = 0.0

    def __init__(self, settings: RouterSettings, path: str, *,
                 connect_timeout_s: float = 600.0,
                 on_lost: Callable[[], None] | None = None) -> None:  # fmt: skip
        self.settings = settings
        self.path = path
        self.connect_timeout_s = connect_timeout_s
        self.on_lost = on_lost
        self.report_s = 0.0 if settings.tpot_feedback == "gap" else settings.stream_report_s
        self.stall_ms = settings.tpot_stall_factor * settings.tpot_slo_ms
        self._pending: dict[int, list[float]] = {}
        self._waiting: dict[int, asyncio.Future] = {}
        self._ids = itertools.count(1)
        self._writer: asyncio.StreamWriter | None = None
        self._tasks: list[asyncio.Task] = []
        self.connected = False

    async def connect(self) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.connect_timeout_s
        while True:
            try:
                reader, self._writer = await asyncio.open_unix_connection(self.path)
                break
            except (FileNotFoundError, ConnectionError):
                if loop.time() >= deadline:
                    raise
                await asyncio.sleep(0.2)
        self.connected = True
        self._tasks = [asyncio.create_task(self._read(reader))]
        if self.report_s > 0:
            self._tasks.append(asyncio.create_task(self._flush_loop()))

    async def close(self) -> None:
        self._flush()
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        if self._writer is not None:
            self._writer.close()
        self.connected = False

    def _send(self, message: Any) -> None:
        if self._writer is not None and not self._writer.is_closing():
            self._writer.write(encode(message))

    async def _request(self, message: tuple) -> Any:
        if not self.connected:
            raise DecisionUnavailable("not connected to the decision process")
        rid = message[1]
        future = asyncio.get_running_loop().create_future()
        self._waiting[rid] = future
        self._send(message)
        try:
            return await future
        except asyncio.CancelledError:
            self._send(("cancel", rid))  # the client went away while it waited
            raise
        finally:
            self._waiting.pop(rid, None)

    async def admit(self, tokens: int, exact: bool, *, streaming: bool = True
                    ) -> RemoteTicket | None:  # fmt: skip
        try:
            return await self._request(("admit", next(self._ids), tokens, exact, streaming))
        except DecisionUnavailable:
            return None  # answered like a rejection (503)

    async def choose(self) -> tuple[str, str]:
        """The baseline's next production pair (the decision process's round robin)."""
        return await self._request(("choose", next(self._ids)))

    def write(self, record: dict[str, Any]) -> None:
        """The baseline's request record (JsonlLog.write), written by the decision
        process."""
        self._send(("log", record))

    def prefill_done(self, ticket: RemoteTicket) -> None:
        self._send(("prefill_done", ticket.id))

    def first_token(self, ticket: RemoteTicket) -> None:
        self._send(("first_token", ticket.id))

    def stream_report(self, ticket: RemoteTicket, gaps_ms: Sequence[float] = ()) -> None:
        self._pending.setdefault(ticket.id, []).extend(gaps_ms)
        if self.report_s == 0 or (gaps_ms and max(gaps_ms) >= self.stall_ms):
            self._flush()

    def finish(self, ticket: RemoteTicket, *, status: str, ttft_ms: float | None,
               tpot_ms: float | None, output_tokens: int,
               timing: Mapping[str, float | None] | None = None) -> None:  # fmt: skip
        gaps = self._pending.pop(ticket.id, [])
        self._send(("finish", ticket.id, gaps, status, ttft_ms, tpot_ms, output_tokens,
                    None if timing is None else dict(timing)))  # fmt: skip

    def _flush(self) -> None:
        if self._pending:
            items, self._pending = list(self._pending.items()), {}
            self._send(("reports", items))

    async def _flush_loop(self) -> None:
        while True:
            await asyncio.sleep(self.report_s)
            self._flush()

    async def _read(self, reader: asyncio.StreamReader) -> None:
        try:
            while True:
                message = await read_message(reader)
                op, rid = message[0], message[1]
                future = self._waiting.get(rid)
                if op == "admitted":
                    _, _, ticket_id, group, prefill, dgroup, decode = message
                    ticket = RemoteTicket(ticket_id, _GroupRef(group, prefill=prefill),
                                          _GroupRef(dgroup, decode=decode))  # fmt: skip
                    if future is None or future.done():  # cancelled meanwhile
                        self.finish(ticket, status="client_disconnected", ttft_ms=None,
                                    tpot_ms=None, output_tokens=0)  # fmt: skip
                    else:
                        future.set_result(ticket)
                elif future is not None and not future.done():
                    future.set_result(None if op == "rejected" else (message[2], message[3]))
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            self.connected = False
            for future in self._waiting.values():
                if not future.done():
                    future.set_exception(DecisionUnavailable("the decision process went away"))
            if self.on_lost is not None:
                self.on_lost()


def create_worker_app(
    config: Mapping[str, Any],
    remote: RemoteRouter,
    *,
    control_uds: str | None = None,
    control_transport: httpx.AsyncBaseTransport | None = None,
    client_factory: Callable[[], httpx.AsyncClient] | None = None,
) -> FastAPI:
    """A proxy worker's app: the proxy routes with `remote` for the Router, and the
    control API (/canatune/*) forwarded to the decision process."""
    cantune = config.get("routing", {}).get("policy", "round_robin") == "cantune"
    proxy_router = create_proxy_router(
        config,
        client_factory=client_factory,
        runtime=SimpleNamespace(router=remote) if cantune else None,
        choose_production=None if cantune else remote.choose,
        request_log=None if cantune else remote,
    )
    transport = control_transport or httpx.AsyncHTTPTransport(uds=control_uds)
    control = httpx.AsyncClient(transport=transport, base_url="http://decision",
                                timeout=120.0, trust_env=False)  # fmt: skip

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        await remote.connect()
        try:
            yield
        finally:
            await remote.close()
            await proxy_router.close_upstream()
            await control.aclose()

    app = FastAPI(title="CanaTune proxy worker", lifespan=lifespan)
    app.include_router(proxy_router)

    @app.api_route("/canatune/{path:path}", methods=["GET", "POST"])
    async def forward(path: str, request: Request) -> Response:
        try:
            answer = await control.request(
                request.method, f"/canatune/{path}", params=request.query_params,
                content=await request.body(),
                headers={"content-type": request.headers.get("content-type",
                                                             "application/json")},
            )  # fmt: skip
        except httpx.HTTPError as error:
            return Response(f'{{"error": "decision process: {error!r}"}}', status_code=502,
                            media_type="application/json")  # fmt: skip
        return Response(answer.content, status_code=answer.status_code,
                        media_type=answer.headers.get("content-type"))  # fmt: skip

    return app


def worker_main() -> None:
    """A proxy worker process (started by the decision process's WorkerPool)."""
    import uvicorn

    config = load_config(os.environ.get("CANATUNE_CONFIG", "config.yaml"))
    listener = socket.socket(fileno=int(os.environ["CANATUNE_PROXY_FD"]))
    remote = RemoteRouter(RouterSettings.from_config(config), os.environ["CANATUNE_IPC"])
    app = create_worker_app(config, remote, control_uds=os.environ["CANATUNE_CONTROL_UDS"])
    server = uvicorn.Server(uvicorn.Config(app, access_log=False, log_level="warning"))

    def lost() -> None:
        server.should_exit = True  # the decision process is gone: so is this worker

    remote.on_lost = lost
    server.run(sockets=[listener])


if __name__ == "__main__":
    worker_main()
