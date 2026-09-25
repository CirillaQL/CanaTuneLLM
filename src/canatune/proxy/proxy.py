"""HTTP proxy for vLLM P2pNcclConnector prefill/decode pairs."""

import asyncio
import json
import os
import re
import time
import uuid
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from canatune.controller.router import prompt_tokens
from canatune.infrastructure.records import JsonlLog

if TYPE_CHECKING:
    from canatune.service import Runtime


class ProxyConfigError(ValueError):
    """Raised when endpoint or route configuration is invalid."""


@dataclass(frozen=True)
class Endpoint:
    name: str
    role: str
    http_host: str
    kv_host: str
    http_port: int
    kv_port: int

    @property
    def completions_url(self) -> str:
        return f"http://{self.http_host}:{self.http_port}/v1/completions"


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProxyConfigError(f"{name} must be a mapping")
    return value


def _port(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 65535:
        raise ProxyConfigError(f"{name} must be a TCP port")
    return value


def _host(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or not re.fullmatch(r"[A-Za-z0-9._-]+", value):
        raise ProxyConfigError(f"{name} must be an IPv4 address or hostname")
    return value


def parse_endpoints(config: Mapping[str, Any]) -> dict[str, Endpoint]:
    """Endpoints with hosts resolved from the node group or environment overrides."""
    topology = _mapping(config.get("topology"), "topology")
    raw_endpoints = _mapping(topology.get("endpoints"), "topology.endpoints")
    prefill_group = _mapping(topology.get("prefill_nodegroup"), "prefill_nodegroup")
    decode_group = _mapping(topology.get("decode_nodegroup"), "decode_nodegroup")
    hosts = {
        "prefill": _host(
            os.environ.get("CANATUNE_PREFILL_HOST") or prefill_group.get("node"),
            "prefill host",
        ),
        "decode": _host(
            os.environ.get("CANATUNE_DECODE_HOST") or decode_group.get("node"),
            "decode host",
        ),
    }
    kv_hosts = {
        "prefill": _host(
            os.environ.get("CANATUNE_PREFILL_KV_HOST") or hosts["prefill"],
            "prefill KV host",
        ),
        "decode": _host(
            os.environ.get("CANATUNE_DECODE_KV_HOST") or hosts["decode"],
            "decode KV host",
        ),
    }
    endpoints: dict[str, Endpoint] = {}
    for name, raw in raw_endpoints.items():
        value = _mapping(raw, f"endpoint {name}")
        role = value.get("role")
        if role not in hosts:
            raise ProxyConfigError(f"endpoint {name} has invalid role")
        endpoints[name] = Endpoint(
            name=name,
            role=role,
            http_host=hosts[role],
            kv_host=kv_hosts[role],
            http_port=_port(value.get("http_port"), f"{name}.http_port"),
            kv_port=_port(value.get("kv_port"), f"{name}.kv_port"),
        )
    return endpoints


class PairSelector:
    """Round robin over configured production pairs; canary is explicit."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        routing = _mapping(config.get("routing"), "routing")
        self.endpoints = parse_endpoints(config)
        self.canary_pair = self._pair(routing.get("canary_pair"))
        raw_pairs = routing.get("production_pairs")
        if not isinstance(raw_pairs, list) or not raw_pairs:
            raise ProxyConfigError("routing.production_pairs must be a nonempty list")
        self.production_pairs = tuple(self._pair(value) for value in raw_pairs)
        if len(set(self.production_pairs)) != len(self.production_pairs):
            raise ProxyConfigError("production pairs must be unique")
        self._index = 0
        self._lock = asyncio.Lock()

    def _pair(self, value: Any) -> tuple[str, str]:
        if not isinstance(value, list) or len(value) != 2:
            raise ProxyConfigError("each route must contain one P and one D endpoint")
        prefill, decode = value
        if prefill not in self.endpoints or decode not in self.endpoints:
            raise ProxyConfigError(f"unknown endpoint in pair {value!r}")
        if self.endpoints[prefill].role != "prefill" or self.endpoints[decode].role != "decode":
            raise ProxyConfigError(f"invalid endpoint roles in pair {value!r}")
        return prefill, decode

    async def choose(self, canary: bool) -> tuple[Endpoint, Endpoint]:
        if canary:
            pair = self.canary_pair
        else:
            async with self._lock:
                pair = self.production_pairs[self._index % len(self.production_pairs)]
                self._index += 1
        return self.endpoints[pair[0]], self.endpoints[pair[1]]


def pd_transport_id(value: str | None, prefill: Endpoint, decode: Endpoint) -> str:
    """Request id carrying both KV addresses (P2pNcclConnector routing)."""
    logical_id = uuid.uuid4().hex if value is None else value
    if len(logical_id) > 256 or not re.fullmatch(r"[A-Za-z0-9._-]+", logical_id):
        raise ValueError(
            "X-Request-Id must contain 1-256 letters, digits, dots, underscores or hyphens"
        )
    return (
        f"___prefill_addr_{prefill.kv_host}:{prefill.kv_port}"
        f"___decode_addr_{decode.kv_host}:{decode.kv_port}_{logical_id}"
    )


class StreamTimer:
    """Finds token events in a vLLM SSE stream split across arbitrary chunks and
    records when each arrives (first token -> TTFT, spacing -> TPOT)."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._buffer = b""
        self.token_times: list[float] = []

    def feed(self, chunk: bytes) -> int:
        self._buffer += chunk
        new = 0
        while True:
            for separator in (b"\r\n\r\n", b"\n\n"):
                index = self._buffer.find(separator)
                if index >= 0:
                    break
            else:
                return new
            event, self._buffer = self._buffer[:index], self._buffer[index + len(separator) :]
            for line in event.splitlines():
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    continue
                try:
                    choices = json.loads(payload).get("choices") or []
                except (ValueError, AttributeError):
                    continue
                # One choice chunk is one generated token, even if its text is "".
                if choices and choices[0].get("text") is not None:
                    self.token_times.append(self._clock())
                    new += 1

    def tpot_ms(self) -> float | None:
        times = self.token_times
        if len(times) < 2:
            return None
        return (times[-1] - times[0]) * 1000.0 / (len(times) - 1)


def create_proxy_router(
    config: Mapping[str, Any],
    client_factory: Callable[[], httpx.AsyncClient] | None = None,
    runtime: "Runtime | None" = None,
) -> APIRouter:
    """Build a completions endpoint from configured P/D pairs.

    With a `runtime` (routing policy `cantune`) each request goes through risk-based
    admission; otherwise production pairs are used round robin. Both paths log one
    record per request when `CANATUNE_RECORD_DIR` is set.
    """
    proxy = _mapping(config.get("proxy"), "proxy")
    if proxy.get("enabled") is not True:
        raise ProxyConfigError("proxy.enabled must be true")
    path = proxy.get("endpoint")
    if not isinstance(path, str) or not path.startswith("/"):
        raise ProxyConfigError("proxy.endpoint must be an absolute URL path")
    experiment = _mapping(config.get("experiment"), "experiment")
    workload = _mapping(experiment.get("workload"), "workload")
    timeout_s = workload.get("request_timeout_s", 900)
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or timeout_s <= 0:
        raise ProxyConfigError("experiment.workload.request_timeout_s must be positive")
    factory = client_factory or (
        lambda: httpx.AsyncClient(timeout=httpx.Timeout(timeout_s, connect=10), trust_env=False)
    )
    selector = PairSelector(config)
    record_dir = os.environ.get("CANATUNE_RECORD_DIR")
    baseline_log = JsonlLog(
        None
        if runtime is not None or not record_dir
        else os.path.join(record_dir, "requests.jsonl")
    )
    chars_per_token = float(config.get("router", {}).get("chars_per_token", 4.0))
    router = APIRouter()

    @router.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "policy": "cantune" if runtime is not None else "round_robin",
            "production_pairs": selector.production_pairs,
        }

    @router.post(path)
    async def completions(request: Request) -> Response:
        arrived = time.monotonic()
        try:
            body = await request.json()
        except ValueError:
            return JSONResponse({"error": "request body must be JSON"}, status_code=400)
        if not isinstance(body, dict):
            return JSONResponse({"error": "request body must be a JSON object"}, status_code=400)

        route = request.headers.get("X-CanaTune-Route", "production")
        if route not in ("production", "canary"):
            return JSONResponse(
                {"error": "X-CanaTune-Route must be production or canary"}, status_code=400
            )
        if not isinstance(body.get("stream", False), bool):
            return JSONResponse({"error": "stream must be a boolean"}, status_code=400)
        try:
            tokens, exact = prompt_tokens(body, chars_per_token)
        except ValueError as error:
            if runtime is not None:
                return JSONResponse({"error": str(error)}, status_code=400)
            tokens, exact = 0, False

        ticket = None
        if runtime is not None and route == "production":
            ticket = await runtime.router.admit(tokens, exact)
            if ticket is None:
                return JSONResponse(
                    {"error": "rejected: no active group can serve this request within the SLO"},
                    status_code=503,
                    headers={"X-CanaTune-Rejected": "1"},
                )
            prefill = selector.endpoints[ticket.group.prefill]
            decode = selector.endpoints[ticket.group.decode]
        else:
            # Explicit Canary probes bypass admission: they are experiments.
            prefill, decode = await selector.choose(canary=route == "canary")
        try:
            transport_id = pd_transport_id(request.headers.get("X-Request-Id"), prefill, decode)
        except ValueError as error:
            if ticket is not None:
                runtime.router.finish(
                    ticket, status="bad_request", ttft_ms=None, tpot_ms=None, output_tokens=0
                )
            return JSONResponse({"error": str(error)}, status_code=400)

        def finish(status: str, timer: StreamTimer | None) -> None:
            times = timer.token_times if timer is not None else []
            ttft_ms = (times[0] - arrived) * 1000.0 if times else None
            tpot_ms = timer.tpot_ms() if timer is not None else None
            if ticket is not None:
                runtime.router.finish(
                    ticket,
                    status=status,
                    ttft_ms=ttft_ms,
                    tpot_ms=tpot_ms,
                    output_tokens=len(times),
                )
            else:
                baseline_log.write(
                    {
                        "event": "request",
                        "route": route,
                        "prefill": prefill.name,
                        "decode": decode.name,
                        "prompt_tokens": tokens,
                        "prompt_exact": exact,
                        "status": status,
                        "ttft_ms": ttft_ms,
                        "tpot_ms": tpot_ms,
                        "output_tokens": len(times),
                    }
                )

        headers = {"X-Request-Id": transport_id}
        response_headers = {
            "X-CanaTune-Prefill-Endpoint": prefill.name,
            "X-CanaTune-Decode-Endpoint": decode.name,
        }
        if ticket is not None:
            response_headers["X-CanaTune-Group"] = ticket.group.name
        prefill_body = dict(body)
        prefill_body["stream"] = False
        prefill_body["max_tokens"] = 1
        if "max_completion_tokens" in prefill_body:
            prefill_body["max_completion_tokens"] = 1
        prefill_body.pop("stream_options", None)

        try:
            async with factory() as client:
                prefill_response = await client.post(
                    prefill.completions_url, json=prefill_body, headers=headers
                )
            if prefill_response.status_code != 200:
                finish("prefill_error", None)
                return JSONResponse(
                    {"error": f"prefill returned HTTP {prefill_response.status_code}"},
                    status_code=502,
                    headers=response_headers,
                )
        except httpx.HTTPError:
            finish("prefill_error", None)
            return JSONResponse(
                {"error": "prefill request failed"}, status_code=502, headers=response_headers
            )

        client = factory()
        try:
            decode_request = client.build_request(
                "POST", decode.completions_url, json=body, headers=headers
            )
            decode_response = await client.send(decode_request, stream=True)
            if decode_response.status_code != 200:
                await decode_response.aclose()
                await client.aclose()
                finish("decode_error", None)
                return JSONResponse(
                    {"error": f"decode returned HTTP {decode_response.status_code}"},
                    status_code=502,
                    headers=response_headers,
                )
            content_type = decode_response.headers.get("content-type", "application/json")
            if body.get("stream") is True:
                if "text/event-stream" not in content_type.lower():
                    await decode_response.aclose()
                    await client.aclose()
                    finish("decode_error", None)
                    return JSONResponse(
                        {"error": "decode did not return an SSE stream"},
                        status_code=502,
                        headers=response_headers,
                    )

                timer = StreamTimer()

                async def stream() -> AsyncIterator[bytes]:
                    status = "client_disconnected"
                    try:
                        async for chunk in decode_response.aiter_raw():
                            if timer.feed(chunk) and ticket is not None:
                                runtime.router.first_token(ticket)
                            yield chunk
                        status = "ok"
                    except httpx.HTTPError:
                        status = "decode_error"
                    finally:
                        finish(status, timer)
                        await decode_response.aclose()
                        await client.aclose()

                return StreamingResponse(
                    stream(), media_type="text/event-stream", headers=response_headers
                )

            result = await decode_response.aread()
            await decode_response.aclose()
            await client.aclose()
            # Non-streaming: TTFT is unknown, so the outcome never enters the risk table.
            finish("ok", None)
            return Response(content=result, media_type=content_type, headers=response_headers)
        except httpx.HTTPError:
            await client.aclose()
            finish("decode_error", None)
            return JSONResponse(
                {"error": "decode request failed"}, status_code=502, headers=response_headers
            )

    return router
