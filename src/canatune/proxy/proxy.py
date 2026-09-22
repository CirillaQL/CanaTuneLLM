"""HTTP proxy for vLLM P2pNcclConnector prefill/decode pairs."""

import asyncio
import os
import re
import uuid
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse


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


class PairSelector:
    """Round robin over configured production pairs; canary is explicit."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        topology = _mapping(config.get("topology"), "topology")
        routing = _mapping(config.get("routing"), "routing")
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
        self.endpoints: dict[str, Endpoint] = {}
        for name, raw in raw_endpoints.items():
            value = _mapping(raw, f"endpoint {name}")
            role = value.get("role")
            if role not in hosts:
                raise ProxyConfigError(f"endpoint {name} has invalid role")
            self.endpoints[name] = Endpoint(
                name=name,
                role=role,
                http_host=hosts[role],
                kv_host=kv_hosts[role],
                http_port=_port(value.get("http_port"), f"{name}.http_port"),
                kv_port=_port(value.get("kv_port"), f"{name}.kv_port"),
            )

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


def _request_id(value: str | None, prefill: Endpoint, decode: Endpoint) -> str:
    logical_id = uuid.uuid4().hex if value is None else value
    if len(logical_id) > 256 or not re.fullmatch(r"[A-Za-z0-9._-]+", logical_id):
        raise ValueError(
            "X-Request-Id must contain 1-256 letters, digits, dots, underscores or hyphens"
        )
    return (
        f"___prefill_addr_{prefill.kv_host}:{prefill.kv_port}"
        f"___decode_addr_{decode.kv_host}:{decode.kv_port}_{logical_id}"
    )


def create_proxy_router(
    config: Mapping[str, Any],
    client_factory: Callable[[], httpx.AsyncClient] | None = None,
) -> APIRouter:
    """Build a completions endpoint from configured P/D pairs."""
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
    router = APIRouter()

    @router.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "production_pairs": selector.production_pairs}

    @router.post(path)
    async def completions(request: Request) -> Response:
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
        prefill, decode = await selector.choose(canary=route == "canary")
        try:
            transport_id = _request_id(request.headers.get("X-Request-Id"), prefill, decode)
        except ValueError as error:
            return JSONResponse({"error": str(error)}, status_code=400)

        headers = {"X-Request-Id": transport_id}
        response_headers = {
            "X-CanaTune-Prefill-Endpoint": prefill.name,
            "X-CanaTune-Decode-Endpoint": decode.name,
        }
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
                return JSONResponse(
                    {"error": f"prefill returned HTTP {prefill_response.status_code}"},
                    status_code=502,
                    headers=response_headers,
                )
        except httpx.HTTPError:
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
                    return JSONResponse(
                        {"error": "decode did not return an SSE stream"},
                        status_code=502,
                        headers=response_headers,
                    )

                async def stream() -> AsyncIterator[bytes]:
                    try:
                        async for chunk in decode_response.aiter_raw():
                            yield chunk
                    finally:
                        await decode_response.aclose()
                        await client.aclose()

                return StreamingResponse(
                    stream(), media_type="text/event-stream", headers=response_headers
                )

            result = await decode_response.aread()
            await decode_response.aclose()
            await client.aclose()
            return Response(content=result, media_type=content_type, headers=response_headers)
        except httpx.HTTPError:
            await client.aclose()
            return JSONResponse(
                {"error": "decode request failed"}, status_code=502, headers=response_headers
            )

    return router
