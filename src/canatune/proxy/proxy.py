"""HTTP proxy for vLLM P2pNcclConnector prefill/decode pairs."""

import asyncio
import json
import math
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

    @property
    def chat_url(self) -> str:
        return f"http://{self.http_host}:{self.http_port}/v1/chat/completions"

    @property
    def models_url(self) -> str:
        return f"http://{self.http_host}:{self.http_port}/v1/models"


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


def is_token_chunk(choice: Mapping[str, Any]) -> bool:
    """One choice chunk is one generated token, even if its text is "". Completions
    carry `text`; chat carries a `delta`, whose first chunk only announces the role
    and whose last may be empty (finish reason): neither is a token."""
    if choice.get("text") is not None:
        return True
    delta = choice.get("delta")
    if not isinstance(delta, Mapping):
        return False
    produced = [delta.get(k) for k in ("content", "reasoning_content", "tool_calls")]
    if "role" in delta and not any(produced):
        return False
    return any(v is not None for v in produced)


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
                if choices and is_token_chunk(choices[0]):
                    self.token_times.append(self._clock())
                    new += 1

    def tpot_ms(self) -> float | None:
        times = self.token_times
        if len(times) < 2:
            return None
        return (times[-1] - times[0]) * 1000.0 / (len(times) - 1)


def load_tokenizer(config: Mapping[str, Any]) -> Any:
    """HF tokenizer of the served model (router.tokenizer, CANATUNE_TOKENIZER or
    MODEL_PATH); None when unavailable: text prompts are then estimated and never
    enter the risk tables."""
    path = (
        os.environ.get("CANATUNE_TOKENIZER")
        or config.get("router", {}).get("tokenizer")
        or os.environ.get("MODEL_PATH")
    )
    if not path:
        return None
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(path)
    except Exception:
        return None


def load_chat_template(config: Mapping[str, Any]) -> str | None:
    """The chat template vLLM serves with (VLLM_CHAT_TEMPLATE, as the launch scripts
    pass it, or router.chat_template): a file or the template itself. None: the
    tokenizer's own template."""
    value = os.environ.get("VLLM_CHAT_TEMPLATE") or config.get("router", {}).get("chat_template")
    if not value:
        return None
    if os.path.isfile(value):
        with open(value, encoding="utf-8") as handle:
            return handle.read()
    return str(value)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # content parts: only text parts count here
        return "".join(str(part.get("text", "")) for part in content if isinstance(part, Mapping))
    return ""


def chat_prompt_tokens(
    body: Mapping[str, Any], tokenizer: Any, template: str | None, chars_per_token: float
) -> tuple[int, bool]:
    """Prompt length of a chat request: the messages rendered with the chat template
    (as vLLM renders them; the template adds the special tokens) and tokenized.
    Without a tokenizer or template the text is estimated (not exact)."""
    messages = body.get("messages")
    if (
        not isinstance(messages, list)
        or not messages
        or not all(isinstance(m, Mapping) for m in messages)
    ):
        raise ValueError("messages must be a nonempty list of message objects")
    template = body.get("chat_template") or template
    if tokenizer is not None and (template or getattr(tokenizer, "chat_template", None)):
        try:
            ids = tokenizer.apply_chat_template(
                messages,
                chat_template=template,
                tools=body.get("tools"),
                add_generation_prompt=body.get("add_generation_prompt", True),
                continue_final_message=body.get("continue_final_message", False),
                tokenize=True,
                **(body.get("chat_template_kwargs") or {}),
            )
            if isinstance(ids, Mapping) or hasattr(ids, "input_ids"):
                ids = ids["input_ids"]
            return max(1, len(ids)), True
        except Exception:
            pass  # vLLM reports the error itself; estimate for the Router meanwhile
    text = "".join(_content_text(m.get("content")) for m in messages)
    return max(1, math.ceil(len(text) / chars_per_token)), False


def stage_ms(arrived: float, stamps: Mapping[str, float]) -> dict[str, float | None]:
    """TTFT breakdown in ms: admission (Router wait), P round trip (proxy -> P -> proxy,
    P queue + prefill + P front end), proxy gap before the decode call, and decode
    first token (D front end + KV wait/load + first step). Missing stages are None."""

    def span(a: str | None, b: str) -> float | None:
        start = arrived if a is None else stamps.get(a)
        end = stamps.get(b)
        return None if start is None or end is None else (end - start) * 1000.0

    return {
        "admit_ms": span(None, "admitted"),
        "to_prefill_ms": span("admitted" if "admitted" in stamps else None, "prefill_sent"),
        "prefill_ms": span("prefill_sent", "prefill_done"),
        "gap_ms": span("prefill_done", "decode_sent"),
        "decode_first_ms": span("decode_sent", "first_token"),
    }


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
    chat_path = proxy.get("chat_endpoint", "/v1/chat/completions")
    models_path = proxy.get("models_endpoint", "/v1/models")
    for name, value in (("chat_endpoint", chat_path), ("models_endpoint", models_path)):
        if value is not None and (not isinstance(value, str) or not value.startswith("/")):
            raise ProxyConfigError(f"proxy.{name} must be an absolute URL path or null")
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
    tokenizer = load_tokenizer(config)

    chat_template = load_chat_template(config)

    def count_tokens(body: Mapping[str, Any], kind: str) -> tuple[int, bool]:
        """Exact prompt length: token ids as given, text through the model's tokenizer
        (special tokens included, as vLLM counts them), chat messages through the chat
        template; estimated without a tokenizer."""
        if kind == "chat":
            return chat_prompt_tokens(body, tokenizer, chat_template, chars_per_token)
        prompt = body.get("prompt")
        if tokenizer is not None and isinstance(prompt, str):
            return max(1, len(tokenizer(prompt).input_ids)), True
        return prompt_tokens(body, chars_per_token)

    router = APIRouter()

    @router.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "policy": "cantune" if runtime is not None else "round_robin",
            "production_pairs": selector.production_pairs,
        }

    async def handle(request: Request, kind: str) -> Response:
        """One request of `kind` (completions or chat): P with max_tokens 1, then the
        same body to D, whose response is returned (streamed or not)."""
        arrived = time.monotonic()
        # Stage boundaries (monotonic) for the per-request TTFT breakdown (E0).
        stamps: dict[str, float] = {}
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
            tokens, exact = count_tokens(body, kind)
        except ValueError as error:
            if runtime is not None:
                return JSONResponse({"error": str(error)}, status_code=400)
            tokens, exact = 0, False

        ticket = None
        if runtime is not None and route == "production":
            ticket = await runtime.router.admit(tokens, exact)
            if ticket is None:
                return JSONResponse(
                    {"error": (
                        "service wait timed out: no active group has capacity"
                        if runtime.router.settings.overload == "best_effort" else
                        "rejected: no active group can serve this request within the SLO"
                    )},
                    status_code=503,
                    headers={"X-CanaTune-Rejected": "1"},
                )
            ticket.streaming = body.get("stream") is True
            stamps["admitted"] = time.monotonic()
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

        finished = False

        def finish(status: str, timer: StreamTimer | None) -> None:
            nonlocal finished
            if finished:
                return
            finished = True
            times = timer.token_times if timer is not None else []
            ttft_ms = (times[0] - arrived) * 1000.0 if times else None
            tpot_ms = timer.tpot_ms() if timer is not None else None
            if times:
                stamps["first_token"] = times[0]
            timing = stage_ms(arrived, stamps)
            if ticket is not None:
                runtime.router.finish(
                    ticket,
                    status=status,
                    ttft_ms=ttft_ms,
                    tpot_ms=tpot_ms,
                    output_tokens=len(times),
                    timing=timing,
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
                        "timing": timing,
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

        # Admitted but not yet streaming: a cancellation (client gone, timeout,
        # shutdown) is no HTTPError, so the reservation is released here.
        handed_off = False
        clients: list[httpx.AsyncClient] = []
        try:
            try:
                async with factory() as client:
                    stamps["prefill_sent"] = time.monotonic()
                    prefill_response = await client.post(
                        prefill.chat_url if kind == "chat" else prefill.completions_url,
                        json=prefill_body,
                        headers=headers,
                    )
                    stamps["prefill_done"] = time.monotonic()
                if prefill_response.status_code == 200 and ticket is not None:
                    runtime.router.prefill_done(ticket)
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
            clients.append(client)
            try:
                decode_request = client.build_request(
                    "POST",
                    decode.chat_url if kind == "chat" else decode.completions_url,
                    json=body,
                    headers=headers,
                )
                stamps["decode_sent"] = time.monotonic()
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
                                new_tokens = timer.feed(chunk)
                                if new_tokens and ticket is not None:
                                    runtime.router.first_token(ticket)
                                    times = timer.token_times
                                    # Recent spacing responds to changed clocks/load;
                                    # a whole-request average retains pre-expansion delays.
                                    recent_tpot = (
                                        (times[-1] - times[-new_tokens - 1]) * 1000 / new_tokens
                                        if len(times) > new_tokens else None
                                    )
                                    runtime.router.token_progress(ticket, recent_tpot)
                                yield chunk
                            status = "ok"
                        except httpx.HTTPError:
                            status = "decode_error"
                        finally:
                            finish(status, timer)
                            await decode_response.aclose()
                            await client.aclose()

                    handed_off = True  # stream() finishes the request from here on
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

        finally:
            if not handed_off:
                for c in clients:
                    if not c.is_closed:
                        await c.aclose()
                finish("client_disconnected", None)  # no-op once finished

    @router.post(path)
    async def completions(request: Request) -> Response:
        return await handle(request, "completions")

    if chat_path is not None:

        @router.post(chat_path)
        async def chat_completions(request: Request) -> Response:
            return await handle(request, "chat")

    if models_path is not None:

        @router.get(models_path)
        async def models() -> Response:
            """Every pair serves the same model: the first production D that answers."""
            for _, decode_name in selector.production_pairs:
                try:
                    async with factory() as client:
                        response = await client.get(selector.endpoints[decode_name].models_url)
                except httpx.HTTPError:
                    continue
                if response.status_code == 200:
                    return Response(
                        content=response.content,
                        media_type=response.headers.get("content-type", "application/json"),
                    )
            return JSONResponse({"error": "no decode endpoint answered"}, status_code=502)

    return router
