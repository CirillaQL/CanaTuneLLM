"""ASGI application construction."""

import os
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI

from canatune.api.control import create_control_router
from canatune.config import load_config
from canatune.proxy.proxy import create_proxy_router, parse_endpoints


def create_app(
    config: Mapping[str, Any] | None = None,
    client_factory: Callable[[], httpx.AsyncClient] | None = None,
    runtime_factory: Callable[..., Any] | None = None,
) -> FastAPI:
    """Create the CanaTune ASGI application.

    `routing.policy` selects `round_robin` (baseline, no clock control) or `cantune`
    (risk-based admission, tier Controller, telemetry and Canary API).
    """
    settings = config
    if settings is None:
        settings = load_config(os.environ.get("CANATUNE_CONFIG", "config.yaml"))
    policy = settings.get("routing", {}).get("policy", "round_robin")
    if policy not in ("round_robin", "cantune"):
        raise ValueError("routing.policy must be round_robin or cantune")

    runtime = None
    if policy == "cantune":
        from canatune.service import build_runtime

        factory = client_factory or (
            lambda: httpx.AsyncClient(timeout=httpx.Timeout(5, connect=2), trust_env=False)
        )
        metrics_urls = {
            name: f"http://{e.http_host}:{e.http_port}/metrics"
            for name, e in parse_endpoints(settings).items()
        }
        runtime = (runtime_factory or build_runtime)(settings, metrics_urls, factory)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        if runtime is not None:
            await runtime.start()
        try:
            yield
        finally:
            if runtime is not None:
                await runtime.shutdown()

    app = FastAPI(
        title="CanaTune",
        description="Proxy and controller for disaggregated vLLM deployments.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.runtime = runtime
    app.include_router(
        create_proxy_router(settings, client_factory=client_factory, runtime=runtime)
    )
    if runtime is not None:
        app.include_router(create_control_router(runtime))
    return app
