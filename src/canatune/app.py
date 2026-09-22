"""ASGI application construction."""

import os
from collections.abc import Callable, Mapping
from typing import Any

import httpx
from fastapi import FastAPI

from canatune.config import load_config
from canatune.proxy.proxy import create_proxy_router


def create_app(
    config: Mapping[str, Any] | None = None,
    client_factory: Callable[[], httpx.AsyncClient] | None = None,
) -> FastAPI:
    """Create the CanaTune ASGI application."""
    settings = config
    if settings is None:
        settings = load_config(os.environ.get("CANATUNE_CONFIG", "config.yaml"))
    app = FastAPI(
        title="CanaTune",
        description="Proxy and controller for disaggregated vLLM deployments.",
        version="0.1.0",
    )
    app.include_router(create_proxy_router(settings, client_factory=client_factory))
    return app


app = create_app()
