"""ASGI application construction."""

from fastapi import FastAPI


def create_app() -> FastAPI:
    """Create the CanaTune ASGI application."""
    return FastAPI(
        title="CanaTune",
        description="Proxy and controller for disaggregated vLLM deployments.",
        version="0.1.0",
    )


app = create_app()
