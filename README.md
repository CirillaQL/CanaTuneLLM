# CanaTune

CanaTune is a proxy/controller service for disaggregated vLLM deployments. It
will coordinate request placement across prefill (P) and decode (D) nodes.

This repository currently contains only the project skeleton. Node discovery,
scheduling, request forwarding, retries, and health management are intentionally
not implemented yet.

## Project layout

```text
src/canatune/
├── api/             # HTTP-facing API boundary
├── controller/      # Request-placement orchestration
├── domain/          # Shared domain models and contracts
├── infrastructure/  # External systems and runtime adapters
├── proxy/           # Upstream vLLM transport boundary
├── app.py           # ASGI application factory
└── __main__.py      # Local process entry point
```

## Development

Python 3.11 or newer is required.

```bash
uv sync --extra dev
uv run python -m canatune
```

Run the project checks with:

```bash
uv run pytest
uv run ruff check .
```
