# CanaTune

CanaTune is a proxy/controller service for disaggregated vLLM deployments. It
coordinates request placement across prefill (P) and decode (D) nodes.

The current proxy accepts OpenAI-compatible `POST /v1/completions` requests.
It sends a one-token prefill request, then forwards the original request to a
decode node using the same vLLM KV-transfer request ID. Non-streaming responses
and real decode SSE streams are returned to the client.

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

Fill in the site-specific paths and node names in `config.yaml`, or set
`CANATUNE_CONFIG` to another YAML file. For jobs with dynamically allocated
nodes, set `CANATUNE_PREFILL_HOST` and `CANATUNE_DECODE_HOST` to the reachable
node IPs or hostnames before starting the proxy. If KV transfer uses another
interface, set `CANATUNE_PREFILL_KV_HOST` and `CANATUNE_DECODE_KV_HOST` as well.
The proxy binds to `proxy.host` and `proxy.port` in the YAML file.

After the P/D servers are running, send a completion request to the proxy:

```bash
curl http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"mistralai/Mistral-7B-v0.1","prompt":"Hello","max_tokens":16}'
```

Production requests rotate through `routing.production_pairs`. An explicit
`X-CanaTune-Route: canary` header selects `routing.canary_pair`. The pair used
is returned in `X-CanaTune-Prefill-Endpoint` and
`X-CanaTune-Decode-Endpoint` response headers. `GET /health` reports the
proxy process configuration; it does not probe vLLM nodes.

The proxy expects the vLLM servers to be started separately, for example with
the scripts in `scripts/`. Only explicitly listed pairs are used. Cross-pair
routing, dynamic health-based selection, frequency tuning, retries, and
`${...}` configuration expansion are not implemented yet.

Run the project checks with:

```bash
uv run pytest
uv run ruff check .
```
