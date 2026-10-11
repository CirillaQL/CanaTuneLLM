# CanaTune

## 1. Project Scope

CanaTune is a Prefill/Decode (P/D) disaggregated inference system for vLLM that
aims to reduce the energy of inference while keeping the SLOs (TTFT and TPOT) as
far as possible. It saves energy through GPU DVFS and P/D request scheduling, and
it finds the clock ranges and SLO judgments that keep the SLOs while saving energy
from online real-time feedback and experiments on a deployed Canary group, not
from an offline-trained model.

## 2. Project Structure

- `src/canatune/api`: HTTP-facing API boundary.
- `src/canatune/controller`: request-placement orchestration.
- `src/canatune/proxy`: upstream vLLM transport boundary.
- `src/canatune/domain`: shared domain models and contracts.
- `src/canatune/infrastructure`: runtime and external-system adapters.

## 3. Configuration

- The project configuration entry point is `config.yaml`.
- Load configuration through `canatune.config.load_config`.
- `${...}` expressions are preserved during YAML loading and are not yet resolved.
- Keep machine-specific paths, node names, credentials, and other sensitive values out of
  committed configuration.

## 4. Development

- Python 3.10 or newer is required (the cluster vLLM env is 3.10).
- Install dependencies with `uv sync --extra dev`.
- Run tests with `uv run pytest`.
- Run static checks with `uv run ruff check .`.
