# CanaTune

CanaTune is the Router and Controller for a prefill/decode (P/D) disaggregated
vLLM deployment. Its goal is to **reduce total GPU energy while meeting the
latency SLO** (TTFT < 500 ms, TPOT ≤ 200 ms) by

1. **concentrating requests** on as few P/D groups as the SLO allows,
2. **parking idle groups** at a locked low clock instead of leaving them unlocked,
3. running active groups at **clock tiers the Canary measured on this cluster**
   (the loaded energy valley of P; the KV-wall-limited D), and
4. **admitting each request** only to a group whose current state is safe
   according to an **online risk table**; requests predicted to violate the SLO
   are rejected (HTTP 503).

No clock value, threshold or capacity is configured. A Canary group measures
them online on the running cluster, so the same code runs on an unknown cluster
(e.g. A100/H100). There is no offline-trained latency or power model. The GPU
count is fixed; the system never scales out.

> Status: v1 (2026-09-25). Design: `CanTuning_系统设计_v2.md`. The tier locator
> was validated offline by replaying K2/K3b/K4a data; it has not run on the
> cluster yet (next: smoke test).

## Deployment assumptions

| Item | Value |
|---|---|
| Model | Mistral-7B-v0.1, vLLM 0.15.1, TP = 1 |
| Prefill | 4 × NVIDIA L40S (P0–P3) |
| Decode | 4 × NVIDIA L4 (D0–D3) |
| KV transfer | `P2pNcclConnector`, `send_type=PUT_ASYNC`, over TCP (`NCCL_IB_DISABLE=1`) |
| Engine flags | prefix cache off, chunked prefill off, max-model-len 4096, D `gpu-memory-utilization 0.82` |
| Groups | G0 = (P0, D0) is the Canary; G1–G3 = (P1–P3, D1–D3) form the production pool. P<sub>i</sub>/D<sub>i</sub> are fixed pairs in v1 |
| Clock control | per-node GPU agent: `sudo -n nvidia-smi -lgc f,f`, NVML read-back |

**Research scope.** The contribution is clock and load control. KV transfer, the
network and the decode first-token wait are treated as **measured environment
parameters**, not as scheduling knobs.

## Design

### Lifecycle

```text
deploy ─> cold start ─────────────────────> publish ─> steady state ─> re-exploration ─┐
          production: MAX clocks,            groups move   Router: risk admission     │
          admit everything                   to H one by   Controller: park / wake /  │
          Canary: calibrate at once          one           L-H                        │
          (~15 min)                                        Canary: serves             │
                                                           ^──────────────────────────┘
```

- **Cold start.** Without a stored tier table, every production group runs at
  MAX (the highest clocks the agents report) and the Router admits every request
  (no table yet, nothing to reject on). The Canary starts calibrating at once and
  never serves during that time. Production samples are recorded at the MAX clock
  point; they are not used as H risk (MAX is faster than H).
- **Publish.** The Canary publishes a tier table. Production groups move to H one
  at a time (`controller.stagger_s`), and the Router switches to risk admission.
  The table is saved with the configuration identity and reused on restart; a
  table from another configuration is ignored.
- **Steady state.** The Canary is an ordinary production group whenever it is not
  experimenting.

### Load unit: equivalent prompt tokens/s

```text
load(group) = Σ over requests admitted in the window (L_in + alpha) / window
```

`alpha` is the fixed per-request prefill cost expressed in tokens. The Canary
fits prefill time = a + b·L_in and sets alpha = a/b (K2 on L40S: 28 ms +
0.0615 ms/token, alpha ≈ 460). Plain RPS ignores prompt length; plain tokens/s
under-counts short requests (a 128-token request costs as much as ~590 tokens).
Capacity `C_H` uses the same unit. Group-level decisions (how many groups, which
tier, Canary load) use this load; per-request admission uses the risk table.

### Tiers (published by the Canary)

| Tier | Meaning | Current cluster (expected) |
|---|---|---|
| MAX | highest supported clocks, cold start only | P 2520 (runs ~2040, power cap), D 1500 |
| **H** | P: upper edge of the loaded energy valley at 0.8·C0, excluding power-limited clocks; D: lowest J/token clock meeting TPOT | P ≈ 1815–2000, D 1050 |
| L (optional) | only if it saves > 2·eps against H at 0.3·C0 | none (one working tier) |
| **Park** | lowest idle power | P 900, D 450 |

D is limited by a **KV wall** (K3b: 48 sequences at 0.82 memory utilization; 64
concurrent gives preemptions and TPOT p95 210–216 ms at both 1050 and 1500).
Raising the D clock does not move it, so D has one clock and a KV admission limit
instead of a boost valve.

### Online risk table (per-request admission)

```text
R(clock point, N_await bucket, L_in bucket, D busy)
    = fraction of admitted requests in that state that violated TTFT >= 500 ms or TPOT > 200 ms
```

- Cells are keyed by the locked clock point (`"1815/1050"`), not by tier name, so
  Canary windows at any clock and production at any tier land in one table.
- Monotone in `N_await` and `L_in`: a sparse cell is bounded by heavier
  well-sampled cells and floored by lighter ones; a cell with neither is
  **unknown = unsafe**.
- Only clean samples are recorded (served OK, TTFT known, exact prompt length, no
  clock change in flight).
- The table starts empty on a new cluster. The Canary's last step fills the H
  cells before production switches to H.

### Router (per request; never changes clocks)

```text
cold start (no table):  admit; send to the least-loaded active group
with a table:
  for each active group g:
      D check:  D running+waiting (or g's in-flight) + 1 <= B*, and KV usage <= kv_limit
      risk:     R(effective clock of g, N_await(g), L_in, D busy(g)) <= theta (10 %)
  admit to the MOST loaded feasible group        # concentrate for batching
  none: retry within max_wait_ms, then reject 503 (X-CanaTune-Rejected: 1)
```

### Controller (every second)

```text
cold start: nothing (everything at MAX)
pressure (Router rejected, or mean active load > 0.85 C_H):
    wake a parked group at H; if none is parked, abort the Canary experiment
L/H (only if L is published): > tau_up -> H at once; < tau_down for t_down -> L
consolidate: total load <= 0.7 C_H x (active - 1) for t_down -> drain the least-loaded
             group (Canary first), then park it at the park clocks
```

While a group's clocks change, the Router assumes the slower of the old and new
clock point, and samples in flight are not recorded.

### Canary (G0)

**Tier locator** (`controller/locator.py`), full run ≈ 40 windows, ≈ 15 min:

| Step | What |
|---|---|
| park | idle power at 5 clocks per GPU; lowest (within noise → lowest clock) |
| alpha | sequential single probes at 5 prompt lengths × 3; linear fit |
| ramp | at MAX: double the load until TTFT p95 > 0.8·SLO or violations, bisect twice → C0; the median busy clock gives the ceiling f_eff (power cap) |
| coarse | 5 clocks from f_eff down at 0.8·C0; stop descending at the first infeasible clock |
| refine | golden section around the cheapest point, or bisection of the SLO edge |
| choose H | highest clock within eps (2 %) of the minimum, skipping clocks power/thermal-limited > 30 % of busy samples |
| tiers | repeat at 0.3·C0; publish L only if it saves > 2·eps |
| decode | J/token over 5 D clocks at concurrency 16; B* (clean: TPOT ≤ SLO, no preemption, no waiting) by doubling + bisection; one window beyond B* at the highest D clock decides KV wall vs frequency step |
| fill | windows at H at 0.3/0.5/0.8/1.0·C0 → risk samples and C_H |

- Feasibility uses the one-sided 90 % Wilson upper bound of the violation rate
  (≤ theta). Windows last until 30 requests are done (20–60 s). A window stops
  early when violations clearly exceed 2·theta.
- Every window result is cached; an aborted run resumes from the cache.

**Probes** (`controller/probe.py`): random token-id prompts (no prefix-cache
hits), lengths drawn jointly from the recent production (prompt, output) pairs
(default pairs until 50 real requests finished), `ignore_eos` so every clock does
the same work, Poisson arrivals (open windows) or a fixed concurrency (closed
windows). They go straight to the Canary pair and are recorded in the risk table.
Energy comes from the NVML energy counters through the agents, sampled every
0.25 s together with the SM clock and clock-limit reasons.

**Scheduler** (`controller/canary.py`):

| | Condition |
|---|---|
| triggers | cold start; length distribution shift (median or p90 > 30 %) → P relocation; production violations above theta (lower confidence bound, ≥ 200 samples) → P relocation; every 30 min → neighbour check of H |
| start | no rejection for 60 s; the other active groups carry the load at ≤ 0.7·C_H; ≥ 10 min since the last experiment; ≤ 10 % of time experimenting. Then drain the Canary (≤ 30 s) |
| stop | finished (publish), or pressure the Controller cannot relieve by waking a parked group (abort; Canary serves at H) |

## Implementation status

Select the policy with `routing.policy`: `round_robin` (baseline: production pairs
in turn, no admission, no clock control) or `cantune`.

| Component | Module | Status |
|---|---|---|
| Transport proxy: one-token prefill, then decode with the same KV-transfer request ID; SSE passthrough; TTFT/TPOT from the stream | `proxy/proxy.py` | done |
| Groups, clock points, tier table and its identity-guarded store | `domain/groups.py` | done |
| Length statistics (probe lengths, shift detection) | `domain/load.py` | done |
| Online risk table keyed by clock point | `domain/risk.py` | done |
| Telemetry: `/metrics` scraper, snapshot age | `infrastructure/telemetry.py` | done |
| GPU agent per node: supported clocks ≥ `min_mhz`, serialized locks, NVML read-back, energy/clock/limit readings, reset on exit | `infrastructure/gpu_agent.py` | done |
| Router: open admission at cold start; risk + D wall admission; equivalent-token load | `controller/router.py` | done |
| Controller: MAX cold start, staggered publish, wake / abort / reject order, L/H, drain and park | `controller/tier_controller.py` | done |
| Tier locator | `controller/locator.py` | done; validated on a replay surrogate (tests), not yet on GPUs |
| Probe backend | `controller/probe.py` | done; tested against mocked vLLM/agents |
| Canary scheduler | `controller/canary.py` | done |
| API: `GET /canatune/state`, `/tiers`, `/risk`; `POST /canatune/canary {"action": full|relocate|recheck|abort}` | `api/control.py` | done |
| Process controller: agents first (`CANATUNE_AGENT_MIN_MHZ`), then P/D, then proxy | `controller/process_controller.py` | done |
| Legacy per-workload frequency table | `controller/frequency_table.py` | superseded; kept for old data |

**Not in v1:** cross-pair P→D routing, several Routers, relaxed SLO for long
prompts, per-iteration clock changes, D frequency steps (detected and reported as
`decode_wall: false`, but D stays single-clock), shadow requests, a passive
energy table for drift detection, a tokenizer for text prompts (their length is
estimated and they never enter the risk table).

**Outputs.** With `CANATUNE_RECORD_DIR` set:

- `requests.jsonl`: one record per production request (admission snapshot, cell,
  risk, TTFT, TPOT, violation).
- `events.jsonl`: tier/clock changes, publishes (with the full evidence),
  locator phases and windows, drains, aborts, energy readings with group state.
- `CANATUNE_RISK_TABLE` and `CANATUNE_TIER_TABLE` persist the two tables.

## Known constraints

- **Clock switch.** Under load the new clock arrives 97–202 ms after the command
  starts; `sudo nvidia-smi` itself takes 139–212 ms (K2b). Unprivileged NVML
  set-clocks returns NoPermission.
- **Power caps.** L40S 350 W: a 2520 lock runs at about 2040 MHz under load.
  L4 72 W.
- **TTFT floor in P/D.** D takes 175–180 ms from request to first chunk regardless
  of length (K4a), so the TTFT budget left for P queueing is small; long prompts
  are rejected more often.
- **Which nodes can lock clocks.** Only nodes whose sudo rules allow `nvidia-smi -lgc`; check this per cluster before assigning P/D nodes.
- **Shared-account process kills.** The older benchmark launcher runs a node-wide
  `pkill -f 'python.*vllm'`; jobs on the same account and node must not match it.

## Experiments behind the design

| ID | What | Main result |
|---|---|---|
| Joint grid (266230) | 5 request types × P 900–2520 × D 450–1500 | plateau with a knee; type-independent band; P and D separable (R² ≥ 0.99) |
| K1 (267570) | P/D with `PUT` | synchronous KV send dominates bursts |
| K2 (267571) | single L40S: clocks × load, idle, switch | 1815 lowest J/request with load (24.7 J vs 27.4 J at 2520); 2520 power-capped; alpha ≈ 460 tokens |
| K1b (267625) | P/D `PUT_ASYNC`, Poisson | TTFT 25–38 % lower than `PUT`; risk strongly depends on L_in |
| K3 (267649) | single L4: clock × concurrency | 1050 lowest J/token; J/token falls ~22× from 1 to 32 sequences |
| K2b + K3b (267654) | switch breakdown; D at 32–128 sequences | switch 97–202 ms under load; D KV wall ~48 sequences, not moved by the clock |
| K4a (267696) | P/D Poisson at P 1305/1815/2520 | P energy 1305 ≈ 1815 (< 2.5 %), 2520 +10–16 %; park idle 62.6 W at 900 |
| Canary replay (offline) | locator on a K2/K3b surrogate, 1000 noisy runs + 400 synthetic clusters | within 2 % of the optimum in ≥ 96 % of runs at ≤ 3 % noise; fixed 1815/2520 ratio only 44 % on unknown clusters |

Next: cluster smoke test (1 pair: cold-start calibration; 2 pairs: publish,
admission, park/wake, abort), then K4 end-to-end (all-max spread vs fixed tier
spread vs CanaTune).

## Project layout

```text
src/canatune/
├── api/             # /canatune state and Canary control
├── controller/      # Router, Controller, tier locator, probes, Canary scheduler, process controller
├── domain/          # groups, clock points, tier table, risk table, length statistics
├── infrastructure/  # vLLM /metrics scraping, GPU agent, clock actuator, JSONL logs
├── proxy/           # Upstream vLLM transport (prefill → KV transfer → decode)
├── app.py           # ASGI application factory
└── __main__.py      # Local process entry point
```

## Development

Python 3.10 or newer is required (the cluster's vLLM env is 3.10).

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

With `routing.policy: round_robin`, production requests rotate through
`routing.production_pairs`. With `cantune`, the Router admits each request to a
group, or answers `503` with `X-CanaTune-Rejected: 1`. Use token-id prompts for
experiments: text prompt lengths are estimated, and those samples never enter the
risk table. Only streaming requests yield TTFT, so only they update the table.
Before the Canary has published tiers (cold start) every request is admitted.

An explicit `X-CanaTune-Route: canary` header selects `routing.canary_pair` and
bypasses admission. The pair used is returned in the
`X-CanaTune-Prefill-Endpoint`, `X-CanaTune-Decode-Endpoint` and (for `cantune`)
`X-CanaTune-Group` response headers. `GET /health` reports the proxy
configuration; it does not probe vLLM nodes.

The proxy expects the vLLM servers to be started separately, for example with
the scripts in `scripts/`. The prefill script takes `PREFILL_KV_SEND_TYPE`
(default `PUT_ASYNC`).

To lock clocks, set `clock_control.enabled: true`. The process controller then
starts one GPU agent per node (port `clock_control.agent_port`) before vLLM, and
stops it last; each agent resets every clock it locked on exit. The agent accepts
any clock the GPU supports at or above `clock_control.min_mhz` (the Canary
discovers tiers, so no clock list is configured). It needs `sudo -n nvidia-smi`
on its node. Without clock control there is no Canary calibration: every group
runs unlocked and every request is admitted.

## Slurm job

Run `uv sync` on the cluster first so `.venv/bin/python` can import CanaTune.
Set `runtime.python` in the YAML file to the vLLM environment's Python, and
provide a model snapshot directory containing `config.json`. Submit from the
repository root with the two intended GPU nodes explicitly assigned:

```bash
export PREFILL_NODE=<prefill-node>
export DECODE_NODE=<decode-node>
export MODEL_PATH=<model-snapshot-directory>
sbatch --nodelist="$PREFILL_NODE,$DECODE_NODE" scripts/run.sbatch
```

The batch script reads GPU IDs, ports and paths from `config.yaml`, then starts
the Python process controller. It launches the existing P/D scripts on their
assigned nodes, waits for all eight HTTP health checks, then starts the proxy
on the batch node. If Slurm stops the job or any service exits unexpectedly,
the controller stops the proxy first, then both P/D node steps. Each process
gets a grace period before forced termination (`SHUTDOWN_GRACE_S`, default 30
seconds for the controller; `NODE_SHUTDOWN_GRACE_S`, default 20 seconds for the
four vLLM instances on each node). Do not run multiple controllers for one job.
`#SBATCH` directives are static defaults; override them with `sbatch` options
when the cluster allocation differs. No experiment Driver or results collector
is wired into this job yet. With the example `proxy.host: 127.0.0.1`, clients
must reach the batch node locally or through a tunnel; use an appropriate bind
address in the site configuration for remote clients.

**Smoke test (Canary cold-start calibration).** `scripts/make_job_config.py`
writes a job config from `config.yaml` with the allocated nodes, GPU indices and
UUIDs (pair 0 = Canary; at least 2 pairs). `scripts/smoke_calibrate.py` starts
the process controller, waits until the Canary publishes a tier table, saves
`/canatune/state`, `/tiers` and `/risk`, then stops everything (agents reset the
clocks). The node scripts start one vLLM per listed GPU (1–8) and check
`PREFILL_GPU_UUIDS` / `DECODE_GPU_UUIDS` before each launch; the agents check
`gpu_uuids` from the config. Set `CANATUNE_STEP_GPUS=1` to request the GPUs for
the node steps (`--gpus-per-node=N --gpu-bind=none`).

Run the project checks with:

```bash
uv run pytest
uv run ruff check .
```
