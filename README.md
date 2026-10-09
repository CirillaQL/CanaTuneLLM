# CanaTune

CanaTune is the Router and Controller for a prefill/decode (P/D) disaggregated
vLLM deployment. Its goal is to **reduce total GPU energy while meeting the
latency SLO** (TTFT < 500 ms, TPOT ≤ 200 ms) by

1. **concentrating requests** on as few P/D groups as the SLO allows,
2. **parking idle groups** at a locked low clock instead of leaving them unlocked,
3. running active groups at **clock tiers the Canary measured on this cluster**
   (the loaded energy valley of P; the KV-wall-limited D), and
4. using **prediction warnings and Production latency feedback**: confirm pressure,
   expand using measured capacity, then reclaim Canary and lock MAX only when
   pressure persists or actual unfinished work is severely stalled. HTTP 503 is
   reserved for the service wait timeout (or the legacy rejection policy).

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
| KV transfer | `NixlConnector` (`kv_both`, `kv_load_failure_policy=fail`): D allocates blocks, then reads P's KV into them; earlier jobs (up to N, homo-l40s): `P2pNcclConnector`, `send_type=PUT_ASYNC`, over TCP (`NCCL_IB_DISABLE=1`) |
| Engine flags | prefix cache off, chunked prefill off, max-model-len 4096, D `gpu-memory-utilization auto` (GPU memory less an absolute reserve, `memory_plan`) |
| Groups | G0 = (P0, D0) is the Canary; G1–G3 = (P1–P3, D1–D3) form the production pool. P<sub>i</sub>/D<sub>i</sub> are fixed pairs in v1 |
| Clock control | per-node GPU agent: `sudo -n nvidia-smi -lgc f,f`, NVML read-back |

**Research scope.** The contribution is clock and load control. KV transfer, the
network and the decode first-token wait are treated as **measured environment
parameters**, not as scheduling knobs.

## Design

Everything the system uses comes from the configuration or from the Canary's own
experiments on the running cluster; nothing is carried over from other clusters or
offline runs.

| Source | What |
|---|---|
| configuration | the model's `config.json` (KV bytes per token), `kv_transfer` (connector; P2pNccl: `kv_buffer_bytes`), topology, the policy: SLOs, theta, `max_wait_ms`, `hold_max_ms`, `t_down_s`, the output cap `canary.max_output_tokens` (cold-start probes generate it: the heaviest decode load, so the first H is conservative) |
| Canary experiments | everything else: tiers, S(L) per clock, D iteration model, power per clock vs load, capacity, the admission predictor, the slack-risk seed, the KV-in-flight gate (P2pNccl only) |
| production, online | predictor refits, slack-risk counts, offered rate; the (prompt, output) lengths of finished requests, which the Canary's probes replay (random tokens, `ignore_eos`) once `length_min_samples` requests finished |

### Lifecycle

```text
deploy ─> cold start ──────────────> publish ─> steady state ──────────────────> re-exploration
          production at MAX,          table +     Router: state-based admission      (verify /
          admit everything            cluster     Controller: solver plan, pressure  relocate /
          Canary: calibrate at once   model       Canary: serves, verifies plans     recheck)
```

- **Cold start.** No table: every production group runs at MAX and the Router
  admits every request. The Canary calibrates at once.
- **Publish.** The tier table carries the Canary's evidence (S(L) per clock, the
  admission calibration and the cluster model). Groups move to the working point
  one at a time; the table is saved with the configuration identity.
- **Steady state.** The Canary serves as an ordinary group when not experimenting.

### Canary calibration (`controller/locator.py`)

| Step | What |
|---|---|
| park | idle power at 5 clocks per GPU; the lowest is the park point |
| service | single requests at 16, 64, 256, ... up to the longest prompt (x3) at MAX: S(L), the KV residence (idle TTFT - prefill), prompts that miss the SLO even when idle |
| ramp | at MAX: double the load until the violation bound exceeds theta, bisect → C0; the busy clock gives the power-capped ceiling |
| search | coarse + refine of the P clock at 0.8·C0 by measured J/request → H |
| tables | S(L) at every coarse P clock (the solver's choices); a higher clock slower than a lower one (beyond eps) is measured again, then bounded by the lower clocks |
| decode | length model at the top D clock: TPOT = α + β·X + γ·K + r95 (X running sequences, K their context tokens; Ramani & Tantawi, arXiv 2609.20957) from closed windows of the short and long prompt halves at two concurrencies; a TPOT SLO that one short sequence misses is reported infeasible; the D clock by J/token at 0.7 of the capacity it implies, refitted there. `decode_model: count` keeps the B* search |
| joint | P at H with the D ladder at 0.8·C0 |
| fill | windows at H over loads → C_H |
| model | from every probe and window of the run (below) |

Every probe records the state it was sent into (prompt lengths at P, KV tokens
between P's return and the first token, requests decoding), its TTFT and SLO
outcome; every window its average P and D power, D busy share and running
sequences. From these (`domain/calibration.py`, `domain/models.py`):

```text
admission   TTFT predictor = least squares over (1, S(L_own), pending S(L), KV in
            flight, decoding) on probes with TTFT <= 2 x SLO; slack-risk counts
            (two-fold: each half predicted by the fit on the other); KV gate = the
            first in-flight share of the buffer whose violation rate is clearly
            above theta (else the whole buffer)
model       S(L) per P clock (plateau + slope); residence per length; D iteration
            alpha + delta x running per D clock; power: idle (park step) + dynamic
            x utilization per clock; D KV capacity (vLLM cache blocks); B*
```

Windows are cached, so an aborted run resumes; feasibility is the one-sided 90 %
Wilson bound of the violation rate <= theta.

### Router (`controller/router.py`, per request; never changes clocks)

Per group it tracks, from the proxy's stage events, the pending P work (sum of
S(L) at the group's clock), the KV bytes in flight and the requests decoding.

```text
hard guards   fresh D telemetry, D KV usage, KV in flight <= gate x kv_buffer_bytes
SLO warning   length model: the TPOT bound of D with this request (sequences on D
              or on their way, the context tokens they hold) exceeds the SLO, or
              its KV passes the wall; count model: D concurrency exceeds B*.
              best_effort still dispatches, including at MAX. Legacy policies
              retain the B* admission ceiling.
risk          predicted TTFT (Canary's fit, refitted online) -> slack = SLO - waited
              - predicted -> violation risk of that slack bucket (Canary's seed +
              production's counts; a sparse bucket is bounded by the nearest
              observed bucket with less slack) <= theta
admit         to the most loaded feasible group (concentration keeps batches large)
overload      best_effort (default): prediction warns; Production confirms;
              measured capacity guides expansion; persistent/severe pressure
              triggers MAX and safely reclaims Canary. Dispatch FIFO even late
              while enforcing physical D/KV limits. Queue until capacity is free;
              hold_max_ms is the service wait timeout, then HTTP 503.
              reject / serve retain legacy rejection and rescue/backfill behavior.
```

Prediction signals are warnings. Actual latency and physical-capacity waits
confirm pressure before resources increase.

### Solver and Controller (`domain/models.py`, `controller/tier_controller.py`)

By default (`controller.solver: false`) every active group runs at the Canary's H
and the Controller only handles production pressure (expansion, MAX, recovery):
on the measured L40S/L4 cluster parking and lower D clocks saved little, and a
lower D clock raised TTFT in a way the model does not capture (jobs C and D). The
solver below stays available with `controller.solver: true`.

Every second, from the offered rate (Router arrivals over `load_window_s`) and the
recent length mix, the solver evaluates every (groups, P clock, D clock) of the
Canary's model. It fills each group up to feasible capacity before opening
another D, matching the Router's concentration policy; D busy power is paid per
working group. The longest offered prompt is retained in the length mix, and
online Router predictor coefficients and slack counts feed the solver:

```text
TTFT     the admission predictor at steady-state feature means (pending P work =
         M/G/1 mean unfinished work, KV in flight and decoding by Little's law) for
         each length of the mix -> violation risk from the slack-risk counts;
         mean risk <= theta
D        running sequences (Little's law on alpha + delta X) <= min(KV capacity /
         mean context, B*, (TPOT SLO - alpha) / delta)
KV       mean KV in flight <= the gate;   idle TTFT of the longest prompt <= SLO
verified a point the Canary saw fail at a rate stays below that rate
power    active groups at their clocks + parked groups at the park clocks
```

The plan is the lowest-power feasible configuration for the peak rate of the last
`t_down_s`. In best-effort mode, increasing resources requires Production feedback
confirmation. Cheaper plans apply only once the same target held for `t_down_s`,
with no pressure in that time, and when savings exceed the Canary's noise band
(`canary.locator.eps`) and amortize the observed clock switching time. Extra groups
drain (the Canary first) and park; missing ones wake. Model-only pressure enters
`warning` and cannot expand or raise clocks. Warning/confirming modes still evaluate
solver targets and finish draining. Warnings do not reset target hold or actual
production calm time; only resource changes reset the target, and only actual
Production pressure delays an otherwise feasible cheaper plan.
Actual near-SLO TTFT/TPOT from at least `feedback_min_samples` recent requests,
or unfinished-request/physical-capacity waits, must persist for `confirm_s`.
Then `expanding` wakes groups at H using the measured per-group capacity (static:
C_H / (mean prompt + alpha); solver: capacity for the current length mix).
After `expansion_grace_s`, persistent fresh feedback triggers `full_effort`;
severe unfinished waits (at least twice the SLO) also require `confirm_s` before
emergency MAX/reclaim. TTFT samples keep first-token timestamps, so pre-expansion
TTFT cannot masquerade as fresh evidence. Prompts impossible at idle/MAX do not
contribute TTFT pressure; their TPOT still counts. Return to energy mode requires
`t_down_s` without actual pressure. These feedback thresholds are configurable. While clocks change the
Router assumes the slower point and in-flight samples are not recorded.

### Canary scheduler (`controller/canary.py`)

| | Condition |
|---|---|
| triggers | cold start or a table without a model (full); production violations above theta (relocation); the length mix shifted or the plan runs a point at a per-group rate the Canary has not confirmed (verify: one window; a failure caps the point below that rate); every 30 min a neighbour check of H |
| start | no pressure for `quiet_s`; the solver finds the other active groups feasible; `min_interval_s` since the last experiment; at most `max_duty` of the time |
| stop | finished, or pressure the Controller cannot relieve by waking a group (abort) |

## Implementation status

Select the policy with `routing.policy`: `round_robin` (baseline: production pairs
in turn, no admission, no clock control) or `cantune`.

| Component | Module | Status |
|---|---|---|
| Transport proxy: one-token prefill, then decode with the same KV-transfer request ID; completions and chat completions, `/v1/models`; SSE passthrough; TTFT/TPOT from the stream | `proxy/proxy.py` | done |
| Groups, clock points, tier table and its identity-guarded store | `domain/groups.py` | done |
| Length statistics (probe lengths, shift detection) | `domain/load.py` | done |
| Online risk table keyed by clock point | `domain/risk.py` | done |
| Telemetry: `/metrics` scraper, snapshot age | `infrastructure/telemetry.py` | done |
| GPU agent per node: supported clocks ≥ `min_mhz`, serialized locks, NVML read-back, energy/clock/limit readings, reset on exit | `infrastructure/gpu_agent.py` | done |
| Router: state-based concentration; SLO-triggered best effort; legacy serve/reject | `controller/router.py` | done |
| Cluster model and solver from the Canary's measurements | `domain/models.py`, `domain/calibration.py` | done; simulated, not yet on GPUs |
| Controller: MAX cold start, solver energy plans, Production-confirmed expansion/MAX, drain and park | `controller/tier_controller.py` | done |
| Tier locator | `controller/locator.py` | done; validated on a replay surrogate (tests), not yet on GPUs |
| Probe backend | `controller/probe.py` | done; tested against mocked vLLM/agents |
| Canary scheduler | `controller/canary.py` | done |
| API: `GET /canatune/state`, `/tiers`, `/risk`; `POST /canatune/canary {"action": full|relocate|recheck|verify|abort}` | `api/control.py` | done |
| Process controller: agents first (`CANATUNE_AGENT_MIN_MHZ`), then P/D, then proxy | `controller/process_controller.py` | done |
| Legacy per-workload frequency table | `controller/frequency_table.py` | superseded; kept for old data |

**Not in v1:** cross-pair P→D routing, several Routers, relaxed SLO for long
prompts, per-iteration clock changes, D frequency steps (detected and reported as
`decode_wall: false`, but D stays single-clock), shadow requests, a passive
energy table for drift detection.

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

GitHub Actions runs `ruff check .` and the full pytest suite on Ubuntu with
Python 3.10 and 3.12 for pushes and pull requests. Dependencies are installed
from `uv.lock` with `uv sync --locked --extra dev`. The tests use simulated
backends and do not require GPUs, Slurm, or running vLLM servers; cluster smoke
tests remain separate. The CI workflow can also be started manually.

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

`POST /v1/chat/completions` works the same way (P and D receive the chat request;
the chat template renders the prompt), and `GET /v1/models` is answered by a
production D, so OpenAI clients and `vllm bench serve` (`openai` and `openai-chat`
backends) can target the proxy directly:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"mistralai/Mistral-7B-v0.1","messages":[{"role":"user","content":"Hello"}],"max_tokens":16,"stream":true}'
```

A base model such as Mistral-7B-v0.1 has no chat template, and vLLM refuses chat
requests without one: set `VLLM_CHAT_TEMPLATE` (a file or the template) for the
launch scripts, which pass it to every P and D; the proxy renders prompts with the
same template for their lengths.

With `routing.policy: round_robin`, production requests rotate through
`routing.production_pairs`. With `cantune`, the Router admits each request to a
group; with the default `best_effort` overload policy a request waits only for
physical capacity and `503` (`X-CanaTune-Rejected: 1`) means the service wait
timed out. Prompt lengths are exact for token ids, for text through the model's
tokenizer (`MODEL_PATH`) and for chat through the chat template; without a
tokenizer they are estimated and those samples never enter the risk table. Only
streaming requests yield TTFT, so only they update the table and the production
feedback; a non-streaming request counts only while it waits at P. Before the
Canary has published tiers (cold start) every request is admitted.

An explicit `X-CanaTune-Route: canary` header selects `routing.canary_pair` and
bypasses admission. The pair used is returned in the
`X-CanaTune-Prefill-Endpoint`, `X-CanaTune-Decode-Endpoint` and (for `cantune`)
`X-CanaTune-Group` response headers. `GET /health` reports the proxy
configuration; it does not probe vLLM nodes.

The proxy expects the vLLM servers to be started separately, for example with
the scripts in `scripts/`. They take `KV_CONNECTOR` (default `NixlConnector`; the
process controller sets it from `kv_transfer.connector`). With NIXL, P is asked to
keep the prompt's KV (`kv_transfer_params.do_remote_decode`) and its response's
`kv_transfer_params` go to D, which reads the blocks; a P response without them is
an error (D would otherwise compute the prompt itself), and so is an error event in
D's stream (a failed KV read). Each endpoint's `kv_port` is its NIXL side-channel
port (`VLLM_NIXL_SIDE_CHANNEL_PORT`, host `CANATUNE_{PREFILL,DECODE}_KV_HOST`).
The `nixl` Python package (NIXL 0.9, vLLM 0.15.1's time) must be importable by
vLLM. With `P2pNcclConnector` the prefill script takes `PREFILL_KV_SEND_TYPE`
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

**Smoke test (2): production path.** `scripts/smoke_production.py` with
`SMOKE_MODE=cantune` runs the cold-start calibration, waits until the production
groups are at H, then plays a load profile through the Router with
`canatune.loadgen`. The profile is `name:seconds:load[:action]` phases, with load
in units of one group's C_H. The default one exercises drain + park, wake,
boost to MAX and back, and a Canary recheck aborted under pressure. The trace is
saved as `trace.json`. `SMOKE_MODE=baseline` replays that trace with the
`round_robin` policy over all pairs, with driver-managed clocks. Both runs read
every GPU's NVML energy counter through the node agents. The results are
`load/requests.jsonl`, `energy.jsonl`, `timeline.jsonl` and `load_summary.json`,
which gives per-phase outcomes, J per request and which Controller paths fired.

**Benchmark with `vllm bench serve`.** `scripts/bench_production.py` starts the
service like the smoke test (`BENCH_MODE=cantune` from a cold start, `static` from
a stored tier table, `baseline` with `round_robin`), then runs `vllm bench serve`
against the proxy once per request rate in `BENCH_RATES`, on a ShareGPT file
(`BENCH_DATASET`) with a fixed seed so every mode gets the same prompts.
`BENCH_OUTPUT_LEN` with `BENCH_IGNORE_EOS=1` (default) fixes the work per request;
set `canary.max_output_tokens` to the same length. Every GPU's NVML energy is
read through the node agents, and a stage's energy window is the benchmark's own
span (its `duration`), not the dataset loading before it. Per stage the results
are vllm's `bench.json` (`--save-detailed`, `--goodput` at the configured SLOs)
under `stages/`, and `bench_summary.json`: goodput, TTFT/TPOT percentiles (TPOT
as vllm defines it), energy per GPU, J per good request and per output token, and
(cantune/static) the Controller modes during the stage.

**Model measurements (E1 + E2).** `scripts/measure_model.py` starts the service
with the `round_robin` policy and clock control, then runs `canatune.measure` on
the Canary pair outside the control loop: a prompt-length table per P clock, idle
power, closed-loop decode windows per D clock, open-loop rate scans per P clock
and for other length mixes. Every request records the P round trip, the gap and
the decode first-token time, so TTFT can be split into its stages. Every window
records vLLM counter deltas, energy per GPU and sampled queue state. Results are
appended to `requests.jsonl` / `windows.jsonl`, and a rerun resumes after the
last finished window. The proxy logs the same stage breakdown (`timing`) for
production requests. `scripts/analysis/sparse_clock_check.py` checks offline
whether a sparse clock design finds the lowest-energy clock of a dense sweep.

Run the project checks with:

```bash
uv run pytest
uv run ruff check .
```
