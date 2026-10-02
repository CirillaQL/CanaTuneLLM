"""Smoke test (2): the production path under a load profile.

SMOKE_MODE=cantune (default): start the service with the `cantune` policy and no
tier table; the Canary calibrates at once, publishes, and the production groups
move to H. Then the load generator plays the profile through the Router, so
admission, consolidation (drain + park), waking, boosting to MAX and back, and a
Canary recheck aborted under pressure all run on real GPUs. The trace is saved
(trace.json) for the baseline.

SMOKE_MODE=static: the same trace (SMOKE_TRACE) through the `cantune` policy with
controller.solver false, starting from the tier table the cantune run published
(CANATUNE_TIER_TABLE, a copy): no new calibration, every group at the Canary's H,
only the pressure path (wake, boost to MAX). The solver's comparison.

SMOKE_MODE=baseline: the same trace (SMOKE_TRACE) through the `round_robin`
policy with clocks left to the driver (the default deployment), for the energy
comparison. Configure every pair as a production pair for this run.

Both modes warm every P/D pair's KV connection first and read the NVML energy
counters of every GPU through the node agents (clock_control.enabled must be true
so the agents run; the baseline never locks a clock).

Environment: everything `canatune.controller.process_controller` needs, plus
  SMOKE_OUT          directory for the results (required)
  SMOKE_MODE         cantune | static | baseline
  SMOKE_DEADLINE_S   cantune: seconds to wait for the publish (default 3000)
  SMOKE_PROFILE      load profile, name:seconds:load[:action],... (default: loadgen's)
  SMOKE_SEED         trace seed (default 20261001)
  SMOKE_TRACE        static, baseline: trace.json written by the cantune run (required)

Exit status: 0 done, 2 no table before the deadline, 3 the service stopped early,
4 the service never became ready, 5 calibration kept failing, 6 the load run failed.
"""

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

from canatune import loadgen
from canatune.config import load_config
from canatune.proxy.proxy import parse_endpoints, pd_transport_id
from canatune.service import cold_start_lengths

READY_TIMEOUT_S = 1800
H_TIMEOUT_S = 90


def fetch(client: httpx.Client, base: str, path: str) -> object | None:
    try:
        response = client.get(base + path, timeout=10)
        response.raise_for_status()
        return response.json()
    except (httpx.HTTPError, ValueError):
        return None


def log(message: str) -> None:
    print(f"smoke2: {message}", flush=True)


def agents_and_names(config: dict) -> tuple[list[str], dict[str, str]]:
    """Agent base URLs and agent|gpu index -> endpoint name."""
    clock = config.get("clock_control") or {}
    if not clock.get("enabled"):
        return [], {}
    raw_port = clock.get("agent_port", 9300)
    urls = {}
    for role, env in (("prefill", "CANATUNE_PREFILL_HOST"), ("decode", "CANATUNE_DECODE_HOST")):
        port = int(raw_port[role] if isinstance(raw_port, dict) else raw_port)
        urls[role] = f"http://{os.environ[env]}:{port}"
    names = {
        f"{urls[e['role']]}|{e['gpu_id']}": name
        for name, e in config["topology"]["endpoints"].items()
    }
    return sorted(set(urls.values())), names


def reset_clocks(client: httpx.Client, agents: list[str]) -> list[dict]:
    """Reset every agent GPU to driver control; -> readings after the reset."""
    readings = []
    for agent in agents:
        for gpu in client.get(f"{agent}/gpus", timeout=10).json():
            client.post(f"{agent}/gpus/{gpu['index']}/reset", timeout=30).raise_for_status()
        time.sleep(3)
        for gpu in client.get(f"{agent}/gpus", timeout=10).json():
            readings.append({"agent": agent, "index": gpu["index"], "sm_mhz": gpu.get("sm_mhz")})
    log(f"clocks reset: {readings}")
    return readings


def model_name(client: httpx.Client, endpoints: dict) -> str:
    prefill = next(e for e in endpoints.values() if e.role == "prefill")
    response = client.get(f"http://{prefill.http_host}:{prefill.http_port}/v1/models", timeout=10)
    response.raise_for_status()
    return response.json()["data"][0]["id"]


def warm_pairs(client: httpx.Client, config: dict, endpoints: dict, model: str) -> None:
    """Three short requests per pair straight to vLLM: the first P2pNccl transfer of
    a pair sets up the connection and would otherwise land in the first phase."""
    routing = config["routing"]
    pairs = [tuple(routing["canary_pair"])] + [tuple(p) for p in routing["production_pairs"]]
    for prefill_name, decode_name in dict.fromkeys(pairs):
        prefill, decode = endpoints[prefill_name], endpoints[decode_name]
        for i in range(3):
            body = {
                "model": model,
                "prompt": loadgen.prompt_ids(7, i, 128),
                "max_tokens": 4,
                "ignore_eos": True,
                "temperature": 0.0,
            }
            headers = {"X-Request-Id": pd_transport_id(f"warm-{i}", prefill, decode)}
            try:
                client.post(
                    prefill.completions_url,
                    json={**body, "max_tokens": 1},
                    headers=headers,
                    timeout=120,
                ).raise_for_status()
                client.post(
                    decode.completions_url, json=body, headers=headers, timeout=120
                ).raise_for_status()
            except httpx.HTTPError as error:
                log(f"warm-up {prefill_name}/{decode_name} failed: {error!r}")
        log(f"warmed {prefill_name}/{decode_name}")


def wait_ready(client: httpx.Client, base: str, mode: str, controller) -> int | None:
    """-> None when ready, else an exit status."""
    started = time.monotonic()
    path = "/canatune/state" if mode != "baseline" else "/health"
    while time.monotonic() - started < READY_TIMEOUT_S:
        if controller.poll() is not None:
            log(f"service stopped early ({controller.returncode})")
            return 3
        if fetch(client, base, path) is not None:
            log(f"ready after {time.monotonic() - started:.0f} s")
            return None
        time.sleep(5)
    return 4


def wait_publish(
    client: httpx.Client, base: str, controller, deadline_s: float, max_failures: int = 3
) -> tuple[int | None, dict | None]:
    started = time.monotonic()
    last_phase = None
    while time.monotonic() - started < deadline_s:
        if controller.poll() is not None:
            log(f"service stopped early ({controller.returncode})")
            return 3, None
        state = fetch(client, base, "/canatune/state") or {}
        history = (state.get("canary") or {}).get("history") or []
        failures = [h for h in history if h.get("outcome") == "failed"]
        if len(failures) >= max_failures:
            log(f"calibration failed {len(failures)} times: {failures[-1].get('error')}")
            return 5, None
        phase = (state.get("canary") or {}).get("phase")
        if phase != last_phase:
            log(f"t={time.monotonic() - started:.0f}s phase={phase}")
            last_phase = phase
        tiers = fetch(client, base, "/canatune/tiers") or {}
        if tiers.get("table") is not None:
            log(f"published after {time.monotonic() - started:.0f} s")
            return None, tiers["table"]
        time.sleep(5)
    return 2, None


def wait_h(client: httpx.Client, base: str, table: dict) -> bool:
    """Every active group runs at the published H (the staggered move finished)."""
    started = time.monotonic()
    while time.monotonic() - started < H_TIMEOUT_S:
        state = fetch(client, base, "/canatune/state") or {}
        groups = (state.get("router") or {}).get("groups") or []
        active = [g for g in groups if g.get("state") == "active"]
        if active and all(g.get("effective") == table["h"] for g in active):
            log(f"production at H {table['h']}: {[g['name'] for g in active]}")
            return True
        time.sleep(2)
    log("active groups did not all reach H in time; playing the profile anyway")
    return False


def event_checks(record_dir: Path, run: dict, meta: dict) -> dict:
    """Which Controller/Canary paths fired, with their time in the trace and phase."""
    events = loadgen._read_jsonl(record_dir / "events.jsonl")
    start = run["started_wall"]
    bounds = [(p["name"], p["start_s"], p["end_s"]) for p in meta["phases"]]

    def phase_of(t: float) -> str | None:
        return next((name for name, a, b in bounds if a <= t < b), None)

    timeline = []
    for e in events:
        if e.get("event") not in (
            "tier",
            "drain",
            "plan",
            "canary_abort",
            "canary_claim",
            "canary_verify",
            "publish",
        ):
            continue
        t = e["wall_time"] - start
        if t < 0 or t > meta["duration_s"] + 300:
            continue
        timeline.append(
            {
                "t_s": round(t, 1),
                "phase": phase_of(t),
                "event": e["event"],
                **{
                    k: e[k]
                    for k in ("group", "tier", "clock", "ok", "reason", "kind", "to", "binding")
                    if k in e
                },
            }
        )
    tiers = [e for e in timeline if e["event"] == "tier"]
    reasons = [e.get("reason") or "" for e in tiers]
    checks = {
        "parked": any(e.get("tier") == "park" and e["reason"] == "drained" for e in tiers),
        "woken": any(r in ("pressure", "plan") for r in reasons),
        "planned": "plan" in reasons,
        "boosted": any(r.startswith("boost") for r in reasons),
        "unboosted": "unboost" in reasons,
        "canary_claimed": any(e["event"] == "canary_claim" for e in timeline),
        "canary_aborted": any(e["event"] == "canary_abort" for e in timeline),
        "plans": sum(1 for e in timeline if e["event"] == "plan"),
        "verifications": sum(1 for e in timeline if e["event"] == "canary_verify"),
        "clock_failures": sum(1 for e in tiers if e.get("ok") is False),
        "clock_errors": sum(1 for e in events if e.get("event") == "clock_error"),
    }
    return {"checks": checks, "timeline": timeline}


def save_service_state(client: httpx.Client, base: str, out: Path, config: dict) -> None:
    for name, endpoint in parse_endpoints(config).items():
        try:
            response = client.get(
                f"http://{endpoint.http_host}:{endpoint.http_port}/metrics", timeout=10
            )
            response.raise_for_status()
            (out / f"metrics_{name}.txt").write_text(response.text)
        except httpx.HTTPError as error:
            log(f"/metrics of {name} failed: {error!r}")
    for name, path in (
        ("state.json", "/canatune/state"),
        ("tiers.json", "/canatune/tiers"),
        ("risk.json", "/canatune/risk"),
    ):
        data = fetch(client, base, path)
        if data is not None:
            (out / name).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def main() -> int:
    out = Path(os.environ["SMOKE_OUT"])
    out.mkdir(parents=True, exist_ok=True)
    mode = os.environ.get("SMOKE_MODE", "cantune")
    if mode not in ("cantune", "static", "baseline"):
        raise SystemExit("SMOKE_MODE must be cantune, static or baseline")
    config = load_config(os.environ["CANATUNE_CONFIG"])
    policy = config["routing"]["policy"]
    if (mode != "baseline") != (policy == "cantune"):
        raise SystemExit(f"SMOKE_MODE={mode} does not match routing.policy={policy}")
    if (mode == "static") == bool(config["controller"].get("solver", False)):
        raise SystemExit(f"SMOKE_MODE={mode} needs controller.solver {mode != 'static'}")
    proxy = config["proxy"]
    host = "127.0.0.1" if proxy["host"] == "0.0.0.0" else proxy["host"]
    base = f"http://{host}:{proxy['port']}"
    slo = config["experiment"]["slo"]
    agents, gpu_names = agents_and_names(config)
    record_dir = Path(os.environ.get("CANATUNE_RECORD_DIR", out / "records"))

    started = time.monotonic()
    controller = subprocess.Popen(
        [sys.executable, "-m", "canatune.controller.process_controller"],
        start_new_session=True,
    )
    status = 4
    summary: dict = {"mode": mode}
    try:
        with httpx.Client(trust_env=False) as client:
            failed = wait_ready(client, base, mode, controller)
            if failed is not None:
                status = failed
                return status
            summary["ready_s"] = round(time.monotonic() - started, 1)
            if mode == "cantune":
                failed, table = wait_publish(
                    client,
                    base,
                    controller,
                    float(os.environ.get("SMOKE_DEADLINE_S", "3000")),
                )
                if failed is not None:
                    status = failed
                    return status
                summary["published_s"] = round(time.monotonic() - started, 1)
                summary["reached_h"] = wait_h(client, base, table)
                pairs = cold_start_lengths(config)
                meta, arrivals = loadgen.build_trace(
                    loadgen.parse_profile(os.environ.get("SMOKE_PROFILE", loadgen.DEFAULT_PROFILE)),
                    pairs,
                    capacity_h=float(table["capacity_h"]),
                    alpha=float(table["alpha_tokens"]),
                    seed=int(os.environ.get("SMOKE_SEED", "20261001")),
                )
                meta["tier_table"] = {
                    k: table.get(k)
                    for k in ("h", "park", "l", "capacity_h", "alpha_tokens", "decode_max_running")
                }
                loadgen.save_trace(out / "trace.json", meta, arrivals)
            elif mode == "static":
                # The stored table is loaded at start: no calibration to wait for.
                failed, table = wait_publish(client, base, controller, 60.0)
                if failed is not None:
                    log("the stored tier table was not loaded")
                    status = failed
                    return status
                summary["reached_h"] = wait_h(client, base, table)
                meta, arrivals = loadgen.load_trace(os.environ["SMOKE_TRACE"])
            else:
                meta, arrivals = loadgen.load_trace(os.environ["SMOKE_TRACE"])
                # Driver-managed clocks: never inherit locks from an earlier run.
                summary["clocks_at_start"] = reset_clocks(client, agents)
            log(
                f"trace: {meta['requests']} requests over {meta['duration_s']:.0f} s, "
                + ", ".join(f"{p['name']}={p['rate_rps']:.2f}rps" for p in meta["phases"])
            )
            endpoints = parse_endpoints(config)
            model = model_name(client, endpoints)
            warm_pairs(client, config, endpoints, model)

            status = 6
            run = asyncio.run(
                loadgen.LoadRun(
                    meta,
                    arrivals,
                    base_url=base,
                    model=model,
                    out_dir=out / "load",
                    agents=agents,
                    state_poll=mode != "baseline",
                    actions=mode != "baseline",
                ).run()
            )
            status = 0
            summary["run"] = run
            result = loadgen.summarize(
                out / "load",
                meta,
                ttft_slo_ms=float(slo["ttft_ms"]),
                tpot_slo_ms=float(slo["tpot_ms"]),
                gpu_names=gpu_names,
            )
            if mode != "baseline":
                result.update(event_checks(record_dir, run, meta))
            (out / "load_summary.json").write_text(json.dumps(result, indent=2) + "\n")
            for name, phase in result["phases"].items():
                log(
                    f"{name}: ok={phase['ok']}/{phase['offered']} rejected={phase['rejected']} "
                    f"violated={phase['violated']} p95={phase['ttft_p95_ms']} "
                    f"J/req={phase['j_per_ok_request']}"
                )
            if "checks" in result:
                log(f"checks: {json.dumps(result['checks'])}")
            save_service_state(client, base, out, config)
            return status
    except Exception as error:
        log(f"error: {error!r}")
        summary["error"] = repr(error)
        return status
    finally:
        if controller.poll() is None:
            controller.send_signal(signal.SIGTERM)  # ordered shutdown, agents reset clocks
            try:
                controller.wait(timeout=180)
            except subprocess.TimeoutExpired:
                os.killpg(controller.pid, signal.SIGKILL)
                controller.wait()
        summary["status"] = {
            0: "done",
            2: "deadline",
            3: "stopped_early",
            4: "never_ready",
            5: "calibration_failed",
            6: "load_failed",
        }[status]
        summary["controller_exit"] = controller.returncode
        (out / "smoke_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        log(json.dumps(summary))


if __name__ == "__main__":
    raise SystemExit(main())
