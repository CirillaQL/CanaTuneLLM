"""Model measurements (E1 + E2) on the P/D pairs, outside the CanaTune control loop.

Starts the process controller with the `round_robin` policy (the Router, Controller
and Canary never run; the proxy is idle) and clock control enabled (the agents run),
then runs `canatune.measure` on the pairs: the Canary pair (P0, D0) first, the
production pairs after it (scans with `pairs: 2` load two pairs at once). Clocks
are locked through the agents, requests go straight to vLLM. Configured clocks
are snapped to the nearest supported ones.

Results go to MEASURE_OUT (requests.jsonl, windows.jsonl, plan.json, summary.json);
rerunning with the same directory resumes after the last finished window.

Environment: everything `canatune.controller.process_controller` needs, plus
  MEASURE_OUT         result directory (required)
  MEASURE_PLAN        optional JSON file overriding `canatune.measure.Plan` fields
  MEASURE_DEADLINE_S  seconds from start after which no new window begins (default 7200)

Exit status: 0 all windows done, 7 stopped for time (partial), 3 the service
stopped early, 4 the service never became ready, 6 a measurement error.
"""

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import httpx

from canatune.config import load_config
from canatune.measure import Measure, Pair, Plan
from canatune.proxy.proxy import parse_endpoints

READY_TIMEOUT_S = 1800


def log(message: str) -> None:
    print(f"measure: {message}", flush=True)


def snap(value: int, supported: list[int]) -> int:
    return min(supported, key=lambda f: (abs(f - value), -f)) if supported else value


def main() -> int:
    out = Path(os.environ["MEASURE_OUT"])
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    deadline = started + float(os.environ.get("MEASURE_DEADLINE_S", "7200"))
    config = load_config(os.environ["CANATUNE_CONFIG"])
    if config["routing"]["policy"] != "round_robin" or not config["clock_control"]["enabled"]:
        raise SystemExit("measure needs routing.policy=round_robin and clock_control.enabled")
    raw_plan = os.environ.get("MEASURE_PLAN")
    plan = Plan.from_json(json.loads(Path(raw_plan).read_text())) if raw_plan else Plan()
    proxy = config["proxy"]
    host = "127.0.0.1" if proxy["host"] == "0.0.0.0" else proxy["host"]
    base = f"http://{host}:{proxy['port']}"
    endpoints = parse_endpoints(config)
    raw_port = config["clock_control"].get("agent_port", 9300)
    hosts = {
        "prefill": os.environ["CANATUNE_PREFILL_HOST"],
        "decode": os.environ["CANATUNE_DECODE_HOST"],
    }

    def agent(name: str) -> tuple[str, int]:
        role = endpoints[name].role
        port = int(raw_port[role] if isinstance(raw_port, dict) else raw_port)
        return f"http://{hosts[role]}:{port}", int(config["topology"]["endpoints"][name]["gpu_id"])

    routing = config["routing"]
    names = [tuple(routing["canary_pair"])] + [tuple(x) for x in routing["production_pairs"]]
    pairs = [
        Pair(f"G{i}", endpoints[pn], endpoints[dn], agent(pn), agent(dn))
        for i, (pn, dn) in enumerate(dict.fromkeys(names))
    ]
    agents = {"prefill": pairs[0].prefill_agent, "decode": pairs[0].decode_agent}

    controller = subprocess.Popen(
        [sys.executable, "-m", "canatune.controller.process_controller"],
        start_new_session=True,
    )
    status = 4
    try:
        with httpx.Client(trust_env=False) as client:
            while time.monotonic() - started < READY_TIMEOUT_S:
                if controller.poll() is not None:
                    log(f"service stopped early ({controller.returncode})")
                    status = 3
                    return status
                try:
                    if client.get(base + "/health", timeout=5).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(5)
            else:
                return status
            log(f"ready after {time.monotonic() - started:.0f} s")
            supported = {}
            for role, (url, gpu) in agents.items():
                supported[role] = client.get(f"{url}/gpus/{gpu}/clocks", timeout=10).json()[
                    "sm_mhz"
                ]
            plan.prefill_clocks = [snap(f, supported["prefill"]) for f in plan.prefill_clocks]
            plan.decode_clocks = [snap(f, supported["decode"]) for f in plan.decode_clocks]
            plan.base_prefill_clock = snap(plan.base_prefill_clock, supported["prefill"])
            plan.base_decode_clock = snap(plan.base_decode_clock, supported["decode"])
            plan.mix_clock = snap(plan.mix_clock, supported["prefill"])
            if plan.lut_clocks:
                plan.lut_clocks = [snap(f, supported["prefill"]) for f in plan.lut_clocks]
            for scan in plan.scans:
                scan["p"] = snap(int(scan["p"]), supported["prefill"])
                scan["d"] = snap(int(scan["d"]), supported["decode"])
            (out / "plan.json").write_text(json.dumps(asdict(plan), indent=2) + "\n")
            (out / "supported_clocks.json").write_text(json.dumps(supported) + "\n")
            log(f"P clocks {plan.prefill_clocks}, D clocks {plan.decode_clocks}")
            prefill = pairs[0].prefill
            response = client.get(
                f"http://{prefill.http_host}:{prefill.http_port}/v1/models", timeout=10
            )
            model = response.json()["data"][0]["id"]

        status = 6
        measure = Measure(plan, pairs, model, out, deadline=deadline)
        log(f"resuming after {len(measure.done)} finished windows" if measure.done else "start")
        result = asyncio.run(measure.run())
        log(json.dumps(result))
        status = 0 if result["status"] == "done" else 7
        return status
    except Exception as error:
        log(f"error: {error!r}")
        return status
    finally:
        with httpx.Client(trust_env=False) as client:
            for pair in pairs:
                for url, gpu in (pair.prefill_agent, pair.decode_agent):
                    try:
                        client.post(f"{url}/gpus/{gpu}/reset", timeout=30)
                    except httpx.HTTPError as error:
                        log(f"reset {url} GPU {gpu} failed: {error!r}")
        if controller.poll() is None:
            controller.send_signal(signal.SIGTERM)  # ordered shutdown; clocks reset first
            try:
                controller.wait(timeout=180)
            except subprocess.TimeoutExpired:
                os.killpg(controller.pid, signal.SIGKILL)
                controller.wait()
        summary = {"status": status, "elapsed_s": round(time.monotonic() - started, 1)}
        (out / "summary.json").write_text(json.dumps(summary) + "\n")
        log(json.dumps(summary))


if __name__ == "__main__":
    raise SystemExit(main())
