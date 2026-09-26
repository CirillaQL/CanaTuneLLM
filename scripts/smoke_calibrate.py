"""Smoke test (1): cold-start Canary calibration on a running P/D deployment.

Starts the process controller (GPU agents -> P/D node steps -> CanaTune proxy),
waits until the Canary publishes a tier table (or the deadline passes), saves
the control-plane state, then stops everything in order (agents reset clocks on
exit). No external traffic is sent: the Canary probes on its own.

Environment: everything `canatune.controller.process_controller` needs, plus
  SMOKE_OUT          directory for the collected results (required)
  SMOKE_DEADLINE_S   seconds to wait for the publish after start (default 2400)
  SMOKE_MAX_FAILURES stop after this many failed calibration attempts (default 3)

Exit status: 0 published, 2 not published before the deadline, 3 the service
stopped early, 4 the service never became ready, 5 calibration kept failing.
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

from canatune.config import load_config


def fetch(client: httpx.Client, base: str, path: str) -> object | None:
    try:
        response = client.get(base + path, timeout=10)
        response.raise_for_status()
        return response.json()
    except (httpx.HTTPError, ValueError):
        return None


def main() -> int:
    out = Path(os.environ["SMOKE_OUT"])
    out.mkdir(parents=True, exist_ok=True)
    deadline_s = float(os.environ.get("SMOKE_DEADLINE_S", "2400"))
    max_failures = int(os.environ.get("SMOKE_MAX_FAILURES", "3"))
    config = load_config(os.environ["CANATUNE_CONFIG"])
    proxy = config["proxy"]
    host = "127.0.0.1" if proxy["host"] == "0.0.0.0" else proxy["host"]
    base = f"http://{host}:{proxy['port']}"

    started = time.monotonic()
    controller = subprocess.Popen(
        [sys.executable, "-m", "canatune.controller.process_controller"],
        start_new_session=True,
    )
    status, ready_at, published = 4, None, None
    last_phase = None
    try:
        with httpx.Client(trust_env=False) as client:
            while time.monotonic() - started < deadline_s:
                if controller.poll() is not None:
                    status = 3
                    print(f"smoke: service stopped early ({controller.returncode})", flush=True)
                    break
                state = fetch(client, base, "/canatune/state")
                if state is not None:
                    if ready_at is None:
                        ready_at = time.monotonic()
                        status = 2
                        print(f"smoke: ready after {ready_at - started:.0f} s", flush=True)
                    history = (state.get("canary") or {}).get("history") or []
                    failures = [h for h in history if h.get("outcome") == "failed"]
                    if len(failures) >= max_failures:
                        status = 5
                        print(
                            f"smoke: calibration failed {len(failures)} times: "
                            f"{failures[-1].get('error')}",
                            flush=True,
                        )
                        break
                    phase = (state.get("canary") or {}).get("phase")
                    if phase != last_phase:
                        print(
                            f"smoke: t={time.monotonic() - started:.0f}s phase={phase}", flush=True
                        )
                        last_phase = phase
                    tiers = fetch(client, base, "/canatune/tiers") or {}
                    if tiers.get("table") is not None:
                        published = time.monotonic()
                        status = 0
                        print(f"smoke: published after {published - started:.0f} s", flush=True)
                        time.sleep(5)  # let the staggered move to H finish
                        break
                time.sleep(5)
            for name, path in (
                ("state.json", "/canatune/state"),
                ("tiers.json", "/canatune/tiers"),
                ("risk.json", "/canatune/risk"),
            ):
                data = fetch(client, base, path)
                if data is not None:
                    (out / name).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    finally:
        if controller.poll() is None:
            controller.send_signal(signal.SIGTERM)  # ordered shutdown, agents reset clocks
            try:
                controller.wait(timeout=180)
            except subprocess.TimeoutExpired:
                os.killpg(controller.pid, signal.SIGKILL)
                controller.wait()
    summary = {
        "status": {
            0: "published",
            2: "deadline",
            3: "stopped_early",
            4: "never_ready",
            5: "calibration_failed",
        }[status],
        "ready_s": None if ready_at is None else round(ready_at - started, 1),
        "published_s": None if published is None else round(published - started, 1),
        "controller_exit": controller.returncode,
    }
    (out / "smoke_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("smoke:", json.dumps(summary), flush=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
