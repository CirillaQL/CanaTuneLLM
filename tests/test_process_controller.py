"""Process lifecycle tests without requiring Slurm or vLLM."""

import signal
import sys
import threading
import time
from pathlib import Path

from canatune.controller.process_controller import ProcessController


def _service_command(
    name: str, events: Path, ready: Path, *, ignore_term: bool = False
) -> list[str]:
    script = """
import signal
import sys
import time
from pathlib import Path

name, events, ready, ignore_term = sys.argv[1:]
def on_term(_signum, _frame):
    with Path(events).open('a') as stream:
        stream.write(name + '\\n')
    if ignore_term != '1':
        raise SystemExit(0)
signal.signal(signal.SIGTERM, on_term)
Path(ready).write_text('ready')
while True:
    time.sleep(0.01)
"""
    return [sys.executable, "-c", script, name, str(events), str(ready), str(int(ignore_term))]


def _wait_for(path: Path) -> None:
    deadline = time.monotonic() + 3
    while not path.exists():
        if time.monotonic() >= deadline:
            raise AssertionError(f"service did not become ready: {path}")
        time.sleep(0.01)


def test_stops_proxy_before_both_node_steps(tmp_path: Path) -> None:
    events = tmp_path / "events"
    controller = ProcessController(health_check=lambda _: True, shutdown_grace_s=0.5)
    try:
        for name in ("prefill", "decode", "proxy"):
            ready = tmp_path / f"{name}.ready"
            controller.start(name, _service_command(name, events, ready), cwd=tmp_path)
            _wait_for(ready)
    finally:
        controller.stop()

    assert events.read_text().splitlines()[0] == "proxy"
    assert set(events.read_text().splitlines()) == {"proxy", "prefill", "decode"}
    assert all(process.poll() is not None for process in controller._processes.values())


def test_kills_service_that_ignores_term(tmp_path: Path) -> None:
    events = tmp_path / "events"
    ready = tmp_path / "proxy.ready"
    controller = ProcessController(health_check=lambda _: True, shutdown_grace_s=0.1)
    controller.start(
        "proxy", _service_command("proxy", events, ready, ignore_term=True), cwd=tmp_path
    )
    try:
        _wait_for(ready)
    finally:
        controller.stop()

    assert controller._processes["proxy"].returncode == -signal.SIGKILL


def test_unexpected_exit_stops_remaining_services(tmp_path: Path) -> None:
    events = tmp_path / "events"
    ready = tmp_path / "decode.ready"
    controller = ProcessController(
        health_check=lambda _: False, shutdown_grace_s=0.5, poll_interval_s=0.01
    )
    status = controller.run(
        cwd=tmp_path,
        prefill_command=(sys.executable, "-c", "import sys; sys.exit(7)"),
        decode_command=_service_command("decode", events, ready),
        proxy_command=(sys.executable, "-c", "raise AssertionError('proxy must not start')"),
        endpoint_urls=("http://unused/health",),
        proxy_health_url="http://unused/health",
        endpoint_timeout_s=3,
        proxy_timeout_s=3,
    )

    assert status == 7
    assert all(process.poll() is not None for process in controller._processes.values())


def test_stop_request_returns_signal_status(tmp_path: Path) -> None:
    events = tmp_path / "events"
    controller = ProcessController(
        health_check=lambda _: True, shutdown_grace_s=0.5, poll_interval_s=0.01
    )
    timer = threading.Timer(0.2, controller._handle_signal, args=(signal.SIGTERM, None))
    timer.start()
    try:
        status = controller.run(
            cwd=tmp_path,
            prefill_command=_service_command("prefill", events, tmp_path / "prefill.ready"),
            decode_command=_service_command("decode", events, tmp_path / "decode.ready"),
            proxy_command=_service_command("proxy", events, tmp_path / "proxy.ready"),
            endpoint_urls=("http://unused/health",),
            proxy_health_url="http://unused/health",
            endpoint_timeout_s=3,
            proxy_timeout_s=3,
        )
    finally:
        timer.join()

    assert status == 128 + signal.SIGTERM
    assert all(process.poll() is not None for process in controller._processes.values())
