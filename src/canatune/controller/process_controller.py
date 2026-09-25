"""Supervise the proxy and both Slurm node steps for one CanaTune job."""

import math
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import httpx

from canatune.config import load_config


class ServiceExited(RuntimeError):
    """A supervised process stopped before the job was asked to stop."""

    def __init__(self, name: str, returncode: int) -> None:
        super().__init__(f"{name} exited with status {returncode}")
        self.returncode = returncode


class StartupError(RuntimeError):
    """The job could not make all services ready."""


def _exit_status(returncode: int) -> int:
    return 128 - returncode if returncode < 0 else returncode or 1


class ProcessController:
    """Own exact child process groups and shut them down in dependency order."""

    def __init__(
        self,
        *,
        health_check: Callable[[str], bool],
        shutdown_grace_s: float = 30.0,
        poll_interval_s: float = 0.2,
    ) -> None:
        if not all(
            math.isfinite(value) and value > 0 for value in (shutdown_grace_s, poll_interval_s)
        ):
            raise ValueError("shutdown grace and poll interval must be positive")
        self._health_check = health_check
        self._shutdown_grace_s = shutdown_grace_s
        self._poll_interval_s = poll_interval_s
        self._processes: dict[str, subprocess.Popen[bytes]] = {}
        self._stop_signal: int | None = None
        self._stop_event = threading.Event()
        self._stopped = False

    def _handle_signal(self, signum: int, _frame: object) -> None:
        self._stop_signal = signum
        self._stop_event.set()

    def start(self, name: str, command: Sequence[str], *, cwd: Path) -> None:
        """Start one owned child in a new process group."""
        if name in self._processes or self._stopped:
            raise ValueError(f"cannot start {name} twice or after shutdown")
        self._processes[name] = subprocess.Popen(list(command), cwd=cwd, start_new_session=True)

    def _exited_service(self) -> ServiceExited | None:
        for name, process in self._processes.items():
            returncode = process.poll()
            if returncode is not None:
                return ServiceExited(name, returncode)
        return None

    def _wait_for_health(self, urls: Sequence[str], timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while not self._stop_event.is_set():
            exited = self._exited_service()
            if exited is not None:
                raise exited
            if all(self._health_check(url) for url in urls):
                return
            if time.monotonic() >= deadline:
                raise StartupError(f"timed out waiting for {', '.join(urls)}")
            self._stop_event.wait(self._poll_interval_s)

    @staticmethod
    def _signal_group(process: subprocess.Popen[bytes], signum: int) -> None:
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            pass
        except PermissionError:
            if process.poll() is None:
                process.send_signal(signum)

    @staticmethod
    def _group_exists(process: subprocess.Popen[bytes]) -> bool:
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return process.poll() is None
        return True

    def _wait_then_kill(self, names: Sequence[str]) -> None:
        processes = [self._processes[name] for name in names if name in self._processes]
        for process in processes:
            self._signal_group(process, signal.SIGTERM)
        deadline = time.monotonic() + self._shutdown_grace_s
        while True:
            for process in processes:
                process.poll()
            if not any(self._group_exists(process) for process in processes):
                break
            if time.monotonic() >= deadline:
                break
            time.sleep(min(self._poll_interval_s, deadline - time.monotonic()))
        for process in processes:
            if self._group_exists(process):
                self._signal_group(process, signal.SIGKILL)
            if process.poll() is None:
                try:
                    process.wait(timeout=self._shutdown_grace_s)
                except subprocess.TimeoutExpired:
                    print(
                        f"CanaTune controller: process {process.pid} did not exit after SIGKILL",
                        file=sys.stderr,
                        flush=True,
                    )

    def stop(self) -> None:
        """Drain the proxy first, then both node steps, then the GPU agents
        (which reset every clock they locked on exit)."""
        if self._stopped:
            return
        self._stopped = True
        self._wait_then_kill(("proxy",))
        self._wait_then_kill(("prefill", "decode"))
        self._wait_then_kill(tuple(name for name in self._processes if name.endswith("_agent")))

    def run(
        self,
        *,
        cwd: Path,
        prefill_command: Sequence[str],
        decode_command: Sequence[str],
        proxy_command: Sequence[str],
        endpoint_urls: Sequence[str],
        proxy_health_url: str,
        endpoint_timeout_s: float,
        proxy_timeout_s: float,
        agent_commands: Mapping[str, Sequence[str]] | None = None,
        agent_health_urls: Sequence[str] = (),
        agent_timeout_s: float = 120.0,
    ) -> int:
        """Start services, supervise them, and return a shell-compatible status."""
        previous_handlers = {
            signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)
        }
        for signum in previous_handlers:
            signal.signal(signum, self._handle_signal)
        try:
            # GPU agents first: they must be up before the Controller locks any clock.
            for name, command in (agent_commands or {}).items():
                self.start(f"{name}_agent", command, cwd=cwd)
            if agent_health_urls:
                self._wait_for_health(agent_health_urls, agent_timeout_s)
                if self._stop_event.is_set():
                    return 128 + (self._stop_signal or signal.SIGTERM)
            self.start("prefill", prefill_command, cwd=cwd)
            self.start("decode", decode_command, cwd=cwd)
            self._wait_for_health(endpoint_urls, endpoint_timeout_s)
            if self._stop_event.is_set():
                return 128 + (self._stop_signal or signal.SIGTERM)

            self.start("proxy", proxy_command, cwd=cwd)
            self._wait_for_health((proxy_health_url,), proxy_timeout_s)
            if not self._stop_event.is_set():
                print(f"CanaTune ready: {proxy_health_url}", flush=True)
            while not self._stop_event.is_set():
                exited = self._exited_service()
                if exited is not None:
                    raise exited
                self._stop_event.wait(self._poll_interval_s)
            return 128 + (self._stop_signal or signal.SIGTERM)
        except ServiceExited as error:
            print(f"CanaTune controller: {error}", file=sys.stderr, flush=True)
            return _exit_status(error.returncode)
        except (OSError, StartupError) as error:
            print(f"CanaTune controller: {error}", file=sys.stderr, flush=True)
            return 1
        finally:
            self.stop()
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)


def _positive_seconds(value: str, name: str) -> float:
    try:
        seconds = float(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive number") from error
    if not 0 < seconds < float("inf"):
        raise ValueError(f"{name} must be a finite positive number")
    return seconds


def main() -> int:
    """Run the P/D steps and the CanaTune service inside one Slurm allocation."""
    root = Path(os.environ["PROJECT_ROOT"])
    config = load_config(os.environ["CANATUNE_CONFIG"])
    prefill_node = os.environ["PREFILL_NODE"]
    decode_node = os.environ["DECODE_NODE"]
    prefill_host = os.environ["CANATUNE_PREFILL_HOST"]
    decode_host = os.environ["CANATUNE_DECODE_HOST"]
    prefill_port = int(os.environ["PREFILL_HTTP_PORT_BASE"])
    decode_port = int(os.environ["DECODE_HTTP_PORT_BASE"])
    proxy = config["proxy"]
    proxy_host = "127.0.0.1" if proxy["host"] == "0.0.0.0" else proxy["host"]
    endpoint_urls = [
        f"http://{host}:{base + index}/health"
        for host, base in ((prefill_host, prefill_port), (decode_host, decode_port))
        for index in range(4)
    ]
    endpoint_timeout_s = _positive_seconds(
        os.environ.get("ENDPOINT_TIMEOUT_S", "6030"), "ENDPOINT_TIMEOUT_S"
    )
    shutdown_grace_s = _positive_seconds(
        os.environ.get("SHUTDOWN_GRACE_S", "30"), "SHUTDOWN_GRACE_S"
    )

    agent_commands: dict[str, list[str]] = {}
    agent_health_urls: list[str] = []
    clock_control = config.get("clock_control", {})
    if clock_control.get("enabled"):
        port = int(clock_control.get("agent_port", 9300))
        topology = config["topology"]
        for role, node, host in (
            ("prefill", prefill_node, prefill_host),
            ("decode", decode_node, decode_host),
        ):
            group = topology[f"{role}_nodegroup"]
            min_mhz = int((clock_control.get("min_mhz") or {}).get(role, 0))
            agent_commands[role] = [
                "srun",
                "--overlap",
                "--nodes=1",
                "--ntasks=1",
                f"--nodelist={node}",
                "env",
                f"CANATUNE_AGENT_GPUS={','.join(map(str, group['gpu_ids']))}",
                f"CANATUNE_AGENT_MIN_MHZ={min_mhz}",
                f"CANATUNE_AGENT_MEMORY_MHZ={group['memory_frequency_mhz']}",
                f"CANATUNE_AGENT_PORT={port}",
                sys.executable,
                "-m",
                "canatune.infrastructure.gpu_agent",
            ]
            agent_health_urls.append(f"http://{host}:{port}/health")
            # The proxy process inherits these and reaches the agents over the network.
            os.environ[f"CANATUNE_{role.upper()}_AGENT"] = f"http://{host}:{port}"

    with httpx.Client(timeout=3, trust_env=False) as client:

        def health_check(url: str) -> bool:
            try:
                return client.get(url).status_code == 200
            except httpx.HTTPError:
                return False

        controller = ProcessController(health_check=health_check, shutdown_grace_s=shutdown_grace_s)
        return controller.run(
            cwd=root,
            prefill_command=(
                "srun",
                "--overlap",
                "--nodes=1",
                "--ntasks=1",
                f"--nodelist={prefill_node}",
                "bash",
                str(root / "scripts/start_4p_prefill_node.sh"),
            ),
            decode_command=(
                "srun",
                "--overlap",
                "--nodes=1",
                "--ntasks=1",
                f"--nodelist={decode_node}",
                "bash",
                str(root / "scripts/start_4d_decode_node.sh"),
            ),
            proxy_command=(sys.executable, "-m", "canatune"),
            endpoint_urls=endpoint_urls,
            proxy_health_url=f"http://{proxy_host}:{proxy['port']}/health",
            endpoint_timeout_s=endpoint_timeout_s,
            proxy_timeout_s=60,
            agent_commands=agent_commands,
            agent_health_urls=agent_health_urls,
        )


if __name__ == "__main__":
    raise SystemExit(main())
