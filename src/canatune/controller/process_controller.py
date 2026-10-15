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

from canatune.config import kv_buffer_device, kv_connector, load_config
from canatune.proxy.proxy import parse_endpoints
from canatune.topology import (
    ROLE_HOST_ENV,
    agent_port,
    check_tensor_parallel,
    host_of,
    node_hosts,
    role_nodes,
    tensor_parallel_size,
)


class ServiceExited(RuntimeError):
    """A supervised process stopped before the job was asked to stop."""

    def __init__(self, name: str, returncode: int) -> None:
        super().__init__(f"{name} exited with status {returncode}")
        self.returncode = returncode


class StartupError(RuntimeError):
    """The job could not make all services ready."""


def _exit_status(returncode: int) -> int:
    return 128 - returncode if returncode < 0 else returncode or 1


NODE_STEPS = ("prefill", "decode")  # step names: prefill, prefill1 (a role's next node), ...


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
        # Called while the agents still run: reset their clocks over HTTP (the agents'
        # own reset on exit may not run when srun ends them with SIGTERM).
        self.before_agents_stop: Callable[[], None] | None = None

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
        self._wait_then_kill(
            tuple(
                n for n in self._processes if n.startswith(NODE_STEPS) and not n.endswith("_agent")
            )
        )
        if self.before_agents_stop is not None:
            try:
                self.before_agents_stop()
            except Exception as error:
                print(f"CanaTune controller: clock reset failed: {error!r}", file=sys.stderr)
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
        extra_node_commands: Mapping[str, Sequence[str]] | None = None,
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
            for name, command in (extra_node_commands or {}).items():  # a role's other nodes
                self.start(name, command, cwd=cwd)
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


def reset_agent_clocks(client: httpx.Client, agent_urls: Sequence[str]) -> None:
    """Reset every GPU of every agent and print the clocks read back."""
    for url in agent_urls:
        for gpu in client.get(f"{url}/gpus", timeout=10).json():
            index = gpu["index"]
            try:
                client.post(f"{url}/gpus/{index}/reset", timeout=30).raise_for_status()
                status = "ok"
            except httpx.HTTPError as error:
                status = f"failed: {error!r}"
            print(f"CanaTune controller: reset {url} GPU {index}: {status}", flush=True)


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
    check_tensor_parallel(config)
    prefill_node = os.environ["PREFILL_NODE"]
    decode_node = os.environ["DECODE_NODE"]
    proxy = config["proxy"]
    proxy_host = "127.0.0.1" if proxy["host"] == "0.0.0.0" else proxy["host"]
    nodes = {role: role_nodes(config, role) for role in NODE_STEPS}
    if len(nodes["prefill"]) == 1:
        nodes["prefill"][0]["node"] = prefill_node
    if len(nodes["decode"]) == 1:
        nodes["decode"][0]["node"] = decode_node
    hosts = node_hosts()
    for role in NODE_STEPS:
        missing = [n["node"] for n in nodes[role] if n["node"] not in hosts]
        if len(nodes[role]) > 1 and missing:
            print(
                f"CanaTune controller: {role} spans several nodes; CANATUNE_NODE_HOSTS has "
                f"no address for {', '.join(missing)}",
                file=sys.stderr,
            )
            return 2
    # Health of every endpoint at its own node's address and configured port.
    endpoint_urls = [
        f"http://{e.http_host}:{e.http_port}/health" for e in parse_endpoints(config).values()
    ]
    # The node steps (srun inherits this environment) start vLLM with the connector
    # the proxy speaks.
    os.environ["KV_CONNECTOR"] = kv_connector(config)
    os.environ["KV_LOAD_FAILURE_POLICY"] = str(
        config["kv_transfer"].get("load_failure_policy", "fail")
    )
    os.environ["KV_BUFFER_DEVICE"] = kv_buffer_device(config)
    # Node steps: optionally request the node's GPUs for the step without binding
    # (each vLLM process selects its own GPUs via CUDA_VISIBLE_DEVICES).
    step_gpus = os.environ.get("CANATUNE_STEP_GPUS") == "1"
    scripts = {"prefill": "start_4p_prefill_node.sh", "decode": "start_4d_decode_node.sh"}

    def node_step(role: str, k: int) -> tuple[str, list[str]]:
        """srun step for the role's k-th node: its GPUs, TP, the index of its first
        endpoint and its own work directory (the first node keeps the plain names)."""
        node = nodes[role][k]
        tp = tensor_parallel_size(config, role)
        prefix = role.upper()
        env = [f"{prefix}_TP={tp}"]
        if len(nodes[role]) > 1:
            offset = sum(len(n["gpu_ids"]) // tp for n in nodes[role][:k])
            host = host_of(config, role, node["node"])
            env += [
                f"{prefix}_GPU_IDS={','.join(map(str, node['gpu_ids']))}",
                f"{prefix}_INDEX_OFFSET={offset}",
                f"{prefix}_ROLE_DIR={role if k == 0 else f'{role}{k}'}",
                f"{ROLE_HOST_ENV[role]}={host}",
                f"CANATUNE_{prefix}_KV_HOST={host}",
            ]
            if node.get("gpu_uuids"):
                env.append(f"{prefix}_GPU_UUIDS={','.join(node['gpu_uuids'])}")
        gpus = [f"--gpus-per-node={len(node['gpu_ids'])}", "--gpu-bind=none"] if step_gpus else []
        name = role if k == 0 else f"{role}{k}"
        command = [
            "srun", "--overlap", "--nodes=1", "--ntasks=1", f"--nodelist={node['node']}",
            *gpus, "env", *env, "bash", str(root / "scripts" / scripts[role]),
        ]
        return name, command

    steps = dict(node_step(role, k) for role in NODE_STEPS for k in range(len(nodes[role])))

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
        topology = config["topology"]
        for role in NODE_STEPS:
            group = topology[f"{role}_nodegroup"]
            min_mhz = int((clock_control.get("min_mhz") or {}).get(role, 0))
            memory = group.get("memory_frequency_mhz")
            for k, node in enumerate(nodes[role]):
                # One port for both roles, or {prefill: ..., decode: ...}.
                port = agent_port(config, role)
                host = (
                    host_of(config, role, node["node"])
                    if len(nodes[role]) > 1
                    else os.environ[ROLE_HOST_ENV[role]]
                )
                agent_commands[role if k == 0 else f"{role}{k}"] = [
                    "srun",
                    "--overlap",
                    "--nodes=1",
                    "--ntasks=1",
                    f"--nodelist={node['node']}",
                    "env",
                    f"CANATUNE_AGENT_GPUS={','.join(map(str, node['gpu_ids']))}",
                    f"CANATUNE_AGENT_MIN_MHZ={min_mhz}",
                    f"CANATUNE_AGENT_PORT={port}",
                    *([f"CANATUNE_AGENT_MEMORY_MHZ={memory}"] if memory else []),
                    *(
                        [f"CANATUNE_AGENT_GPU_UUIDS={','.join(node['gpu_uuids'])}"]
                        if node.get("gpu_uuids")
                        else []
                    ),
                    sys.executable,
                    "-m",
                    "canatune.infrastructure.gpu_agent",
                ]
                agent_health_urls.append(f"http://{host}:{port}/health")
                if k == 0:
                    # The proxy process inherits these and reaches the agents over the
                    # network; a role's other nodes' agents follow from CANATUNE_NODE_HOSTS.
                    os.environ[f"CANATUNE_{role.upper()}_AGENT"] = f"http://{host}:{port}"

    with httpx.Client(timeout=3, trust_env=False) as client:

        def health_check(url: str) -> bool:
            try:
                return client.get(url).status_code == 200
            except httpx.HTTPError:
                return False

        controller = ProcessController(health_check=health_check, shutdown_grace_s=shutdown_grace_s)
        agent_urls = [url.removesuffix("/health") for url in agent_health_urls]
        controller.before_agents_stop = lambda: reset_agent_clocks(client, agent_urls)
        return controller.run(
            cwd=root,
            prefill_command=steps.pop("prefill"),
            decode_command=steps.pop("decode"),
            proxy_command=(sys.executable, "-m", "canatune"),
            endpoint_urls=endpoint_urls,
            proxy_health_url=f"http://{proxy_host}:{proxy['port']}/health",
            endpoint_timeout_s=endpoint_timeout_s,
            proxy_timeout_s=60,
            agent_commands=agent_commands,
            agent_health_urls=agent_health_urls,
            extra_node_commands=steps,
        )


if __name__ == "__main__":
    raise SystemExit(main())
