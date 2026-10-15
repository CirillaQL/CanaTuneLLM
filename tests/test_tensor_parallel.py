"""Tensor-parallel endpoints and roles on several nodes: job config, clock refs and
locks, merged readings, per-node hosts, energy per endpoint, node steps, launch script."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import bench_production  # noqa: E402
import make_job_config  # noqa: E402
import smoke_production  # noqa: E402

from canatune.config import ConfigError  # noqa: E402
from canatune.controller import process_controller  # noqa: E402
from canatune.infrastructure.clocks import (  # noqa: E402
    AgentClockActuator,
    GpuRef,
    endpoint_reading,
    gpu_refs,
)
from canatune.proxy.proxy import parse_endpoints  # noqa: E402
from canatune.service import identity  # noqa: E402
from canatune.topology import check_tensor_parallel  # noqa: E402

HOSTS = "node-a=10.0.0.5,node-b=10.0.0.3,node-c=10.0.0.2"


def tp_config(tmp_path: Path, *sets: str) -> dict:
    """4 x TP2 P on node-a + node-b, 4 x TP2 D on node-c (8 GPUs)."""
    out = tmp_path / "job.yaml"
    argv = [
        "--base", str(ROOT / "config.yaml"), "--out", str(out),
        "--prefill-node", "node-a,node-b", "--prefill-gpus", "0,1,2,3;4,5,6,7",
        "--prefill-uuids", "U0,U1,U2,U3;N4,N5,N6,N7",
        "--decode-node", "node-c", "--decode-gpus", "0,1,2,3,4,5,6,7",
        "--tp", "2", "--python", "python", "--work-dir", str(tmp_path / "w"),
        "--set", "kv_transfer.connector=P2pNcclConnector", "--set", "clock_control.enabled=true",
    ]  # fmt: skip
    for item in sets:
        argv += ["--set", item]
    assert make_job_config.main(argv) == 0
    return yaml.safe_load(out.read_text())


def test_job_config_places_tp_endpoints_on_their_nodes(tmp_path) -> None:
    config = tp_config(tmp_path)
    e = config["topology"]["endpoints"]
    assert sorted(e) == ["D0", "D1", "D2", "D3", "P0", "P1", "P2", "P3"]
    assert [(e[k]["node"], e[k]["gpu_ids"]) for k in ("P0", "P1", "P2", "P3")] == [
        ("node-a", [0, 1]), ("node-a", [2, 3]), ("node-b", [4, 5]), ("node-b", [6, 7]),
    ]  # fmt: skip
    # Ports restart on every node; KV ports step by the TP size (rank r: kv_port + r).
    assert [e[k]["http_port"] for k in ("P0", "P1", "P2", "P3")] == [8100, 8101, 8100, 8101]
    assert [e[k]["kv_port"] for k in ("P0", "P1", "P2", "P3")] == [14579, 14581, 14579, 14581]
    assert [e[k]["gpu_ids"] for k in ("D0", "D3")] == [[0, 1], [6, 7]]
    assert [e[k]["kv_port"] for k in ("D0", "D1", "D2", "D3")] == [14579, 14581, 14583, 14585]
    prefill = config["topology"]["prefill_nodegroup"]
    assert prefill["tensor_parallel_size"] == 2
    assert prefill["nodes"][1] == {
        "node": "node-b", "gpu_ids": [4, 5, 6, 7], "gpu_uuids": ["N4", "N5", "N6", "N7"],
    }  # fmt: skip
    assert "nodes" not in config["topology"]["decode_nodegroup"]
    assert config["routing"]["production_pairs"] == [["P1", "D1"], ["P2", "D2"], ["P3", "D3"]]


def test_job_config_rejects_gpus_that_do_not_split_into_tp(tmp_path) -> None:
    argv = [
        "--base", str(ROOT / "config.yaml"), "--out", str(tmp_path / "x.yaml"),
        "--prefill-node", "n", "--prefill-gpus", "0,1,2", "--decode-node", "m",
        "--decode-gpus", "0,1,2", "--tp", "2", "--python", "p", "--work-dir", str(tmp_path),
    ]  # fmt: skip
    with pytest.raises(SystemExit):
        make_job_config.main(argv)


def test_refs_use_each_nodes_agent_and_every_rank(tmp_path) -> None:
    config = tp_config(tmp_path, "clock_control.agents.decode=http://10.0.0.2:9300")
    refs = gpu_refs(config, {"CANATUNE_NODE_HOSTS": HOSTS})
    assert refs["P1"] == GpuRef("http://10.0.0.5:9300", 2, (3,))
    assert refs["P2"] == GpuRef("http://10.0.0.3:9300", 4, (5,))
    assert refs["D3"].gpus == (6, 7)


def test_tp1_single_node_refs_are_unchanged() -> None:
    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    config["clock_control"]["agents"] = {"prefill": "http://p:1", "decode": "http://d:2"}
    refs = gpu_refs(config, {})
    assert refs["P2"] == GpuRef("http://p:1", 2) and refs["D0"] == GpuRef("http://d:2", 0)


def test_lock_and_clocks_cover_every_gpu_of_the_endpoint() -> None:
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        gpu = int(request.url.path.split("/")[2])
        if request.url.path.endswith("/clocks"):
            return httpx.Response(200, json={"sm_mhz": [600, 1080, 2520] + ([1500] if gpu else [])})
        seen.append((gpu, json.loads(request.content)["mhz"]))
        return httpx.Response(
            200,
            json={"ok": True, "gpu": gpu, "target_mhz": 1080, "cmd_ms": 5.0 + gpu,
                  "reach_ms": 20.0 * (gpu + 1), "sm_mhz": 1080 - 15 * gpu},
        )  # fmt: skip

    actuator = AgentClockActuator(
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://a")
    )
    ref = GpuRef("http://a", 0, (1,))

    import asyncio

    result = asyncio.run(actuator.lock(ref, 1080))
    assert sorted(seen) == [(0, 1080), (1, 1080)]
    assert result["ok"] and result["gpus"] == [0, 1]
    assert result["sm_mhz"] == 1065 and result["reach_ms"] == 40.0 and result["cmd_ms"] == 6.0
    assert asyncio.run(actuator.supported_clocks(ref)) == [600, 1080, 2520]


def test_reading_sums_energy_and_power_of_the_ranks() -> None:
    readings = [
        {"index": 0, "sm_mhz": 1080, "power_w": 100.0, "energy_mj": 1000.0, "throttle_reasons": 0},
        {"index": 1, "sm_mhz": 1065, "power_w": 90.0, "energy_mj": 900.0, "throttle_reasons": 4},
        {"index": 2, "sm_mhz": 2520, "power_w": 50.0, "energy_mj": 5.0, "throttle_reasons": 0},
    ]
    merged = endpoint_reading(GpuRef("a", 0, (1,)), readings)
    assert merged["energy_mj"] == 1900.0 and merged["power_w"] == 190.0
    assert merged["sm_mhz"] == 1065 and merged["throttle_reasons"] == 4
    assert endpoint_reading(GpuRef("a", 2), readings) is readings[2]
    assert endpoint_reading(GpuRef("a", 2, (3,)), readings) is None


def test_endpoints_resolve_to_their_nodes_address(tmp_path, monkeypatch) -> None:
    config = tp_config(tmp_path)
    monkeypatch.setenv("CANATUNE_NODE_HOSTS", HOSTS)
    monkeypatch.setenv("CANATUNE_PREFILL_HOST", "10.9.9.9")
    endpoints = parse_endpoints(config)
    assert (endpoints["P0"].http_host, endpoints["P3"].kv_host) == ("10.0.0.5", "10.0.0.3")
    assert endpoints["D2"].http_host == "10.0.0.2"


def test_bench_energy_sums_the_gpus_of_each_endpoint(tmp_path, monkeypatch) -> None:
    config = tp_config(tmp_path)
    monkeypatch.setenv("CANATUNE_NODE_HOSTS", HOSTS)
    monkeypatch.setenv("CANATUNE_DECODE_HOST", "10.0.0.2")  # one node: the role's host
    agents, names = smoke_production.agents_and_names(config)
    assert agents == ["http://10.0.0.2:9300", "http://10.0.0.3:9300", "http://10.0.0.5:9300"]
    assert names["http://10.0.0.3:9300|5"] == "P2" and names["http://10.0.0.2:9300|7"] == "D3"
    series = {
        "http://10.0.0.5:9300|0": [(0.0, 0.0), (10.0, 1_000_000.0)],
        "http://10.0.0.5:9300|1": [(0.0, 0.0), (10.0, 500_000.0)],
    }
    stage = bench_production.Stage(name="s", workload="w", rate_rps=1.0, num_prompts=1)
    summary = bench_production.stage_summary(
        stage, (0.0, 10.0), None, series, names, [], ttft_slo_ms=600, tpot_slo_ms=100
    )
    assert summary["energy_by_gpu_j"] == {"P0": 1500.0}
    assert summary["energy_j"] == 1500.0


def test_p2p_nccl_needs_symmetric_tp(tmp_path) -> None:
    config = tp_config(tmp_path)
    check_tensor_parallel(config)
    config["topology"]["decode_nodegroup"]["tensor_parallel_size"] = 1
    with pytest.raises(ConfigError):
        check_tensor_parallel(config)


def test_identity_names_tp_only_when_used(tmp_path) -> None:
    base = yaml.safe_load((ROOT / "config.yaml").read_text())
    assert "tensor_parallel_size" not in identity(base)
    assert identity(tp_config(tmp_path))["tensor_parallel_size"] == [2, 2]


def test_controller_starts_a_step_and_an_agent_per_node(tmp_path, monkeypatch) -> None:
    tp_config(tmp_path)
    path = tmp_path / "job.yaml"
    monkeypatch.setenv("PROJECT_ROOT", str(ROOT))
    monkeypatch.setenv("CANATUNE_CONFIG", str(path))
    monkeypatch.setenv("PREFILL_NODE", "node-a")
    monkeypatch.setenv("DECODE_NODE", "node-c")
    monkeypatch.setenv("CANATUNE_PREFILL_HOST", "10.0.0.5")
    monkeypatch.setenv("CANATUNE_DECODE_HOST", "10.0.0.2")
    monkeypatch.setenv("CANATUNE_NODE_HOSTS", HOSTS)
    monkeypatch.setenv("CANATUNE_STEP_GPUS", "1")
    for name in ("KV_CONNECTOR", "KV_LOAD_FAILURE_POLICY", "KV_BUFFER_DEVICE",
                 "CANATUNE_PREFILL_AGENT", "CANATUNE_DECODE_AGENT"):  # fmt: skip
        monkeypatch.delenv(name, raising=False)
    captured = {}

    def fake_run(self, **kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(process_controller.ProcessController, "run", fake_run)
    assert process_controller.main() == 0

    first = captured["prefill_command"]
    assert "--nodelist=node-a" in first and "--gpus-per-node=4" in first
    assert "PREFILL_TP=2" in first and "PREFILL_GPU_IDS=0,1,2,3" in first
    assert "PREFILL_INDEX_OFFSET=0" in first and "PREFILL_ROLE_DIR=prefill" in first
    second = captured["extra_node_commands"]["prefill1"]
    assert "--nodelist=node-b" in second and "PREFILL_GPU_IDS=4,5,6,7" in second
    assert "PREFILL_INDEX_OFFSET=2" in second and "PREFILL_ROLE_DIR=prefill1" in second
    assert "CANATUNE_PREFILL_KV_HOST=10.0.0.3" in second
    assert "PREFILL_GPU_UUIDS=N4,N5,N6,N7" in second
    decode = captured["decode_command"]
    assert "DECODE_TP=2" in decode and "--gpus-per-node=8" in decode
    assert not any(a.startswith("DECODE_GPU_IDS=") for a in decode)  # one node: inherited
    assert set(captured["agent_commands"]) == {"prefill", "prefill1", "decode"}
    assert "CANATUNE_AGENT_GPUS=4,5,6,7" in captured["agent_commands"]["prefill1"]
    assert sorted(captured["agent_health_urls"]) == [
        "http://10.0.0.2:9300/health", "http://10.0.0.3:9300/health", "http://10.0.0.5:9300/health",
    ]  # fmt: skip
    assert "http://10.0.0.3:8101/health" in captured["endpoint_urls"]
    assert len(captured["endpoint_urls"]) == 8
    assert os.environ["CANATUNE_PREFILL_AGENT"] == "http://10.0.0.5:9300"


def test_controller_needs_addresses_for_a_multi_node_role(tmp_path, monkeypatch) -> None:
    tp_config(tmp_path)
    monkeypatch.setenv("PROJECT_ROOT", str(ROOT))
    monkeypatch.setenv("CANATUNE_CONFIG", str(tmp_path / "job.yaml"))
    monkeypatch.setenv("PREFILL_NODE", "node-a")
    monkeypatch.setenv("DECODE_NODE", "node-c")
    monkeypatch.setenv("CANATUNE_NODE_HOSTS", "node-a=10.0.0.5")
    assert process_controller.main() == 2


def test_launch_script_groups_gpus_into_tp_instances(tmp_path) -> None:
    """The prefill script with stub vLLM / nvidia-smi / curl: devices, TP, ports, names."""
    bin_dir, model = tmp_path / "bin", tmp_path / "model"
    bin_dir.mkdir()
    model.mkdir()
    calls = tmp_path / "calls"
    (bin_dir / "curl").write_text("#!/bin/sh\nexit 0\n")
    (bin_dir / "nvidia-smi").write_text("#!/bin/sh\necho GPU-x\n")
    python = bin_dir / "python"
    python.write_text(
        f'#!/bin/sh\necho "CVD=$CUDA_VISIBLE_DEVICES $*" >> {calls}\nexec sleep 30\n'
    )
    for f in bin_dir.iterdir():
        f.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "PYTHON_BIN": str(python), "MODEL_PATH": str(model), "PD_WORK_DIR": str(tmp_path / "pd"),
        "PREFILL_GPU_IDS": "4,5,6,7", "PREFILL_HTTP_PORT_BASE": "8100",
        "PREFILL_KV_PORT_BASE": "14579", "PREFILL_TP": "2", "PREFILL_INDEX_OFFSET": "2",
        "PREFILL_ROLE_DIR": "prefill1", "KV_CONNECTOR": "P2pNcclConnector",
        "NCCL_SOCKET_IFNAME": "^lo", "CANATUNE_PREFILL_KV_HOST": "10.0.0.3",
        "NODE_SHUTDOWN_GRACE_S": "1",
    }  # fmt: skip
    process = subprocess.Popen(
        ["bash", str(ROOT / "scripts/start_4p_prefill_node.sh")],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )  # fmt: skip
    lines = []
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            line = process.stdout.readline()
            lines.append(line)
            if not line or line.startswith("All "):
                break
        # The stub health check passes at once: wait until both stub servers have run.
        while time.monotonic() < deadline and (
            not calls.exists() or len(calls.read_text().splitlines()) < 2
        ):
            time.sleep(0.05)
    finally:
        process.terminate()
        process.wait(timeout=10)
    out = "".join(lines)
    assert "P2 ready: GPU=4,5 HTTP=8100 KV=14579" in out
    assert "P3 ready: GPU=6,7 HTTP=8101 KV=14581" in out
    assert "All 2 Prefill instances are ready (TP 2)" in out
    launched = sorted(calls.read_text().splitlines())  # the stubs may log in either order
    assert launched[0].startswith("CVD=4,5 ") and "--tensor-parallel-size 2" in launched[0]
    assert '"kv_port":14579' in launched[0]
    assert '"kv_port":14581' in launched[1] and launched[1].startswith("CVD=6,7 ")
    assert (tmp_path / "pd/prefill1/logs/prefill_3.log").exists()
