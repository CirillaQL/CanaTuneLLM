"""scripts/make_job_config.py: node-group overrides reach the derived endpoints."""

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import make_job_config  # noqa: E402

BASE = str(Path(__file__).resolve().parents[1] / "config.yaml")


def run(tmp_path, *sets: str) -> dict:
    out = tmp_path / "job.yaml"
    argv = [
        "--base", BASE, "--out", str(out), "--prefill-node", "n", "--decode-node", "n",
        "--prefill-gpus", "0,1", "--decode-gpus", "2,3", "--python", "python",
        "--work-dir", str(tmp_path / "w"),
    ]  # fmt: skip
    for item in sets:
        argv += ["--set", item]
    old = sys.argv
    sys.argv = ["make_job_config.py", *argv]
    try:
        assert make_job_config.main() == 0
    finally:
        sys.argv = old
    return yaml.safe_load(out.read_text())


def test_single_host_ports_reach_the_endpoints(tmp_path) -> None:
    config = run(
        tmp_path,
        "topology.decode_nodegroup.kv_port_base=14679",
        "clock_control.agent_port={prefill: 9450, decode: 9451}",
        "topology.decode_nodegroup.gpu_type=NVIDIA L40S",
    )
    e = config["topology"]["endpoints"]
    assert [e[k]["kv_port"] for k in ("P0", "P1", "D0", "D1")] == [14579, 14580, 14679, 14680]
    assert [e[k]["gpu_id"] for k in ("P0", "P1", "D0", "D1")] == [0, 1, 2, 3]
    assert config["clock_control"]["agent_port"] == {"prefill": 9450, "decode": 9451}
    assert config["topology"]["decode_nodegroup"]["gpu_type"] == "NVIDIA L40S"
    assert config["routing"]["production_pairs"] == [["P1", "D1"]]


def test_an_explicit_endpoint_override_still_wins(tmp_path) -> None:
    config = run(
        tmp_path,
        "topology.decode_nodegroup.kv_port_base=14679",
        "topology.endpoints.D1.kv_port=15000",
    )
    assert config["topology"]["endpoints"]["D0"]["kv_port"] == 14679
    assert config["topology"]["endpoints"]["D1"]["kv_port"] == 15000
