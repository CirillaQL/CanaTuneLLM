"""D GPU memory utilization from the absolute reserve (canatune.infrastructure.memory_plan)."""

import copy

import pytest

from canatune.config import load_config
from canatune.infrastructure import memory_plan as mp

L4_MIB, L40S_MIB = 23034, 46068  # nvidia-smi memory.total


def test_the_default_reserve_keeps_the_same_headroom_on_any_gpu() -> None:
    config = load_config()
    assert config["topology"]["decode_nodegroup"]["gpu_memory_utilization"] == "auto"
    assert mp.peers(config) == 1  # fixed pairing
    reserve = mp.reserve_bytes(config)
    assert reserve == pytest.approx(1e9 + 3e9 + 0.25e9 + 1e9)
    for mib in (L4_MIB, L40S_MIB):
        total = mib * 1024 * 1024
        assert (1 - mp.utilization(config, total)) * total == pytest.approx(reserve)
    assert mp.utilization(config, L4_MIB * 2**20) == pytest.approx(0.783, abs=1e-3)
    assert mp.utilization(config, L40S_MIB * 2**20) == pytest.approx(0.891, abs=1e-3)


def test_dynamic_pairing_reserves_one_communicator_per_prefill() -> None:
    config = load_config()
    config["router"]["pairing"] = "dynamic"
    prefills = sum(1 for e in config["topology"]["endpoints"].values() if e["role"] == "prefill")
    assert mp.peers(config) == prefills
    assert mp.reserve_bytes(config) == pytest.approx(5.0e9 + prefills * 0.25e9)


def test_fixed_fraction_and_bounds() -> None:
    config = copy.deepcopy(load_config())
    config["topology"]["decode_nodegroup"]["gpu_memory_utilization"] = 0.82
    assert mp.utilization(config, 1.0) == 0.82  # a number is used as it is
    config["topology"]["decode_nodegroup"]["gpu_memory_utilization"] = "auto"
    with pytest.raises(mp.MemoryPlanError):
        mp.utilization(config, 8 * 2**30)  # an 8 GiB GPU cannot hold the reserve
    config["topology"]["decode_nodegroup"]["memory_reserve"] = {"overflow_bytes": -1}
    with pytest.raises(mp.MemoryPlanError):
        mp.reserve_bytes(config)


def test_command_line_prints_the_fraction(tmp_path, capsys) -> None:
    assert mp.main(["--config", "config.yaml", "--total-mib", str(L40S_MIB)]) == 0
    out = capsys.readouterr()
    assert float(out.out.strip()) == pytest.approx(0.891, abs=1e-3)
    assert "reserve 5.25 GB" in out.err
    assert mp.main(["--config", "config.yaml", "--total-mib", "4096"]) == 2
