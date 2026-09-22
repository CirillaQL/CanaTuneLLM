"""Schema and persistence behavior for the Canary decision table."""

import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from canatune.controller.frequency_table import (
    WORKLOAD_IDS,
    FrequencyTableError,
    RevisionConflict,
    WorkloadFrequencyTable,
)


def measured(**changes):
    value = {
        "prefill_frequency_mhz": 1710,
        "decode_frequency_mhz": 975,
        "measured_power_w": 220.5,
        "measured_energy_j": 18.25,
        "ttft_ms": 499.9,
        "tpot_ms": 200.0,
        "prefill_endpoint_id": "P0",
        "decode_endpoint_id": "D0",
        "sample_count": 3,
        "updated_unix_s": 1_700_000_000.25,
        "source": "joint_pd_confirmation",
        "slo_met": True,
        "fallback_reason": None,
    }
    return {**value, **changes}


def fallback(**changes):
    value = measured(
        prefill_frequency_mhz=2520,
        decode_frequency_mhz=1500,
        measured_power_w=None,
        measured_energy_j=None,
        ttft_ms=None,
        tpot_ms=None,
        sample_count=0,
        source="safe_high_after_slo_exhausted",
        slo_met=False,
        fallback_reason="no candidate satisfied the SLO",
    )
    return {**value, **changes}


def test_initializes_all_workloads_and_reloads_without_reset(tmp_path) -> None:
    path = tmp_path / "frequency_table.json"
    table = WorkloadFrequencyTable(path)
    document = json.loads(path.read_text(encoding="utf-8"))

    assert document["schema_version"] == 1
    assert tuple(document["entries"]) == WORKLOAD_IDS
    assert all(
        entry == {"workload_id": key, "revision": 0, "value": None}
        for key, entry in document["entries"].items()
    )

    table.write("small_light", measured(), expected_revision=0)
    reloaded = WorkloadFrequencyTable(path)
    assert reloaded.read("small_light").revision == 1
    assert reloaded.read("small_light").value.measured_energy_j == 18.25
    assert reloaded.snapshot() == json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "invalid",
    [
        {"ttft_ms": 500.0},
        {"tpot_ms": 200.1},
        {"sample_count": 0},
        {"measured_energy_j": 0},
        {"measured_power_w": float("nan")},
        {"prefill_frequency_mhz": True},
        {"prefill_endpoint_id": "P1"},
        {"fallback_reason": "unexpected"},
        {"source": "safe_high_after_slo_exhausted"},
    ],
)
def test_rejects_unconfirmed_measurements(tmp_path, invalid) -> None:
    table = WorkloadFrequencyTable(tmp_path / "frequency_table.json")
    with pytest.raises(FrequencyTableError):
        table.write("small_light", measured(**invalid))
    assert table.read("small_light").revision == 0


def test_fallback_has_no_metrics_and_is_replaced_by_confirmed_measurement(tmp_path) -> None:
    table = WorkloadFrequencyTable(tmp_path / "frequency_table.json")
    initial = table.write("small_light", fallback())
    assert initial.revision == 1
    assert initial.value.slo_met is False

    confirmed = table.write("small_light", measured(), expected_revision=1)
    assert confirmed.revision == 2
    assert confirmed.value.slo_met is True
    assert table.write("small_light", fallback()) == confirmed


@pytest.mark.parametrize(
    "invalid",
    [
        {"sample_count": 1},
        {"ttft_ms": 0.0},
        {"fallback_reason": None},
        {"source": "unknown_fallback"},
        {"slo_met": True},
    ],
)
def test_rejects_malformed_fallbacks(tmp_path, invalid) -> None:
    table = WorkloadFrequencyTable(tmp_path / "frequency_table.json")
    with pytest.raises(FrequencyTableError):
        table.write("small_light", fallback(**invalid))


def test_only_lower_energy_measurement_replaces_existing_value(tmp_path) -> None:
    table = WorkloadFrequencyTable(tmp_path / "frequency_table.json")
    first = table.write("decode_heavy", measured(), expected_revision=0)

    assert table.write("decode_heavy", measured(measured_energy_j=19), expected_revision=1) == first
    assert table.write("decode_heavy", measured(measured_energy_j=18.25)) == first
    better = table.write("decode_heavy", measured(measured_energy_j=17), expected_revision=1)
    assert better.revision == 2

    with pytest.raises(RevisionConflict):
        table.write("decode_heavy", measured(measured_energy_j=16), expected_revision=1)
    assert table.read("decode_heavy") == better


def test_conditional_writes_are_serialized(tmp_path) -> None:
    table = WorkloadFrequencyTable(tmp_path / "frequency_table.json")

    def attempt(energy):
        try:
            return table.write(
                "both_heavy", measured(measured_energy_j=energy), expected_revision=0
            )
        except RevisionConflict:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(attempt, (17, 16)))
    assert sum(result == "conflict" for result in results) == 1
    assert table.read("both_heavy").revision == 1


def test_failed_replace_rolls_back_memory_and_keeps_previous_file(tmp_path, monkeypatch) -> None:
    path = tmp_path / "frequency_table.json"
    table = WorkloadFrequencyTable(path)
    before = path.read_bytes()

    def fail_replace(*args):
        raise OSError("simulated disk failure")

    monkeypatch.setattr("canatune.controller.frequency_table.os.replace", fail_replace)
    with pytest.raises(OSError, match="simulated disk failure"):
        table.write("small_light", measured())

    assert table.read("small_light").revision == 0
    assert path.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))


def test_invalid_existing_file_is_not_overwritten(tmp_path) -> None:
    path = tmp_path / "frequency_table.json"
    path.write_text('{"schema_version": 2, "entries": {}}', encoding="utf-8")
    with pytest.raises(FrequencyTableError):
        WorkloadFrequencyTable(path)
    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == 2
