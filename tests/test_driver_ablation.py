"""Driver-clock readiness and immutable calibration export for ablation arms."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import bench_production as bench  # noqa: E402
import smoke_production as smoke  # noqa: E402
from test_router_controller import published  # noqa: E402

from canatune.api.control import create_control_router  # noqa: E402
from canatune.config import load_config  # noqa: E402
from canatune.service import build_runtime  # noqa: E402


def test_export_is_a_separate_preproduction_copy(tmp_path):
    tier = tmp_path / "tier_table.json"
    tier.write_text('{"h": "900/1050"}')
    risk = {"calibrated": True}
    slack = {"counts": {}, "seed": {"1": [50, 0]}}

    def respond(request):
        payload = {"table": risk} if request.url.path == "/canatune/risk" else {
            "router": {"slack": slack}}
        return httpx.Response(200, json=payload)

    snapshot = tmp_path / "snapshot"
    # No risk files exist: the live API is the source, not shutdown artifacts.
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        bench.export_calibration(snapshot, tier, client, "http://service")
        with pytest.raises(FileExistsError):
            bench.export_calibration(snapshot, tier, client, "http://service")
    tier.write_text("production tier update")
    risk["production"] = True
    assert (snapshot / tier.name).read_text() == '{"h": "900/1050"}'
    assert json.loads((snapshot / "risk_table.json").read_text()) == {"calibrated": True}
    assert json.loads((snapshot / "risk_table.json.slack.json").read_text()) == slack


def test_export_rejects_incomplete_calibration(tmp_path):
    tier = tmp_path / "tier_table.json"
    with httpx.Client() as client, pytest.raises(RuntimeError, match="published tier"):
        bench.export_calibration(tmp_path / "snapshot", tier, client, "http://service")
    assert not (tmp_path / "snapshot").exists()


def test_live_snapshot_loads_for_both_ablations_without_shutdown(monkeypatch, tmp_path):
    config = load_config()
    config["telemetry"]["enabled"] = False
    config["clock_control"]["enabled"] = False
    tier, risk = tmp_path / "tier_table.json", tmp_path / "risk_table.json"
    monkeypatch.setenv("CANATUNE_TIER_TABLE", str(tier))
    monkeypatch.setenv("CANATUNE_RISK_TABLE", str(risk))
    runtime = build_runtime(config, {}, lambda: httpx.AsyncClient())
    runtime.tiers.table = published()
    runtime.controller.store.save(runtime.tiers.table)
    runtime.router._apply_table()
    assert not risk.exists()
    app = FastAPI()
    app.include_router(create_control_router(runtime))
    snapshot = tmp_path / "snapshot"
    with TestClient(app) as client:
        bench.export_calibration(snapshot, tier, client, "http://testserver")
    monkeypatch.setenv("CANATUNE_TIER_TABLE", str(snapshot / tier.name))
    monkeypatch.setenv("CANATUNE_RISK_TABLE", str(snapshot / risk.name))
    for choice, policy in (("round_robin", "canary"), ("room", "driver")):
        config["router"]["group_choice"] = choice
        config["controller"]["production_clocks"] = policy
        restored = build_runtime(config, {}, lambda: httpx.AsyncClient())
        restored.router._apply_table()
        assert restored.tiers.table == runtime.tiers.table
        assert restored.router.table.to_json() == runtime.router.table.to_json()
        assert restored.router.slack.to_json() == runtime.router.slack.to_json()


@pytest.mark.parametrize("policy,ready,effective,expected", [
    ("driver", True, "1800/1410", True),
    ("driver", False, "1800/1410", False),
    ("max", False, "1800/1410", True),
    ("canary", False, "900/1050", True),
    ("canary", True, "1800/1410", False),
])
def test_wait_uses_policy_confirmation_instead_of_always_waiting_for_h(
    monkeypatch, policy, ready, effective, expected,
):
    state = {
        "controller": {"production_clocks": policy, "driver_clocks_ready": ready,
                       "max_point": "1800/1410"},
        "router": {"groups": [{"name": "G0", "state": "active", "effective": effective}]},
    }
    monkeypatch.setattr(smoke.time, "sleep", lambda _: None)
    ticks = iter((0.0, 0.0, smoke.H_TIMEOUT_S + 1))
    monkeypatch.setattr(smoke.time, "monotonic", lambda: next(ticks))
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json=state))
    with httpx.Client(transport=transport) as c:
        assert smoke.wait_h(c, "http://service", {"h": "900/1050"}, policy) is expected


def test_benchmark_stops_before_warmup_if_driver_reset_is_unconfirmed(monkeypatch, tmp_path):
    config = load_config()
    config["routing"]["policy"] = "cantune"
    config["controller"]["production_clocks"] = "driver"
    dataset = tmp_path / "dataset.json"
    dataset.write_text("[]")
    for key, value in {"BENCH_OUT": str(tmp_path), "BENCH_MODE": "static",
                       "BENCH_DATASET": str(dataset), "BENCH_WORKLOADS": "",
                       "BENCH_CLOCK_GRID": "", "CANATUNE_CONFIG": "unused"}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(bench, "load_config", lambda _: config)
    monkeypatch.setattr(smoke, "agents_and_names", lambda _: ([], {}))
    monkeypatch.setattr(bench.subprocess, "Popen", lambda *a, **k:
                        SimpleNamespace(returncode=0, poll=lambda: 0))
    monkeypatch.setattr(smoke, "wait_ready", lambda *a: None)
    monkeypatch.setattr(smoke, "wait_publish", lambda *a: (None, {"h": "900/1050"}))
    monkeypatch.setattr(smoke, "wait_h", lambda *a: False)
    monkeypatch.setattr(smoke, "warm_pairs", lambda *a: pytest.fail("warmup after failed reset"))
    monkeypatch.setattr(smoke, "model_name", lambda *a:
                        pytest.fail("model lookup after failed reset"))
    assert bench.main() == 4
    assert "reset not confirmed" in json.loads((tmp_path / "bench_run.json").read_text())["error"]
