"""Clock-grid evaluation hooks: overriding tier H's working clocks, freezing the Canary
after its first publish, the /canatune/working endpoint and the bench's grid stages."""

import asyncio
import sys
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from canatune.api.control import create_control_router
from canatune.config import load_config
from canatune.controller.canary import CanaryScheduler, SchedulerSettings
from canatune.controller.tier_controller import ControllerConfigError
from canatune.domain.groups import ClockPoint, GroupState, Tier
from canatune.infrastructure.clocks import NullClockActuator
from canatune.service import build_runtime

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import bench_production as bench  # noqa: E402
from test_bench import FAKE_BENCH  # noqa: E402
from test_router_controller import MAX, H, activate, published, setup  # noqa: E402


class SteppedClocks(NullClockActuator):
    """Supported clocks in 30 MHz steps (P) and 15 MHz steps (D)."""

    async def supported_clocks(self, ref):
        return list(range(600, 2521, 30)) if ref.gpu % 2 == 0 else list(range(300, 2041, 15))


def test_override_moves_every_group_at_h_and_snaps_to_supported_clocks() -> None:
    _, _, groups, tiers, _, controller, _ = setup(table=published())
    controller.actuator = SteppedClocks()
    activate(groups)
    groups[1].tier, groups[1].effective = Tier.MAX, MAX  # full effort: stays at MAX
    clock = asyncio.run(controller.set_working(ClockPoint(1094, 1171)))
    assert clock == ClockPoint(1080, 1170)  # nearest supported (P steps of 30 from 600)
    assert tiers.clocks(Tier.H) == clock
    assert [g.effective for g in groups] == [clock, MAX] + [clock] * (len(groups) - 2)
    locked = {(ref.gpu, mhz) for ref, mhz in controller.actuator.calls}
    assert locked == {(0, 1080), (1, 1170)}  # G0's refs; G1 (at MAX) is not touched
    # Back to the table's H; a group returning from MAX lands on the working point.
    assert asyncio.run(controller.set_working(None)) == H
    assert groups[0].effective == H


def test_override_is_refused_while_the_solver_chooses() -> None:
    _, _, _, _, _, controller, _ = setup(table=published(model=True), solver=True)
    with pytest.raises(ControllerConfigError):
        asyncio.run(controller.set_working(ClockPoint(1080, 1170)))


def test_a_frozen_canary_runs_only_the_cold_start() -> None:
    _, _, _, tiers, _, controller, _ = setup(table=None)
    from canatune.domain.load import LengthStats

    frozen = SchedulerSettings(recalibrate=False, periodic_s=0.0)
    scheduler = CanaryScheduler(controller, None, LengthStats([(512, 64)]), frozen, theta=0.1)
    assert scheduler.trigger() == ("full", "cold_start")
    tiers.table = published()
    assert scheduler.trigger() is None  # periodic_s = 0 would otherwise recheck at once
    live = CanaryScheduler(controller, None, LengthStats([(512, 64)]),
                           SchedulerSettings(periodic_s=0.0), theta=0.1)  # fmt: skip
    assert live.trigger() == ("recheck", "periodic")


def test_working_endpoint() -> None:
    config = load_config()
    config["routing"]["policy"] = "cantune"
    config["telemetry"]["enabled"] = False
    runtime = build_runtime(config, {}, lambda: httpx.AsyncClient())
    app = FastAPI()
    app.include_router(create_control_router(runtime))
    client = TestClient(app)
    assert client.post("/canatune/working", json={"prefill_mhz": 1080, "decode_mhz": 1170}
                       ).json()["ok"] is False  # no table yet  # fmt: skip
    runtime.tiers.table = published()
    group = runtime.controller.groups[1]
    group.state, group.tier = GroupState.ACTIVE, Tier.H
    reply = client.post("/canatune/working", json={"prefill_mhz": 1080, "decode_mhz": 1170})
    assert reply.json() == {"ok": True, "clock": "1080/1170"}
    assert group.effective == ClockPoint(1080, 1170)
    assert client.post("/canatune/working", json={"prefill_mhz": 1080}).json()["ok"] is False
    assert client.post("/canatune/working", json={}).json()["clock"] == H.key()


def test_grid_runs_every_stage_per_point_and_moves_h_first(tmp_path) -> None:
    fake = tmp_path / "fake_bench.py"
    fake.write_text(FAKE_BENCH)
    s = bench.BenchSettings.from_env({
        "BENCH_DATASET": "/d", "BENCH_RATES": "1,2", "BENCH_STAGE_S": "4", "BENCH_GAP_S": "0",
        "BENCH_CMD": f"{sys.executable} {fake}", "BENCH_GRID_SETTLE_S": "0",
        "BENCH_CLOCK_GRID": "[[1080, 1170], [1560, 735]]",
    })  # fmt: skip
    assert [st.name for st in s.stages()] == [
        "g00_1080_1170_r0_1rps", "g00_1080_1170_r1_2rps",
        "g01_1560_735_r0_1rps", "g01_1560_735_r1_2rps",
    ]  # fmt: skip
    moves = []

    async def set_clock(clock):
        moves.append(clock)
        return {"ok": True, "clock": f"{clock[0]}/{clock[1]}"}

    stages, finished = asyncio.run(
        bench.run_stages(
            s, base_url="u", model="m", out=tmp_path, agents=[], state_poll=False,
            ttft_slo_ms=1000, tpot_slo_ms=200, set_clock=set_clock,
        )
    )  # fmt: skip
    assert moves == [(1080, 1170), (1560, 735)]  # once per point, before its first stage
    assert all(code == 0 for *_, code in stages) and len(stages) == 4
    assert (tmp_path / "stages/g01_1560_735_r0_1rps/clock.json").exists()
    summary = bench.summarize(tmp_path, stages, finished, {}, ttft_slo_ms=1000, tpot_slo_ms=200)
    assert summary["stages"]["g01_1560_735_r1_2rps"]["clock"] == "1560/735"


def test_a_refused_move_stops_the_grid(tmp_path) -> None:
    s = bench.BenchSettings.from_env({
        "BENCH_DATASET": "/d", "BENCH_RATES": "1", "BENCH_GRID_SETTLE_S": "0",
        "BENCH_CLOCK_GRID": "[[1080, 1170]]",
    })  # fmt: skip

    async def refuse(clock):
        return {"ok": False, "reason": "no tier table yet"}

    with pytest.raises(RuntimeError, match="could not move H"):
        asyncio.run(
            bench.run_stages(
                s, base_url="u", model="m", out=tmp_path, agents=[], state_poll=False,
                ttft_slo_ms=1000, tpot_slo_ms=200, set_clock=refuse,
            )
        )  # fmt: skip
