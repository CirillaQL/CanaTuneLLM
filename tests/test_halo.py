"""halo.enabled: the HALO-style bandit comparison (zones, emergency upshift, LinUCB)."""

import asyncio

import httpx
import pytest

from canatune.config import load_config
from canatune.controller.halo import HaloController, HaloSettings, LinUcbArm, PhaseBandit, solve


def runtime(**halo):
    from canatune.service import build_runtime

    config = load_config()
    config["routing"]["policy"] = "cantune"
    config["telemetry"]["enabled"] = False
    config["halo"] = {"enabled": True, **halo}
    rt = build_runtime(config, {}, lambda: httpx.AsyncClient())
    asyncio.run(rt.controller.start())
    for g in rt.groups:
        g.state = type(g.state).ACTIVE
        g.effective = rt.tiers.max_point
    return rt


def bandit(**kw) -> PhaseBandit:
    b = PhaseBandit("decode", 100.0, HaloSettings(enabled=True, **kw))
    b.ladder = [2040, 1890, 1740, 1590, 1440, 1290, 1140, 990, 840]
    return b


def test_runtime_uses_halo_with_round_robin() -> None:
    rt = runtime()
    assert isinstance(rt.feedback, HaloController)
    names = [rt.router.try_admit(128, True).group.name for _ in range(8)]
    assert names == ["G0", "G1", "G2", "G3"] * 2


def test_both_comparisons_cannot_run_together() -> None:
    from canatune.service import build_runtime

    config = load_config()
    config["routing"]["policy"] = "cantune"
    config["telemetry"]["enabled"] = False
    config["halo"] = {"enabled": True}
    config["feedback"] = {"enabled": True}
    with pytest.raises(ValueError):
        build_runtime(config, {}, lambda: httpx.AsyncClient())


def test_solve_and_linucb_prefer_the_better_arm() -> None:
    assert solve([[2.0, 0.0], [0.0, 4.0]], [2.0, 2.0]) == pytest.approx([1.0, 0.5])
    good, bad = LinUcbArm.new(2, 1.0), LinUcbArm.new(2, 1.0)
    for _ in range(20):
        good.update([1.0, 0.5], -0.1)
        bad.update([1.0, 0.5], -1.0)
    assert good.ucb([1.0, 0.5], 0.1) > bad.ucb([1.0, 0.5], 0.1)


def test_green_only_downshifts_red_only_upshifts_with_one_emergency() -> None:
    b = bandit()
    d = b.step([20.0] * 50, 1000.0, 50.0, 0.5, 2040)  # p95 20 / 100: green
    assert d["zone"] == "green" and d["action"] in (-2, -1) and b.level > 0
    b.level = 6
    d = b.step([99.0] * 50, 1000.0, 50.0, 0.5, 2040)  # first entry into red
    assert (d["zone"], d["how"], d["action"]) == ("red", "emergency", 4) and b.level == 2
    d = b.step([99.0] * 50, 1000.0, 50.0, 0.5, 2040)  # still red: bandit, upshifts only
    assert d["how"] == "bandit" and d["action"] in (1, 2)


def test_cost_penalises_p95_above_065_slo() -> None:
    b = bandit()
    calm = b.cost(2000.0, 50.0, 60.0)
    near = b.cost(2000.0, 50.0, 90.0)
    assert calm == pytest.approx(4 * 0.05) and near > calm


def test_window_scores_latencies_and_moves_clocks() -> None:
    rt = runtime()
    halo = rt.feedback
    asyncio.run(halo.setup())
    t = halo._clock()
    rt.router.finished_outcomes.extend([(t, False, False, 100.0, 20.0)] * 30)  # both green
    asyncio.run(halo.window({"prefill": 500.0, "decode": 500.0}))
    assert halo.point().prefill_mhz <= rt.tiers.max_point.prefill_mhz
