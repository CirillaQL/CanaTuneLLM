"""feedback.enabled: the comparison heuristic without the Canary (watermark routing,
SLO-feedback clocks)."""

import asyncio

import httpx

from canatune.config import load_config
from canatune.controller.feedback import FeedbackClockController, FeedbackRouter, ladder
from canatune.domain.groups import ClockPoint


class FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def runtime(**feedback):
    from canatune.service import build_runtime

    config = load_config()
    config["routing"]["policy"] = "cantune"
    config["telemetry"]["enabled"] = False
    config["feedback"] = {"enabled": True, **feedback}
    rt = build_runtime(config, {}, lambda: httpx.AsyncClient())
    asyncio.run(rt.controller.start())  # production at MAX, the Canary exploring
    for g in rt.groups:  # what Runtime.start does without a locator
        g.state = type(g.state).ACTIVE
        g.effective = rt.tiers.max_point
    return rt


def test_runtime_uses_the_heuristic_without_a_canary() -> None:
    rt = runtime()
    assert isinstance(rt.router, FeedbackRouter)
    assert isinstance(rt.feedback, FeedbackClockController)
    assert rt.canary.locator is None and rt.tiers.table is None


def test_watermark_routing_fills_groups_in_fixed_order_with_the_canary_last() -> None:
    rt = runtime(p_watermark=2)
    tickets = [rt.router.try_admit(128, True) for _ in range(9)]
    assert [t.group.name for t in tickets] == [
        "G1", "G1", "G2", "G2", "G3", "G3", "G0", "G0", "G0"]  # fmt: skip
    rt.router.prefill_done(tickets[0])  # G1 has room again
    assert rt.router.try_admit(128, True).group.name == "G1"


def test_ladder_snaps_to_supported_clocks() -> None:
    supported = list(range(210, 2521, 15))
    out = ladder(supported, 600, 2520, 4)
    assert out[0] == 2520 and out[-1] == 600 and len(out) == 5
    assert ladder([1500], 300, 1500, 4) == [1500]


def test_clocks_step_down_when_quiet_and_return_to_max_on_violations() -> None:
    rt = runtime(quiet_windows=2, min_samples=5, theta=0.1)
    clock = FakeClock()
    fb = FeedbackClockController(rt.groups, rt.router, rt.controller, rt.feedback.settings,
                                 clock=clock)
    fb.ladders = {"prefill": [2520, 2000, 1500], "decode": [2040, 1500, 1000]}
    for _ in range(4):  # quiet windows, no requests at all
        clock.t += 10
        fb.decide(clock.t)
    assert fb.point() == ClockPoint(1500, 1000)
    # 2 of 10 requests miss the TTFT SLO: prefill back to MAX, decode stays
    clock.t += 10
    rt.router.finished_outcomes.extend(
        [(clock.t, i < 2, False) for i in range(10)])  # fmt: skip
    assert fb.decide(clock.t) == ["prefill_up"]
    assert fb.point() == ClockPoint(2520, 1000)
    # too few samples never raise a clock
    clock.t += 20
    rt.router.finished_outcomes.extend([(clock.t, True, True)] * 3)
    assert "decode_up" not in fb.decide(clock.t)


def test_apply_locks_every_group_and_bumps_the_clock_epochs() -> None:
    rt = runtime()
    fb = rt.feedback
    fb.ladders = {"prefill": [2520, 1500], "decode": [1500, 900]}
    fb.level = {"prefill": 1, "decode": 1}
    before = dict(rt.router.clock_epochs)
    asyncio.run(fb.apply("test"))
    assert all(g.effective == ClockPoint(1500, 900) for g in rt.groups)
    assert all(rt.router.clock_epochs[g.name] == before[g.name] + 2 for g in rt.groups)
