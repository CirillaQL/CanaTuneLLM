"""Redirect before full effort (router.group_choice fill, controller.redirect): confirmed
production pressure first cools the groups whose own latency shows it and sends new
requests to the groups with room; MAX only once no group has room."""

import asyncio
import dataclasses

from test_router_controller import activate, published, setup

from canatune.domain.calibration import KneeCounts
from canatune.domain.groups import Tier


def fill_setup(**router):
    clock, _, groups, _, r, controller, _ = setup(
        table=published(), admission="slack", overload="best_effort",
        router={"group_choice": "fill", **router},
    )
    controller.settings = dataclasses.replace(controller.settings, solver=False)
    activate(groups)  # static: every group serves, nothing parked
    return clock, groups, r, controller


def serve(router, group, ttft, count=5, tokens=128):
    """`count` finished requests admitted to `group` with this TTFT."""
    for _ in range(count):
        ticket = router.try_admit(tokens, True, best_effort=True)
        assert ticket is not None
        if ticket.group is not group:  # place it where the test wants it
            ticket.group.n_at_p -= 1
            ticket.group.n_await -= 1
            ticket.group.n_inflight -= 1
            ticket.group.t_await -= tokens
            ticket.group = group
            group.n_at_p += 1
            group.n_await += 1
            group.n_inflight += 1
            group.t_await += tokens
        router.finish(ticket, status="ok", ttft_ms=ttft, tpot_ms=50, output_tokens=64)


def confirm(clock, controller):
    asyncio.run(controller.tick())
    clock.now += controller.settings.confirm_s
    asyncio.run(controller.tick())


def test_feedback_names_the_group_whose_own_latency_is_near_the_slo() -> None:
    clock, groups, router, controller = fill_setup()
    serve(router, groups[1], ttft=950)
    serve(router, groups[2], ttft=200)
    s = controller.settings
    f = router.production_feedback(s.feedback_window_s, s.feedback_min_samples,
                                   s.feedback_near_slo)
    assert f["pressure"] and f["pressured_groups"] == ["G1"]
    assert f["groups"]["G2"]["near_fraction"] == 0.0


def test_a_hot_group_is_cooled_and_new_requests_go_on_without_full_effort() -> None:
    clock, groups, router, controller = fill_setup()
    serve(router, groups[1], ttft=950)
    confirm(clock, controller)
    assert controller.mode == "expanding" and not router.full_effort
    assert router.is_cooled("G1") and all(g.tier is Tier.H for g in groups)
    assert router.try_admit(128, True).group.name == "G2"  # G1 has no room meanwhile
    clock.now += controller.settings.feedback_window_s + 1
    assert router.try_admit(128, True).group.name == "G1"  # cooled only for a while


def test_requests_after_the_redirect_still_late_cool_the_next_group() -> None:
    clock, groups, router, controller = fill_setup()
    serve(router, groups[1], ttft=950)
    confirm(clock, controller)
    clock.now += controller.settings.expansion_grace_s
    serve(router, groups[2], ttft=950)  # the new routing: G2 near the SLO too
    confirm(clock, controller)
    assert controller.mode == "expanding" and router.is_cooled("G2")
    assert router.try_admit(128, True).group.name == "G3"
    clock.now += controller.settings.expansion_grace_s
    serve(router, groups[3], ttft=950)
    serve(router, groups[0], ttft=950)
    confirm(clock, controller)  # no group with room is left
    assert controller.mode == "full_effort" and router.full_effort
    assert all(g.tier is Tier.MAX for g in groups)


def test_pressure_in_every_group_leaves_no_room_and_goes_to_full_effort() -> None:
    clock, groups, router, controller = fill_setup()
    for g in groups:  # one near request in each group
        serve(router, g, ttft=950, count=1)
        serve(router, g, ttft=200, count=3)
    confirm(clock, controller)
    assert controller.mode == "full_effort"


def test_one_near_request_points_at_its_group_even_with_few_samples() -> None:
    clock, groups, router, controller = fill_setup()
    serve(router, groups[0], ttft=200, count=5)
    serve(router, groups[2], ttft=200, count=1)
    serve(router, groups[2], ttft=950, count=1)  # 1 of 7 overall: past theta
    confirm(clock, controller)
    assert controller.mode == "expanding" and router.is_cooled("G2")
    assert not router.is_cooled("G1") and not router.full_effort


def test_concentrate_keeps_going_straight_to_full_effort() -> None:
    clock, _, groups, _, router, controller, _ = setup(
        table=published(), admission="slack", overload="best_effort",
        router={"group_choice": "concentrate"},
    )
    controller.settings = dataclasses.replace(controller.settings, solver=False)
    activate(groups)
    serve(router, groups[1], ttft=950)
    confirm(clock, controller)
    assert controller.mode == "full_effort"


def test_a_prompt_slow_even_when_idle_counts_only_once_it_is_late() -> None:
    clock, _, groups, _, router, controller, _ = setup(
        table=published(), admission="slack", overload="best_effort",
        router={"group_choice": "fill"},
    )
    slo = router.settings.ttft_slo_ms
    # idle (top clock) TTFT of a 4096-token prompt at 0.93 x SLO, near it before any load
    router.tiers.table.evidence["idle_ttft_ms"] = {"128": 0.2 * slo, "4096": 0.93 * slo}
    activate(groups)
    s = controller.settings
    near = lambda: router.production_feedback(  # noqa: E731
        s.feedback_window_s, s.feedback_min_samples, s.feedback_near_slo)["near_fraction"]
    serve(router, groups[1], ttft=0.97 * slo, tokens=4096)  # near the SLO, as when idle
    assert near() == 0.0
    serve(router, groups[1], ttft=1.1 * slo, tokens=4096)  # late: it counts
    assert near() == 0.5
    serve(router, groups[2], ttft=0.97 * slo, tokens=128, count=10)  # a short one: load
    assert near() > 0.5


def test_a_merged_sparse_block_puts_the_knee_at_its_first_bad_bin() -> None:
    # Job 275524's Canary: 10-20 % of the buffer 1 of 16 violated, 20-30 % 7 of 18; the
    # merged block passes, but the knee is 20 %, not 10 %.
    bins = {"0": [1474, 66], "1": [16, 1], "2": [18, 7], "3": [3, 3], "5": [9, 7]}
    assert KneeCounts(bins).knee(0.1, 20) == 2
