"""router.group_choice fill: groups in a fixed order up to the P slot cap the Canary
measured, spread once every P is at it; the congestion knees (KV gate, P slot cap)
from the Canary's counts and updated online from production."""

import dataclasses

import pytest
from test_router_controller import H, activate, make_safe, published, setup

from canatune.config import load_config
from canatune.controller.router import RouterSettings
from canatune.domain.admission import CostBook
from canatune.domain.calibration import (
    KneeCounts,
    ProbeSample,
    calibrate_admission,
    p_slot_cap,
)


def counts(cells: dict[int, tuple[int, int]]) -> KneeCounts:
    return KneeCounts({str(b): list(c) for b, c in cells.items()})


def test_knee_merges_a_sparse_tail_instead_of_missing_it() -> None:
    # Job 275385 r3: every sample above 30 % of the buffer violated, but no bin there
    # held 20 samples on its own.
    knee = counts({0: (200, 4), 1: (150, 3), 2: (80, 2), **{b: (4, 4) for b in range(3, 10)}})
    assert knee.knee(0.1, 20) == 3


def test_knee_needs_a_jump_over_the_bins_below() -> None:
    # Violations as frequent at every level (long prompts, slow clocks) are no knee.
    flat = counts({b: (100, 30) for b in range(6)})
    assert flat.knee(0.1, 20) is None
    # The lowest bin alone is no knee either, however bad.
    assert counts({0: (100, 60), 1: (100, 5)}).knee(0.1, 20) is None
    # Deep overload far above does not pull the knee down into a calm bin.
    deep = counts({0: (100, 2), 1: (100, 3), 2: (100, 4), 3: (100, 60), 9: (500, 500)})
    assert deep.knee(0.1, 20) == 3


def sample(at_p: int, violated: bool, mhz: int = 1800) -> ProbeSample:
    return ProbeSample(mhz, 1050, 256, (256,) * at_p, 0, 0, 1500.0 if violated else 300.0,
                       violated)


def test_p_slot_cap_is_the_count_at_p_from_which_requests_violate() -> None:
    calm = [sample(n, i % 25 == 0) for n in range(4) for i in range(50)]
    blocked = [sample(n, i % 2 == 0) for n in range(4, 8) for i in range(30)]
    cap, source, bins = p_slot_cap(calm + blocked, theta=0.1)
    assert (cap, source) == (4, "measured") and bins["4"] == [30, 15]
    assert p_slot_cap(calm, theta=0.1)[:2] == (None, "none")


def test_calibration_takes_the_knees_from_the_clocks_production_runs_at() -> None:
    at_h = [sample(n, n >= 4 and i % 2 == 0) for n in range(8) for i in range(40)]
    slow = [sample(n, n >= 1, mhz=900) for n in range(8) for i in range(40)]  # slow prefill
    costs = CostBook({900: {"256": 120.0}, 1800: {"256": 60.0}})
    args = dict(prior=None, ttft_slo_ms=1000, theta=0.1, kv_bytes_per_token=131072,
                buffer_bytes=None)
    everything = calibrate_admission(at_h + slow, costs, **args)
    at_or_above_h = calibrate_admission(at_h + slow, costs, prefill_mhz=1800, **args)
    assert everything["p_slot_cap"] == 1  # the slow clock's violations
    assert at_or_above_h["p_slot_cap"] == 4 and at_or_above_h["knee_samples"] == len(at_h)


def fill_router(cap: int | None, **router):
    table = published()
    table.evidence["admission"]["p_slot_cap"] = cap
    _, risk, groups, _, r, _, _ = setup(table=table, router={"group_choice": "fill", **router})
    activate(groups)
    make_safe(risk, H)
    return groups, r


def test_fill_takes_groups_in_order_up_to_the_cap_and_returns_when_one_drains() -> None:
    groups, router = fill_router(cap=2)
    tickets = [router.try_admit(128, True) for _ in range(5)]
    assert [t.group.name for t in tickets] == ["G1", "G1", "G2", "G2", "G3"]
    router.prefill_done(tickets[0])  # G1's P returned one: G1 is first again
    assert router.try_admit(128, True).group.name == "G1"


def test_fill_spreads_by_the_fewest_at_p_once_every_p_is_at_its_cap() -> None:
    groups, router = fill_router(cap=1)
    names = [router.try_admit(128, True).group.name for _ in range(4)]
    assert names == ["G1", "G2", "G3", "G0"]  # the Canary last
    for g, n in zip(groups, (2, 1, 2, 2)):
        g.n_at_p = n
    assert router.try_admit(128, True).group.name == "G1"


def test_fill_without_a_measured_cap_keeps_filling_the_first_group() -> None:
    _, router = fill_router(cap=None)
    assert {router.try_admit(128, True).group.name for _ in range(3)} == {"G1"}


def test_fill_moves_on_once_a_d_nears_the_tpot_slo() -> None:
    groups, router = fill_router(cap=None)
    near = router.settings.near_slo * router.settings.tpot_slo_ms
    bounds = {"G1": near + 1.0, "G2": near - 1.0}  # D1 past the near-SLO share
    candidates = router.candidates

    def with_bounds(*args, **kwargs):
        out = candidates(*args, **kwargs)
        router._tpot_by_group.update(bounds)
        return out

    router.candidates = with_bounds
    assert router.try_admit(128, True).group.name == "G2"
    bounds["G2"] = near + 2.0  # G3 next in order (no bound: its D is not watched)
    assert router.try_admit(128, True).group.name == "G3"


def test_fill_full_effort_ignores_the_kv_gate_and_draws_two_groups() -> None:
    groups, router = fill_router(cap=2, fill_p2c_groups=2)
    router._apply_table()
    router.baseline_mode = True
    for g in groups:
        g.inflight_bytes = 0.9e9  # above any gate
    for g in (groups[0], groups[2], groups[3]):
        g.n_at_p = 50
    picks = [router._admit_baseline(128, True, 0.0).group.name for _ in range(30)]
    # Of every two drawn the less loaded P: G1 whenever it is drawn (half of the pairs).
    assert 10 <= picks.count("G1") <= 25 and set(picks) <= {"G0", "G1", "G2", "G3"}


def test_round_robin_baseline_keeps_the_kv_gate() -> None:
    _, risk, groups, _, router, _, _ = setup(
        table=published(), router={"group_choice": "concentrate"},
        connector="P2pNcclConnector",
    )
    activate(groups)
    router._apply_table()
    router.baseline_mode = True
    for g in groups:
        g.inflight_bytes = 0.9e9  # > 0.5 x 1 GB buffer
    assert router._admit_baseline(128, True, 0.0) is None


def test_production_violations_at_p_lower_the_slot_cap_online() -> None:
    groups, router = fill_router(cap=None)
    router._apply_table()
    router._slot_knee = KneeCounts({"0": [200, 4]})
    router.table.record = lambda *args: None  # keep admission open; the knee is the point
    seen = []
    for i in range(60):
        groups[1].n_at_p = 0 if i % 2 else 3  # admitted with three already at P
        t = router.try_admit(128, True)
        bad = t.snapshot["n_at_p"] >= 3
        router.first_token(t)
        router.finish(t, status="ok", ttft_ms=1500 if bad else 200, tpot_ms=50,
                      output_tokens=8)
        seen.append(router.p_slot_cap)
    assert seen[0] is None and router.p_slot_cap == 3


def test_baseline_admissions_do_not_move_the_knees() -> None:
    groups, router = fill_router(cap=2)
    router._apply_table()
    router.baseline_mode = True
    before = router._slot_knee
    for _ in range(40):
        t = router._admit_baseline(128, True, 0.0)
        router.first_token(t)
        router.finish(t, status="ok", ttft_ms=5000, tpot_ms=50, output_tokens=8)
    assert router.p_slot_cap == 2 and before is router._slot_knee


def test_fill_is_a_valid_group_choice() -> None:
    config = load_config()
    config["router"]["group_choice"] = "fill"
    settings = RouterSettings.from_config(config)
    assert settings.group_choice == "fill" and settings.knee_update
    assert dataclasses.replace(settings, fill_p2c_groups=8).fill_p2c_groups == 8
    config["router"]["fill_p2c_groups"] = 0
    with pytest.raises(ValueError):
        RouterSettings.from_config(config)
