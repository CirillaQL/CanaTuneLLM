"""The Canary's calibration (admission parameters, cluster model) and the solver.
Everything is fitted from probe samples and windows; no constants of a cluster."""

import asyncio
import random

import pytest

from canatune.config import load_config
from canatune.controller.router import CanaTuneRouter, RouterSettings
from canatune.domain.admission import CostBook, SlackRisk, TtftPredictor, slack_edges
from canatune.domain.calibration import (
    ProbeSample,
    calibrate_admission,
    fit_plateau_slope,
    kv_gate_fraction,
)
from canatune.domain.groups import ClockPoint, GroupState, Tier, TierState, TierTable
from canatune.domain.models import ClusterModel, Solver, fit_decode, fit_power
from canatune.domain.risk import RiskTable
from canatune.service import build_groups, identity

KV = 131072


def test_plateau_slope_fit() -> None:
    table = {n: max(30.0, 5 + 0.07 * n) for n in (16, 64, 256, 512, 1024, 2048)}
    fit = fit_plateau_slope(table)
    assert fit["plateau_ms"] == pytest.approx(30, abs=1)
    assert fit["slope_us_per_token"] == pytest.approx(70, rel=0.05)
    assert 300 <= fit["knee_tokens"] <= 400


def synthetic_samples(n=1500, seed=1, buffer=1e9):
    """A pair whose TTFT follows a state model and that collapses once more than
    40 % of the receive buffer is in flight."""
    rng = random.Random(seed)
    cost = lambda L: max(30.0, 5 + 0.07 * L)  # noqa: E731
    out = []
    for _ in range(n):
        L = rng.choice((128, 512, 1024, 2048))
        at_p = tuple(rng.choice((128, 512, 1024)) for _ in range(rng.randint(0, 6)))
        inflight = rng.randint(0, 6000)
        decoding = rng.randint(0, 30)
        ttft = 150 + cost(L) + 0.8 * sum(map(cost, at_p)) + 400 * inflight * KV / 1e9
        ttft += 2 * decoding + rng.gauss(0, 25)
        if inflight * KV / buffer > 0.4:
            ttft += 1500  # buffer overflow: P stalls
        out.append(ProbeSample(1545, 1170, L, at_p, inflight, decoding, ttft, ttft > 1000))
    return out, cost


def test_kv_gate_finds_the_overflow_knee_or_falls_back_to_the_buffer() -> None:
    samples, _ = synthetic_samples()
    gate, source, bins = kv_gate_fraction(
        samples, kv_bytes_per_token=KV, buffer_bytes=1e9, theta=0.1
    )
    assert gate == pytest.approx(0.4, abs=0.1) and source == "measured" and bins
    calm = [s for s in samples if s.inflight_tokens * KV < 0.2e9]
    gate, source, _ = kv_gate_fraction(calm, kv_bytes_per_token=KV, buffer_bytes=1e9, theta=0.1)
    assert gate == 1.0 and source == "buffer"  # never reached: the connector's own limit


def test_calibration_fits_the_state_model_without_a_prior_and_seeds_slack() -> None:
    samples, cost = synthetic_samples()
    calm = [s for s in samples if s.inflight_tokens * KV <= 0.4e9]  # below the cliff
    table = {n: cost(n) for n in (16, 128, 512, 1024, 2048)}
    result = calibrate_admission(
        calm,
        CostBook({1545: table}),
        prior=None,
        ttft_slo_ms=1000,
        theta=0.1,
        kv_bytes_per_token=KV,
        buffer_bytes=1e9,
    )
    intercept, own, pending, inflight, decoding = result["predictor_coef"]
    assert pending == pytest.approx(0.8, abs=0.1)
    assert inflight == pytest.approx(400, rel=0.25)
    assert decoding == pytest.approx(2, abs=1)
    assert result["residual_ms_rms"] < 40 and result["predictor_prior"] is None
    slack = SlackRisk(slack_edges(1000))
    slack.set_seed(result["slack_counts"])
    assert slack.estimate(600).source == "observed" and slack.estimate(600).risk < 0.05
    assert slack.safe_slack(0.05) is not None


def test_probe_records_the_state_each_request_was_sent_into() -> None:
    from test_probe import FakeAgent, make_probe

    probe, _ = make_probe([], FakeAgent())
    probe._at_prefill = {99: 512}
    probe._inflight_tokens = 300
    probe._decoding = 2

    async def run():
        async with probe.client_factory() as client:
            return await probe.request(client, 128, 3)

    outcome = asyncio.run(run())
    assert outcome.at_prefill == (512,) and outcome.inflight_tokens == 300
    assert outcome.decoding_at_send == 2
    assert probe._at_prefill == {99: 512} and probe._inflight_tokens == 300
    assert probe._decoding == 2  # its own stages were undone


def test_router_applies_the_published_calibration_and_per_clock_costs() -> None:
    config = load_config()
    config["router"]["admission"] = "slack"
    groups = build_groups(config)
    for g in groups:
        g.state, g.tier, g.effective = GroupState.ACTIVE, Tier.H, ClockPoint(1545, 1170)
    tiers = TierState(max_point=ClockPoint(2520, 1500))
    risk = RiskTable.from_config(config["risk"], identity(config))
    router = CanaTuneRouter(groups, risk, RouterSettings.from_config(config), tiers)
    assert router.kv_fraction == 1.0  # the whole buffer until the Canary measured a gate
    assert router.prefill_cost(1545)(2048) == 0.0  # no table yet: admission is open
    assert router.backfill_slack_ms() == router.slack.edges[-1]  # nothing observed yet
    counts = {"8": [200, 60], "10": [200, 0], "11": [200, 0], "12": [200, 0], "13": [200, 0]}
    tiers.table = TierTable(
        park=ClockPoint(600, 300),
        h=ClockPoint(1545, 1170),
        capacity_h=5000,
        alpha_tokens=500,
        published_at=1.0,
        evidence={
            "prefill_ms_by_clock": {
                "2520": {"16": 30, "2048": 150},
                "1545": {"16": 30, "2048": 250},
            },
            "admission": {
                "predictor_coef": [90.0, 1.0, 0.7, 500.0, 3.0],
                "slack_counts": counts,
                "kv_gate_fraction": 0.3,
            },
        },
    )
    assert router.prefill_cost(1545)(2048) == pytest.approx(250)
    assert router.prefill_cost(2520)(2048) == pytest.approx(150)
    assert router.prefill_cost(1800)(2048) == pytest.approx(250)  # nearest at or below
    assert router.predictor.prior == [90.0, 1.0, 0.7, 500.0, 3.0]
    assert router.kv_fraction == 0.3
    # lowest bucket from which all observed ones are safe: 10 = [500, 600) ms
    assert router.backfill_slack_ms() == 500.0
    t = router.try_admit(2048, True)
    assert t is not None and t.s_own_ms == pytest.approx(250)
    groups[2].inflight_bytes = 0.35e9  # above the measured 30 % gate
    assert not router._kv_blocked(groups[1]) and router._kv_blocked(groups[2])


def test_predictor_fit_with_and_without_a_prior() -> None:
    rows = [(TtftPredictor.features(50, p, 0, 0), 200 + 0.5 * p) for p in range(0, 500, 5)]
    plain = TtftPredictor().fit(rows)
    assert plain[2] == pytest.approx(0.5, abs=0.01)
    anchored = TtftPredictor((100.0, 1.0, 1.0, 800.0, 1.0), refit_every=10)
    coef = anchored.fit(rows)
    assert coef[2] == pytest.approx(0.5, abs=0.15) and anchored.coef[2] == 1.0
    anchored.set_prior(coef)
    assert anchored.coef == coef


# ---- cluster model and solver --------------------------------------------------------


def test_decode_and_power_fits() -> None:
    fits = fit_decode({1170: [(8, 66.0), (16, 77.0)], 735: [(12, 90.0)]})
    assert fits[1170] == pytest.approx([55.0, 1.375])
    assert fits[735][1] == pytest.approx(1.375)  # one load level: the shared slope
    power = fit_power({1000: 40.0, 2000: 60.0}, [(1500, 150.0, 0.5), (1500, 250.0, 1.0)])
    assert power[1500][0] == pytest.approx(50.0)  # idle interpolated
    assert power[1500][1] == pytest.approx(200.0)  # (150 - 50) / 0.5 and (250 - 50) / 1


def model(**kwargs) -> ClusterModel:
    values = dict(
        prefill={
            1080: {16: 30.0, 256: 40.0, 2048: 240.0},
            1545: {16: 30.0, 256: 36.0, 2048: 175.0},
            2520: {16: 30.0, 256: 33.0, 2048: 150.0},
        },
        residence={128: 120.0, 2048: 260.0},
        decode={735: [60.0, 2.2], 1170: [57.0, 1.4], 2040: [55.0, 1.4]},
        power_prefill={1080: [40.0, 150.0], 1545: [45.0, 210.0], 2520: [55.0, 330.0]},
        power_decode={735: [18.0, 25.0], 1170: [20.0, 35.0], 2040: [24.0, 50.0]},
        park_power_w=55.0,
        rho_prefill=0.5,
        ttft_slo_ms=1000.0,
        tpot_slo_ms=200.0,
        kv_bytes_per_token=KV,
        kv_gate_bytes=0.5e9,
        kv_capacity_tokens=30000,
    )
    values.update(kwargs)
    return ClusterModel(**values)


def test_solver_concentrates_at_low_load_and_spreads_at_high_load() -> None:
    solver = Solver(model(), groups=3)
    mix = [(128, 64), (512, 64), (1024, 64)]
    low = solver.solve(0.5, mix)
    assert low.feasible and low.n == 1 and low.point == ClockPoint(1080, 735)
    high = solver.solve(12.0, mix)
    assert high.feasible and high.n >= 2
    assert solver.solve(500.0, mix).feasible is False  # beyond every configuration


def test_solver_raises_p_for_long_prompts_and_d_for_long_outputs() -> None:
    solver = Solver(model(), groups=1)
    short = solver.solve(3.0, [(128, 64)])
    long_prompts = solver.solve(3.0, [(2048, 64)])
    assert long_prompts.point.prefill_mhz > short.point.prefill_mhz
    long_outputs = solver.solve(1.2, [(128, 400)])
    assert long_outputs.point.decode_mhz > short.point.decode_mhz or long_outputs.n > 1
    assert long_prompts.binding in ("prefill", "kv", "ttft", "decode")


def test_verified_failure_caps_a_point() -> None:
    m = model()
    solver = Solver(m, groups=1)
    mix = [(512, 64)]
    first = solver.solve(2.0, mix)
    m.caps[first.point.key()] = 2.0  # the Canary saw that point fail at 2 req/s
    second = Solver(ClusterModel.from_json(m.to_json()), groups=1).solve(2.0, mix)
    assert second.point != first.point and second.feasible
