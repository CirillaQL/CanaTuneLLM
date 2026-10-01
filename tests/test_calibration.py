"""Deployment priors and the Canary's admission calibration (no offline fits)."""

import asyncio
import json
import random

import pytest

from canatune.config import load_config
from canatune.controller.router import CanaTuneRouter, RouterSettings
from canatune.domain.admission import CostBook, SlackRisk, TtftPredictor
from canatune.domain.calibration import (
    ProbeSample,
    calibrate_admission,
    fit_plateau_slope,
    kv_gate_fraction,
)
from canatune.domain.groups import ClockPoint, GroupState, Tier, TierState, TierTable
from canatune.domain.priors import cluster_priors, gpu_spec, model_spec
from canatune.domain.risk import RiskTable
from canatune.service import build_groups, identity

MISTRAL = {
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "intermediate_size": 14336,
    "vocab_size": 32000,
    "max_position_embeddings": 32768,
}
KV = 131072


def test_model_and_gpu_specs() -> None:
    m = model_spec(MISTRAL)
    assert m.parameters == pytest.approx(7.24e9, rel=0.01)
    assert m.kv_bytes_per_token == KV
    assert gpu_spec("NVIDIA L40S").name == "L40S"
    assert gpu_spec("NVIDIA L4").name == "L4"
    assert gpu_spec("A100-SXM4-80GB").memory_gb == 80
    assert gpu_spec("Mystery GPU") is None
    assert gpu_spec("Mystery GPU", {"memory_gb": 16, "bandwidth_gbs": 500, "bf16_tflops": 90})


def test_cluster_priors_from_model_dir_and_topology(tmp_path, monkeypatch) -> None:
    (tmp_path / "config.json").write_text(json.dumps(MISTRAL))
    monkeypatch.setenv("CANATUNE_MODEL_PATH", str(tmp_path))
    priors = cluster_priors(load_config())
    assert priors is not None
    # right scale against r6b (plateau 30-45 ms, 68-113 us/token, D 55-60 ms, KV ~26-33k)
    assert 25 < priors.prefill_plateau_ms < 50
    assert 50 < priors.prefill_ms_per_token * 1e3 < 130
    assert 45 < priors.decode_iteration_ms < 90
    assert 20_000 < priors.decode_kv_tokens < 40_000
    assert priors.predictor_coef()[3] == pytest.approx(800.0)  # 1 GB over 10 Gb/s
    # slower clock: the slope grows, the plateau does not
    assert priors.prefill_ms(16, 0.5) == priors.prefill_ms(16)
    assert priors.prefill_ms(2048, 0.5) > 1.5 * priors.prefill_ms(2048)
    monkeypatch.setenv("CANATUNE_MODEL_PATH", str(tmp_path / "missing"))
    assert cluster_priors(load_config()) is None


def test_plateau_slope_fit() -> None:
    table = {n: max(30.0, 5 + 0.07 * n) for n in (16, 64, 256, 512, 1024, 2048)}
    fit = fit_plateau_slope(table)
    assert fit["plateau_ms"] == pytest.approx(30, abs=1)
    assert fit["slope_us_per_token"] == pytest.approx(70, rel=0.05)
    assert 300 <= fit["knee_tokens"] <= 400


def synthetic_samples(n=1500, seed=1, buffer=1e9):
    """A pair whose TTFT follows a known state model and that collapses once more
    than 40 % of the receive buffer is in flight."""
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


def test_kv_gate_finds_the_overflow_knee_and_keeps_the_prior_without_evidence() -> None:
    samples, _ = synthetic_samples()
    gate, source, bins = kv_gate_fraction(
        samples, kv_bytes_per_token=KV, buffer_bytes=1e9, theta=0.1, prior=0.5
    )
    assert gate == pytest.approx(0.4, abs=0.1) and source == "measured" and bins
    calm = [s for s in samples if s.inflight_tokens * KV < 0.2e9]
    gate, source, _ = kv_gate_fraction(
        calm, kv_bytes_per_token=KV, buffer_bytes=1e9, theta=0.1, prior=0.5
    )
    assert gate == 0.5 and source == "prior"  # never reached: the prior stays


def test_calibration_recovers_the_state_model_and_seeds_slack() -> None:
    samples, cost = synthetic_samples()
    calm = [s for s in samples if s.inflight_tokens * KV <= 0.4e9]  # below the cliff
    table = {n: cost(n) for n in (16, 128, 512, 1024, 2048)}
    result = calibrate_admission(
        calm,
        CostBook({1545: table}),
        prior=(100.0, 1.0, 1.0, 800.0, 1.0),
        ttft_slo_ms=1000,
        theta=0.1,
        kv_bytes_per_token=KV,
        buffer_bytes=1e9,
        gate_prior=0.5,
    )
    intercept, own, pending, inflight, decoding = result["predictor_coef"]
    assert pending == pytest.approx(0.8, abs=0.1)
    assert inflight == pytest.approx(400, rel=0.25)
    assert decoding == pytest.approx(2, abs=1)
    assert result["residual_ms_rms"] < 40
    slack = SlackRisk()
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
    assert router.kv_fraction == 0.5
    assert router.backfill_slack_ms() == pytest.approx(400.0)  # 0.4 x SLO until observed
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


def test_predictor_set_prior_and_fit() -> None:
    p = TtftPredictor((100.0, 1.0, 1.0, 800.0, 1.0), refit_every=10)
    rows = [(p.features(50, pending, 0, 0), 200 + 0.5 * pending) for pending in range(0, 500, 5)]
    coef = p.fit(rows)
    assert coef[2] == pytest.approx(0.5, abs=0.15) and p.coef[2] == 1.0  # fit leaves p alone
    p.set_prior(coef)
    assert p.coef == coef
