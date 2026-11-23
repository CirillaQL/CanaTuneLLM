"""router.admission predict (ablation): admit on the point TTFT prediction alone, i.e.
while wait + predicted TTFT <= the TTFT SLO, without the slack risk, the KV in-flight
gate, the D length model or the measured occupancy limits (P slot cap)."""

import pytest
from test_router_controller import activate, published, setup

from canatune.config import load_config
from canatune.controller.router import RouterConfigError, RouterSettings


def predict_router(predicted: dict[str, float], admission: str = "predict", cap: int | None = 1):
    table = published()
    table.evidence["admission"]["p_slot_cap"] = cap
    # every slack bucket violated: the slack risk would admit nothing
    table.evidence["admission"]["slack_counts"] = {str(b): [50, 50] for b in range(14)}
    table.decode_length = {"1050": [1.0]}  # a length model is published
    _, _, groups, _, router, _, _ = setup(
        table=table, admission=admission, overload="best_effort",
        router={"group_choice": "fill"},
    )
    activate(groups)
    predict = router._predict

    def fixed(group, s_own, dg):
        x, _ = predict(group, s_own, dg)
        return x, predicted[group.name]

    router._predict = fixed
    return groups, router


def test_predict_is_a_valid_admission_and_others_are_rejected() -> None:
    config = load_config()
    config["router"]["admission"] = "predict"
    assert RouterSettings.from_config(config).admission == "predict"
    config["router"]["admission"] = "guess"
    with pytest.raises(RouterConfigError):
        RouterSettings.from_config(config)


def test_predict_admits_in_fill_order_while_the_prediction_meets_the_slo() -> None:
    predicted = {"G0": 100.0, "G1": 100.0, "G2": 100.0, "G3": 100.0}
    _, router = predict_router(predicted)
    # the slack risk says 100 %, the P slot cap is 1: neither applies, G1 keeps filling
    tickets = [router.try_admit(128, True) for _ in range(3)]
    assert [t.group.name for t in tickets] == ["G1", "G1", "G1"]
    assert all(t.estimate.source == "predict" and t.overflow is None for t in tickets)
    assert router.p_slot_cap is None
    # G1 predicted past the SLO: the next group in order takes it
    predicted["G1"] = router.settings.ttft_slo_ms + 1.0
    assert router.try_admit(128, True).group.name == "G2"


def test_predict_counts_the_wait_already_spent() -> None:
    _, router = predict_router({"G0": 300.0, "G1": 300.0, "G2": 300.0, "G3": 300.0})
    slo = router.settings.ttft_slo_ms
    assert router.try_admit(128, True, waited_ms=slo - 300.0) is not None  # exactly at the SLO
    assert router.try_admit(128, True, waited_ms=slo - 299.0) is None  # no group fits: waits


def test_predict_overflows_best_effort_when_no_prediction_fits() -> None:
    _, router = predict_router({g: 1e6 for g in ("G0", "G1", "G2", "G3")})
    ticket = router.try_admit(128, True, best_effort=True)
    assert ticket is not None and ticket.overflow == "best_effort"
    assert ticket.estimate.source == "overflow_best_effort" and ticket.estimate.risk == 1.0


def test_predict_ignores_the_kv_gate_and_the_d_length_model() -> None:
    groups, router = predict_router({"G0": 100.0, "G1": 100.0, "G2": 100.0, "G3": 100.0})
    for g in groups:
        g.inflight_bytes = 0.99e9  # above the calibrated gate (0.5 of the buffer)

    def no_length_model(*args, **kwargs):
        raise AssertionError("the D length model must not be consulted")

    router._decode_risk = no_length_model
    assert router.try_admit(128, True).group.name == "G1"


def test_slack_admission_still_consults_the_d_length_model() -> None:
    # control for the test above: the same setup in slack mode does use it
    _, router = predict_router({"G0": 100.0, "G1": 100.0, "G2": 100.0, "G3": 100.0},
                               admission="slack")
    calls = []

    def length_model(group, tokens, table):
        calls.append(group.name)
        return None

    router._decode_risk = length_model
    router.try_admit(128, True)
    assert calls
