"""router.admission capacity (ablation): admit on the Canary's measured occupancy limits
alone (P slot cap, plus slack's D-side checks), without a TTFT prediction or slack risk."""

import pytest
from test_router_controller import activate, published, setup

from canatune.config import load_config
from canatune.controller.router import RouterSettings


def capacity_router(predicted: dict[str, float], cap: int | None = 1, admission="capacity",
                    connector: str | None = None):
    table = published()
    table.evidence["admission"]["p_slot_cap"] = cap
    # every slack bucket violated: the slack risk would admit nothing
    table.evidence["admission"]["slack_counts"] = {str(b): [50, 50] for b in range(14)}
    table.decode_length = {"1050": [1.0]}  # a length model is published
    _, _, groups, _, router, _, _ = setup(
        table=table, admission=admission, overload="best_effort",
        router={"group_choice": "fill"}, connector=connector,
    )
    activate(groups)
    predict = router._predict

    def fixed(group, s_own, dg):
        x, _ = predict(group, s_own, dg)
        return x, predicted[group.name]

    router._predict = fixed
    router._decode_risk = lambda *args, **kwargs: None  # the length model allows every D
    return groups, router


def test_capacity_is_a_valid_admission() -> None:
    config = load_config()
    config["router"]["admission"] = "capacity"
    assert RouterSettings.from_config(config).admission == "capacity"


def test_capacity_ignores_the_ttft_prediction_and_the_slack_risk() -> None:
    slo = 600.0
    groups, router = capacity_router({g: 1e6 for g in ("G0", "G1", "G2", "G3")})
    ticket = router.try_admit(128, True, waited_ms=slo)  # predicted far past the SLO
    assert router.p_slot_cap == 1  # kept, unlike capacity_limits=false or predict
    assert ticket is not None and ticket.group.name == "G1"
    assert ticket.estimate.source == "capacity" and ticket.overflow is None


def test_capacity_fills_in_order_up_to_the_slot_cap_then_overflows() -> None:
    groups, router = capacity_router({g: 100.0 for g in ("G0", "G1", "G2", "G3")})
    names = [router.try_admit(128, True).group.name for _ in range(4)]
    assert names == ["G1", "G2", "G3", "G0"]  # one per P, production before the Canary
    assert all(g.n_at_p == 1 for g in groups)
    assert router.try_admit(128, True) is None  # every P at its cap: not feasible
    ticket = router.try_admit(128, True, best_effort=True)
    assert ticket is not None and ticket.overflow == "best_effort"
    assert ticket.estimate.source == "overflow_best_effort"


def test_capacity_overflow_ranks_by_requests_at_p_not_by_prediction() -> None:
    predicted = {"G0": 100.0, "G1": 100.0, "G2": 100.0, "G3": 100.0}
    groups, router = capacity_router(predicted, cap=1)
    for _ in range(4):
        router.try_admit(128, True)
    router.try_admit(128, True, best_effort=True)  # G1 now holds 2
    by_name = {g.name: g for g in groups}
    assert by_name["G1"].n_at_p == 2
    predicted["G1"] = 1.0  # lowest prediction, but the most requests at P
    ticket = router.try_admit(128, True, best_effort=True)
    assert ticket.group.name != "G1"


def test_capacity_keeps_the_kv_in_flight_gate() -> None:
    groups, router = capacity_router({g: 100.0 for g in ("G0", "G1", "G2", "G3")},
                                     connector="P2pNcclConnector")
    assert router.settings.kv_buffer_bytes  # P2pNccl: D has a KV receive buffer
    for g in groups:
        g.inflight_bytes = 0.99e9  # above the calibrated gate (0.5 of the buffer)
    assert router.try_admit(128, True) is None
    groups[1].inflight_bytes = 0.0
    assert router.try_admit(128, True).group is groups[1]


def test_capacity_consults_the_d_length_model() -> None:
    _, router = capacity_router({g: 100.0 for g in ("G0", "G1", "G2", "G3")})
    calls = []

    def length_model(group, tokens, table):
        calls.append(group.name)
        return None

    router._decode_risk = length_model
    router.try_admit(128, True)
    assert calls


@pytest.mark.parametrize("admission", ["slack", "predict"])
def test_other_admissions_unchanged_on_overflow(admission) -> None:
    _, router = capacity_router({g: 1e6 for g in ("G0", "G1", "G2", "G3")}, admission=admission)
    ticket = router.try_admit(128, True, best_effort=True)
    assert ticket is not None and ticket.overflow == "best_effort"
