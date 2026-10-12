"""router.group_choice (fixed pairing): concentrate (default) or round_robin (ablation A1)."""

import pytest
from test_router_controller import H, activate, make_safe, published, setup

from canatune.controller.router import RouterConfigError, RouterSettings


def admitted_groups(group_choice: str, n: int) -> list[str]:
    _, risk, groups, _, router, _, _ = setup(
        table=published(), router={"group_choice": group_choice}
    )
    activate(groups)
    make_safe(risk, H)  # every group can take up to three queued requests
    return [router.try_admit(128, True).group.name for _ in range(n)]


def test_round_robin_takes_the_feasible_groups_in_turn() -> None:
    names = admitted_groups("round_robin", 5)
    assert len(set(names[:4])) == 4  # each of the four groups once
    assert names[4] == names[0]  # then the first again


def test_concentrate_keeps_filling_the_most_loaded_group() -> None:
    names = admitted_groups("concentrate", 2)
    assert names[0] == names[1] == "G1"  # production before Canary, then the busiest


def test_group_choice_is_validated() -> None:
    from canatune.config import load_config

    config = load_config()
    config["router"]["group_choice"] = "spread"
    with pytest.raises(RouterConfigError):
        RouterSettings.from_config(config)
