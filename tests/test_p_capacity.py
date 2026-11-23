"""Distribution lookup must distinguish shape, extrapolate, and survive persistence."""

import pytest

from canatune.domain.groups import ClockPoint, TierStore, TierTable
from canatune.domain.p_capacity import nearest_p_rho, p_profiles


def bank():
    return {name: {"prompt_lengths": list(lengths), "p_rho": rho}
            for (name, lengths), rho in zip(p_profiles().items(), (1.5, .9, .85, .7, .65))}


@pytest.mark.parametrize("name", list(p_profiles()))
def test_exact_profile_selects_its_own_measured_capacity(name):
    profiles = bank()
    assert nearest_p_rho(profiles[name]["prompt_lengths"], profiles) == (
        name, profiles[name]["p_rho"],
    )


def test_unseen_bimodal_uses_nearest_even_outside_the_old_distance_gate():
    # Nearest TV is .4, beyond the old .25 gate; two nearest profiles tie.
    assert nearest_p_rho([128, 2048], bank()) == ("long_heavy", .7)


def test_equal_means_do_not_collapse_different_distributions():
    bimodal = [128] * 8 + [2048] * 7
    assert sum(bimodal) / len(bimodal) == 1024
    assert nearest_p_rho([1024] * 15, bank()) == ("mixed", .85)
    assert nearest_p_rho(bimodal, bank()) == ("short_heavy", .9)


def test_arbitrary_lengths_map_to_nearest_measured_anchors():
    assert nearest_p_rho([1900, 2300], bank()) == ("long", .65)
    assert nearest_p_rho([100, 200], bank()) == ("short", 1.5)
    assert nearest_p_rho([320], bank()) == ("short", 1.5)  # midpoint -> shorter


def test_unknown_capacities_are_never_fabricated():
    profiles = bank()
    profiles["short"]["p_rho"] = None
    assert nearest_p_rho([128], profiles) == ("short_heavy", .9)
    assert nearest_p_rho([], profiles) is None
    assert nearest_p_rho([128], {}) is None


def test_profiles_respect_the_measured_prompt_limit():
    assert len(p_profiles(512)) == 5
    assert all(max(lengths) <= 512 for lengths in p_profiles(512).values())


def test_bank_survives_tier_store_roundtrip(tmp_path):
    store = TierStore(tmp_path / "tiers.json", {"cluster": "test"})
    table = TierTable(ClockPoint(500, 500), ClockPoint(1500, 1000), 1000, 100,
                      evidence={"room": {"p_rho": .8, "p_profiles": bank()}})
    store.save(table)
    restored = store.load()
    assert restored.evidence == table.evidence
    assert nearest_p_rho([128, 2048], restored.evidence["room"]["p_profiles"]) == (
        "long_heavy", .7,
    )
