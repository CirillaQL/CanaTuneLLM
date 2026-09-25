"""Online risk table: buckets, seed, monotone bounds, persistence."""

import pytest

from canatune.config import load_config
from canatune.domain.groups import ClockPoint
from canatune.domain.risk import Buckets, RiskTable, RiskTableError, is_violation
from canatune.service import identity

H = ClockPoint(1815, 1050)
L = ClockPoint(1305, 1050)


def table(**kwargs) -> RiskTable:
    return RiskTable(Buckets((0, 1, 2, 3, 5), (128, 512, 1024, 2048)), min_samples=3, **kwargs)


def test_bucket_uses_upper_edge() -> None:
    b = Buckets((0, 1, 2, 3, 5), (128, 512, 1024, 2048))
    assert [b.n_await_bucket(n) for n in (0, 1, 3, 4, 5, 6)] == [0, 1, 3, 4, 4, 5]
    assert [b.prompt_bucket(n) for n in (1, 128, 129, 2048, 3000)] == [0, 0, 1, 3, 4]


def test_config_table_starts_empty_and_unknown_is_unsafe() -> None:
    config = load_config()
    t = RiskTable.from_config(config["risk"], identity(config))
    estimate = t.lookup(H, 0, 128, True)
    assert (estimate.source, estimate.risk) == ("unknown", 1.0)


def test_seed_is_keyed_by_clock_point() -> None:
    config = load_config()
    raw = dict(config["risk"])
    raw["seed"] = {"1815/1050": [[0.05] * 5 for _ in range(6)]}
    t = RiskTable.from_config(raw, identity(config))
    assert t.lookup(H, 0, 128, True).risk == pytest.approx(0.05)
    assert t.lookup(L, 0, 128, True).source == "unknown"


def test_sparse_cell_uses_heavier_bound_and_lighter_floor() -> None:
    t = table()
    heavy = t.cell(L, 3, 1024, False)
    for violated in (False, False, True):  # 1/3 observed, enough samples
        t.record(heavy, violated)
    light = t.cell(L, 0, 128, False)
    for _ in range(3):
        t.record(light, False)
    middle = t.estimate(t.cell(L, 1, 512, False))
    assert middle.source == "heavier_bound"
    assert middle.risk == pytest.approx(1 / 3)
    # A cell heavier than every sample has no upper bound: unsafe.
    assert t.estimate(t.cell(L, 5, 2048, False)).risk == 1.0
    # Other clock points are independent.
    assert t.estimate(t.cell(H, 1, 512, False)).source == "unknown"


def test_sparse_observations_can_raise_but_not_lower_a_seed() -> None:
    t = table()
    t.load_seed(H, [[0.2] * 5 for _ in range(6)])
    cell = t.cell(H, 0, 128, True)
    t.record(cell, False)
    assert t.estimate(cell).risk == pytest.approx(0.2)
    t.record(cell, True)
    assert t.estimate(cell).risk == pytest.approx(0.5)
    t.record(cell, False)  # now 3 samples: observed 1/3 replaces the seed
    assert t.estimate(cell).source == "observed"


def test_clock_usable_needs_a_safe_well_sampled_cell() -> None:
    t = table()
    assert not t.clock_usable(L, 0.1)
    cell = t.cell(L, 0, 128, False)
    for _ in range(3):
        t.record(cell, False)
    assert t.clock_usable(L, 0.1)


def test_persistence_roundtrip_and_identity_guard(tmp_path) -> None:
    path = tmp_path / "risk.json"
    t = table(identity={"model": "m"}, path=path)
    cell = t.cell(L, 1, 512, True)
    t.record(cell, True)
    t.lookup(L, 1, 512, True)
    t.save()
    loaded = table(identity={"model": "m"}, path=path)
    loaded.load()
    assert loaded.to_json()["cells"][cell.key()]["admitted"] == 1
    assert cell.key() == "1305/1050|1|1|1"
    assert loaded.exploration_candidates(0.1)[0]["cell"] == cell.key()
    with pytest.raises(RiskTableError):
        table(identity={"model": "other"}, path=path).load()


def test_violation_rule() -> None:
    kw = {"ttft_slo_ms": 500, "tpot_slo_ms": 200}
    assert is_violation(499, 199, **kw) is False
    assert is_violation(500, 10, **kw) is True
    assert is_violation(10, 201, **kw) is True
    assert is_violation(None, 10, **kw) is None
