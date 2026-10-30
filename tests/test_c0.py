"""C0 robustness in the tier locator: borderline or saturated ramp decisions are
measured again and pooled, the search load backs off when the ceiling clock misses
the SLO, the ramp bisects to a narrow bracket, and a retry draws other traces."""

import asyncio
import dataclasses
import sys
from pathlib import Path

import pytest

from canatune.controller.locator import LocatorError, LocatorRun, TierLocator, WindowResult
from canatune.domain.groups import ClockPoint

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_locator import settings  # noqa: E402

TOP = ClockPoint(2520, 2040)


class Scripted:
    """open_window from a script: (load, rep) -> (violations, requests, KV peak,
    preemptions); unscripted loads pass cleanly below `wall` and fail above it."""

    def __init__(self, script=None, wall=3000.0):
        self.script = script or {}
        self.wall = wall
        self.calls = []

    async def open_window(self, clock, eq_tps, alpha, seconds, abort_above, rep=0):
        self.calls.append((round(eq_tps), rep))
        k, n, kv, pre = self.script.get(
            (round(eq_tps), rep), (0, 60, 0.5, 0) if eq_tps <= self.wall else (30, 60, 1.0, 5)
        )
        return WindowResult(clock=clock, kind="open", load=eq_tps, duration_s=30.0,
                            requests=n, violations=k, decode_kv_max=kv,
                            decode_preemptions=pre)  # fmt: skip


def locator(backend) -> TierLocator:
    loc = TierLocator(backend, settings())
    loc.run = LocatorRun(started_at=0.0)
    return loc


def test_a_borderline_pass_is_measured_again_and_pooled() -> None:
    # job 275914 r3: 2/60 at the edge (bound 0.093), then the same load breaks down
    backend = Scripted({(3905, 0): (2, 60, 0.98, 0), (3905, 1): (19, 60, 1.0, 0)})
    ok, pooled = asyncio.run(locator(backend).decided(TOP, 3905.0, 500.0))
    assert not ok and (pooled.violations, pooled.requests) == (21, 120)
    assert backend.calls == [(3905, 0), (3905, 1)]


def test_a_clear_pass_needs_one_window() -> None:
    backend = Scripted({(2000, 0): (0, 60, 0.6, 0)})
    ok, _ = asyncio.run(locator(backend).decided(TOP, 2000.0, 500.0))
    assert ok and backend.calls == [(2000, 0)]


def test_a_pass_that_saturated_d_must_hold_twice() -> None:
    backend = Scripted({(2400, 0): (0, 60, 0.99, 1), (2400, 1): (1, 60, 0.9, 0)})
    ok, pooled = asyncio.run(locator(backend).decided(TOP, 2400.0, 500.0))
    assert ok and pooled.requests == 120 and len(backend.calls) == 2


def test_pooling_stops_at_the_window_cap() -> None:
    script = {(2600, r): (3, 60, 0.7, 0) for r in range(5)}  # always borderline
    backend = Scripted(script)
    ok, pooled = asyncio.run(locator(backend).decided(TOP, 2600.0, 500.0))
    assert pooled.requests == 60 * settings().ramp_max_windows
    assert ok  # 9/180: bound 0.075 <= 0.10


def test_the_search_load_backs_off_instead_of_failing() -> None:
    backend = Scripted(wall=2700.0)  # C0 3905 was lucky: the ceiling holds only ~2700
    loc = locator(backend)
    c0 = asyncio.run(loc.hold_target(TOP, 3905.0, 500.0))
    assert c0 == pytest.approx(3905.0 * 0.85)  # 0.8 x 3319 = 2655 holds
    assert loc.run.evidence["c0_backoffs"] == 1


def test_the_search_load_gives_up_only_after_the_back_offs() -> None:
    loc = locator(Scripted(wall=100.0))
    with pytest.raises(LocatorError, match="no feasible prefill clock"):
        asyncio.run(loc.hold_target(TOP, 3905.0, 500.0))


def test_the_ramp_bisects_to_a_narrow_bracket() -> None:
    backend = Scripted(wall=3000.0)
    loc = locator(backend)
    s = dataclasses.replace(loc.s, ramp_start_rps=1.0)
    loc.s = s

    class Hw:
        prefill_clocks = (1080, 2520)

    c0, _ = asyncio.run(loc.ramp(Hw(), TOP, 500.0, 500.0))
    # start 1000, doubled to 2000 (pass) and 4000 (fail); bisected until high/low <= 1.1
    assert 2700 <= c0 <= 3000


def test_a_retry_draws_other_traces() -> None:
    backend = Scripted()
    loc = locator(backend)
    asyncio.run(loc.open(TOP, 2000.0, 500.0))
    loc.attempt = 1  # as after a LocatorError in locate()
    asyncio.run(loc.open(TOP, 2000.0, 500.0))
    assert backend.calls == [(2000, 0), (2000, 100)]  # not the cached window again
