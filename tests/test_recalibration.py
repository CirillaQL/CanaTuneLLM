"""Canary re-calibration: a length shift relocates the P side when no solver plan exists,
only once the shift has settled and with the new mix only, and the cold-start
calibration does not use up the experiment budget."""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from canatune.controller.canary import CanaryScheduler, SchedulerSettings
from canatune.domain.groups import GroupState
from canatune.domain.load import LengthStats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_router_controller import published, setup  # noqa: E402

COLD = [(128, 256), (512, 256), (1024, 256), (2048, 256)]


def scheduler(solver: bool = False, **settings) -> CanaryScheduler:
    settings.setdefault("length_window", 200)  # blocks of 200 keep the tests short
    table = published(model=solver)
    _, _, _, _, _, controller, _ = setup(table=table, solver=solver)
    lengths = LengthStats(COLD, min_samples=50)
    s = CanaryScheduler(controller, None, lengths, SchedulerSettings(**settings), theta=0.1)
    s._reference = lengths.summary()  # the cold start calibrated with the default lengths
    return s


def shift(s: CanaryScheduler, prompt: int, output: int, n: int = 400) -> None:
    for _ in range(n):
        s.lengths.record(prompt, output)


def test_a_length_shift_relocates_without_the_solver() -> None:
    s = scheduler(periodic_s=1e9)
    assert s.trigger() is None
    shift(s, 200, 57)  # BurstGPT's short requests against the default probe mix
    assert s.trigger() == ("relocate", "length_shift")


def test_with_the_solver_a_length_shift_verifies_its_plan() -> None:
    s = scheduler(solver=True, periodic_s=1e9)
    shift(s, 2450, 200)
    assert s.trigger() == ("verify", "length_shift")


def test_no_trigger_before_enough_real_requests() -> None:
    s = scheduler(periodic_s=1e9)
    shift(s, 200, 57, n=49)  # still the default lengths (min_samples 50)
    assert s.lengths.using_default and s.trigger() is None


def test_no_trigger_while_the_length_mix_is_still_changing() -> None:
    # job-like long-then-short: the reference is the long mix, short requests arrive
    s = scheduler(periodic_s=1e9)
    shift(s, 2450, 200)
    s._reference = s.lengths.summary()
    shift(s, 200, 57, n=199)  # latest 200: 1 long + 199 short, the 200 before: long
    assert s.lengths.summary().shifted(s._reference, 0.3)  # the old whole-window rule fired here
    assert s.trigger() is None
    shift(s, 200, 57, n=201)  # 400 short: settled
    assert s.trigger() == ("relocate", "length_shift")


def test_a_gradual_drift_still_triggers() -> None:
    s = scheduler(periodic_s=1e9)
    s._reference = s.lengths.summary()  # default lengths, prompt mean 928
    for prompt in (900, 800, 700, 600, 500, 400):  # each block within 30 % of the last
        shift(s, prompt, 256, n=200)
    assert s.trigger() == ("relocate", "length_shift")


def test_the_relocation_measures_only_the_new_mix() -> None:
    s = scheduler(periodic_s=1e9)
    shift(s, 2450, 200, n=1200)
    s._reference = s.lengths.summary()
    shift(s, 200, 57)
    assert s.trigger() == ("relocate", "length_shift")
    s._pending = ("relocate", "length_shift")
    s.canary.state = GroupState.DRAINING
    s.canary.n_inflight = 0
    s.locator = SimpleNamespace()  # the windows are not this test's subject
    s._experiment = _noop
    asyncio.run(s.step())
    assert s.lengths.pairs() == [(200, 57)] * 400


async def _noop(kind: str, reason: str) -> None:
    return None


def test_a_recheck_does_not_absorb_a_length_shift() -> None:
    s = scheduler(periodic_s=1e9)
    reference = s._reference
    table = s.controller.tiers.table
    s.controller.publish = _noop_publish
    asyncio.run(s._adopt(table, None, "periodic"))  # a recheck's result
    assert s._reference is reference
    shift(s, 2450, 200)
    assert s.trigger() == ("relocate", "length_shift")


async def _noop_publish(table, reason) -> None:
    return None


def test_the_cold_start_calibration_is_outside_the_duty_budget() -> None:
    s = scheduler()
    clock = s._clock
    s._started_at = clock() - 3300.0  # a 55-minute cold start
    asyncio.run(s._finish("full", "cold_start", "published", None))
    assert s.explore_s == 0.0
    s._started_at = clock() - 1200.0
    asyncio.run(s._finish("relocate", "length_shift", "published", None))
    assert s.explore_s == pytest.approx(1200.0, abs=1.0)
