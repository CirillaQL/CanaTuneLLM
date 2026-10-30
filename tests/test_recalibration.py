"""Canary re-calibration: a length shift relocates the P side when no solver plan exists,
and the cold-start calibration does not use up the experiment budget."""

import asyncio
import sys
from pathlib import Path

import pytest

from canatune.controller.canary import CanaryScheduler, SchedulerSettings
from canatune.domain.load import LengthStats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_router_controller import published, setup  # noqa: E402

COLD = [(128, 256), (512, 256), (1024, 256), (2048, 256)]


def scheduler(solver: bool = False, **settings) -> CanaryScheduler:
    table = published(model=solver)
    _, _, _, _, _, controller, _ = setup(table=table, solver=solver)
    lengths = LengthStats(COLD, min_samples=50)
    s = CanaryScheduler(controller, None, lengths, SchedulerSettings(**settings), theta=0.1)
    s._reference = lengths.summary()  # the cold start calibrated with the default lengths
    return s


def shift(s: CanaryScheduler, prompt: int, output: int, n: int = 200) -> None:
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


def test_the_cold_start_calibration_is_outside_the_duty_budget() -> None:
    s = scheduler()
    clock = s._clock
    s._started_at = clock() - 3300.0  # a 55-minute cold start
    asyncio.run(s._finish("full", "cold_start", "published", None))
    assert s.explore_s == 0.0
    s._started_at = clock() - 1200.0
    asyncio.run(s._finish("relocate", "length_shift", "published", None))
    assert s.explore_s == pytest.approx(1200.0, abs=1.0)
