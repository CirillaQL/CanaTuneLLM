"""Canary scheduler: when to run the tier locator, when to stop it.

The Canary is an ordinary production group whenever it is not experimenting.

Start (all of):
* a trigger: cold start (no table: start immediately, production runs at MAX
  and admits everything meanwhile), a shift of the production length
  distribution (median or p90 by more than `length_shift`), production
  violations drifting above theta, or the periodic neighbour check of H;
* the rest of the pool can carry production: no rejection for `quiet_s`, and
  the load of the active groups without the Canary <= `others_load_max * C_H`;
* budget: `min_interval_s` since the last experiment and at most `max_duty` of
  the time spent experimenting.
The Canary then drains (no new requests; in-flight ones finish) and explores.

Stop: the locator finishes (the new table is published), or the Controller
reports pressure it cannot relieve by waking a parked group (abort: the
Canary locks back to H and serves). Window results are cached in the locator,
so an aborted run resumes where it stopped next time.
"""

import asyncio
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from canatune.controller.locator import LocatorError, TierLocator, wilson_ucb
from canatune.controller.tier_controller import TierController
from canatune.domain.groups import Group, GroupState, Tier
from canatune.domain.load import LengthStats, LengthSummary


@dataclass(frozen=True)
class SchedulerSettings:
    period_s: float = 1.0
    quiet_s: float = 60.0
    others_load_max: float = 0.7
    min_interval_s: float = 600.0
    max_duty: float = 0.10
    periodic_s: float = 1800.0
    length_shift: float = 0.30
    drift_min_samples: int = 200
    drain_timeout_s: float = 30.0

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "SchedulerSettings":
        raw = {k: v for k, v in config.get("canary", {}).items() if k in cls.__dataclass_fields__}
        return cls(**raw)


class CanaryScheduler:
    def __init__(
        self,
        controller: TierController,
        locator: TierLocator | None,
        lengths: LengthStats,
        settings: SchedulerSettings,
        *,
        theta: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.controller = controller
        self.locator = locator
        self.lengths = lengths
        self.s = settings
        self.theta = theta
        self._clock = clock
        self._t0 = clock()
        self.task: asyncio.Task | None = None
        self.kind: str | None = None
        self.reason: str | None = None
        self.last_end: float | None = None
        self.explore_s = 0.0
        self._started_at: float | None = None
        self._rejections_seen = controller.router.rejections
        self._last_rejection_at = float("-inf")
        self._reference: LengthSummary | None = None
        self._draining_since: float | None = None
        self._pending: tuple[str, str] | None = None
        self.history: list[dict[str, Any]] = []
        controller.on_pressure = self.abort

    @property
    def canary(self) -> Group | None:
        return next((g for g in self.controller.groups if g.canary), None)

    @property
    def running(self) -> bool:
        return self.task is not None and not self.task.done()

    # ---- conditions -----------------------------------------------------------------

    def _note_rejections(self) -> None:
        rejections = self.controller.router.rejections
        if rejections != self._rejections_seen:
            self._rejections_seen = rejections
            self._last_rejection_at = self._clock()

    def trigger(self) -> tuple[str, str] | None:
        """-> (kind, reason): kind is "full", "relocate" or "recheck"."""
        table = self.controller.tiers.table
        if table is None:
            return "full", "cold_start"
        if self._reference is not None and not self.lengths.using_default:
            if self.lengths.summary().shifted(self._reference, self.s.length_shift):
                return "relocate", "length_shift"
        outcomes = self.controller.router.outcomes
        if len(outcomes) >= self.s.drift_min_samples:
            k = sum(outcomes)
            # Lower confidence bound above theta: production really violates more.
            if (
                k / len(outcomes) > self.theta
                and 1 - wilson_ucb(len(outcomes) - k, len(outcomes), 1.28) > self.theta
            ):
                return "relocate", "violation_drift"
        last = self.last_end if self.last_end is not None else self._t0
        if self._clock() - last >= self.s.periodic_s:
            return "recheck", "periodic"
        return None

    def may_start(self) -> tuple[bool, str]:
        table = self.controller.tiers.table
        now = self._clock()
        if table is None:
            # Cold start: go at once; after a failed attempt retry every quiet_s.
            if self.last_end is not None and now - self.last_end < self.s.quiet_s:
                return False, "cold_start_retry_wait"
            return True, "cold_start"
        if self.last_end is not None and now - self.last_end < self.s.min_interval_s:
            return False, "min_interval"
        elapsed = max(now - self._t0, 1.0)
        if self.explore_s / elapsed > self.s.max_duty:
            return False, "duty_budget"
        if now - self._last_rejection_at < self.s.quiet_s:
            return False, "recent_rejections"
        canary = self.canary
        if canary is None:
            return False, "no_canary"
        loads = self.controller.loads(now)
        others = [
            g for g in self.controller.groups if g is not canary and g.state is GroupState.ACTIVE
        ]
        total = sum(loads[g.name] for g in self.controller.groups if g.state is GroupState.ACTIVE)
        if not others:
            return False, "no_other_active_group"
        if total > self.s.others_load_max * table.capacity_h * len(others):
            return False, "others_cannot_carry"
        return True, "ok"

    # ---- control --------------------------------------------------------------------------

    def abort(self, reason: str) -> None:
        """Called by the Controller under pressure it cannot relieve otherwise."""
        if self.controller.tiers.table is None:
            return  # cold start: production is at MAX and admits everything
        if self.running:
            assert self.task is not None
            self.task.cancel()
            self.controller.log.write({"event": "canary_abort", "reason": reason})
        if self._pending is not None:
            self._pending = None
            canary = self.canary
            if canary is not None and canary.state is GroupState.DRAINING:
                canary.state = GroupState.ACTIVE

    def request(self, kind: str, reason: str) -> tuple[bool, str]:
        """Manual start through the control API (start conditions still apply)."""
        if self.running or self._pending is not None:
            return False, "already_running"
        ok, why = self.may_start()
        if ok:
            self._begin(kind, reason)
        return ok, why

    def _begin(self, kind: str, reason: str) -> None:
        canary = self.canary
        if canary is None or self.locator is None:
            return
        self._pending = (kind, reason)
        if canary.state is GroupState.ACTIVE:
            canary.state = GroupState.DRAINING  # no new production requests
            self._draining_since = self._clock()
        self.controller.log.write(
            {"event": "canary_claim", "kind": kind, "reason": reason, "state": canary.state.value}
        )

    async def step(self) -> None:
        self._note_rejections()
        canary = self.canary
        if canary is None or self.locator is None or self.running:
            return
        if self._pending is None:
            trig = self.trigger()
            if trig is None:
                return
            ok, _ = self.may_start()
            if not ok:
                return
            self._begin(*trig)
        # Wait until the Canary has no production request in flight.
        if canary.state is GroupState.DRAINING and canary.n_inflight:
            if self._clock() - (self._draining_since or self._clock()) > self.s.drain_timeout_s:
                canary.state = GroupState.ACTIVE  # could not drain: give up this time
                self._pending = None
            return
        assert self._pending is not None
        kind, reason = self._pending
        self._pending = None
        canary.state = GroupState.EXPLORING
        self.kind, self.reason = kind, reason
        self._started_at = self._clock()
        self.task = asyncio.create_task(self._experiment(kind, reason))

    def _prompts(self) -> list[int]:
        pairs = self.lengths.pairs()
        prompts = sorted(p for p, _ in pairs)
        picks = {prompts[int(q * (len(prompts) - 1))] for q in (0.1, 0.3, 0.5, 0.7, 0.9)}
        if len(picks) < 2:
            only = next(iter(picks))
            picks = {max(1, only // 2), only, only * 2}
        return sorted(picks)

    async def _experiment(self, kind: str, reason: str) -> None:
        assert self.locator is not None
        controller = self.controller
        table = controller.tiers.table
        outcome = "failed"
        try:
            summary = self.lengths.summary()
            if kind == "recheck" and table is not None:
                new = await self.locator.recheck(table, self._prompts())
            else:
                if kind == "relocate" or reason == "length_shift":
                    self.locator.forget()  # old windows used another workload
                new = await self.locator.locate(
                    summary.prompt_mean,
                    self._prompts(),
                    previous=table if kind == "relocate" else None,
                )
            self._reference = summary
            controller.router.outcomes.clear()
            await controller.publish(new, reason)
            outcome = "published"
        except asyncio.CancelledError:
            outcome = "aborted"
        except LocatorError as error:
            controller.log.write({"event": "locator_error", "error": str(error)})
        except Exception as error:  # probes hit real services: never kill the control loop
            controller.log.write({"event": "locator_error", "error": repr(error)})
        finally:
            now = self._clock()
            self.explore_s += now - (self._started_at or now)
            self.last_end = now
            self.history.append({"kind": kind, "reason": reason, "outcome": outcome, "end": now})
            canary = self.canary
            if canary is not None:
                canary.state = GroupState.ACTIVE
                if controller.tiers.table is not None:
                    await controller.set_tier(canary, Tier.H, f"canary_{outcome}")
                else:
                    await controller.set_tier(canary, Tier.MAX, f"canary_{outcome}")

    async def run(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            try:
                await self.step()
            except Exception as error:
                self.controller.log.write({"event": "canary_error", "error": repr(error)})
            try:
                await asyncio.wait_for(stop.wait(), timeout=self.s.period_s)
            except TimeoutError:
                pass
        if self.running:
            assert self.task is not None
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)

    def state(self) -> dict[str, Any]:
        run = None if self.locator is None else self.locator.run
        return {
            "running": self.running,
            "kind": self.kind,
            "reason": self.reason,
            "pending": self._pending,
            "phase": None if run is None else run.phase,
            "windows": None if run is None else run.windows,
            "reused_windows": None if run is None else run.reused,
            "explore_s": self.explore_s,
            "history": self.history[-20:],
            "lengths_default": self.lengths.using_default,
        }
