"""Feedback heuristic: a comparison system without the Canary.

`feedback.enabled: true` replaces the Canary's calibration with two online rules that
use only production feedback and fixed operator settings:

* Routing (FeedbackRouter): the groups in the same fixed order as CanaTune's fill
  (production first, the former Canary last); each request goes to the first group
  below both watermarks, at most `p_watermark` requests outstanding at its P and its
  D's KV cache usage (vLLM /metrics) below `d_kv_watermark`; with no such group, to the
  group with the fewest requests at P, then the lightest D.
* Clocks (FeedbackClockController): every `window_s`, the requests that finished in
  the window are scored against the TTFT and the TPOT SLO separately. A phase whose
  violation share exceeds theta goes back to its maximum clock at once; a phase whose
  share stayed at or below theta / 2 for `quiet_windows` windows in a row moves one
  step down its clock ladder. The ladder spans the configured clock floor
  (clock_control.min_mhz) to the maximum the agents report, in `steps` steps. All
  groups share the clocks of each phase.

No tier table is ever published: the Router stays in open admission (every routable
group is a candidate) and the tier Controller's own control loop does not run.
"""

import asyncio
import math
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from canatune.controller.router import CanaTuneRouter, Route, Ticket
from canatune.controller.tier_controller import TierController
from canatune.domain.groups import ClockPoint, Group, GroupState
from canatune.infrastructure.records import JsonlLog


class FeedbackConfigError(ValueError):
    """Feedback heuristic settings are invalid."""


@dataclass(frozen=True)
class FeedbackSettings:
    enabled: bool = False
    p_watermark: int = 4
    d_kv_watermark: float = 0.9
    window_s: float = 10.0
    min_samples: int = 5
    quiet_windows: int = 3
    steps: int = 8
    theta: float = 0.10
    prefill_min_mhz: int = 600
    decode_min_mhz: int = 300
    telemetry_max_age_s: float = 2.0

    def __post_init__(self) -> None:
        if (self.p_watermark < 1 or not 0 < self.d_kv_watermark <= 1
                or not (math.isfinite(self.window_s) and self.window_s > 0)
                or self.min_samples < 1 or self.quiet_windows < 1 or self.steps < 1
                or not 0 < self.theta < 1 or self.prefill_min_mhz < 1 or self.decode_min_mhz < 1
                or not self.telemetry_max_age_s > 0):
            raise FeedbackConfigError("invalid feedback settings")

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "FeedbackSettings":
        raw = config.get("feedback") or {}
        controller = config.get("controller") or {}
        floors = (config.get("clock_control") or {}).get("min_mhz") or {}
        return cls(
            enabled=bool(raw.get("enabled", False)),
            p_watermark=int(raw.get("p_watermark", 4)),
            d_kv_watermark=float(raw.get("d_kv_watermark", 0.9)),
            # Defaults: the production feedback window and sample floor CanaTune uses.
            window_s=float(raw.get("window_s", controller.get("feedback_window_s", 10))),
            min_samples=int(raw.get("min_samples", controller.get("feedback_min_samples", 5))),
            quiet_windows=int(raw.get("quiet_windows", 3)),
            steps=int(raw.get("steps", 8)),
            theta=float(raw.get("theta", (config.get("router") or {}).get("theta", 0.10))),
            prefill_min_mhz=int(raw.get("prefill_min_mhz", floors.get("prefill", 600))),
            decode_min_mhz=int(raw.get("decode_min_mhz", floors.get("decode", 300))),
            telemetry_max_age_s=float(raw.get("telemetry_max_age_s", 2.0)),
        )


class FeedbackRouter(CanaTuneRouter):
    """CanaTune's Router in open admission with watermark routing (or round robin, the
    HALO comparison); it also keeps the latency and SLO outcome of every finished
    request for the clock controller."""

    def __init__(self, *args: Any, feedback: FeedbackSettings, routing: str = "watermark",
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if routing not in ("watermark", "round_robin"):
            raise FeedbackConfigError("routing must be watermark or round_robin")
        self.feedback = feedback
        self.routing = routing
        self._next = 0  # round robin: the next group index in turn
        # (finish time, TTFT violated, TPOT violated, TTFT ms, TPOT ms) of finished requests
        self.finished_outcomes: deque[tuple[float, bool, bool, float | None, float | None]] = (
            deque())

    def kv_usage(self, group: Group) -> float | None:
        if self.telemetry is None:
            return None
        snapshot = self.telemetry.fresh(group.decode, self.feedback.telemetry_max_age_s)
        return None if snapshot is None else snapshot.kv_usage

    def below_watermarks(self, route: Route) -> bool:
        if route.p.n_at_p >= self.feedback.p_watermark:
            return False
        usage = self.kv_usage(route.d)
        # No fresh reading: count only the requests already sent to that D.
        return usage is None or usage < self.feedback.d_kv_watermark

    def _choose(self, feasible):  # type: ignore[no-untyped-def]
        index = self.groups.index
        if self.routing == "round_robin":  # the pairs in turn, as the baseline routes
            n = len(self.groups)
            item = min(feasible, key=lambda item: (index(item[0].p) - self._next) % n)
            self._next = (index(item[0].p) + 1) % n
            return item
        order = sorted(feasible, key=lambda item: (item[0].p.canary, index(item[0].p)))
        for item in order:
            if self.below_watermarks(item[0]):
                return item
        return min(order, key=lambda item: (
            item[0].p.n_at_p, item[0].d.n_await + item[0].d.n_decoding,
            item[0].p.pending_ms, index(item[0].p)))  # fmt: skip

    def finish(self, ticket: Ticket, *, status: str, ttft_ms: float | None,
               tpot_ms: float | None, output_tokens: int,
               timing: Mapping[str, float | None] | None = None) -> None:
        if not ticket.finished:
            s = self.settings
            if status != "ok":
                outcome = (True, True)  # a failed request counts as a violation of both
            else:
                outcome = (ttft_ms is None or ttft_ms > s.ttft_slo_ms,
                           tpot_ms is not None and tpot_ms > s.tpot_slo_ms)
            ok = status == "ok"
            self.finished_outcomes.append((self._clock(), *outcome,
                                           ttft_ms if ok else None, tpot_ms if ok else None))
        super().finish(ticket, status=status, ttft_ms=ttft_ms, tpot_ms=tpot_ms,
                       output_tokens=output_tokens, timing=timing)


def ladder(supported: Sequence[int], floor: int, top: int, steps: int) -> list[int]:
    """Clocks from `top` down to `floor` in `steps` even steps, each snapped to the
    highest supported clock at or below its target (duplicates dropped)."""
    usable = sorted({f for f in supported if floor <= f <= top}, reverse=True)
    if not usable:
        return [top]
    out: list[int] = []
    for i in range(steps + 1):
        target = top - (top - floor) * i / steps
        pick = next((f for f in usable if f <= target + 1e-9), usable[-1])
        if pick not in out:
            out.append(pick)
    return out


async def lock_all(groups: Sequence[Group], router: CanaTuneRouter, controller: TierController,
                   target: ClockPoint) -> bool:
    """Lock every active group's P and D to `target`; the Router assumes the slower of the
    old and new point meanwhile, and the clock epochs keep in-flight samples apart."""
    epochs = router.clock_epochs
    active = [g for g in groups if g.state is GroupState.ACTIVE]
    for group in active:
        epochs[group.name] = epochs.get(group.name, 0) + 1
        if group.effective is not None:
            group.effective = ClockPoint(min(group.effective.prefill_mhz, target.prefill_mhz),
                                         min(group.effective.decode_mhz, target.decode_mhz))
    results = await asyncio.gather(*(
        controller._lock(endpoint, mhz) for group in active
        for endpoint, mhz in ((group.prefill, target.prefill_mhz),
                              (group.decode, target.decode_mhz))))
    for group in active:
        group.effective = target
        epochs[group.name] += 1
    return all(results)


class FeedbackClockController:
    def __init__(
        self,
        groups: Sequence[Group],
        router: FeedbackRouter,
        controller: TierController,
        settings: FeedbackSettings,
        *,
        log: JsonlLog | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.groups = list(groups)
        self.router = router
        self.controller = controller
        self.settings = settings
        self.log = log or JsonlLog(None)
        self._clock = clock
        self.ladders: dict[str, list[int]] = {"prefill": [], "decode": []}
        self.level = {"prefill": 0, "decode": 0}  # index into the ladder, 0 = MAX
        self.quiet = {"prefill": 0, "decode": 0}
        self._applied: ClockPoint | None = None

    async def setup(self) -> None:
        """Ladders from the clocks the agents report (the configured MAX alone when
        clock control is off or the agents do not answer)."""
        top = self.controller.tiers.max_point
        refs = self.controller.gpu_refs
        g = self.groups[0]
        for phase, endpoint, floor, mhz in (
                ("prefill", g.prefill, self.settings.prefill_min_mhz, top.prefill_mhz),
                ("decode", g.decode, self.settings.decode_min_mhz, top.decode_mhz)):
            supported: list[int] = []
            if endpoint in refs:
                try:
                    supported = await self.controller.actuator.supported_clocks(refs[endpoint])
                except Exception as error:
                    self.log.write({"event": "feedback_clock_error", "phase": phase,
                                    "error": repr(error)})
            self.ladders[phase] = ladder(supported or [mhz], floor, mhz, self.settings.steps)
        self.log.write({"event": "feedback_setup", "ladders": self.ladders,
                        "settings": self.settings.__dict__})
        await self.apply("start")

    def point(self) -> ClockPoint:
        return ClockPoint(self.ladders["prefill"][self.level["prefill"]],
                          self.ladders["decode"][self.level["decode"]])

    async def apply(self, reason: str) -> None:
        target = self.point()
        if target == self._applied:
            return
        ok = await lock_all(self.groups, self.router, self.controller, target)
        self._applied = target
        self.log.write({"event": "feedback_clock", "clock": target.key(), "reason": reason,
                        "ok": ok})

    def window(self, now: float) -> tuple[int, int, int]:
        """-> (requests, TTFT violations, TPOT violations) finished in the last window."""
        outcomes = self.router.finished_outcomes
        while outcomes and outcomes[0][0] < now - self.settings.window_s:
            outcomes.popleft()
        return (len(outcomes), sum(o[1] for o in outcomes), sum(o[2] for o in outcomes))

    def decide(self, now: float) -> list[str]:
        """One window's decision; -> the reasons of the clock changes it makes."""
        s = self.settings
        n, ttft_bad, tpot_bad = self.window(now)
        changes = []
        for phase, bad in (("prefill", ttft_bad), ("decode", tpot_bad)):
            share = bad / n if n else 0.0
            if n >= s.min_samples and share > s.theta:
                self.quiet[phase] = 0
                if self.level[phase] != 0:
                    self.level[phase] = 0
                    changes.append(f"{phase}_up")
            elif share <= s.theta / 2:
                self.quiet[phase] += 1
                if (self.quiet[phase] >= s.quiet_windows
                        and self.level[phase] < len(self.ladders[phase]) - 1):
                    self.level[phase] += 1
                    self.quiet[phase] = 0
                    changes.append(f"{phase}_down")
            else:
                self.quiet[phase] = 0
        self.log.write({"event": "feedback_window", "requests": n, "ttft_violations": ttft_bad,
                        "tpot_violations": tpot_bad, "clock": self.point().key(),
                        "changes": changes})
        return changes

    async def run(self, stop: asyncio.Event) -> None:
        await self.setup()
        next_window = self._clock() + self.settings.window_s
        while not stop.is_set():
            try:
                now = self._clock()
                await self.controller._log_energy(now)
                if now >= next_window:
                    next_window = now + self.settings.window_s
                    changes = self.decide(now)
                    if changes:
                        await self.apply(",".join(changes))
            except Exception as error:  # a bad tick must not stop control
                self.log.write({"event": "feedback_error", "error": repr(error)})
            try:
                await asyncio.wait_for(stop.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                pass

    def state(self) -> dict[str, Any]:
        return {"clock": self.point().key(), "level": dict(self.level),
                "ladders": self.ladders, "quiet": dict(self.quiet)}
