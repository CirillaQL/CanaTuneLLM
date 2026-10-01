"""Seconds-scale Controller: how many groups run, at which clocks.

Cold start (no tier table): every production group is active at MAX and the
Router admits everything; the Controller does nothing else.

With a table, every tick:
* pressure (the Router rejected, held or overflowed requests): wake a parked
  group; with none left ask the Canary to abort its experiment; else raise the
  most loaded working group to MAX. Boosted groups return to the working point
  after `t_down_s` without pressure.
* plan (the Canary's cluster model, `domain.models.Solver`): from the offered
  request rate (Router arrivals over `load_window_s`; its burst factor over
  `burst_window_s` is logged) and the length mix of recent requests, the
  lowest-power feasible (groups, P clock, D clock). Switching rules:
    target the configuration for the peak demand of the last `t_down_s`
          (hysteresis from the observed demand itself): it stays feasible while
          the demand moves below that peak
    up    (the current configuration cannot carry the current demand): at once
    down  (fewer groups or slower clocks): once the same target held for
          `t_down_s`, with no pressure in that time, and when it saves more than
          `switch_gain` of the current power (the Canary's measurement noise band)
  Extra groups are drained (the Canary first, so it is free to explore) and then
  parked; missing ones are woken (production first). A clock change takes
  ~0.2-0.5 s; meanwhile the Router assumes the slower of old and new clock point,
  and every change bumps the group's clock epoch so in-flight samples are not
  written to the risk table.
"""

import asyncio
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from canatune.controller.router import CanaTuneRouter
from canatune.domain.groups import Group, GroupState, Tier, TierState, TierStore, TierTable, slower
from canatune.domain.models import ClusterModel, Evaluation, Solver
from canatune.infrastructure.clocks import ClockActuator, GpuRef
from canatune.infrastructure.records import JsonlLog


class ControllerConfigError(ValueError):
    """Controller settings are invalid."""


@dataclass(frozen=True)
class ControllerSettings:
    period_s: float = 1.0
    load_window_s: float = 10.0
    t_down_s: float = 30.0
    min_active_groups: int = 1
    stagger_s: float = 2.0
    energy_log_period_s: float = 5.0
    burst_window_s: float = 120.0
    burst_bucket_s: float = 10.0
    switch_gain: float = 0.02  # = canary.locator.eps (measurement noise band)
    solver: bool = True  # False: every group stays at the Canary's H (static comparison)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "ControllerSettings":
        raw = config.get("controller")
        if not isinstance(raw, Mapping):
            raise ControllerConfigError("controller section is required")
        return cls(
            period_s=float(raw.get("period_s", 1.0)),
            load_window_s=float(raw.get("load_window_s", 10.0)),
            t_down_s=float(raw.get("t_down_s", 30.0)),
            min_active_groups=int(raw.get("min_active_groups", 1)),
            burst_window_s=float(raw.get("burst_window_s", 120.0)),
            burst_bucket_s=float(raw.get("burst_bucket_s", 10.0)),
            switch_gain=float(
                config.get("canary", {}).get("locator", {}).get("eps", raw.get("switch_gain", 0.02))
            ),
            stagger_s=float(raw.get("stagger_s", 2.0)),
            energy_log_period_s=float(raw.get("energy_log_period_s", 5.0)),
            solver=bool(raw.get("solver", True)),
        )


class TierController:
    def __init__(
        self,
        groups: Sequence[Group],
        router: CanaTuneRouter,
        settings: ControllerSettings,
        actuator: ClockActuator,
        gpu_refs: Mapping[str, GpuRef],
        tiers: TierState,
        *,
        store: TierStore | None = None,
        log: JsonlLog | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.groups = list(groups)
        self.router = router
        self.settings = settings
        self.actuator = actuator
        self.gpu_refs = dict(gpu_refs)
        self.tiers = tiers
        self.store = store
        self.log = log or JsonlLog(None)
        self._clock = clock
        self._calm_since: float | None = None
        self._target: tuple | None = None  # (n, point) the solver keeps choosing
        self._target_since: float | None = None
        self.plan: Evaluation | None = None  # the solver's last answer
        self._solver_key: object = None
        self._solver: Solver | None = None
        self._demand: deque[tuple[float, float]] = deque()  # (t, rate x burst) for down
        router.arrival_keep_s = max(settings.burst_window_s, settings.load_window_s) + 1
        self._pressure_at = float("-inf")
        self._rejections_seen = 0
        self._last_energy_log = float("-inf")
        # Groups whose last clock lock failed: name -> (next retry time, backoff s).
        self._retry: dict[str, tuple[float, float]] = {}
        # Set by the Canary scheduler: abort an experiment; -> whether one was aborted.
        self.on_pressure: Callable[[str], bool] | None = None

    # ---- actuation ---------------------------------------------------------------

    async def _lock(self, endpoint: str, mhz: int) -> bool:
        ref = self.gpu_refs.get(endpoint)
        if ref is None:
            return True  # clock control not configured for this endpoint
        try:
            result = await self.actuator.lock(ref, mhz)
        except Exception as error:
            self.log.write(
                {"event": "clock_error", "endpoint": endpoint, "mhz": mhz, "error": str(error)}
            )
            return False
        self.log.write({"event": "clock", "endpoint": endpoint, "mhz": mhz, **result})
        return bool(result.get("ok"))

    async def set_tier(self, group: Group, tier: Tier, reason: str) -> None:
        """Lock P and D to `tier`; the Router sees the conservative point meanwhile."""
        old = group.effective
        target = self.tiers.clocks(tier)
        group.tier = tier
        epochs = self.router.clock_epochs
        epochs[group.name] = epochs.get(group.name, 0) + 1
        group.effective = None if tier is Tier.PARK or old is None else slower(old, target)
        ok = all(
            await asyncio.gather(
                self._lock(group.prefill, target.prefill_mhz),
                self._lock(group.decode, target.decode_mhz),
            )
        )
        if group.tier is tier:
            if ok:
                self._retry.pop(group.name, None)
            else:
                # Retry from tick(), backing off from period_s up to t_down_s.
                _, delay = self._retry.get(group.name, (0.0, self.settings.period_s / 2))
                delay = min(2 * delay, self.settings.t_down_s)
                self._retry[group.name] = (self._clock() + delay, delay)
        if tier is not Tier.PARK and group.tier is tier:
            # On failure keep the conservative point (None after Park: not routable).
            if ok:
                group.effective = target
            elif old is not None:
                group.effective = slower(old, target)
        epochs[group.name] += 1
        self.log.write(
            {
                "event": "tier",
                "group": group.name,
                "tier": tier.value,
                "clock": target.key(),
                "ok": ok,
                "reason": reason,
            }
        )

    # ---- lifecycle --------------------------------------------------------------------

    async def start(self) -> None:
        """Production active (MAX at cold start, H with a stored table); the Canary
        starts calibrating right away when there is no table, else it serves."""
        cold = self.tiers.table is None
        for group in sorted(self.groups, key=lambda g: g.canary):  # production first
            if group.canary and cold:
                group.state = GroupState.EXPLORING  # the locator drives its clocks
                continue
            group.state = GroupState.ACTIVE
            await self.set_tier(group, Tier.MAX if cold else Tier.H, "start")

    async def publish(self, table: TierTable, reason: str) -> None:
        """Adopt a new tier table; active groups move to it one at a time so that
        not every group is slowed by a clock change at once."""
        self.tiers.table = table
        self.tiers.working = None  # the solver chooses again on the new model
        self._target = self._target_since = None
        if self.store is not None:
            self.store.save(table)
        self.log.write({"event": "publish", "reason": reason, "table": table.to_json()})
        for group in self.groups:
            if group.state is not GroupState.ACTIVE:
                continue
            tier = Tier.H
            if group.effective != self.tiers.clocks(tier):
                await self.set_tier(group, tier, f"publish:{reason}")
                await asyncio.sleep(self.settings.stagger_s)

    async def wake(self, reason: str) -> Group | None:
        parked = [g for g in self.groups if g.state is GroupState.PARK]
        if not parked:
            return None
        group = sorted(parked, key=lambda g: g.canary)[0]  # production first
        group.state = GroupState.ACTIVE
        await self.set_tier(group, Tier.H, reason)
        return group

    def loads(self, now: float) -> dict[str, float]:
        alpha = self.tiers.alpha_tokens
        return {g.name: g.load(now, self.settings.load_window_s, alpha) for g in self.groups}

    # ---- one control step -------------------------------------------------------------

    def solver(self) -> Solver | None:
        """Solver over the published cluster model (rebuilt when a table is published
        or the Canary's verification capped a point)."""
        raw = self.tiers.model_json() if self.settings.solver else None
        key = None if raw is None else (self.tiers.table.published_at, str(raw.get("caps")))
        if key != self._solver_key:
            self._solver_key = key
            self._solver = (
                None if raw is None else Solver(ClusterModel.from_json(raw), len(self.groups))
            )
        return self._solver

    def demand(self, now: float) -> tuple[float, float, list[tuple[int, int]]]:
        """-> (offered requests/s, burst factor, recent (prompt, output) lengths)."""
        s = self.settings
        rate, _ = self.router.offered(now, s.load_window_s, s.burst_bucket_s)
        _, burst = self.router.offered(now, s.burst_window_s, s.burst_bucket_s)
        lengths = self.router.lengths.pairs() if self.router.lengths is not None else []
        return rate, burst, lengths

    def can_carry(self, n: int, now: float | None = None) -> bool:
        """Would n groups at the working point carry the current demand?"""
        solver = self.solver()
        if solver is None or n < 1:
            return False
        now = self._clock() if now is None else now
        rate, _, lengths = self.demand(now)
        if rate <= 0:
            return True  # no demand to carry
        if not lengths:
            return False  # demand without length data: cannot tell
        peak = max([rate] + [d for _, d in self._demand])
        point = self.tiers.clocks(Tier.H)
        return solver.evaluate(peak, lengths, n, point).feasible

    # ---- one control step -------------------------------------------------------------

    async def tick(self) -> None:
        now = self._clock()
        await self._log_energy(now)
        await self._retry_locks(now)  # also on a cold start: a group locked to MAX
        table = self.tiers.table
        if table is None:
            return  # cold start: everything at MAX, nothing to decide
        new_pressure = self.router.pressure - self._rejections_seen
        self._rejections_seen = self.router.pressure
        active = [g for g in self.groups if g.state is GroupState.ACTIVE]

        # Pressure: wake a parked group; with none left, the Canary must come back;
        # with the Canary serving too, raise a group to MAX before the Router rejects.
        if new_pressure > 0:
            self._calm_since = None
            self._pressure_at = now
            woken = await self.wake("pressure")
            if woken is not None:
                active.append(woken)
            elif not (self.on_pressure is not None and self.on_pressure("pressure")):
                await self._boost(active, "pressure")
        else:
            await self._unboost(active, now)
        await self._plan(now)
        await self._finish_draining()

    async def _retry_locks(self, now: float) -> None:
        """Lock again the groups whose last lock failed (their target is unchanged)."""
        for group in self.groups:
            due = self._retry.get(group.name)
            if due is None or now < due[0]:
                continue
            if group.state in (GroupState.ACTIVE, GroupState.PARK):
                await self.set_tier(group, group.tier, "retry")
            else:
                self._retry.pop(group.name, None)  # draining/exploring: not ours to lock

    async def _boost(self, active: Sequence[Group], reason: str) -> None:
        """Raise the most loaded working group to MAX (one per tick)."""
        working = [g for g in active if g.tier is Tier.H]
        if working:
            group = max(working, key=lambda g: (g.n_inflight, g.pending_ms))
            await self.set_tier(group, Tier.MAX, f"boost:{reason}")

    async def _unboost(self, active: Sequence[Group], now: float) -> None:
        """Return one MAX group to the working point after t_down_s without pressure."""
        boosted = [g for g in active if g.tier is Tier.MAX]
        if not boosted:
            self._calm_since = None
            return
        if self._calm_since is None:
            self._calm_since = now
            return
        if now - self._calm_since < self.settings.t_down_s:
            return
        self._calm_since = now  # the next group waits another t_down_s
        group = min(boosted, key=lambda g: (g.n_inflight, g.pending_ms))
        await self.set_tier(group, Tier.H, "unboost")

    async def _plan(self, now: float) -> None:
        solver = self.solver()
        if solver is None:
            return
        s = self.settings
        rate, burst, lengths = self.demand(now)
        if not lengths:
            return
        serving = [g for g in self.groups if g.state in (GroupState.ACTIVE, GroupState.PARK)]
        demand = rate  # variability enters through the peak over t_down_s below
        self._demand.append((now, demand))
        while self._demand and self._demand[0][0] < now - s.t_down_s:
            self._demand.popleft()
        active = [g for g in self.groups if g.state is GroupState.ACTIVE]
        # Capacity counts only routable groups (a failed lock after Park leaves none).
        routable = [g for g in active if g.effective is not None]
        point = self.tiers.clocks(Tier.H)
        current = solver.evaluate(demand, lengths, max(len(routable), 1), point)
        up = not current.feasible  # the current configuration cannot carry the demand
        # Plan for the peak demand of the last t_down_s (both directions): the chosen
        # configuration then stays feasible while the demand moves below that peak.
        peak = max(d for _, d in self._demand)
        target = solver.solve(peak, lengths, min_groups=s.min_active_groups, groups=len(serving))
        self.plan = target
        key = (target.n, target.point)
        if key != self._target:
            self._target, self._target_since = key, now
        held = now - (self._target_since or now) >= s.t_down_s
        calm = now - self._pressure_at >= s.t_down_s
        gain = current.power_w - target.power_w > s.switch_gain * current.power_w
        if key == (len(routable), point):
            return
        if not (up or (held and calm and gain)):
            return
        self.log.write(
            {
                "event": "plan",
                "t": round(now, 1),
                "rate_rps": round(rate, 3),
                "burst": round(burst, 3),
                "from": {"n": len(routable), "point": point.key(), "power_w": current.power_w},
                "to": {"n": target.n, "point": target.point.key(), "power_w": target.power_w},
                "feasible": target.feasible,
                "binding": target.binding,
                "use": {k: round(v, 3) for k, v in target.use.items()},
                "reason": "up" if up else "down",
            }
        )
        if target.point != point:
            self.tiers.working = target.point
            for group in active:
                if group.tier is Tier.H and group.effective != target.point:
                    await self.set_tier(group, Tier.H, "plan")
        while len(routable) < target.n:
            woken = await self.wake("plan")
            if woken is None:
                break
            active.append(woken)
            if woken.effective is not None:
                routable.append(woken)
        if len(routable) > max(target.n, s.min_active_groups):
            # Drain one group per step: the Canary first so it is free to explore.
            victim = min(active, key=lambda g: (not g.canary, g.n_inflight, g.pending_ms))
            victim.state = GroupState.DRAINING
            self.log.write({"event": "drain", "group": victim.name, "reason": "plan"})

    async def _finish_draining(self) -> None:
        for group in self.groups:
            if group.state is GroupState.DRAINING and group.n_inflight == 0:
                group.state = GroupState.PARK
                await self.set_tier(group, Tier.PARK, "drained")

    async def _log_energy(self, now: float) -> None:
        if now - self._last_energy_log < self.settings.energy_log_period_s:
            return
        self._last_energy_log = now
        agents = sorted({ref.agent_url for ref in self.gpu_refs.values()})
        by_gpu = {(ref.agent_url, ref.gpu): name for name, ref in self.gpu_refs.items()}
        state = {g.prefill: g for g in self.groups} | {g.decode: g for g in self.groups}
        for agent in agents:
            try:
                readings = await self.actuator.readings(agent)
            except Exception as error:
                self.log.write({"event": "energy_error", "agent": agent, "error": str(error)})
                continue
            for reading in readings:
                endpoint = by_gpu.get((agent, reading.get("index")))
                if endpoint is not None:
                    group = state.get(endpoint)
                    self.log.write(
                        {
                            "event": "energy",
                            "endpoint": endpoint,
                            "group": None if group is None else group.name,
                            "group_state": None if group is None else group.state.value,
                            **reading,
                        }
                    )

    async def run(self, stop: asyncio.Event) -> None:
        """Control loop; `start()` must have run (the Runtime does it)."""
        while not stop.is_set():
            try:
                await self.tick()
            except Exception as error:  # a bad tick must not stop control
                self.log.write({"event": "controller_error", "error": repr(error)})
            try:
                await asyncio.wait_for(stop.wait(), timeout=self.settings.period_s)
            except asyncio.TimeoutError:
                pass

    def state(self) -> dict[str, Any]:
        table = self.tiers.table
        return {
            "tiers": None if table is None else table.to_json(),
            "max_point": self.tiers.max_point.key(),
            "loads": self.loads(self._clock()),
        }
