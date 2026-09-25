"""Seconds-scale Controller: group tiers, waking and parking.

Cold start (no tier table): every production group is active at MAX and the
Router admits everything; the Controller neither consolidates nor switches.
Once the Canary publishes a table, production groups move to H one at a time,
and from then on the Controller
* wakes a parked group when the Router rejected requests or the mean active
  load exceeds `wake_load_fraction * C_H`; if nothing is parked it asks the
  Canary to abort its experiment (pressure order: wake -> abort Canary -> reject),
* switches L/H per group when the Canary published an L tier,
* drains and parks the least-loaded group when the rest could carry the total
  at `park_load_fraction * C_H` for `t_down_s`.

Loads are equivalent prompt tokens/s. A clock change takes ~0.2-0.5 s, so the
Controller acts on windowed trends; while a group's clock changes the Router
assumes the slower of the old and new clock point, and every change bumps the
group's clock epoch so in-flight samples are not written to the risk table.
"""

import asyncio
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from canatune.controller.router import CanaTuneRouter
from canatune.domain.groups import Group, GroupState, Tier, TierState, TierStore, TierTable, slower
from canatune.infrastructure.clocks import ClockActuator, GpuRef
from canatune.infrastructure.records import JsonlLog


class ControllerConfigError(ValueError):
    """Controller settings are invalid."""


def _fraction(raw: Mapping[str, Any], key: str, default: float) -> float:
    value = raw.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= 1:
        raise ControllerConfigError(f"controller.{key} must be in (0, 1]")
    return float(value)


@dataclass(frozen=True)
class ControllerSettings:
    period_s: float = 1.0
    load_window_s: float = 10.0
    t_down_s: float = 30.0
    wake_load_fraction: float = 0.85
    park_load_fraction: float = 0.7
    min_active_groups: int = 1
    stagger_s: float = 2.0
    energy_log_period_s: float = 5.0

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "ControllerSettings":
        raw = config.get("controller")
        if not isinstance(raw, Mapping):
            raise ControllerConfigError("controller section is required")
        return cls(
            period_s=float(raw.get("period_s", 1.0)),
            load_window_s=float(raw.get("load_window_s", 10.0)),
            t_down_s=float(raw.get("t_down_s", 30.0)),
            wake_load_fraction=_fraction(raw, "wake_load_fraction", 0.85),
            park_load_fraction=_fraction(raw, "park_load_fraction", 0.7),
            min_active_groups=int(raw.get("min_active_groups", 1)),
            stagger_s=float(raw.get("stagger_s", 2.0)),
            energy_log_period_s=float(raw.get("energy_log_period_s", 5.0)),
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
        self._low_since: dict[str, float] = {}
        self._consolidate_since: float | None = None
        self._rejections_seen = 0
        self._last_energy_log = float("-inf")
        self.on_pressure: Callable[[str], None] | None = None  # set by the Canary scheduler

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
        if tier is not Tier.PARK and group.tier is tier:
            # On failure keep the conservative point; a later tick retries.
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
        if self.store is not None:
            self.store.save(table)
        self.log.write({"event": "publish", "reason": reason, "table": table.to_json()})
        for group in self.groups:
            if group.state is not GroupState.ACTIVE:
                continue
            tier = Tier.H if group.tier in (Tier.MAX, Tier.H) or table.l is None else group.tier
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

    async def tick(self) -> None:
        now = self._clock()
        await self._log_energy(now)
        table = self.tiers.table
        if table is None:
            return  # cold start: everything at MAX, nothing to decide
        s = self.settings
        new_rejections = self.router.rejections - self._rejections_seen
        self._rejections_seen = self.router.rejections
        loads = self.loads(now)
        active = [g for g in self.groups if g.state is GroupState.ACTIVE]
        capacity = table.capacity_h
        mean_load = sum(loads[g.name] for g in active) / len(active) if active else float("inf")

        # Pressure: wake a parked group; with none left, the Canary must come back.
        if new_rejections > 0 or mean_load > s.wake_load_fraction * capacity:
            reason = "rejections" if new_rejections else "load_above_wake"
            woken = await self.wake(reason)
            if woken is not None:
                active.append(woken)
            elif self.on_pressure is not None:
                self.on_pressure(reason)

        await self._switch_l_h(table, active, loads, now)
        await self._consolidate(table, active, loads, now)
        await self._finish_draining()

    async def _switch_l_h(
        self, table: TierTable, active: Sequence[Group], loads: Mapping[str, float], now: float
    ) -> None:
        if table.l is None or table.tau_up is None or table.tau_down is None:
            for group in active:
                if group.tier is not Tier.H:
                    await self.set_tier(group, Tier.H, "single_working_tier")
            return
        for group in active:
            load = loads[group.name]
            if group.tier is not Tier.H and (group.tier is not Tier.L or load > table.tau_up):
                self._low_since.pop(group.name, None)
                await self.set_tier(group, Tier.H, "load_above_tau_up")
            elif group.tier is Tier.H:
                if load < table.tau_down:
                    since = self._low_since.setdefault(group.name, now)
                    if now - since >= self.settings.t_down_s:
                        self._low_since.pop(group.name)
                        await self.set_tier(group, Tier.L, "load_below_tau_down")
                else:
                    self._low_since.pop(group.name, None)

    async def _consolidate(
        self, table: TierTable, active: Sequence[Group], loads: Mapping[str, float], now: float
    ) -> None:
        s = self.settings
        if len(active) <= s.min_active_groups:
            self._consolidate_since = None
            return
        total = sum(loads[g.name] for g in active)
        if total > s.park_load_fraction * table.capacity_h * (len(active) - 1):
            self._consolidate_since = None
            return
        if self._consolidate_since is None:
            self._consolidate_since = now
            return
        if now - self._consolidate_since < s.t_down_s:
            return
        self._consolidate_since = None
        # Drain the least-loaded group; the Canary first so it is free to explore.
        victim = min(active, key=lambda g: (loads[g.name], not g.canary, g.n_inflight))
        victim.state = GroupState.DRAINING
        self.log.write({"event": "drain", "group": victim.name, "total_load": total})

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
