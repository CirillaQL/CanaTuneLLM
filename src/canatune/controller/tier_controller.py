"""Seconds-scale Controller: how many groups run, at which clocks.

Cold start (no tier table): every production group is active at MAX and the
Router admits everything; the Controller does nothing else.

With a table, every tick:
* best-effort stages: prediction warning -> persistent Production latency confirms
  pressure -> expand at measured working capacity -> MAX/reclaim Canary only after
  expansion fails or unfinished work is severely stalled. Recover after a quiet
  t_down_s using actual feedback, not a model-only feasibility decision.
  Legacy overload policies retain incremental wake/abort/boost behavior.
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
import math
import time
from collections import deque
from collections.abc import Awaitable, Callable, Mapping, Sequence
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
    feedback_window_s: float = 10.0
    feedback_min_samples: int = 5
    feedback_near_slo: float = 0.9
    confirm_s: float = 2.0
    expansion_grace_s: float = 3.0
    solver: bool = True  # False: every group stays at the Canary's H (static comparison)

    def __post_init__(self) -> None:
        durations = (self.feedback_window_s, self.confirm_s, self.expansion_grace_s)
        if (not all(math.isfinite(v) for v in durations)
                or self.feedback_window_s <= 0 or self.feedback_min_samples < 1
                or not 0 < self.feedback_near_slo <= 1
                or self.confirm_s < 0 or self.expansion_grace_s < 0):
            raise ControllerConfigError("invalid production feedback settings")

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
            feedback_window_s=float(raw.get("feedback_window_s", 10)),
            feedback_min_samples=int(raw.get("feedback_min_samples", 5)),
            feedback_near_slo=float(raw.get("feedback_near_slo", 0.9)),
            confirm_s=float(raw.get("confirm_s", 2)),
            expansion_grace_s=float(raw.get("expansion_grace_s", 3)),
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
        self.reclaim_canary: Callable[[], Awaitable[None]] | None = None
        self._full_effort = False
        self._control_lock = asyncio.Lock()
        self._switch_s = 0.0
        self.mode = "energy"
        self._warning_seen = 0
        self._warning_at = float("-inf")
        self._confirm_since: float | None = None
        self._severe_since: float | None = None
        self._expanded_at: float | None = None
        self.feedback: dict[str, Any] = {}

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
        self._switch_s = max(
            self._switch_s, float(result.get("confirm_ms") or result.get("cmd_ms") or 0) / 1000
        )
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
            tier = Tier.MAX if self._full_effort else Tier.H
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
        if raw is not None and raw.get("predictor_coef") is not None:
            self.router._apply_table()
            raw = dict(raw)
            raw["predictor_coef"] = list(self.router.predictor.coef)
            slack = self.router.slack.to_json()
            counts = {k: list(v) for k, v in slack["seed"].items()}
            for k, v in slack["counts"].items():
                old = counts.setdefault(k, [0, 0])
                old[0] += v[0]
                old[1] += v[1]
            raw["slack_counts"] = counts
        key = None if raw is None else (
            self.tiers.table.published_at, str(raw.get("caps")),
            str(raw.get("predictor_coef")), str(raw.get("slack_counts")),
        )
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
        if self.router.settings.overload == "best_effort":
            recent, _ = self.router.offered(now, max(s.period_s, 1), max(s.period_s, 1))
            rate = max(rate, recent)  # react to bursts before the long window fills
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
        async with self._control_lock:
            await self._tick()

    async def _tick(self) -> None:
        now = self._clock()
        if self.router.settings.overload == "best_effort":
            await self._feedback_control(now)
            now = self._clock()  # clock changes above can take seconds; arrivals went on
            await self._log_energy(now)
            await self._retry_locks(now)
            if self.tiers.table is not None:
                if self.mode in ("energy", "warning", "confirming"):
                    await self._plan(now)  # observe targets even while latency is confirming
                await self._finish_draining()  # lifecycle cleanup never waits for energy mode
            return
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

    def _mode(self, mode: str, reason: str) -> None:
        if self.mode != mode:
            previous = self.mode
            self.mode = mode
            if mode in ("expanding", "full_effort") or previous in ("expanding", "full_effort"):
                self._target = self._target_since = None  # resources changed, target must re-settle
            self.log.write({"event": "control_mode", "mode": mode, "reason": reason})

    async def _feedback_control(self, now: float) -> None:
        s = self.settings
        warning = self.router.risk_signals != self._warning_seen
        self._warning_seen = self.router.risk_signals
        if warning:
            self._warning_at = now
            if self.mode == "energy":
                self._mode("warning", "prediction")
        since = self._expanded_at if self.mode == "expanding" else None
        f = self.feedback = self.router.production_feedback(
            s.feedback_window_s, s.feedback_min_samples, s.feedback_near_slo,
            since=float("-inf") if since is None else since,
        )
        if f["pressure"]:
            self._pressure_at = now
            if self._confirm_since is None:
                self._confirm_since = now
            if self.mode in ("energy", "warning"):
                self._mode("confirming", "production_latency")
        else:
            self._confirm_since = None
        if f["severe"]:
            if self._severe_since is None:
                self._severe_since = now
        else:
            self._severe_since = None
        confirmed = self._confirm_since is not None and now - self._confirm_since >= s.confirm_s
        emergency = self._severe_since is not None and now - self._severe_since >= s.confirm_s
        if self._full_effort:
            await self._enter_full_effort("reclaim_retry")
            if f["pressure"] or now - self._pressure_at < s.t_down_s:
                return
            if any(g.state is GroupState.EXPLORING for g in self.groups):
                return
            if self.tiers.table is None and any(g.n_inflight for g in self.groups):
                return
            self._full_effort = self.router.full_effort = False
            self._expanded_at = None
            for group in self.groups:
                if group.state is GroupState.ACTIVE and self.tiers.table is not None:
                    await self.set_tier(group, Tier.H, "recovered")
            self._mode("energy", "production_recovered")
            return
        if emergency:
            await self._enter_full_effort("severe_production_backlog")
            return
        if self.mode == "expanding":
            assert self._expanded_at is not None
            if now - self._expanded_at < s.expansion_grace_s:
                return
            if confirmed:
                await self._enter_full_effort("expansion_insufficient")
                return
        elif confirmed:
            if not any(
                g.state not in (GroupState.ACTIVE, GroupState.EXPLORING) for g in self.groups
            ):
                # Every group already serves: expansion cannot add capacity, waiting
                # out its grace only delays MAX.
                await self._enter_full_effort("nothing_to_expand")
                return
            self._mode("expanding", "production_confirmed")
            await self._expand_capacity(now)
            self._expanded_at = self._clock()  # grace starts after locks finish
            self._confirm_since = None
            return
        quiet_after = (max(self._pressure_at, self._warning_at)
                       if self.mode in ("warning", "confirming") else self._pressure_at)
        if not f["pressure"] and now - quiet_after >= s.t_down_s:
            self._expanded_at = None
            self._mode("energy", "production_recovered")

    async def _expand_capacity(self, now: float) -> None:
        """Wake enough groups at the measured working point; static uses C_H."""
        active = [g for g in self.groups if g.routable]
        rate, _, mix = self.demand(now)
        solver = self.solver()
        capacity = 0.0
        if solver is not None and mix:
            capacity = solver.capacity(mix, self.tiers.clocks(Tier.H))
        elif self.tiers.table is not None and mix:
            mean = sum(p for p, _ in mix) / len(mix)
            capacity = self.tiers.table.capacity_h / (mean + self.tiers.alpha_tokens)
        available = sorted(
            (g for g in self.groups if g.state is not GroupState.EXPLORING),
            key=lambda g: g.canary,
        )
        needed = math.ceil(rate / capacity) if capacity > 0 else len(active) + 1
        needed = min(len(available), max(len(active) + 1, needed))
        self.log.write({"event": "capacity_expand", "rate_rps": rate,
                        "capacity_rps": capacity, "target_groups": needed})
        for group in available:
            if len(active) >= needed:
                break
            if group not in active:
                group.state = GroupState.ACTIVE
                await self.set_tier(
                    group, Tier.H if self.tiers.table is not None else Tier.MAX,
                    "production_capacity",
                )
                if group.routable:
                    active.append(group)

    async def _enter_full_effort(self, reason: str) -> None:
        self._mode("full_effort", reason)
        if not self._full_effort:
            self._full_effort = self.router.full_effort = True
            self._target = self._target_since = None
            self.log.write({"event": "control_mode", "mode": "full_effort", "reason": reason})
            if self.on_pressure is not None:
                self.on_pressure(reason)
        changes = []
        for group in self.groups:
            if group.state is GroupState.EXPLORING:
                continue  # experiment cleanup owns this group until it returns
            group.state = GroupState.ACTIVE
            if group.tier is not Tier.MAX:
                changes.append(self.set_tier(group, Tier.MAX, reason))
        await asyncio.gather(*changes)
        # Wake production first: draining a cancelled experiment may take seconds.
        if self.reclaim_canary is not None:
            try:
                await self.reclaim_canary()
            except Exception as error:
                self.log.write({"event": "canary_reclaim_error", "error": repr(error)})
                return  # remains EXPLORING; retry on the next full-effort tick
            for group in self.groups:
                if group.canary and group.state is GroupState.ACTIVE and (
                    group.tier is not Tier.MAX or group.effective is None
                ):
                    await self.set_tier(group, Tier.MAX, reason)

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
        held = now - (now if self._target_since is None else self._target_since) >= s.t_down_s
        calm = now - self._pressure_at >= s.t_down_s
        gain = current.power_w - target.power_w > s.switch_gain * current.power_w
        gain = gain and (
            (current.power_w - target.power_w) * s.t_down_s
            > 2 * self._switch_s * max(current.power_w, target.power_w)
        )
        if self.router.settings.overload == "best_effort" and (
            not target.feasible or up or target.n > len(routable)
            or target.point.prefill_mhz > point.prefill_mhz
            or target.point.decode_mhz > point.decode_mhz
        ):
            if self.mode == "energy":
                self.router.signal_pressure("model_capacity_warning")
                self._warning_at = now
                self._mode("warning", "model_capacity")
            return  # model-only pressure never expands or enters MAX
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
            self.router.pressure_event.clear()
            try:
                await self.tick()
            except Exception as error:  # a bad tick must not stop control
                self.log.write({"event": "controller_error", "error": repr(error)})
            try:
                stopping = asyncio.create_task(stop.wait())
                pressure = asyncio.create_task(self.router.pressure_event.wait())
                try:
                    await asyncio.wait(
                        (stopping, pressure), timeout=self.settings.period_s,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                finally:
                    stopping.cancel()
                    pressure.cancel()
                    await asyncio.gather(stopping, pressure, return_exceptions=True)
            except asyncio.TimeoutError:
                pass

    def state(self) -> dict[str, Any]:
        table = self.tiers.table
        return {
            "mode": self.mode,
            "production_feedback": dict(self.feedback),
            "tiers": None if table is None else table.to_json(),
            "max_point": self.tiers.max_point.key(),
            "loads": self.loads(self._clock()),
        }
