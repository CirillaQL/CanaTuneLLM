"""Assemble groups, tables, telemetry, Router, Controller and Canary from config."""

import asyncio
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import httpx

from canatune.controller.canary import CanaryScheduler, SchedulerSettings
from canatune.controller.locator import LocatorSettings, ProbeBackend, TierLocator
from canatune.controller.probe import CanaryProbe, ProbeSettings
from canatune.controller.router import CanaTuneRouter, RouterSettings
from canatune.controller.tier_controller import ControllerSettings, TierController
from canatune.domain.groups import (
    ClockPoint,
    Group,
    GroupState,
    Tier,
    TierState,
    TierStore,
    TierTableError,
)
from canatune.domain.load import LengthStats
from canatune.domain.risk import RiskTable, RiskTableError
from canatune.infrastructure.clocks import (
    AgentClockActuator,
    ClockActuator,
    GpuRef,
    NullClockActuator,
    gpu_refs,
)
from canatune.infrastructure.records import JsonlLog
from canatune.infrastructure.telemetry import Telemetry
from canatune.proxy.proxy import parse_endpoints


@dataclass
class Runtime:
    """Everything the proxy needs for the `cantune` routing policy."""

    groups: list[Group]
    router: CanaTuneRouter
    controller: TierController
    canary: CanaryScheduler
    telemetry: Telemetry | None
    events: JsonlLog
    tiers: TierState
    lengths: LengthStats
    tasks: list[asyncio.Task] = field(default_factory=list)
    stop: asyncio.Event = field(default_factory=asyncio.Event)

    async def discover_max(self) -> None:
        """MAX = highest clocks the Canary pair's agents report (unknown cluster);
        the configured values stay when clock control is off."""
        canary = self.canary.canary
        refs = self.controller.gpu_refs
        if canary is None or canary.prefill not in refs or canary.decode not in refs:
            return
        try:
            p = await self.controller.actuator.supported_clocks(refs[canary.prefill])
            d = await self.controller.actuator.supported_clocks(refs[canary.decode])
        except Exception as error:
            self.events.write({"event": "max_clock_error", "error": repr(error)})
            return
        if p and d:
            self.tiers.max_point = ClockPoint(max(p), max(d))
            self.events.write({"event": "max_clock", "clock": self.tiers.max_point.key()})

    async def start(self) -> None:
        await self.discover_max()
        await self.controller.start()
        canary = self.canary.canary
        if self.canary.locator is None and canary is not None:
            # No agents, no calibration: the Canary serves like any production group.
            if canary.state is GroupState.EXPLORING:
                canary.state = GroupState.ACTIVE
                await self.controller.set_tier(canary, Tier.MAX, "no_locator")
        if self.telemetry is not None:
            self.tasks.append(asyncio.create_task(self.telemetry.run(self.stop)))
        self.tasks.append(asyncio.create_task(self.controller.run(self.stop)))
        self.tasks.append(asyncio.create_task(self.canary.run(self.stop)))

    async def shutdown(self) -> None:
        self.stop.set()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        self.router.table.save()


def identity(config: Mapping[str, Any]) -> dict[str, Any]:
    """Configuration identity of the risk and tier tables: a change starts new ones."""
    return {
        "model": config["model"]["name"],
        "vllm_version": config["runtime"].get("vllm_version"),
        "connector": config["kv_transfer"]["connector"],
        "send_type": config["kv_transfer"]["prefill_send_type"],
        "prefill_gpu": config["topology"]["prefill_nodegroup"]["gpu_type"],
        "decode_gpu": config["topology"]["decode_nodegroup"]["gpu_type"],
        "max_model_len": config["model"]["max_model_len"],
    }


def build_groups(config: Mapping[str, Any]) -> list[Group]:
    routing = config["routing"]
    groups = [
        Group(
            name="G0",
            prefill=routing["canary_pair"][0],
            decode=routing["canary_pair"][1],
            canary=True,
        )
    ]
    for index, (prefill, decode) in enumerate(routing["production_pairs"], start=1):
        groups.append(Group(name=f"G{index}", prefill=prefill, decode=decode))
    for group in groups:
        group.state = GroupState.PARK
        group.tier = Tier.PARK
    return groups


def configured_max(config: Mapping[str, Any]) -> ClockPoint:
    topology = config["topology"]
    return ClockPoint(
        int(topology["prefill_nodegroup"]["high_frequency_mhz"]),
        int(topology["decode_nodegroup"]["high_frequency_mhz"]),
    )


def _env_override(raw: Mapping[str, Any], env: str, key: str) -> Any:
    return os.environ.get(env) or raw.get(key)


def build_runtime(
    config: Mapping[str, Any],
    metrics_urls: Mapping[str, str],
    client_factory: Callable[[], httpx.AsyncClient],
    *,
    actuator: ClockActuator | None = None,
    refs: Mapping[str, GpuRef] | None = None,
    probe_backend: ProbeBackend | None = None,
) -> Runtime:
    record_dir = os.environ.get("CANATUNE_RECORD_DIR")
    requests_log = JsonlLog(None if not record_dir else os.path.join(record_dir, "requests.jsonl"))
    events = JsonlLog(None if not record_dir else os.path.join(record_dir, "events.jsonl"))
    ident = identity(config)

    risk_raw = dict(config["risk"])
    risk_raw["path"] = _env_override(risk_raw, "CANATUNE_RISK_TABLE", "path")
    try:
        table = RiskTable.from_config(risk_raw, ident)
    except RiskTableError as error:
        # A table from another configuration is neither reused nor overwritten.
        events.write({"event": "risk_table_ignored", "error": str(error)})
        table = RiskTable.from_config({**risk_raw, "path": None}, ident)

    canary_raw = config.get("canary", {})
    store = TierStore(_env_override(canary_raw, "CANATUNE_TIER_TABLE", "tier_table_path"), ident)
    tiers = TierState(max_point=configured_max(config))
    try:
        tiers.table = store.load()
    except TierTableError as error:
        events.write({"event": "tier_table_ignored", "error": str(error)})
    lengths = LengthStats(
        [tuple(pair) for pair in canary_raw.get("default_lengths", [[512, 64]])],
        min_samples=int(canary_raw.get("length_min_samples", 50)),
    )

    telemetry = None
    if config.get("telemetry", {}).get("enabled", True):
        telemetry = Telemetry(
            metrics_urls,
            period_s=float(config.get("telemetry", {}).get("period_s", 0.25)),
            client_factory=client_factory,
        )

    groups = build_groups(config)
    router_settings = RouterSettings.from_config(config)
    router = CanaTuneRouter(
        groups,
        table,
        router_settings,
        tiers,
        lengths=lengths,
        telemetry=telemetry,
        log=requests_log,
    )

    clock_control = dict(config.get("clock_control", {}))
    agents = dict(clock_control.get("agents") or {})
    for role, env in (("prefill", "CANATUNE_PREFILL_AGENT"), ("decode", "CANATUNE_DECODE_AGENT")):
        agents[role] = os.environ.get(env) or agents.get(role)
    clock_control["agents"] = agents
    if actuator is None:
        actuator = (
            AgentClockActuator(client_factory)
            if clock_control.get("enabled")
            else NullClockActuator()
        )
    if refs is None:
        refs = (
            gpu_refs({**config, "clock_control": clock_control})
            if clock_control.get("enabled")
            else {}
        )

    controller = TierController(
        groups,
        router,
        ControllerSettings.from_config(config),
        actuator,
        refs,
        tiers,
        store=store,
        log=events,
    )

    locator = None
    canary_group = groups[0]
    if probe_backend is None and canary_group.prefill in refs and canary_group.decode in refs:
        slo = config["experiment"]["slo"]
        probe_backend = CanaryProbe(
            canary_group,
            parse_endpoints(config),
            actuator,
            refs,
            lengths,
            table,
            ProbeSettings.from_config(config),
            client_factory=lambda: httpx.AsyncClient(
                timeout=httpx.Timeout(120, connect=5), trust_env=False
            ),
            telemetry=telemetry,
            ttft_slo_ms=float(slo["ttft_ms"]),
            tpot_slo_ms=float(slo["tpot_ms"]),
            log=events,
        )
    if probe_backend is not None:
        locator = TierLocator(probe_backend, LocatorSettings.from_config(config), log=events)
    canary = CanaryScheduler(
        controller,
        locator,
        lengths,
        SchedulerSettings.from_config(config),
        theta=router_settings.theta,
    )
    return Runtime(groups, router, controller, canary, telemetry, events, tiers, lengths)
