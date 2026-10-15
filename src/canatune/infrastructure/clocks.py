"""Controller-side access to the per-node GPU agents."""

import asyncio
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from canatune.topology import agent_urls, endpoint_gpus, endpoint_node


@dataclass(frozen=True)
class GpuRef:
    """One endpoint's physical GPUs and the agent that manages them: `gpu`, plus the
    other ranks' GPUs of a tensor-parallel endpoint in `extra` (same node, same agent)."""

    agent_url: str
    gpu: int
    extra: tuple[int, ...] = ()

    @property
    def gpus(self) -> tuple[int, ...]:
        return (self.gpu, *self.extra)


def merge_locks(results: list[dict[str, Any]]) -> dict[str, Any]:
    """One lock result for a TP endpoint: ok when every rank's GPU locked; the slowest
    confirmation and the lowest clock read back (the ranks run in lock step)."""
    if len(results) == 1:
        return results[0]
    reach = [r.get("reach_ms") for r in results]
    errors = [str(r["error"]) for r in results if r.get("error")]
    return {
        "ok": all(r.get("ok") for r in results),
        "gpu": results[0].get("gpu"),
        "gpus": [r.get("gpu") for r in results],
        "target_mhz": results[0].get("target_mhz"),
        "cmd_ms": max(float(r.get("cmd_ms") or 0) for r in results),
        "reach_ms": None if None in reach else max(float(x) for x in reach),
        "sm_mhz": min(int(r.get("sm_mhz") or 0) for r in results),
        "error": "; ".join(errors) or None,
    }


def merge_readings(readings: list[dict[str, Any]]) -> dict[str, Any]:
    """One reading for a TP endpoint: energy and power summed over its GPUs, the
    lowest SM clock, and any GPU's throttle reasons."""
    if len(readings) == 1:
        return readings[0]
    reasons = [r.get("throttle_reasons") for r in readings]
    energy = [r.get("energy_mj") for r in readings]
    return {
        "index": readings[0].get("index"),
        "indices": [r.get("index") for r in readings],
        "sm_mhz": min(int(r.get("sm_mhz") or 0) for r in readings),
        "power_w": sum(float(r.get("power_w") or 0) for r in readings),
        "energy_mj": None if None in energy else sum(float(e) for e in energy),
        "throttle_reasons": None
        if all(x is None for x in reasons)
        else _or(int(x) for x in reasons if x is not None),
    }


def _or(values: Iterable[int]) -> int:
    out = 0
    for value in values:
        out |= value
    return out


def endpoint_reading(ref: GpuRef, readings: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The merged reading of `ref`'s GPUs from an agent's /gpus list (None if any is
    missing)."""
    by_index = {r.get("index"): r for r in readings}
    if not all(g in by_index for g in ref.gpus):
        return None
    return merge_readings([by_index[g] for g in ref.gpus])


class ClockActuator(Protocol):
    async def lock(self, ref: GpuRef, mhz: int) -> dict[str, Any]: ...

    async def readings(self, agent_url: str) -> list[dict[str, Any]]: ...

    async def supported_clocks(self, ref: GpuRef) -> list[int]: ...


class AgentClockActuator:
    def __init__(self, client_factory: Callable[[], httpx.AsyncClient]) -> None:
        self._client_factory = client_factory

    async def lock(self, ref: GpuRef, mhz: int) -> dict[str, Any]:
        async with self._client_factory() as client:

            async def one(gpu: int) -> dict[str, Any]:
                response = await client.post(f"{ref.agent_url}/gpus/{gpu}/lock", json={"mhz": mhz})
                response.raise_for_status()
                return response.json()

            return merge_locks(list(await asyncio.gather(*(one(g) for g in ref.gpus))))

    async def readings(self, agent_url: str) -> list[dict[str, Any]]:
        async with self._client_factory() as client:
            response = await client.get(f"{agent_url}/gpus")
            response.raise_for_status()
            return response.json()

    async def supported_clocks(self, ref: GpuRef) -> list[int]:
        """Clocks every GPU of the endpoint supports."""
        common: set[int] | None = None
        async with self._client_factory() as client:
            for gpu in ref.gpus:
                response = await client.get(f"{ref.agent_url}/gpus/{gpu}/clocks")
                response.raise_for_status()
                clocks = {int(f) for f in response.json()["sm_mhz"]}
                common = clocks if common is None else common & clocks
        return sorted(common or ())


class NullClockActuator:
    """Clock control disabled: every lock is reported as confirmed without acting.
    Used for the round-robin baseline and CPU-only tests."""

    def __init__(self) -> None:
        self.calls: list[tuple[GpuRef, int]] = []

    async def lock(self, ref: GpuRef, mhz: int) -> dict[str, Any]:
        self.calls.append((ref, mhz))
        return {"ok": True, "gpu": ref.gpu, "target_mhz": mhz, "reach_ms": 0.0, "sm_mhz": mhz}

    async def readings(self, agent_url: str) -> list[dict[str, Any]]:
        return []

    async def supported_clocks(self, ref: GpuRef) -> list[int]:
        return []


def gpu_refs(
    config: Mapping[str, Any], env: Mapping[str, str] | None = None
) -> dict[str, GpuRef]:
    """Map endpoint name -> GpuRef from `topology` and `clock_control.agents`: a role's
    agent URL, or, for a role on several nodes, the agent on the endpoint's node
    (CANATUNE_NODE_HOSTS and clock_control.agent_port)."""
    topology = config["topology"]
    agents = config.get("clock_control", {}).get("agents", {}) or {}
    by_node = agent_urls(config, env)
    multi = {
        role: len({node for r, node in by_node if r == role}) > 1 for role in ("prefill", "decode")
    }
    refs = {}
    for name, endpoint in topology["endpoints"].items():
        role = endpoint["role"]
        url = by_node.get((role, endpoint_node(config, endpoint))) if multi[role] else None
        url = url or agents.get(role)
        if url:
            first, *rest = endpoint_gpus(endpoint)
            refs[name] = GpuRef(agent_url=str(url).rstrip("/"), gpu=first, extra=tuple(rest))
    return refs
