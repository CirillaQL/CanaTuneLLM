"""Controller-side access to the per-node GPU agents."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import httpx


@dataclass(frozen=True)
class GpuRef:
    """One endpoint's physical GPU and the agent that manages it."""

    agent_url: str
    gpu: int


class ClockActuator(Protocol):
    async def lock(self, ref: GpuRef, mhz: int) -> dict[str, Any]: ...

    async def readings(self, agent_url: str) -> list[dict[str, Any]]: ...

    async def supported_clocks(self, ref: GpuRef) -> list[int]: ...


class AgentClockActuator:
    def __init__(self, client_factory: Callable[[], httpx.AsyncClient]) -> None:
        self._client_factory = client_factory

    async def lock(self, ref: GpuRef, mhz: int) -> dict[str, Any]:
        async with self._client_factory() as client:
            response = await client.post(f"{ref.agent_url}/gpus/{ref.gpu}/lock", json={"mhz": mhz})
            response.raise_for_status()
            return response.json()

    async def readings(self, agent_url: str) -> list[dict[str, Any]]:
        async with self._client_factory() as client:
            response = await client.get(f"{agent_url}/gpus")
            response.raise_for_status()
            return response.json()

    async def supported_clocks(self, ref: GpuRef) -> list[int]:
        async with self._client_factory() as client:
            response = await client.get(f"{ref.agent_url}/gpus/{ref.gpu}/clocks")
            response.raise_for_status()
            return [int(f) for f in response.json()["sm_mhz"]]


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


def gpu_refs(config: Mapping[str, Any]) -> dict[str, GpuRef]:
    """Map endpoint name -> GpuRef from `topology` and `clock_control.agents`."""
    topology = config["topology"]
    agents = config.get("clock_control", {}).get("agents", {})
    refs = {}
    for name, endpoint in topology["endpoints"].items():
        role = endpoint["role"]
        url = agents.get(role)
        if url:
            refs[name] = GpuRef(agent_url=str(url).rstrip("/"), gpu=int(endpoint["gpu_id"]))
    return refs
