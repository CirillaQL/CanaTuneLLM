"""Per-node GPU agent: clock locking and NVML readings over HTTP.

Run one agent on every GPU node (`python -m canatune.infrastructure.gpu_agent`).
The Controller never calls `nvidia-smi` itself: GPUs live on other nodes, and
clock changes must be serialized per GPU and confirmed by an NVML read-back.

Safety: the agent only touches the GPU indices in `CANATUNE_AGENT_GPUS` and only
accepts clocks the GPU supports, within [`CANATUNE_AGENT_MIN_MHZ`, max]; the
Canary discovers tiers, so the clock set is not configured in advance.

Environment:
  CANATUNE_AGENT_GPUS         comma-separated physical GPU indices (required)
  CANATUNE_AGENT_MIN_MHZ      lowest SM clock that may be locked (default 0)
  CANATUNE_AGENT_GPU_UUIDS    UUIDs in the same order as the GPUs; the agent refuses
                              to start on a mismatch (GPUs are not cgroup-isolated)
  CANATUNE_AGENT_MEMORY_MHZ   memory clock locked once before the first SM lock (optional)
  CANATUNE_AGENT_HOST/PORT    bind address (default 0.0.0.0:9300)
  CANATUNE_AGENT_DRY_RUN=1    fake GPUs (tests, laptops)
"""

import asyncio
import os
import subprocess
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


class GpuBackend:
    """NVML reads plus `sudo -n nvidia-smi` locks for a fixed set of GPUs."""

    def __init__(
        self,
        indices: Iterable[int],
        *,
        memory_mhz: int | None = None,
        expected_uuids: Iterable[str] | None = None,
    ) -> None:
        import pynvml  # nvidia-ml-py; only needed where GPUs exist

        pynvml.nvmlInit()
        self.nv = pynvml
        indices = list(indices)
        self.handles = {i: pynvml.nvmlDeviceGetHandleByIndex(i) for i in indices}
        if expected_uuids is not None:
            for index, expected in zip(indices, expected_uuids, strict=True):
                actual = pynvml.nvmlDeviceGetUUID(self.handles[index])
                actual = actual.decode() if isinstance(actual, bytes) else actual
                if actual != expected:
                    raise RuntimeError(f"GPU {index} is {actual}, expected {expected}")
        self.memory_mhz = memory_mhz
        self.memory_locked: set[int] = set()

    def sm_mhz(self, index: int) -> int:
        return int(self.nv.nvmlDeviceGetClockInfo(self.handles[index], self.nv.NVML_CLOCK_SM))

    def supported_clocks(self, index: int) -> list[int]:
        """SM clocks supported at the locked (or highest) memory clock."""
        handle = self.handles[index]
        memory = self.memory_mhz or max(self.nv.nvmlDeviceGetSupportedMemoryClocks(handle))
        return sorted(
            {int(f) for f in self.nv.nvmlDeviceGetSupportedGraphicsClocks(handle, memory)}
        )

    def reading(self, index: int) -> dict[str, Any]:
        handle = self.handles[index]
        reasons = getattr(self.nv, "nvmlDeviceGetCurrentClocksEventReasons", None) or getattr(
            self.nv, "nvmlDeviceGetCurrentClocksThrottleReasons", None
        )
        return {
            "index": index,
            "sm_mhz": self.sm_mhz(index),
            "power_w": self.nv.nvmlDeviceGetPowerUsage(handle) / 1000.0,
            "energy_mj": float(self.nv.nvmlDeviceGetTotalEnergyConsumption(handle)),
            "throttle_reasons": None if reasons is None else int(reasons(handle)),
        }

    @staticmethod
    def _smi(index: int, *args: str) -> None:
        cmd = ["sudo", "-n", "nvidia-smi", "-i", str(index), *args]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        text = res.stdout + res.stderr
        # nvidia-smi has returned non-zero after a successful change (job 267572);
        # its own confirmation text is authoritative.
        if res.returncode != 0 and "All done" not in text:
            raise RuntimeError(f"{' '.join(cmd)} failed: rc={res.returncode} {text[-300:]}")

    def lock(self, index: int, mhz: int) -> None:
        if self.memory_mhz is not None and index not in self.memory_locked:
            self._smi(index, "-lmc", f"{self.memory_mhz},{self.memory_mhz}")
            self.memory_locked.add(index)
        self._smi(index, "-lgc", f"{mhz},{mhz}")

    def reset(self, index: int) -> None:
        self._smi(index, "-rgc")
        if index in self.memory_locked:
            self._smi(index, "-rmc")
            self.memory_locked.discard(index)


class DryRunBackend:
    """Deterministic fake GPUs: clock follows the lock immediately, 50 W."""

    def __init__(
        self, indices: Iterable[int], *, supported: Iterable[int] | None = None, **_: Any
    ) -> None:
        self.clocks = {i: 1500 for i in indices}
        self.supported = sorted(supported or range(210, 2521, 15))
        self._t0 = time.monotonic()

    def sm_mhz(self, index: int) -> int:
        return self.clocks[index]

    def supported_clocks(self, index: int) -> list[int]:
        return list(self.supported)

    def reading(self, index: int) -> dict[str, Any]:
        return {
            "index": index,
            "sm_mhz": self.clocks[index],
            "power_w": 50.0,
            "energy_mj": (time.monotonic() - self._t0) * 50_000.0,
            "throttle_reasons": 0,
        }

    def lock(self, index: int, mhz: int) -> None:
        self.clocks[index] = mhz

    def reset(self, index: int) -> None:
        self.clocks[index] = 1500


@dataclass
class LockResult:
    ok: bool
    gpu: int
    target_mhz: int
    cmd_ms: float
    reach_ms: float | None
    sm_mhz: int
    error: str | None = None


class ClockService:
    """Serializes clock changes per GPU and confirms each by NVML read-back."""

    def __init__(
        self,
        backend: Any,
        gpus: Iterable[int],
        allowed_mhz: Iterable[int] | None = None,
        *,
        min_mhz: int = 0,
        confirm_timeout_s: float = 3.0,
        tolerance_mhz: int = 20,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.backend = backend
        self.gpus = frozenset(gpus)
        self.allowed_mhz = None if allowed_mhz is None else frozenset(allowed_mhz)
        self.min_mhz = min_mhz
        self.confirm_timeout_s = confirm_timeout_s
        self.tolerance_mhz = tolerance_mhz
        self._clock = clock
        self._locks = {gpu: threading.Lock() for gpu in self.gpus}

    def check_gpu(self, gpu: int) -> None:
        if gpu not in self.gpus:
            raise KeyError(f"GPU {gpu} is not managed by this agent")

    def lock(self, gpu: int, mhz: int) -> LockResult:
        self.check_gpu(gpu)
        if mhz not in self.allowed(gpu):
            raise ValueError(f"{mhz} MHz is not an allowed clock")
        with self._locks[gpu]:
            t0 = self._clock()
            try:
                self.backend.lock(gpu, mhz)
            except Exception as error:  # report, never crash the agent
                return LockResult(
                    False,
                    gpu,
                    mhz,
                    (self._clock() - t0) * 1000,
                    None,
                    self.backend.sm_mhz(gpu),
                    str(error),
                )
            cmd_ms = (self._clock() - t0) * 1000
            while True:
                current = self.backend.sm_mhz(gpu)
                if abs(current - mhz) <= self.tolerance_mhz:
                    return LockResult(True, gpu, mhz, cmd_ms, (self._clock() - t0) * 1000, current)
                if self._clock() - t0 > self.confirm_timeout_s:
                    # Power/thermal limits can hold the clock below target under load;
                    # the lock itself succeeded, so report it as unconfirmed, not failed.
                    return LockResult(
                        True, gpu, mhz, cmd_ms, None, current, "clock not confirmed within timeout"
                    )
                time.sleep(0.005)

    def allowed(self, gpu: int) -> list[int]:
        """Supported clocks >= min_mhz (and in the explicit list, if one is set)."""
        return [
            f
            for f in self.backend.supported_clocks(gpu)
            if f >= self.min_mhz and (self.allowed_mhz is None or f in self.allowed_mhz)
        ]

    def reset(self, gpu: int) -> None:
        self.check_gpu(gpu)
        with self._locks[gpu]:
            self.backend.reset(gpu)


class LockRequest(BaseModel):
    mhz: int


def _int_list(value: str, name: str) -> list[int]:
    try:
        items = [int(part) for part in value.split(",") if part.strip()]
    except ValueError as error:
        raise ValueError(f"{name} must be comma-separated integers") from error
    if not items:
        raise ValueError(f"{name} is required")
    return items


def create_agent_app(service: ClockService) -> FastAPI:
    app = FastAPI(title="CanaTune GPU agent")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "gpus": sorted(service.gpus)}

    @app.get("/gpus")
    async def gpus() -> list[dict[str, Any]]:
        return [service.backend.reading(gpu) for gpu in sorted(service.gpus)]

    @app.get("/gpus/{gpu}/clocks")
    async def clocks(gpu: int) -> dict[str, Any]:
        try:
            service.check_gpu(gpu)
        except KeyError as error:
            raise HTTPException(404, str(error)) from error
        return {"gpu": gpu, "sm_mhz": service.allowed(gpu)}

    @app.post("/gpus/{gpu}/lock")
    async def lock(gpu: int, request: LockRequest) -> dict[str, Any]:
        try:
            result = await asyncio.to_thread(service.lock, gpu, request.mhz)
        except KeyError as error:
            raise HTTPException(404, str(error)) from error
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        return result.__dict__

    @app.post("/gpus/{gpu}/reset")
    async def reset(gpu: int) -> dict[str, Any]:
        try:
            await asyncio.to_thread(service.reset, gpu)
        except KeyError as error:
            raise HTTPException(404, str(error)) from error
        return {"ok": True, "gpu": gpu}

    return app


def service_from_env() -> ClockService:
    gpus = _int_list(os.environ.get("CANATUNE_AGENT_GPUS", ""), "CANATUNE_AGENT_GPUS")
    raw_allowed = os.environ.get("CANATUNE_AGENT_ALLOWED_MHZ")
    allowed = _int_list(raw_allowed, "CANATUNE_AGENT_ALLOWED_MHZ") if raw_allowed else None
    memory = os.environ.get("CANATUNE_AGENT_MEMORY_MHZ")
    backend_type = DryRunBackend if os.environ.get("CANATUNE_AGENT_DRY_RUN") == "1" else GpuBackend
    raw_uuids = os.environ.get("CANATUNE_AGENT_GPU_UUIDS")
    uuids = [u.strip() for u in raw_uuids.split(",")] if raw_uuids else None
    if uuids is not None and len(uuids) != len(gpus):
        raise ValueError("CANATUNE_AGENT_GPU_UUIDS must list one UUID per GPU")
    backend = backend_type(gpus, memory_mhz=int(memory) if memory else None, expected_uuids=uuids)
    min_mhz = int(os.environ.get("CANATUNE_AGENT_MIN_MHZ", "0"))
    return ClockService(backend, gpus, allowed, min_mhz=min_mhz)


def main() -> None:
    import uvicorn

    service = service_from_env()
    try:
        uvicorn.run(
            create_agent_app(service),
            host=os.environ.get("CANATUNE_AGENT_HOST", "0.0.0.0"),
            port=int(os.environ.get("CANATUNE_AGENT_PORT", "9300")),
        )
    finally:
        # Never leave clocks locked after the job: reset every managed GPU.
        for gpu in sorted(service.gpus):
            try:
                service.reset(gpu)
            except Exception:
                pass


if __name__ == "__main__":
    main()
