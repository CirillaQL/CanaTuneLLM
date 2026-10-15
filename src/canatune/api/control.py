"""State and Canary control endpoints under `/canatune`."""

from typing import TYPE_CHECKING, Any

from fastapi import APIRouter
from pydantic import BaseModel

from canatune.domain.groups import ClockPoint

if TYPE_CHECKING:
    from canatune.service import Runtime


class CanaryRequest(BaseModel):
    action: str  # "full", "relocate", "recheck", "verify" or "abort"


class WorkingRequest(BaseModel):
    prefill_mhz: int | None = None  # both None: back to the table's H
    decode_mhz: int | None = None


def create_control_router(runtime: "Runtime") -> APIRouter:
    router = APIRouter(prefix="/canatune")

    @router.get("/state")
    async def state() -> dict[str, Any]:
        return {
            "router": runtime.router.state(),
            "controller": runtime.controller.state(),
            "canary": runtime.canary.state(),
            "lengths": runtime.lengths.summary().__dict__,
        }

    @router.get("/tiers")
    async def tiers() -> dict[str, Any]:
        table = runtime.tiers.table
        return {
            "max_point": runtime.tiers.max_point.key(),
            "table": None if table is None else table.to_json(),
        }

    @router.get("/risk")
    async def risk() -> dict[str, Any]:
        return {
            "table": runtime.router.table.to_json(),
            "exploration_queue": runtime.router.table.exploration_candidates(
                runtime.router.settings.theta
            )[:50],
        }

    @router.post("/canary")
    async def canary(request: CanaryRequest) -> dict[str, Any]:
        """Start an experiment now (start conditions still apply) or abort one."""
        if request.action == "abort":
            runtime.canary.abort("api")
            return {"ok": True}
        if request.action in ("full", "relocate", "recheck", "verify"):
            ok, reason = runtime.canary.request(request.action, "api")
            return {"ok": ok, "reason": reason}
        return {"ok": False, "reason": "action must be full, relocate, recheck, verify or abort"}

    @router.post("/working")
    async def working(request: WorkingRequest) -> dict[str, Any]:
        """Lock tier H to these clocks (snapped to supported ones) on every group at H;
        an evaluation hook for clock grids, see TierController.set_working."""
        if (request.prefill_mhz is None) != (request.decode_mhz is None):
            return {"ok": False, "reason": "give both clocks, or neither to clear"}
        if runtime.tiers.table is None:
            return {"ok": False, "reason": "no tier table yet"}
        point = (
            None
            if request.prefill_mhz is None
            else ClockPoint(int(request.prefill_mhz), int(request.decode_mhz))
        )
        try:
            clock = await runtime.controller.set_working(point, "api")
        except ValueError as error:
            return {"ok": False, "reason": str(error)}
        return {"ok": True, "clock": clock.key()}

    return router
