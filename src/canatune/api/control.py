"""State and Canary control endpoints under `/canatune`."""

from typing import TYPE_CHECKING, Any

from fastapi import APIRouter
from pydantic import BaseModel

if TYPE_CHECKING:
    from canatune.service import Runtime


class CanaryRequest(BaseModel):
    action: str  # "full", "relocate", "recheck" or "abort"


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
        if request.action in ("full", "relocate", "recheck"):
            ok, reason = runtime.canary.request(request.action, "api")
            return {"ok": ok, "reason": reason}
        return {"ok": False, "reason": "action must be full, relocate, recheck or abort"}

    return router
