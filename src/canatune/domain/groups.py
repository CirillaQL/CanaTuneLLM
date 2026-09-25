"""P/D groups, clock points and the published tier table.

Tiers are not configured: the Canary measures them on the running cluster and
publishes a `TierTable`. Until then every production group runs at MAX (the
highest clock the hardware supports) and the Router admits every request.
"""

import json
import math
import os
import tempfile
import threading
from collections import deque
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class Tier(StrEnum):
    """Clock tier of one group.

    MAX  cold start (no published table): highest supported clocks.
    H/L  working tiers measured by the Canary; L exists only when the low-load
         and target-load energy bands do not overlap.
    PARK idle clocks of a parked group.
    """

    PARK = "park"
    L = "L"
    H = "H"
    MAX = "max"


class GroupState(StrEnum):
    PARK = "park"
    ACTIVE = "active"
    DRAINING = "draining"
    EXPLORING = "exploring"  # Canary only: running probes, no production traffic


@dataclass(frozen=True, order=True)
class ClockPoint:
    """Locked SM clocks of one P/D pair. The risk table is keyed by clock point,
    so samples stay valid when a tier later moves to another clock."""

    prefill_mhz: int
    decode_mhz: int

    def key(self) -> str:
        return f"{self.prefill_mhz}/{self.decode_mhz}"

    @classmethod
    def from_key(cls, key: str) -> "ClockPoint":
        prefill, decode = key.split("/")
        return cls(int(prefill), int(decode))


def slower(a: ClockPoint, b: ClockPoint) -> ClockPoint:
    """The more conservative of two points while a clock change is in flight."""
    return ClockPoint(min(a.prefill_mhz, b.prefill_mhz), min(a.decode_mhz, b.decode_mhz))


SCHEMA_VERSION = 1


class TierTableError(ValueError):
    """The tier table file is invalid or belongs to another configuration."""


@dataclass
class TierTable:
    """Tiers and capacities published by the Canary (design v2 §6).

    Loads are in equivalent prompt tokens per second: sum over requests of
    (prompt tokens + alpha), where alpha is the per-request fixed prefill cost
    expressed in tokens.
    """

    park: ClockPoint
    h: ClockPoint
    capacity_h: float  # equivalent tokens/s per group at H (TTFT p95 <= target)
    alpha_tokens: float
    l: ClockPoint | None = None  # noqa: E741 - tier name
    capacity_l: float | None = None
    tau_up: float | None = None  # L -> H above this load (equivalent tokens/s)
    tau_down: float | None = None  # H -> L below this load
    decode_max_running: int | None = None  # B*: clean D concurrency (KV wall)
    decode_kv_limit: float | None = None  # admit only while D KV usage <= this
    decode_wall: bool = True  # False: D has a frequency step (reported, not used in v1)
    published_at: float = 0.0  # wall time
    evidence: dict[str, Any] = field(default_factory=dict)

    def clocks(self, tier: Tier, max_point: ClockPoint) -> ClockPoint:
        if tier is Tier.MAX:
            return max_point
        if tier is Tier.PARK:
            return self.park
        if tier is Tier.L:
            if self.l is None:
                raise KeyError("no L tier published")
            return self.l
        return self.h

    def capacity(self, tier: Tier) -> float:
        if tier is Tier.L and self.capacity_l is not None:
            return self.capacity_l
        return self.capacity_h

    def to_json(self) -> dict[str, Any]:
        raw = asdict(self)
        for key in ("park", "h", "l"):
            point = getattr(self, key)
            raw[key] = None if point is None else point.key()
        return raw

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> "TierTable":
        values = dict(raw)
        for key in ("park", "h", "l"):
            if values.get(key) is not None:
                values[key] = ClockPoint.from_key(values[key])
        return cls(**values)


class TierStore:
    """Atomic JSON persistence of the published table, guarded by the
    configuration identity (a table from another cluster is refused)."""

    def __init__(self, path: str | Path | None, identity: Mapping[str, Any]) -> None:
        self.path = None if path is None else Path(path)
        self.identity = dict(identity)
        self._lock = threading.Lock()

    def load(self) -> TierTable | None:
        if self.path is None or not self.path.exists():
            return None
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        if raw.get("schema_version") != SCHEMA_VERSION:
            raise TierTableError("unsupported tier table schema")
        if raw.get("identity") != self.identity:
            raise TierTableError(f"tier table {self.path} belongs to another configuration")
        return TierTable.from_json(raw["table"])

    def save(self, table: TierTable) -> None:
        if self.path is None:
            return
        payload = {
            "schema_version": SCHEMA_VERSION,
            "identity": self.identity,
            "table": table.to_json(),
        }
        text = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temp = tempfile.mkstemp(dir=self.path.parent, prefix=".tiers-")
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                    stream.write(text)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temp, self.path)
            except BaseException:
                Path(temp).unlink(missing_ok=True)
                raise


@dataclass
class TierState:
    """What the Router and Controller share: the highest clocks and the tier
    table currently in force (None = cold start: MAX, admit everything)."""

    max_point: ClockPoint
    table: TierTable | None = None

    @property
    def alpha_tokens(self) -> float:
        return 0.0 if self.table is None else self.table.alpha_tokens

    def clocks(self, tier: Tier) -> ClockPoint:
        if tier is Tier.MAX or self.table is None:
            return self.max_point
        return self.table.clocks(tier, self.max_point)


@dataclass
class Group:
    """Mutable runtime view of one fixed (P_i, D_i) pair.

    `tier` is the target set by the Controller; `effective` is the clock point
    the Router may assume. It only moves to the target once the agents confirm
    the new clocks, and is the slower of old and new while a change is in flight
    (None while parking/waking: the group is not routable then).
    """

    name: str
    prefill: str
    decode: str
    canary: bool = False
    state: GroupState = GroupState.ACTIVE
    tier: Tier = Tier.MAX
    effective: ClockPoint | None = None
    n_await: int = 0
    t_await: int = 0
    n_inflight: int = 0
    admissions: deque[tuple[float, int]] = field(default_factory=deque)

    @property
    def routable(self) -> bool:
        return self.state is GroupState.ACTIVE and self.effective is not None

    def record_admission(self, now: float, prompt_tokens: int, window_s: float) -> None:
        self.admissions.append((now, prompt_tokens))
        self.trim(now, window_s)

    def trim(self, now: float, window_s: float) -> None:
        while self.admissions and self.admissions[0][0] < now - window_s:
            self.admissions.popleft()

    def load(self, now: float, window_s: float, alpha_tokens: float) -> float:
        """Windowed load in equivalent prompt tokens/s: sum(prompt + alpha) / window."""
        if not math.isfinite(window_s) or window_s <= 0:
            raise ValueError("window_s must be positive")
        self.trim(now, window_s)
        tokens = sum(t for _, t in self.admissions)
        return (tokens + alpha_tokens * len(self.admissions)) / window_s

    def snapshot(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "prefill": self.prefill,
            "decode": self.decode,
            "canary": self.canary,
            "state": self.state.value,
            "tier": self.tier.value,
            "effective": None if self.effective is None else self.effective.key(),
            "n_await": self.n_await,
            "t_await": self.t_await,
            "n_inflight": self.n_inflight,
        }
