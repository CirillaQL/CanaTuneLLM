"""Online risk table R(clock point, N_await, L_in, D busy) for per-request admission.

A cell's risk is the fraction of admitted requests in that state that violated
the SLO. Cells are keyed by the locked clock point ("1815/1050"), not by tier
name, so Canary probe windows at any clock and production traffic at any tier
all land in the same table. The table is monotone in N_await and L_in (K1b), which gives safe
bounds for sparse cells:

* a heavier cell (more waiting requests, longer prompt) has risk >= this cell,
  so the smallest well-sampled heavier risk is a conservative estimate;
* a lighter cell has risk <= this cell, so the largest well-sampled lighter risk
  is a lower bound that the estimate never goes below.

A cell with neither its own samples nor a heavier bound is unknown and treated
as unsafe (risk 1.0).
"""

import bisect
import json
import math
import os
import tempfile
import threading
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from canatune.domain.groups import ClockPoint

SCHEMA_VERSION = 1
UNSAFE = 1.0


class RiskTableError(ValueError):
    """The risk table configuration or file is invalid."""


def _edges(value: Any, name: str) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in value)
        or list(value) != sorted(set(value))
    ):
        raise RiskTableError(f"{name} must be a nonempty strictly increasing list of integers")
    return tuple(value)


@dataclass(frozen=True)
class Buckets:
    """Upper bucket edges. A value maps to the first edge >= value (upper edge =
    conservative); values above the last edge fall into an overflow bucket."""

    n_await: tuple[int, ...]
    prompt_tokens: tuple[int, ...]

    @staticmethod
    def _index(edges: Sequence[int], value: int) -> int:
        return bisect.bisect_left(edges, value)

    def n_await_bucket(self, value: int) -> int:
        return self._index(self.n_await, value)

    def prompt_bucket(self, value: int) -> int:
        return self._index(self.prompt_tokens, value)

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.n_await) + 1, len(self.prompt_tokens) + 1


@dataclass(frozen=True)
class Cell:
    clock: str  # ClockPoint.key()
    n_await: int  # bucket index
    prompt: int  # bucket index
    d_busy: bool

    def key(self) -> str:
        return f"{self.clock}|{self.n_await}|{self.prompt}|{int(self.d_busy)}"

    @classmethod
    def from_key(cls, key: str) -> "Cell":
        clock, n_await, prompt, d_busy = key.split("|")
        return cls(clock, int(n_await), int(prompt), d_busy == "1")


@dataclass
class CellStats:
    admitted: int = 0
    violated: int = 0
    seed_risk: float | None = None  # prior from an offline-measured seed (K1b); not a model

    @property
    def observed(self) -> float | None:
        return self.violated / self.admitted if self.admitted else None


@dataclass(frozen=True)
class RiskEstimate:
    risk: float
    source: str  # "observed", "seed", "heavier_bound", "unknown"
    samples: int


class RiskTable:
    """Thread-safe risk table with optional atomic JSON persistence."""

    def __init__(
        self,
        buckets: Buckets,
        *,
        min_samples: int,
        identity: Mapping[str, Any] | None = None,
        path: str | Path | None = None,
    ) -> None:
        if isinstance(min_samples, bool) or not isinstance(min_samples, int) or min_samples < 1:
            raise RiskTableError("min_samples must be a positive integer")
        self.buckets = buckets
        self.min_samples = min_samples
        self.identity = dict(identity or {})
        self.path = None if path is None else Path(path)
        self._cells: dict[Cell, CellStats] = {}
        self._visits: dict[Cell, int] = {}
        self._lock = threading.Lock()

    # ---- construction ---------------------------------------------------

    @classmethod
    def from_config(cls, raw: Mapping[str, Any], identity: Mapping[str, Any]) -> "RiskTable":
        buckets = Buckets(
            n_await=_edges(raw.get("n_await_edges"), "risk.n_await_edges"),
            prompt_tokens=_edges(raw.get("prompt_token_edges"), "risk.prompt_token_edges"),
        )
        table = cls(
            buckets,
            min_samples=raw.get("min_samples", 20),
            identity=identity,
            path=raw.get("path"),
        )
        seed = raw.get("seed") or {}
        if not isinstance(seed, Mapping):
            raise RiskTableError("risk.seed must be a mapping")
        for clock_key, rows in seed.items():
            table.load_seed(ClockPoint.from_key(str(clock_key)), rows)
        if table.path is not None and table.path.exists():
            table.load()
        return table

    def load_seed(self, clock: ClockPoint, rows: Any) -> None:
        """Seed rows are indexed [n_await bucket][prompt bucket]; applied to both D states."""
        n_rows, n_cols = self.buckets.shape
        if (
            not isinstance(rows, list)
            or len(rows) != n_rows
            or any(not isinstance(row, list) or len(row) != n_cols for row in rows)
        ):
            raise RiskTableError(f"seed for {clock.key()} must be a {n_rows}x{n_cols} matrix")
        with self._lock:
            for i, row in enumerate(rows):
                for j, value in enumerate(row):
                    if value is None:
                        continue
                    if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
                        raise RiskTableError("seed risks must be in [0, 1] or null")
                    for d_busy in (False, True):
                        cell = Cell(clock.key(), i, j, d_busy)
                        stats = self._cells.setdefault(cell, CellStats())
                        stats.seed_risk = float(value)

    # ---- lookups ----------------------------------------------------------

    def cell(self, clock: ClockPoint, n_await: int, prompt_tokens: int, d_busy: bool) -> Cell:
        return Cell(
            clock.key(),
            self.buckets.n_await_bucket(n_await),
            self.buckets.prompt_bucket(prompt_tokens),
            d_busy,
        )

    def _own(self, cell: Cell) -> RiskEstimate | None:
        stats = self._cells.get(cell)
        if stats is None:
            return None
        if stats.admitted >= self.min_samples:
            return RiskEstimate(stats.observed or 0.0, "observed", stats.admitted)
        if stats.seed_risk is not None:
            # Sparse observations can only raise a seed, never lower it.
            observed = stats.observed
            risk = stats.seed_risk if observed is None else max(stats.seed_risk, observed)
            return RiskEstimate(risk, "seed", stats.admitted)
        return None

    def estimate(self, cell: Cell) -> RiskEstimate:
        with self._lock:
            return self._estimate(cell)

    def _estimate(self, cell: Cell) -> RiskEstimate:
        own = self._own(cell)
        n_rows, n_cols = self.buckets.shape
        lower = 0.0
        for i in range(cell.n_await + 1):
            for j in range(cell.prompt + 1):
                if (i, j) == (cell.n_await, cell.prompt):
                    continue
                other = self._own(Cell(cell.clock, i, j, cell.d_busy))
                if other is not None:
                    lower = max(lower, other.risk)
        if own is not None:
            return RiskEstimate(max(own.risk, lower), own.source, own.samples)
        upper = None
        for i in range(cell.n_await, n_rows):
            for j in range(cell.prompt, n_cols):
                if (i, j) == (cell.n_await, cell.prompt):
                    continue
                other = self._own(Cell(cell.clock, i, j, cell.d_busy))
                if other is not None:
                    upper = other.risk if upper is None else min(upper, other.risk)
        if upper is None:
            return RiskEstimate(UNSAFE, "unknown", 0)
        return RiskEstimate(max(upper, lower), "heavier_bound", 0)

    def lookup(
        self, clock: ClockPoint, n_await: int, prompt_tokens: int, d_busy: bool
    ) -> RiskEstimate:
        cell = self.cell(clock, n_await, prompt_tokens, d_busy)
        with self._lock:
            self._visits[cell] = self._visits.get(cell, 0) + 1
            return self._estimate(cell)

    def clock_usable(self, clock: ClockPoint, theta: float) -> bool:
        """A clock point is usable once at least one of its cells is well sampled
        and safe; until then the Controller keeps groups off it."""
        with self._lock:
            return any(
                cell.clock == clock.key()
                and stats.admitted >= self.min_samples
                and (stats.observed or 0.0) <= theta
                for cell, stats in self._cells.items()
            )

    # ---- updates --------------------------------------------------------------

    def record(self, cell: Cell, violated: bool) -> None:
        with self._lock:
            stats = self._cells.setdefault(cell, CellStats())
            stats.admitted += 1
            stats.violated += int(violated)

    def exploration_candidates(self, theta: float) -> list[dict[str, Any]]:
        """Visited cells whose estimate is not backed by enough own samples,
        most-visited first: the Canary's exploration queue."""
        with self._lock:
            out = []
            for cell, visits in self._visits.items():
                stats = self._cells.get(cell)
                if stats is not None and stats.admitted >= self.min_samples:
                    continue
                estimate = self._estimate(cell)
                out.append(
                    {
                        "cell": cell.key(),
                        "visits": visits,
                        "samples": 0 if stats is None else stats.admitted,
                        "risk": estimate.risk,
                        "source": estimate.source,
                        "near_boundary": abs(estimate.risk - theta) <= 0.1,
                    }
                )
        out.sort(key=lambda item: (-item["visits"], not item["near_boundary"]))
        return out

    # ---- persistence -------------------------------------------------------------

    def to_json(self) -> dict[str, Any]:
        with self._lock:
            return {
                "schema_version": SCHEMA_VERSION,
                "identity": self.identity,
                "buckets": {
                    "n_await_edges": list(self.buckets.n_await),
                    "prompt_token_edges": list(self.buckets.prompt_tokens),
                },
                "cells": {
                    cell.key(): {
                        "admitted": s.admitted,
                        "violated": s.violated,
                        "seed_risk": s.seed_risk,
                    }
                    for cell, s in sorted(self._cells.items(), key=lambda kv: kv[0].key())
                },
                "visits": {cell.key(): n for cell, n in self._visits.items()},
            }

    def save(self) -> None:
        if self.path is None:
            return
        payload = json.dumps(self.to_json(), indent=2, sort_keys=True) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temp = tempfile.mkstemp(dir=self.path.parent, prefix=".risk-", suffix=".json")
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, self.path)
        except BaseException:
            Path(temp).unlink(missing_ok=True)
            raise

    def load(self) -> None:
        """Merge observed counts from the file. A table measured under another
        configuration identity (model, engine, connector...) is refused."""
        assert self.path is not None
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        if raw.get("schema_version") != SCHEMA_VERSION:
            raise RiskTableError("unsupported risk table schema")
        if raw.get("identity") != self.identity:
            raise RiskTableError(
                f"risk table {self.path} belongs to another configuration: {raw.get('identity')}"
            )
        buckets = raw.get("buckets", {})
        if (
            tuple(buckets.get("n_await_edges", ())) != self.buckets.n_await
            or tuple(buckets.get("prompt_token_edges", ())) != self.buckets.prompt_tokens
        ):
            raise RiskTableError("risk table buckets differ from the configuration")
        with self._lock:
            for key, value in raw.get("cells", {}).items():
                stats = self._cells.setdefault(Cell.from_key(key), CellStats())
                stats.admitted = int(value["admitted"])
                stats.violated = int(value["violated"])
            for key, visits in raw.get("visits", {}).items():
                self._visits[Cell.from_key(key)] = int(visits)

    def cells(self) -> Iterable[tuple[Cell, CellStats]]:
        with self._lock:
            return list(self._cells.items())


def is_violation(
    ttft_ms: float | None, tpot_ms: float | None, *, ttft_slo_ms: float, tpot_slo_ms: float
) -> bool | None:
    """SLO outcome of one request; None when TTFT is unknown (not recorded)."""
    if ttft_ms is None or not math.isfinite(ttft_ms):
        return None
    if ttft_ms >= ttft_slo_ms:
        return True
    return tpot_ms is not None and tpot_ms > tpot_slo_ms
