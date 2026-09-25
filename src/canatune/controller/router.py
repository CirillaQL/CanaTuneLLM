"""Per-request admission and group choice (never changes clocks).

Cold start (no tier table published): every request is admitted and spread to
the least-loaded active group; all production groups run at MAX meanwhile.

With a published table, for each routable group the Router checks
* the online risk table at the group's effective clock point (risk <= theta),
* the D-side KV wall: D concurrency below B* and KV usage below the limit the
  Canary measured,
and admits to the most loaded feasible group (concentration keeps batches large).
Otherwise it waits briefly within the TTFT budget, then rejects (HTTP 503).
Reservations are taken synchronously inside the event loop, so two requests can
never both see the same free slot.
"""

import asyncio
import itertools
import math
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from canatune.domain.groups import Group, TierState
from canatune.domain.load import LengthStats
from canatune.domain.risk import Cell, RiskEstimate, RiskTable, is_violation
from canatune.infrastructure.records import JsonlLog
from canatune.infrastructure.telemetry import Telemetry


class RouterConfigError(ValueError):
    """Router settings are invalid."""


def _positive(raw: Mapping[str, Any], key: str, default: float) -> float:
    value = raw.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0:
        raise RouterConfigError(f"router.{key} must be positive")
    return float(value)


@dataclass(frozen=True)
class RouterSettings:
    theta: float
    ttft_slo_ms: float
    tpot_slo_ms: float
    max_wait_ms: float
    retry_period_ms: float
    snapshot_max_age_s: float
    d_busy_min_running: int
    load_window_s: float
    chars_per_token: float

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "RouterSettings":
        raw = config.get("router", {})
        slo = config["experiment"]["slo"]
        controller = config.get("controller", {})
        theta = raw.get("theta", 0.1)
        if isinstance(theta, bool) or not isinstance(theta, (int, float)) or not 0 <= theta < 1:
            raise RouterConfigError("router.theta must be in [0, 1)")
        return cls(
            theta=float(theta),
            ttft_slo_ms=float(slo["ttft_ms"]),
            tpot_slo_ms=float(slo["tpot_ms"]),
            max_wait_ms=_positive(raw, "max_wait_ms", 100),
            retry_period_ms=_positive(raw, "retry_period_ms", 20),
            snapshot_max_age_s=_positive(raw, "snapshot_max_age_s", 1.0),
            d_busy_min_running=int(raw.get("d_busy_min_running", 1)),
            load_window_s=float(controller.get("load_window_s", 10.0)),
            chars_per_token=_positive(raw, "chars_per_token", 4.0),
        )


def prompt_tokens(body: Mapping[str, Any], chars_per_token: float) -> tuple[int, bool]:
    """Prompt length in tokens and whether it is exact. Token-id prompts are exact;
    text prompts are estimated (the Router does not tokenize)."""
    prompt = body.get("prompt")
    if isinstance(prompt, list) and prompt and all(isinstance(t, int) for t in prompt):
        return len(prompt), True
    if isinstance(prompt, str):
        return max(1, math.ceil(len(prompt) / chars_per_token)), False
    if isinstance(prompt, list) and prompt and all(isinstance(t, str) for t in prompt):
        return max(1, math.ceil(sum(map(len, prompt)) / chars_per_token)), False
    raise ValueError("prompt must be a string or a list of token ids")


@dataclass
class Ticket:
    """One admitted request: its reservation and the admission snapshot."""

    id: int
    group: Group
    cell: Cell
    estimate: RiskEstimate | None  # None: cold-start (open) admission
    prompt_tokens: int
    prompt_exact: bool
    admitted_at: float
    wait_ms: float
    clock_epoch: int
    snapshot: dict[str, Any] = field(default_factory=dict)
    first_token_at: float | None = None
    finished: bool = False


class CanaTuneRouter:
    def __init__(
        self,
        groups: Sequence[Group],
        table: RiskTable,
        settings: RouterSettings,
        tiers: TierState,
        *,
        lengths: LengthStats | None = None,
        telemetry: Telemetry | None = None,
        log: JsonlLog | None = None,
        clock: Callable[[], float] = time.monotonic,
        clock_epochs: Mapping[str, int] | None = None,
    ) -> None:
        self.groups = list(groups)
        self.table = table
        self.settings = settings
        self.tiers = tiers
        self.lengths = lengths
        self.telemetry = telemetry
        self.log = log or JsonlLog(None)
        self._clock = clock
        self._ids = itertools.count(1)
        self.clock_epochs: dict[str, int] = dict(clock_epochs or {g.name: 0 for g in groups})
        self.rejections = 0
        self.admitted = 0
        # SLO outcomes of risk-admitted production requests (drift detection).
        self.outcomes: deque[bool] = deque(maxlen=500)

    @property
    def open_admission(self) -> bool:
        """Cold start: no tier table yet, admit everything."""
        return self.tiers.table is None

    # ---- state ---------------------------------------------------------------------

    def _decode_state(self, group: Group) -> tuple[bool, str | None]:
        """(D busy, reason D cannot take one more request or None)."""
        decoding = group.n_inflight - group.n_await
        running = waiting = 0.0
        kv = None
        if self.telemetry is not None:
            snapshot = self.telemetry.fresh(group.decode, self.settings.snapshot_max_age_s)
            if snapshot is None:
                return False, "decode_snapshot_stale"
            running = snapshot.running or 0.0
            waiting = snapshot.waiting or 0.0
            kv = snapshot.kv_usage
        busy = max(running, decoding) >= self.settings.d_busy_min_running
        table = self.tiers.table
        if table is not None and table.decode_max_running is not None:
            # Every admitted request of this group ends up decoding on its D.
            if max(running + waiting, group.n_inflight) + 1 > table.decode_max_running:
                return busy, "decode_concurrency_wall"
        if table is not None and table.decode_kv_limit is not None and kv is not None:
            if kv > table.decode_kv_limit:
                return busy, "decode_kv_wall"
        return busy, None

    def candidates(self, tokens: int) -> list[tuple[Group, Cell, RiskEstimate | None, bool]]:
        """Groups that may take this request now, with their risk cell."""
        out = []
        for group in self.groups:
            if not group.routable:
                continue
            assert group.effective is not None
            if self.open_admission:
                d_busy = group.n_inflight > group.n_await
                cell = self.table.cell(group.effective, group.n_await, tokens, d_busy)
                out.append((group, cell, None, d_busy))
                continue
            d_busy, blocked = self._decode_state(group)
            if blocked is not None:
                continue
            cell = self.table.cell(group.effective, group.n_await, tokens, d_busy)
            estimate = self.table.lookup(group.effective, group.n_await, tokens, d_busy)
            if estimate.risk <= self.settings.theta:
                out.append((group, cell, estimate, d_busy))
        return out

    def try_admit(self, tokens: int, exact: bool, waited_ms: float = 0.0) -> Ticket | None:
        feasible = self.candidates(tokens)
        if not feasible:
            return None
        if self.open_admission:
            # Spread: least outstanding prompt work first, production before Canary.
            group, cell, estimate, d_busy = min(
                feasible, key=lambda item: (item[0].t_await, item[0].n_inflight, item[0].canary)
            )
        else:
            # Most loaded feasible group; production before Canary on ties.
            group, cell, estimate, d_busy = max(
                feasible,
                key=lambda item: (
                    item[0].n_inflight,
                    not item[0].canary,
                    -self.groups.index(item[0]),
                ),
            )
        now = self._clock()
        ticket = Ticket(
            id=next(self._ids),
            group=group,
            cell=cell,
            estimate=estimate,
            prompt_tokens=tokens,
            prompt_exact=exact,
            admitted_at=now,
            wait_ms=waited_ms,
            clock_epoch=self.clock_epochs.get(group.name, 0),
            snapshot={
                "n_await": group.n_await,
                "t_await": group.t_await,
                "n_inflight": group.n_inflight,
                "d_busy": d_busy,
                "tier": group.tier.value,
                "clock": None if group.effective is None else group.effective.key(),
                "open_admission": self.open_admission,
            },
        )
        group.n_await += 1
        group.t_await += tokens
        group.n_inflight += 1
        group.record_admission(now, tokens, self.settings.load_window_s)
        self.admitted += 1
        return ticket

    async def admit(self, tokens: int, exact: bool) -> Ticket | None:
        """Admit now, or retry until the wait budget is spent, then reject.
        The budget runs on real time because the retries really sleep."""
        start = time.monotonic()
        budget_s = self.settings.max_wait_ms / 1000.0
        retry_s = self.settings.retry_period_ms / 1000.0
        while True:
            waited_ms = (time.monotonic() - start) * 1000.0
            ticket = self.try_admit(tokens, exact, waited_ms)
            if ticket is not None:
                return ticket
            if time.monotonic() - start + retry_s > budget_s:
                self.rejections += 1
                self.log.write({"event": "reject", "prompt_tokens": tokens, "waited_ms": waited_ms})
                return None
            await asyncio.sleep(retry_s)

    # ---- request lifecycle -----------------------------------------------------------

    def first_token(self, ticket: Ticket) -> None:
        if ticket.first_token_at is not None or ticket.finished:
            return
        ticket.first_token_at = self._clock()
        ticket.group.n_await -= 1
        ticket.group.t_await -= ticket.prompt_tokens

    def finish(
        self,
        ticket: Ticket,
        *,
        status: str,
        ttft_ms: float | None,
        tpot_ms: float | None,
        output_tokens: int,
    ) -> None:
        if ticket.finished:
            return
        if ticket.first_token_at is None:
            ticket.group.n_await -= 1
            ticket.group.t_await -= ticket.prompt_tokens
        ticket.group.n_inflight -= 1
        ticket.finished = True

        violated = is_violation(
            ttft_ms,
            tpot_ms,
            ttft_slo_ms=self.settings.ttft_slo_ms,
            tpot_slo_ms=self.settings.tpot_slo_ms,
        )
        # Only clean samples update the table: served OK, TTFT known, exact prompt
        # length, and no clock change on this group while the request was in flight.
        skip = None
        if status != "ok":
            skip = "request_failed"
        elif violated is None:
            skip = "ttft_unknown"
        elif not ticket.prompt_exact:
            skip = "prompt_length_estimated"
        elif self.clock_epochs.get(ticket.group.name, 0) != ticket.clock_epoch:
            skip = "clock_changed"
        if skip is None:
            self.table.record(ticket.cell, bool(violated))
            if ticket.estimate is not None:
                self.outcomes.append(bool(violated))
        if status == "ok" and self.lengths is not None:
            self.lengths.record(ticket.prompt_tokens, output_tokens)
        self.log.write(
            {
                "event": "request",
                "id": ticket.id,
                "group": ticket.group.name,
                "prefill": ticket.group.prefill,
                "decode": ticket.group.decode,
                "cell": ticket.cell.key(),
                "risk": None if ticket.estimate is None else ticket.estimate.risk,
                "risk_source": None if ticket.estimate is None else ticket.estimate.source,
                "prompt_tokens": ticket.prompt_tokens,
                "prompt_exact": ticket.prompt_exact,
                "wait_ms": ticket.wait_ms,
                "snapshot": ticket.snapshot,
                "status": status,
                "ttft_ms": ttft_ms,
                "tpot_ms": tpot_ms,
                "output_tokens": output_tokens,
                "violated": violated,
                "table_skip": skip,
            }
        )

    def state(self) -> dict[str, Any]:
        return {
            "open_admission": self.open_admission,
            "admitted": self.admitted,
            "rejections": self.rejections,
            "groups": [g.snapshot() for g in self.groups],
        }
