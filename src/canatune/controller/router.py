"""Per-request admission and group choice (never changes clocks).

Cold start (no tier table published): every request is admitted and spread to
the least-loaded active group; all production groups run at MAX meanwhile.

With a published table, for each routable group the Router checks
* hard limits: D concurrency below B*, D KV usage below the Canary's limit, and
  (admission `slack`) the KV bytes in flight to its D below a fraction of the
  connector's receive buffer (r7: the capacity cliff and the D crashes came from
  that buffer overflowing),
* the SLO risk: `slack` predicts the request's TTFT from the group's state (own
  prefill cost, pending P work, KV in flight, requests decoding) and looks up the
  violation risk of that predicted slack; `cells` (v1) looks up the risk table
  cell (clock point, queue, prompt bucket, D busy),
and admits to the most loaded feasible group (concentration keeps batches large).
Otherwise it waits briefly within the TTFT budget, then rejects (HTTP 503).
Reservations are taken synchronously inside the event loop, so two requests can
never both see the same free slot.
"""

import asyncio
import itertools
import math
import os
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from canatune.domain.admission import (
    PrefillCost,
    SlackRisk,
    TtftPredictor,
    kv_bytes_from_model_dir,
)
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
    admission: str = "slack"  # slack (state-based) or cells (v1 risk table)
    kv_bytes_per_token: float = 131072.0
    kv_buffer_bytes: float = 1e9  # the KV connector's receive buffer on D
    kv_inflight_fraction: float = 0.5  # r6b/r7: > 0.5 buffer in flight -> 58-67 % violated

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
            admission=_admission(raw),
            kv_bytes_per_token=float(kv_bytes_setting(config)),
            kv_buffer_bytes=float(config.get("kv_transfer", {}).get("kv_buffer_bytes", 1e9)),
            kv_inflight_fraction=_positive(raw, "kv_inflight_fraction", 0.5),
        )


def _admission(raw: Mapping[str, Any]) -> str:
    value = raw.get("admission", "slack")
    if value not in ("slack", "cells"):
        raise RouterConfigError("router.admission must be slack or cells")
    return value


def kv_bytes_setting(config: Mapping[str, Any]) -> float:
    """kv_transfer.kv_bytes_per_token, else from the model's config.json
    (MODEL_PATH / CANATUNE_MODEL_PATH), else Mistral-7B bf16."""
    value = config.get("kv_transfer", {}).get("kv_bytes_per_token")
    if value:
        return float(value)
    for env in ("CANATUNE_MODEL_PATH", "MODEL_PATH"):
        found = kv_bytes_from_model_dir(os.environ.get(env))
        if found:
            return float(found)
    return 131072.0


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
    stage: str = "prefill"  # prefill -> transfer (P returned) -> decode (first token)
    s_own_ms: float = 0.0
    features: tuple[float, ...] = ()
    predicted_ms: float | None = None


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
        predictor: TtftPredictor | None = None,
        slack: SlackRisk | None = None,
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
        self.predictor = predictor or TtftPredictor()
        self.slack = slack or SlackRisk(min_samples=table.min_samples)
        self._cost_key: object = None
        self._cost = PrefillCost()
        self._slack_by_group: dict[str, tuple[tuple[float, ...], float, Any]] = {}

    @property
    def open_admission(self) -> bool:
        """Cold start: no tier table yet, admit everything."""
        return self.tiers.table is None

    def prefill_cost(self) -> PrefillCost:
        """S(L) from the Canary's single-request length table in the published tier
        table (evidence.alpha_fit.prefill_ms_by_length); a generic curve before."""
        table = self.tiers.table
        key = None if table is None else table.published_at
        if key != self._cost_key:
            lut = None
            if table is not None:
                lut = (table.evidence.get("alpha_fit") or {}).get("prefill_ms_by_length")
            self._cost = PrefillCost({int(k): float(v) for k, v in (lut or {}).items()})
            self._cost_key = key
        return self._cost

    def _predict(self, group: Group, s_own_ms: float) -> tuple[tuple[float, ...], float]:
        x = self.predictor.features(
            s_own_ms, group.pending_ms, group.inflight_bytes, group.n_decoding
        )
        return x, self.predictor.predict(x)

    def _kv_blocked(self, group: Group) -> bool:
        s = self.settings
        return group.inflight_bytes >= s.kv_inflight_fraction * s.kv_buffer_bytes

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
        """Groups that may take this request now, with their risk cell (the slack
        estimate, when that decided, is kept in self._slack_by_group)."""
        out = []
        self._slack_by_group: dict[str, tuple[tuple[float, ...], float, Any]] = {}
        s_own = self.prefill_cost()(tokens)
        for group in self.groups:
            if not group.routable:
                continue
            assert group.effective is not None
            x, predicted = self._predict(group, s_own)
            if self.open_admission:
                d_busy = group.n_inflight > group.n_await
                cell = self.table.cell(group.effective, group.n_await, tokens, d_busy)
                self._slack_by_group[group.name] = (x, predicted, None)
                out.append((group, cell, None, d_busy))
                continue
            d_busy, blocked = self._decode_state(group)
            if blocked is not None:
                continue
            cell = self.table.cell(group.effective, group.n_await, tokens, d_busy)
            if self.settings.admission == "slack":
                if self._kv_blocked(group):
                    continue
                slack = self.slack.estimate(self.settings.ttft_slo_ms - predicted)
                self._slack_by_group[group.name] = (x, predicted, slack)
                if slack.risk <= self.settings.theta:
                    estimate = RiskEstimate(slack.risk, f"slack_{slack.source}", slack.samples)
                    out.append((group, cell, estimate, d_busy))
                continue
            estimate = self.table.lookup(group.effective, group.n_await, tokens, d_busy)
            self._slack_by_group[group.name] = (x, predicted, None)
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
        x, predicted, _ = self._slack_by_group.get(group.name, ((), None, None))
        s_own = self.prefill_cost()(tokens)
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
                "pending_ms": round(group.pending_ms, 1),
                "inflight_mb": round(group.inflight_bytes / 1e6, 1),
                "n_decoding": group.n_decoding,
                "predicted_ms": None if predicted is None else round(predicted, 1),
            },
            s_own_ms=s_own,
            features=tuple(x),
            predicted_ms=predicted,
        )
        group.n_at_p += 1
        group.pending_ms += s_own
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

    def _kv_bytes(self, ticket: Ticket) -> float:
        return ticket.prompt_tokens * self.settings.kv_bytes_per_token

    def _leave_stage(self, ticket: Ticket) -> None:
        group = ticket.group
        if ticket.stage == "prefill":
            group.n_at_p -= 1
            group.pending_ms = max(0.0, group.pending_ms - ticket.s_own_ms)
        elif ticket.stage == "transfer":
            group.inflight_bytes = max(0.0, group.inflight_bytes - self._kv_bytes(ticket))
        elif ticket.stage == "decode":
            group.n_decoding -= 1

    def prefill_done(self, ticket: Ticket) -> None:
        """P returned: its KV now travels to (or waits in the buffer of) the D."""
        if ticket.finished or ticket.stage != "prefill":
            return
        self._leave_stage(ticket)
        ticket.stage = "transfer"
        ticket.group.inflight_bytes += self._kv_bytes(ticket)

    def first_token(self, ticket: Ticket) -> None:
        if ticket.first_token_at is not None or ticket.finished:
            return
        ticket.first_token_at = self._clock()
        ticket.group.n_await -= 1
        ticket.group.t_await -= ticket.prompt_tokens
        self._leave_stage(ticket)
        ticket.stage = "decode"
        ticket.group.n_decoding += 1

    def finish(
        self,
        ticket: Ticket,
        *,
        status: str,
        ttft_ms: float | None,
        tpot_ms: float | None,
        output_tokens: int,
        timing: Mapping[str, float | None] | None = None,
    ) -> None:
        if ticket.finished:
            return
        if ticket.first_token_at is None:
            ticket.group.n_await -= 1
            ticket.group.t_await -= ticket.prompt_tokens
        self._leave_stage(ticket)
        ticket.stage = "done"
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
            if ticket.features and ticket.predicted_ms is not None and ttft_ms is not None:
                # Slack as predicted at admission (the model in force then), then refit.
                self.slack.record(self.settings.ttft_slo_ms - ticket.predicted_ms, bool(violated))
                self.predictor.record(ticket.features, ttft_ms)
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
                "predicted_ms": ticket.predicted_ms,
                "timing": None if timing is None else dict(timing),
            }
        )

    def state(self) -> dict[str, Any]:
        return {
            "admission": self.settings.admission,
            "predictor": self.predictor.state(),
            "slack": self.slack.to_json(),
            "open_admission": self.open_admission,
            "admitted": self.admitted,
            "rejections": self.rejections,
            "groups": [g.snapshot() for g in self.groups],
        }
