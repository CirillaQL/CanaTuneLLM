"""Per-request admission and group choice (never changes clocks).

Cold start (no tier table published): every request is admitted and spread to
the least-loaded active group; all production groups run at MAX meanwhile.

With a published table, for each routable group the Router checks
* hard guards: fresh D telemetry, D KV usage below the Canary's limit, and
  (admission `slack`) the KV bytes in flight to its D below a fraction of the
  connector's receive buffer (the gate the Canary measured; overflowing that
  buffer stalls P and can crash D),
* the SLO risk: `slack` predicts the request's TTFT from the group's state (own
  prefill cost, pending P work, KV in flight, requests decoding) and looks up the
  violation risk of that predicted slack; `cells` (v1) looks up the risk table
  cell (clock point, queue, prompt bucket, D busy),
and admits to the most loaded feasible group (concentration keeps batches large).
B* is a clean-window SLO bound, not a hardware concurrency limit: best-effort
uses it only to warn, counting D work rather than requests still pre-filling.
Legacy reject/serve policies retain their original calibrated admission ceiling.
With `best_effort` (default), prediction warns but does not force a clock change.
Actual production feedback confirms pressure, measured capacity guides expansion,
and only persistent/severe pressure reclaims Canary and locks MAX. Requests
dispatch FIFO whenever physical capacity is free, even if predicted unsafe or late.
Only the service wait timeout (`hold_max_ms`) rejects them. Physical KV and
concurrency limits remain enforced. Legacy `reject` and `serve` policies remain
available for comparisons; `serve` uses rescue and deadline/backfill priority.
The predicted TTFT includes the time already spent waiting at the proxy (TTFT
counts from arrival); the predictor learns the part after dispatch.
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
    CostBook,
    PrefillCost,
    SlackRisk,
    TtftPredictor,
    kv_bytes_from_model_dir,
    slack_edges,
)
from canatune.domain.groups import Group, TierState
from canatune.domain.load import LengthStats
from canatune.domain.models import interpolate
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
    overload: str = "best_effort"  # risk triggers resources; queue only at hard limits
    doomed: str = "backfill"  # serve: backfill (wait for spare capacity) or dispatch
    backfill_slack_ms: float | None = None  # backfill only where a fresh request keeps this
    # slack; None (auto): the lowest slack whose observed risk is <= theta / 2
    hold_max_ms: float = 30000.0  # serve: client timeout (policy): 503 after this long

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
            overload=_choice(raw, "overload", ("reject", "serve", "best_effort"), "best_effort"),
            hold_max_ms=_positive(raw, "hold_max_ms", 30000),
            doomed=_choice(raw, "doomed", ("backfill", "dispatch"), "backfill"),
            backfill_slack_ms=_auto_ms(raw, "backfill_slack_ms"),
        )


def _admission(raw: Mapping[str, Any]) -> str:
    return _choice(raw, "admission", ("slack", "cells"), "slack")


def _auto_ms(raw: Mapping[str, Any], key: str) -> float | None:
    value = raw.get(key, "auto")
    if value in (None, "auto"):
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise RouterConfigError(f"router.{key} must be auto or a non-negative number")
    return float(value)


def _choice(raw: Mapping[str, Any], key: str, allowed: Sequence[str], default: str) -> str:
    value = raw.get(key, default)
    if value not in allowed:
        raise RouterConfigError(f"router.{key} must be one of {', '.join(allowed)}")
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
    last_token_at: float | None = None
    finished: bool = False
    stage: str = "prefill"  # prefill -> transfer (P returned) -> decode (first token)
    s_own_ms: float = 0.0
    features: tuple[float, ...] = ()
    predicted_ms: float | None = None
    overflow: str | None = None  # serve: "rescue", "doomed" or "backfill" (late, spare slot)
    slack_ms: float | None = None  # SLO - waited - predicted, at admission


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
        self.overflows = {"rescue": 0, "doomed": 0, "backfill": 0, "best_effort": 0}
        self._holding: deque[int] = deque()  # serve: waiters past max_wait, in arrival order
        self._held = 0  # waiters that ever entered the hold queue
        self._fresh: set[int] = set()  # waiters still within max_wait
        self._expires: dict[int, float] = {}  # holder -> clock time its SLO is lost
        self._idle_ttft: dict[int, float] = {}  # Canary: TTFT of one request when idle
        self._waiters = itertools.count(1)
        self.arrivals: deque[float] = deque(maxlen=100_000)  # offered requests (times)
        self.arrival_keep_s = 600.0  # >= every window asked of `offered` (Controller sets it)
        # SLO outcomes of risk-admitted production requests (drift detection).
        self.outcomes: deque[bool] = deque(maxlen=500)
        self.predictor = predictor or TtftPredictor(clip_ms=2 * settings.ttft_slo_ms)
        self.slack = slack or SlackRisk(
            slack_edges(settings.ttft_slo_ms), min_samples=table.min_samples
        )
        # Share of D's receive buffer allowed in flight: the Canary's measured gate;
        # the whole buffer (the connector's own limit) until then.
        self.kv_fraction = 1.0
        self._applied_key: object = None
        self._costs = self._cost_book(None)
        self._slack_by_group: dict[str, tuple[tuple[float, ...], float, Any]] = {}
        self.full_effort = False
        self.pressure_event = asyncio.Event()
        self.risk_signals = 0
        self._signalled: set[int] = set()
        self._queue: deque[int] = deque()
        self._waiting_since: dict[int, float] = {}
        self._live: dict[int, Ticket] = {}
        # One sample per request: live stream updates replace rather than multiply it.
        self._production: dict[int, tuple[float, float, float, float]] = {}

    def signal_pressure(self, reason: str) -> None:
        """Wake control immediately; this signal never changes GPU clocks here."""
        self.risk_signals += 1
        self.pressure_event.set()
        self.log.write({"event": "pressure", "reason": reason})

    @property
    def open_admission(self) -> bool:
        """Cold start: no tier table yet, admit everything."""
        return self.tiers.table is None

    def _cost_book(self, evidence: Mapping[str, Any] | None) -> CostBook:
        """S(L) per P clock: the Canary's tables (evidence.prefill_ms_by_clock, or the
        MAX table of older tier tables); none before the Canary publishes."""
        tables: dict[int, dict[int, float]] = {}
        if evidence:
            for mhz, lut in (evidence.get("prefill_ms_by_clock") or {}).items():
                tables[int(mhz)] = {int(k): float(v) for k, v in lut.items()}
            old = (evidence.get("alpha_fit") or {}).get("prefill_ms_by_length")
            if not tables and old:
                tables[self.tiers.max_point.prefill_mhz] = {
                    int(k): float(v) for k, v in old.items()
                }
        return CostBook(tables)

    def _apply_table(self) -> None:
        """On a newly published tier table: its S(L) tables, and the admission
        parameters the Canary calibrated (predictor prior, slack seed, KV gate)."""
        table = self.tiers.table
        key = None if table is None else table.published_at
        if key == self._applied_key:
            return
        self._applied_key = key
        evidence = None if table is None else table.evidence
        self._costs = self._cost_book(evidence)
        self._idle_ttft = {
            int(k): float(v) for k, v in ((evidence or {}).get("idle_ttft_ms") or {}).items()
        }
        admission = (evidence or {}).get("admission")
        if admission:
            self.predictor.set_prior(admission["predictor_coef"])
            self.slack.set_seed(admission.get("slack_counts") or {})
            self.kv_fraction = float(admission.get("kv_gate_fraction") or 1.0)
            self.log.write({"event": "admission_calibration", "published_at": key, **admission})

    def prefill_cost(self, mhz: int | None = None) -> PrefillCost:
        """S(L) at P clock `mhz` (None: the highest measured)."""
        self._apply_table()
        return self._costs.for_clock(mhz)

    def deadline_ms(self, tokens: int) -> float:
        """Longest wait after which the SLO can still be met: TTFT SLO minus the
        Canary's idle TTFT at this length (the SLO itself before that is measured)."""
        self._apply_table()
        idle = interpolate(self._idle_ttft, tokens) or 0.0
        return max(0.0, self.settings.ttft_slo_ms - idle)

    def backfill_slack_ms(self) -> float:
        """Spare-capacity margin for backfilling doomed requests: configured, or the
        lowest slack from which the risk table is safe at theta / 2 (until that is
        observed, the top slack edge: backfill only into clearly idle groups)."""
        s = self.settings
        if s.backfill_slack_ms is not None:
            return s.backfill_slack_ms
        safe = self.slack.safe_slack(s.theta / 2)
        return max(0.0, safe) if safe is not None else self.slack.edges[-1]

    def _predict(self, group: Group, s_own_ms: float) -> tuple[tuple[float, ...], float]:
        x = self.predictor.features(
            s_own_ms, group.pending_ms, group.inflight_bytes, group.n_decoding
        )
        return x, self.predictor.predict(x)

    def _kv_blocked(self, group: Group, tokens: int = 0) -> bool:
        """KV in flight would pass the gate. A request alone on an empty link is
        never blocked: the gate is a measured congestion knee, not a size limit,
        and a prompt larger than it would otherwise wait until the timeout."""
        s = self.settings
        gate = self.kv_fraction * s.kv_buffer_bytes
        if group.inflight_bytes >= gate:
            return True
        return group.inflight_bytes > 0 and (
            group.inflight_bytes + tokens * s.kv_bytes_per_token > gate
        )

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
        if (table is not None and table.decode_max_running is not None
                and self.settings.overload != "best_effort"):
            # Legacy policies retain the calibrated admission ceiling. In best
            # effort B* is an SLO warning, not the hardware's concurrency limit.
            if max(running + waiting, group.n_inflight) + 1 > table.decode_max_running:
                return busy, "decode_concurrency_wall"
        if table is not None and table.decode_kv_limit is not None and kv is not None:
            if kv > table.decode_kv_limit:
                return busy, "decode_kv_wall"
        return busy, None

    def candidates(
        self, tokens: int, waited_ms: float = 0.0
    ) -> list[tuple[Group, Cell, RiskEstimate | None, bool]]:
        """Groups that may take this request now, with their risk cell (the slack
        estimate, when that decided, is kept in self._slack_by_group). The slack
        is what remains of the TTFT budget after `waited_ms` and the prediction."""
        out = []
        self._slack_by_group: dict[str, tuple[tuple[float, ...], float, Any]] = {}
        self._within_limits: list[tuple[Group, Cell, bool]] = []
        for group in self.groups:
            if not group.routable:
                continue
            assert group.effective is not None
            s_own = self.prefill_cost(group.effective.prefill_mhz)(tokens)
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
            if self._kv_blocked(group, tokens) and self.settings.admission == "slack":
                continue
            self._within_limits.append((group, cell, d_busy))
            tier_table = self.tiers.table
            if (self.settings.overload == "best_effort" and tier_table is not None
                    and tier_table.decode_max_running is not None):
                # P requests are not running D sequences. Telemetry covers D's
                # own queue; stage reservations cover unobserved running work.
                snapshot = (self.telemetry.fresh(group.decode, self.settings.snapshot_max_age_s)
                            if self.telemetry is not None else None)
                d_load = max(group.n_decoding, (snapshot.running or 0) + (snapshot.waiting or 0)
                             if snapshot is not None else 0)
                if d_load + 1 > tier_table.decode_max_running:
                    self._slack_by_group[group.name] = (x, predicted, None)
                    continue  # warn, but remains eligible for best-effort dispatch
            model = self.tiers.model_json()
            if model and self.settings.overload == "best_effort":
                decode = model.get("decode") or {}
                f = group.effective.decode_mhz
                alpha = interpolate({int(k): v[0] for k, v in decode.items()}, f) or 0
                delta = interpolate({int(k): v[1] for k, v in decode.items()}, f) or 0
                if alpha + delta * (group.n_inflight + 1) > self.settings.tpot_slo_ms:
                    self._slack_by_group[group.name] = (x, predicted, None)
                    continue  # TPOT risk requests resources, not a hard rejection
            if self.settings.admission == "slack":
                slack = self.slack.estimate(self.settings.ttft_slo_ms - waited_ms - predicted)
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

    def try_admit(
        self,
        tokens: int,
        exact: bool,
        waited_ms: float = 0.0,
        *,
        rescue: bool = False,
        backfill: bool = False,
        best_effort: bool = False,
    ) -> Ticket | None:
        """Admit to a feasible group. With `rescue` (serve mode, wait budget spent):
        to the group within the hard limits with the lowest predicted TTFT if that
        still meets the SLO. With `backfill` too (the oldest holder, no fresh request
        waiting) a doomed request takes a group feasible for a fresh request, or
        (doomed: dispatch) the lowest predicted group anyway."""
        feasible = self.candidates(tokens, waited_ms)
        kind = None
        if not feasible:
            if not (rescue or best_effort) or not self._within_limits:
                return None
            ranked = []
            for group, cell, d_busy in self._within_limits:
                predicted = self._slack_by_group[group.name][1]
                order = (predicted, group.canary, self.groups.index(group))
                ranked.append((*order, group, cell, d_busy))
            predicted, _, _, group, cell, d_busy = min(ranked, key=lambda r: r[:3])
            slack = self._slack_by_group[group.name][2]
            if best_effort:
                kind = "best_effort"
            elif waited_ms + predicted <= self.settings.ttft_slo_ms:
                kind = "rescue"
            elif not backfill:
                return None
            elif self.settings.doomed == "dispatch":
                kind = "doomed"
            else:
                # Spare capacity: feasible for a fresh request with backfill_slack_ms
                # to spare, so the late request does not take an on-time one's place.
                spare = self.candidates(tokens, self.backfill_slack_ms())
                if not spare:
                    return None
                kind = "backfill"
                group, cell, _, d_busy = min(
                    spare, key=lambda item: self._slack_by_group[item[0].name][1]
                )
                slack = self._slack_by_group[group.name][2]
            risk = 1.0 if slack is None else slack.risk
            samples = 0 if slack is None else slack.samples
            feasible = [(group, cell, RiskEstimate(risk, f"overflow_{kind}", samples), d_busy)]
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
        s_own = self.prefill_cost(None if group.effective is None else group.effective.prefill_mhz)(
            tokens
        )
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
            overflow=kind,
            slack_ms=None
            if predicted is None
            else self.settings.ttft_slo_ms - waited_ms - predicted,
        )
        if kind is not None:
            self.overflows[kind] += 1  # a doomed request counts once, when it leaves
            ticket.snapshot["overflow"] = kind
        group.n_at_p += 1
        group.pending_ms += s_own
        group.n_await += 1
        group.t_await += tokens
        group.n_inflight += 1
        group.record_admission(now, tokens, self.settings.load_window_s)
        self.admitted += 1
        if self.settings.overload == "best_effort":
            self._live[ticket.id] = ticket
        return ticket

    @property
    def pressure(self) -> int:
        """Requests the current configuration could not serve within the SLO risk:
        rejections, overflows and requests held past the wait budget (the
        Controller's and Canary's pressure signal)."""
        return self.rejections + sum(self.overflows.values()) + self._held + self.risk_signals

    def new_waiter(self) -> int:
        """One per arriving request (also the offered-rate record for the solver)."""
        self.arrivals.append(self._clock())
        return next(self._waiters)

    def offered(self, now: float, window_s: float, bucket_s: float) -> tuple[float, float]:
        """-> (offered requests/s over `window_s`, burst factor: the 90th percentile
        of the rate over `bucket_s` buckets in that window, divided by the mean)."""
        while self.arrivals and self.arrivals[0] < now - self.arrival_keep_s:
            self.arrivals.popleft()
        times = [t for t in self.arrivals if t >= now - window_s]
        if not times:
            return 0.0, 1.0
        rate = len(times) / window_s
        buckets = max(1, int(window_s // bucket_s))
        counts = [0] * buckets
        for t in times:
            counts[min(buckets - 1, int((now - t) // bucket_s))] += 1
        rates = sorted(c / bucket_s for c in counts)
        p90 = rates[min(len(rates) - 1, int(0.9 * len(rates)))]
        return rate, max(1.0, p90 / rate)

    def step(self, waiter: int, tokens: int, exact: bool, waited_ms: float) -> Ticket | str:
        """One admission attempt of a waiting request -> Ticket, "wait" or "reject".
        `admit` drives it on real time; simulations drive it on their own clock."""
        s = self.settings
        if s.overload == "best_effort":
            if waited_ms >= s.hold_max_ms:
                self._release(waiter)
                self.rejections += 1
                self.log.write({"event": "reject", "reason": "service_timeout"})
                return "reject"
            if waiter not in self._queue:
                self._queue.append(waiter)
                self._waiting_since[waiter] = self._clock() - waited_ms / 1000
            feasible = self.candidates(tokens, waited_ms)
            if not feasible and waiter not in self._signalled:
                self._signalled.add(waiter)
                self.signal_pressure("slo_risk_or_capacity")
            # FIFO prevents late requests from starving behind fresh arrivals.
            if self._queue[0] == waiter:
                ticket = self.try_admit(
                    tokens, exact, waited_ms, best_effort=True
                )
                if ticket is not None:
                    self._release(waiter)
                    return ticket
            return "wait"
        serve = s.overload == "serve" and not self.open_admission
        past_wait = waited_ms + s.retry_period_ms > s.max_wait_ms
        now = self._clock()
        if serve and past_wait and waiter not in self._holding:
            self._holding.append(waiter)
            self._expires[waiter] = now + (self.deadline_ms(tokens) - waited_ms) / 1000.0
            self._held += 1
            self._fresh.discard(waiter)
        elif not past_wait:
            self._fresh.add(waiter)
        # Priority: requests that can still meet the SLO first. Any holder may be
        # rescued. A doomed one is served best effort only once its deadline has
        # passed, as the oldest such holder, and only while no fresh request and no
        # holder within its deadline waits for the same capacity (head-of-line
        # blocking behind doomed requests halved the goodput under overload in the
        # sim). doomed: dispatch keeps the oldest holder dispatching at once.
        holder = serve and past_wait
        if not holder or self._fresh:
            head = False
        elif s.doomed == "dispatch":
            head = self._holding[0] == waiter
        else:
            expired = [w for w in self._holding if self._expires[w] <= now]
            live = len(expired) < len(self._holding)
            head = bool(expired) and expired[0] == waiter and not live
        ticket = self.try_admit(tokens, exact, waited_ms, rescue=holder, backfill=head)
        if ticket is not None:
            self._release(waiter)
            return ticket
        if past_wait and (not serve or waited_ms + s.retry_period_ms > s.hold_max_ms):
            self._release(waiter)
            self.rejections += 1
            self.log.write({"event": "reject", "prompt_tokens": tokens, "waited_ms": waited_ms})
            return "reject"
        return "wait"

    def _release(self, waiter: int) -> None:
        self._fresh.discard(waiter)
        self._signalled.discard(waiter)
        self._waiting_since.pop(waiter, None)
        try:
            self._queue.remove(waiter)
        except ValueError:
            pass
        self._expires.pop(waiter, None)
        try:
            self._holding.remove(waiter)
        except ValueError:
            pass

    async def admit(self, tokens: int, exact: bool) -> Ticket | None:
        """Admit now, or retry until the wait budget is spent, then reject (or, in
        serve mode, overflow / hold). The budget runs on real time because the
        retries really sleep."""
        start = time.monotonic()
        retry_s = self.settings.retry_period_ms / 1000.0
        waiter = self.new_waiter()
        try:
            while True:
                waited_ms = (time.monotonic() - start) * 1000.0
                result = self.step(waiter, tokens, exact, waited_ms)
                if isinstance(result, Ticket):
                    return result
                if result == "reject":
                    return None
                await asyncio.sleep(retry_s)
        finally:
            self._release(waiter)  # a cancelled (client gone) waiter leaves the queue

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
        ticket.last_token_at = ticket.first_token_at
        ticket.group.n_await -= 1
        ticket.group.t_await -= ticket.prompt_tokens
        self._leave_stage(ticket)
        ticket.stage = "decode"
        ticket.group.n_decoding += 1

    def _actionable_ttft(self, tokens: int) -> bool:
        """Do not scale the cluster for prompts already impossible at idle/MAX."""
        self._apply_table()
        idle = interpolate(self._idle_ttft, tokens)
        return idle is None or idle < self.settings.ttft_slo_ms

    def observe_latency(
        self, ticket: Ticket, ttft_ms: float | None, tpot_ms: float | None,
    ) -> None:
        if self.settings.overload != "best_effort":
            return
        now = self._clock()
        ttft_time, ttft_ratio, tpot_time, tpot_ratio = self._production.get(
            ticket.id, (float("-inf"), 0.0, float("-inf"), 0.0),
        )
        if ttft_ms is not None:
            # TTFT belongs to first-token time, not every subsequent token/finish.
            ttft_time = ticket.first_token_at if ticket.first_token_at is not None else now
            ttft_ratio = (ttft_ms / self.settings.ttft_slo_ms
                          if self._actionable_ttft(ticket.prompt_tokens) else 0.0)
        if tpot_ms is not None:
            tpot_time, tpot_ratio = now, tpot_ms / self.settings.tpot_slo_ms
        self._production[ticket.id] = (ttft_time, ttft_ratio, tpot_time, tpot_ratio)
        if max(ttft_ratio, tpot_ratio) >= 0.9:
            self.pressure_event.set()  # feedback wakes control without becoming a prediction

    def token_progress(self, ticket: Ticket, tpot_ms: float | None) -> None:
        ticket.last_token_at = self._clock()
        ttft = ((ticket.first_token_at - ticket.admitted_at) * 1000 + ticket.wait_ms
                if ticket.first_token_at is not None else None)
        self.observe_latency(ticket, ttft, tpot_ms)

    def production_feedback(
        self, window_s: float, min_samples: int, near: float, *, since: float = float("-inf"),
    ) -> dict[str, Any]:
        """Recent actual latency plus unfinished-request/physical-capacity waits."""
        now = self._clock()
        cutoff = max(now - window_s, since)
        self._production = {
            k: v for k, v in self._production.items() if max(v[0], v[2]) >= now - window_s
        }
        ratios = [
            max(r if t >= cutoff else 0, dr if dt >= cutoff else 0)
            for t, r, dt, dr in self._production.values() if max(t, dt) >= cutoff
        ]
        live_ratio = 0.0
        for ticket in self._live.values():
            if ticket.first_token_at is None:
                if self._actionable_ttft(ticket.prompt_tokens):
                    age = (now - ticket.admitted_at) * 1000 + ticket.wait_ms
                    live_ratio = max(live_ratio, age / self.settings.ttft_slo_ms)
            elif ticket.last_token_at is not None:
                live_ratio = max(
                    live_ratio, (now - ticket.last_token_at) * 1000 / self.settings.tpot_slo_ms,
                )
        wait_ratio = max(
            ((now - t) * 1000 / self.settings.ttft_slo_ms
             for t in self._waiting_since.values()), default=0.0,
        )
        fraction = sum(r >= near for r in ratios) / len(ratios) if ratios else 0.0
        return {
            "samples": len(ratios), "near_fraction": fraction,
            "live_ratio": live_ratio, "wait_ratio": wait_ratio,
            "pressure": (len(ratios) >= min_samples and fraction > self.settings.theta)
            or max(live_ratio, wait_ratio) >= near,
            "severe": max(live_ratio, wait_ratio) >= 2,
        }

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
        self._live.pop(ticket.id, None)
        if status == "ok":
            recent = self._production.get(ticket.id)
            # Keep recent stream spacing; completion's cumulative TPOT can be old.
            feedback_tpot = None if recent and recent[2] != float("-inf") else tpot_ms
            self.observe_latency(ticket, ttft_ms, feedback_tpot)
        else:
            self._production.pop(ticket.id, None)

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
            if ticket.estimate is not None and ticket.overflow is None:
                self.outcomes.append(bool(violated))
            if ticket.features and ticket.slack_ms is not None and ttft_ms is not None:
                # Slack as predicted at admission (the model in force then), then refit
                # on the time after dispatch (the wait at the proxy is known exactly).
                self.slack.record(ticket.slack_ms, bool(violated))
                self.predictor.record(ticket.features, max(0.0, ttft_ms - ticket.wait_ms))
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
                "overflow": ticket.overflow,
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
            "overload": self.settings.overload,
            "full_effort": self.full_effort,
            "risk_signals": self.risk_signals,
            "queued": len(self._queue),
            "kv_gate_fraction": self.kv_fraction,
            "backfill_slack_ms": self.backfill_slack_ms(),
            "overflows": dict(self.overflows),
            "holding": len(self._holding),
            "holding_expired": sum(1 for w in self._holding if self._expires[w] <= self._clock()),
            "groups": [g.snapshot() for g in self.groups],
        }
