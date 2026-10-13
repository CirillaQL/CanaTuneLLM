"""Admission parameters from the Canary's own windows (no offline analysis).

Every Canary probe records the state it was sent into: the prompt lengths still at
P, the KV tokens between P's return and the first token, the requests decoding;
plus its TTFT and SLO outcome. The tier locator's windows (ramp to overload,
clock search, fill) cover idle to overloaded:

  fit_plateau_slope   S(L) per P clock: plateau (bandwidth-bound) + slope
  predictor           least-squares fit of the TTFT model (TTFT <= 2 x SLO)
  slack seed          violations by predicted-slack bucket, two-fold (each half
                      predicted by the fit on the other half), seeding the risk table
  kv gate             the share of D's receive buffer in flight above which
                      requests violate more than theta (on whatever buffer and
                      link this cluster has); the whole buffer if none is reached
  P slot cap          the number of requests already at P (sent, P not returned)
                      from which requests violate more than theta; under
                      PUT_ASYNC a send blocked by D holds its request at P, so the
                      count sees both P queueing and P blocked by D; no cap if
                      none is reached

Both knees come from the same violation counts per bin (`KneeCounts`): the Canary's
samples seed them and the Router adds production's (online update).
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from canatune.domain.admission import (
    CostBook,
    SlackRisk,
    TtftPredictor,
    slack_edges,
    wilson_ucb,
)


@dataclass(frozen=True)
class ProbeSample:
    """One Canary request and the state it was sent into."""

    prefill_mhz: int
    decode_mhz: int
    prompt_tokens: int
    at_prefill: tuple[int, ...]  # prompt lengths of the requests still at P
    inflight_tokens: int  # prompt tokens of requests between P's return and first token
    decoding: int
    ttft_ms: float | None
    violated: bool | None

    def to_json(self) -> dict[str, Any]:
        return {
            "p": self.prefill_mhz,
            "d": self.decode_mhz,
            "L": self.prompt_tokens,
            "at_p": list(self.at_prefill),
            "inflight": self.inflight_tokens,
            "decoding": self.decoding,
            "ttft_ms": self.ttft_ms,
            "violated": self.violated,
        }


def fit_plateau_slope(table: Mapping[int, float]) -> dict[str, float] | None:
    """S(L) = max(plateau, c + slope * L) on per-length medians: the plateau is the
    mean of the shortest lengths, the slope a least-squares line through the rest;
    the split with the least squared error wins. -> plateau_ms, slope_us_per_token,
    knee_tokens (L where the line meets the plateau)."""
    points = sorted((int(k), float(v)) for k, v in table.items())
    if len(points) < 3:
        return None
    best = None
    for k in range(1, len(points) - 1):
        flat, rest = points[:k], points[k:]
        plateau = sum(v for _, v in flat) / len(flat)
        n = len(rest)
        mx = sum(x for x, _ in rest) / n
        my = sum(y for _, y in rest) / n
        sxx = sum((x - mx) ** 2 for x, _ in rest)
        if sxx <= 0:
            continue
        slope = sum((x - mx) * (y - my) for x, y in rest) / sxx
        if slope <= 0:
            continue
        c = my - slope * mx
        sse = sum((max(plateau, c + slope * x) - y) ** 2 for x, y in points)
        if best is None or sse < best[0]:
            best = (sse, plateau, slope, c)
    if best is None:
        return None
    _, plateau, slope, c = best
    return {
        "plateau_ms": round(plateau, 2),
        "slope_us_per_token": round(slope * 1e3, 2),
        "knee_tokens": round(max(0.0, (plateau - c) / slope)),
    }


def _features(sample: ProbeSample, costs: CostBook, kv_bytes_per_token: float) -> tuple[float, ...]:
    cost = costs.for_clock(sample.prefill_mhz)
    return TtftPredictor.features(
        cost(sample.prompt_tokens),
        sum(cost(n) for n in sample.at_prefill),
        sample.inflight_tokens * kv_bytes_per_token,
        sample.decoding,
    )


KV_GATE_BINS = 10  # the gate in tenths of the receive buffer
P_SLOT_BINS = 32  # requests at P are counted up to this (more share the last bin)


def kv_gate_bin(inflight_bytes: float, buffer_bytes: float) -> int:
    return max(0, min(KV_GATE_BINS - 1, int(inflight_bytes / buffer_bytes * KV_GATE_BINS)))


def gate_of_bin(b: int) -> float:
    return max(1.0 / KV_GATE_BINS, b / KV_GATE_BINS)


def p_slot_bin(at_prefill: int) -> int:
    return max(0, min(P_SLOT_BINS - 1, int(at_prefill)))


class KneeCounts:
    """[requests, violations] per bin of a congestion signal: a seed (the Canary's
    counts) plus what is recorded later (production)."""

    def __init__(self, seed: Mapping[str, Sequence[int]] | None = None) -> None:
        self.seed = {int(b): [int(c[0]), int(c[1])] for b, c in (seed or {}).items()}
        self.live: dict[int, list[int]] = {}

    def record(self, b: int, violated: bool) -> None:
        cell = self.live.setdefault(b, [0, 0])
        cell[0] += 1
        cell[1] += int(violated)

    def counts(self) -> dict[int, list[int]]:
        out = {b: list(c) for b, c in self.seed.items()}
        for b, (n, k) in self.live.items():
            cell = out.setdefault(b, [0, 0])
            cell[0] += n
            cell[1] += k
        return dict(sorted(out.items()))

    def to_json(self) -> dict[str, list[int]]:
        return {str(b): c for b, c in self.counts().items()}

    def knee(self, theta: float, min_samples: int) -> int | None:
        """The lowest bin from which requests violate clearly more than theta and
        more than below it: bins are merged upwards from each bin until the block
        holds min_samples (so a sparse tail still counts, without the deep overload
        far above diluting into it), and the block's Wilson lower bound must pass
        both theta and the violation rate of all bins below (violations that are
        as frequent below, e.g. of long prompts, are not this signal's knee; nor is
        the lowest observed bin, which has nothing below to compare with). In a
        merged block the knee is the first bin whose own rate passes that floor.
        None: no such bin was observed."""
        counts = self.counts()
        bins = list(counts)
        head_n = head_k = 0
        for i, b in enumerate(bins):
            if i == 0:
                head_n, head_k = counts[b]
                continue
            n = k = 0
            block = []
            for c in bins[i:]:
                n += counts[c][0]
                k += counts[c][1]
                block.append(c)
                if n >= min_samples:
                    break
            if n < min_samples:
                return None  # the rest together is too sparse to tell
            lower = 1 - wilson_ucb(n - k, n)  # lower bound of the violation rate
            floor = max(theta, head_k / head_n if head_n else 0.0)
            if lower > floor:
                # A merged block: the knee is its first bin that is itself above the
                # floor (the violations of a higher bin are not the lower one's).
                return next((c for c in block if counts[c][1] > floor * counts[c][0]), b)
            head_n += counts[b][0]
            head_k += counts[b][1]
        return None


def _at_or_above(
    samples: Sequence[ProbeSample], prefill_mhz: int | None, min_samples: int
) -> Sequence[ProbeSample]:
    """Samples sent at P clocks from `prefill_mhz` up (where production runs: the
    low clocks of the search add violations of slow prefill, not of congestion);
    all of them when those are too few."""
    if prefill_mhz is None:
        return samples
    kept = [s for s in samples if s.prefill_mhz >= prefill_mhz]
    return kept if len(kept) >= 2 * min_samples else samples


def kv_gate_fraction(
    samples: Sequence[ProbeSample],
    *,
    kv_bytes_per_token: float,
    buffer_bytes: float | None,
    theta: float,
    min_samples: int = 20,
) -> tuple[float | None, str, dict[str, list[int]]]:
    """-> (gate, source, bins). Share of the receive buffer in flight up to which
    violations stay within theta: the lower edge of the knee bin (`KneeCounts.knee`,
    source "measured"); if the windows never reached one, the buffer itself (1.0,
    source "buffer": the connector's own limit from the config). No receive buffer
    (NixlConnector): no gate (None, source "none")."""
    if buffer_bytes is None:
        return None, "none", {}
    counts = KneeCounts()
    for s in samples:
        if s.violated is not None:
            counts.record(kv_gate_bin(s.inflight_tokens * kv_bytes_per_token, buffer_bytes),
                          s.violated)
    b = counts.knee(theta, min_samples)
    if b is None:
        return 1.0, "buffer", counts.to_json()
    return gate_of_bin(b), "measured", counts.to_json()


def p_slot_cap(
    samples: Sequence[ProbeSample], *, theta: float, min_samples: int = 20
) -> tuple[int | None, str, dict[str, list[int]]]:
    """-> (cap, source, bins). A group takes a request while fewer than `cap` are at
    its P: the knee of the violations by the number of requests already at P when
    each was sent (source "measured"); None (source "none") if never reached."""
    counts = KneeCounts()
    for s in samples:
        if s.violated is not None:
            counts.record(p_slot_bin(len(s.at_prefill)), s.violated)
    b = counts.knee(theta, min_samples)
    if b is None:
        return None, "none", counts.to_json()
    return max(1, b), "measured", counts.to_json()


def calibrate_admission(
    samples: Sequence[ProbeSample],
    costs: CostBook,
    *,
    prior: Sequence[float] | None,
    ttft_slo_ms: float,
    theta: float,
    kv_bytes_per_token: float,
    buffer_bytes: float | None,
    min_samples: int = 20,
    prefill_mhz: int | None = None,
) -> dict[str, Any] | None:
    """-> evidence["admission"]: predictor coefficients, slack-bucket counts (the
    risk-table seed), the KV gate and the P slot cap (both from the samples at P
    clocks from `prefill_mhz`, H's, up); None without enough clean samples."""
    clean = [s for s in samples if s.ttft_ms is not None and s.violated is not None]
    if len(clean) < 2 * min_samples:
        return None
    rows = [(_features(s, costs, kv_bytes_per_token), float(s.ttft_ms)) for s in clean]
    # The ramp drives the pair into collapse on purpose (TTFT of tens of seconds);
    # the linear model is for the region where admission decides, so it is fitted
    # on TTFT <= 2 x SLO. Every sample still counts for the slack seed and the gate.
    near = [(x, y) for x, y in rows if y <= 2 * ttft_slo_ms]
    if len(near) < 2 * min_samples:
        return None
    base = TtftPredictor(prior, clip_ms=2 * ttft_slo_ms)
    coef = base.fit(near)
    # Two-fold slack: each half predicted by the fit on the other half, so the seed
    # reflects prediction error on unseen requests, not the in-sample residual.
    halves = (rows[0::2], rows[1::2])
    outcomes = (clean[0::2], clean[1::2])
    slack = SlackRisk(slack_edges(ttft_slo_ms), min_samples=min_samples)
    for i in (0, 1):
        fold = base.fit([(x, y) for x, y in halves[1 - i] if y <= 2 * ttft_slo_ms])
        for (x, _), s in zip(halves[i], outcomes[i]):
            predicted = sum(c * v for c, v in zip(fold, x))
            slack.record(ttft_slo_ms - predicted, bool(s.violated))
    knees = _at_or_above(clean, prefill_mhz, min_samples)
    gate, gate_source, gate_bins = kv_gate_fraction(
        knees,
        kv_bytes_per_token=kv_bytes_per_token,
        buffer_bytes=buffer_bytes,
        theta=theta,
        min_samples=min_samples,
    )
    slot, slot_source, slot_bins = p_slot_cap(knees, theta=theta, min_samples=min_samples)
    residuals = [y - sum(c * v for c, v in zip(coef, x)) for x, y in near]
    return {
        "samples": len(clean),
        "fit_samples": len(near),
        "violations": sum(bool(s.violated) for s in clean),
        "predictor_coef": [round(c, 4) for c in coef],
        "predictor_prior": None if prior is None else [round(c, 4) for c in prior],
        "residual_ms_rms": round(math.sqrt(sum(r * r for r in residuals) / len(residuals)), 1),
        "slack_counts": slack.to_json()["counts"],
        "kv_gate_fraction": gate,
        "kv_gate_source": gate_source,
        "kv_gate_bins": gate_bins,
        "p_slot_cap": slot,
        "p_slot_source": slot_source,
        "p_slot_bins": slot_bins,
        "knee_samples": len(knees),
    }
