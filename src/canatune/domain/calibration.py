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


def kv_gate_fraction(
    samples: Sequence[ProbeSample],
    *,
    kv_bytes_per_token: float,
    buffer_bytes: float,
    theta: float,
    min_samples: int = 20,
    bins: int = 10,
) -> tuple[float, str, dict[str, list[int]]]:
    """-> (gate, source, bins). Share of the receive buffer in flight up to which
    violations stay within theta: the lower edge of the first well-sampled bin
    whose violation rate is clearly above theta (its Wilson lower bound, source
    "measured"); if the windows never reached such a bin, the buffer itself (1.0,
    source "buffer": the connector's own limit from the config)."""
    counts: dict[int, list[int]] = {}
    for s in samples:
        if s.violated is None:
            continue
        share = s.inflight_tokens * kv_bytes_per_token / buffer_bytes
        b = min(bins - 1, int(share * bins))
        cell = counts.setdefault(b, [0, 0])
        cell[0] += 1
        cell[1] += int(s.violated)
    for b in sorted(counts):
        n, k = counts[b]
        if n < min_samples:
            continue
        lower = 1 - wilson_ucb(n - k, n)  # lower bound of the violation rate
        if lower > theta:
            table = {str(b): c for b, c in sorted(counts.items())}
            return max(1.0 / bins, b / bins), "measured", table
    table = {str(b): c for b, c in sorted(counts.items())}
    return 1.0, "buffer", table


def calibrate_admission(
    samples: Sequence[ProbeSample],
    costs: CostBook,
    *,
    prior: Sequence[float] | None,
    ttft_slo_ms: float,
    theta: float,
    kv_bytes_per_token: float,
    buffer_bytes: float,
    min_samples: int = 20,
) -> dict[str, Any] | None:
    """-> evidence["admission"]: predictor coefficients, slack-bucket counts (the
    risk-table seed), the KV gate; None without enough clean samples."""
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
    gate, gate_source, gate_bins = kv_gate_fraction(
        clean,
        kv_bytes_per_token=kv_bytes_per_token,
        buffer_bytes=buffer_bytes,
        theta=theta,
        min_samples=min_samples,
    )
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
    }
