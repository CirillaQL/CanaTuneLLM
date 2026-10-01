"""State-based admission: KV geometry, prefill cost, an online TTFT predictor and a
risk table over predicted slack. Every number comes from the config (model KV
geometry, SLO) or from the Canary's calibration on this cluster.

Per request the Router knows, at arrival, for each group:
  own prefill cost S(L)             from the Canary's single-request length table
  pending P work                    sum S(L) of requests sent to P, not yet returned
  KV bytes in flight                prompt tokens x KV bytes/token of requests whose P
                                    returned and whose first token has not arrived
                                    (the D receive buffer holds these)
  requests decoding on D
It predicts TTFT with a linear model refitted online on its own clean samples,
and admits when the violation risk of the predicted-slack bucket (slack = SLO -
prediction) is within theta. Slack buckets are shared by all clock points and
prompt lengths, so they fill much faster than the v1 cells (queue x prompt x D).
"""

import json
import math
import os
import tempfile
import threading
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FEATURES = ("intercept", "s_own_ms", "pending_ms", "inflight_gb", "decoding")


def slack_edges(ttft_slo_ms: float) -> tuple[float, ...]:
    """Slack bucket edges in tenths of the TTFT SLO, from -0.4 to +0.8 SLO."""
    return tuple(ttft_slo_ms * k / 10 for k in range(-4, 9))


def kv_bytes_per_token(model_config: Mapping[str, Any], dtype_bytes: int = 2) -> int:
    """K and V of every layer: 2 x layers x KV heads x head dim x dtype bytes
    (HF config.json names; Mistral-7B bf16: 131072)."""
    layers = int(model_config["num_hidden_layers"])
    heads = int(model_config["num_attention_heads"])
    kv_heads = int(model_config.get("num_key_value_heads", heads))
    head_dim = int(model_config.get("head_dim") or model_config["hidden_size"] // heads)
    return 2 * layers * kv_heads * head_dim * dtype_bytes


def kv_bytes_from_model_dir(path: str | os.PathLike | None) -> int | None:
    if not path:
        return None
    try:
        raw = json.loads((Path(path) / "config.json").read_text(encoding="utf-8"))
        return kv_bytes_per_token(raw)
    except (OSError, ValueError, KeyError):
        return None


class PrefillCost:
    """Single-request prefill time S(L) in ms: piecewise linear over the Canary's
    length table (plateau + slope), extrapolated with the last segment. Without a
    table (cold start, admission open) it is 0."""

    def __init__(self, table: Mapping[int, float] | None = None) -> None:
        self.points = sorted((int(k), float(v)) for k, v in (table or {}).items())

    def __call__(self, length: int) -> float:
        pts = self.points
        if len(pts) < 2:
            return 0.0
        if length <= pts[0][0]:
            return pts[0][1]
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            if length <= x1:
                return y0 + (y1 - y0) * (length - x0) / (x1 - x0)
        (x0, y0), (x1, y1) = pts[-2], pts[-1]
        return y1 + (y1 - y0) * (length - x1) / (x1 - x0)


class CostBook:
    """S(L) per P clock: the Canary's tables measured at several clocks; a clock
    without one uses the nearest measured clock at or below it (prefill time only
    shortens with the clock, so that is the conservative choice)."""

    def __init__(self, tables: Mapping[int, Mapping[int, float]] | None = None) -> None:
        self.costs = {int(f): PrefillCost(t) for f, t in (tables or {}).items() if len(t) >= 2}

    def for_clock(self, mhz: int | None) -> PrefillCost:
        if not self.costs:
            return PrefillCost()
        if mhz is None:
            return self.costs[max(self.costs)]
        below = [f for f in self.costs if f <= mhz]
        return self.costs[max(below) if below else min(self.costs)]


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float] | None:
    n = len(vector)
    m = [row[:] + [vector[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            return None
        m[col], m[pivot] = m[pivot], m[col]
        for r in range(n):
            if r != col:
                f = m[r][col] / m[col][col]
                m[r] = [a - f * b for a, b in zip(m[r], m[col])]
    return [m[i][n] / m[i][i] for i in range(n)]


class TtftPredictor:
    """Linear TTFT model over FEATURES. The Canary's calibration fits it (plain
    least squares, no prior); production refits it on a sliding window of clean
    samples, ridge-regularized towards the Canary's fit so a sparse window cannot
    swing the coefficients far. Until the Canary publishes, admission is open and
    the predictor is not used (prior None: all coefficients 0)."""

    def __init__(
        self,
        prior: Sequence[float] | None = None,
        *,
        window: int = 2000,
        refit_every: int = 50,
        ridge: float = 50.0,
        clip_ms: float = math.inf,
    ) -> None:
        if prior is not None and len(prior) != len(FEATURES):
            raise ValueError(f"prior needs {len(FEATURES)} coefficients")
        self.has_prior = prior is not None
        self.prior = [float(c) for c in (prior or [0.0] * len(FEATURES))]
        self.coef = list(self.prior)
        self.samples: deque[tuple[tuple[float, ...], float]] = deque(maxlen=window)
        self.refit_every = refit_every
        self.ridge = ridge
        self.clip_ms = clip_ms
        self._since_fit = 0
        self._lock = threading.Lock()

    @staticmethod
    def features(
        s_own_ms: float, pending_ms: float, inflight_bytes: float, decoding: int
    ) -> tuple[float, ...]:
        return (1.0, float(s_own_ms), float(pending_ms), inflight_bytes / 1e9, float(decoding))

    def predict(self, x: Sequence[float]) -> float:
        with self._lock:
            return sum(c * v for c, v in zip(self.coef, x))

    def set_prior(self, prior: Sequence[float]) -> None:
        """New prior (the Canary's calibration): the coefficients restart from it and
        the production samples kept so far refit towards it."""
        with self._lock:
            self.prior = [float(c) for c in prior]
            self.coef = list(self.prior)
            self.has_prior = True
            if len(self.samples) >= self.refit_every:
                self._refit()

    def fit(self, samples: Sequence[tuple[Sequence[float], float]]) -> list[float]:
        """Least-squares fit of `samples` (features, TTFT ms); -> coefficients (the
        predictor itself is not changed). Towards the prior when there is one, with
        a ridge worth a handful of samples, else plain least squares."""
        probe = TtftPredictor(
            self.prior if self.has_prior else None,
            window=max(len(samples), 1),
            ridge=5.0 if self.has_prior else 1e-6,
            clip_ms=self.clip_ms,
        )
        for x, y in samples:
            probe.samples.append((tuple(x), min(float(y), self.clip_ms)))
        if probe.samples:
            probe._refit()
        return list(probe.coef)

    def record(self, x: Sequence[float], ttft_ms: float) -> None:
        with self._lock:
            self.samples.append((tuple(x), min(float(ttft_ms), self.clip_ms)))
            self._since_fit += 1
            if self.has_prior and self._since_fit >= self.refit_every:
                self._since_fit = 0
                self._refit()

    def _refit(self) -> None:
        n = len(FEATURES)
        # Scale-aware ridge: penalize (coef - prior) relative to each feature's spread.
        ata = [[0.0] * n for _ in range(n)]
        aty = [0.0] * n
        for x, y in self.samples:
            for i in range(n):
                aty[i] += x[i] * y
                for j in range(n):
                    ata[i][j] += x[i] * x[j]
        for i in range(n):
            scale = ata[i][i] / max(len(self.samples), 1) or 1.0
            ata[i][i] += self.ridge * scale
            aty[i] += self.ridge * scale * self.prior[i]
        solved = _solve(ata, aty)
        if solved is not None and all(math.isfinite(c) for c in solved):
            self.coef = solved

    def state(self) -> dict[str, Any]:
        with self._lock:
            return {"coef": dict(zip(FEATURES, self.coef)), "samples": len(self.samples)}


def wilson_ucb(violated: int, n: int, z: float = 1.28) -> float:
    if n == 0:
        return 1.0
    p = violated / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c + s) / d


@dataclass
class SlackEstimate:
    bucket: int
    risk: float
    samples: int
    source: str  # "observed", "bounded" (by a lower-slack bucket), "unknown"


class SlackRisk:
    """Violation risk by predicted-slack bucket: Wilson UCB (the calibration's
    feasibility bound), over production's own samples plus the Canary's seed. A
    bucket with `min_samples` uses its own record. A sparse bucket is bounded by the
    nearest observed bucket with less slack (risk does not grow with slack, so that
    is an upper bound); with none below it the risk is unknown (1.0)."""

    def __init__(
        self,
        edges_ms: Sequence[float] = slack_edges(1000.0),
        *,
        min_samples: int = 20,
        path: str | Path | None = None,
    ) -> None:
        self.edges = sorted(float(e) for e in edges_ms)
        self.min_samples = min_samples
        self.path = None if path is None else Path(path)
        self.counts: dict[int, list[int]] = {}  # bucket -> [admitted, violated]
        self.seed: dict[int, list[int]] = {}  # the Canary's calibration windows
        self._lock = threading.Lock()

    def set_seed(self, counts: Mapping[Any, Sequence[int]]) -> None:
        """Counts from the Canary's calibration windows, added to production's own
        (replaced, not accumulated, by each new calibration)."""
        with self._lock:
            self.seed = {int(b): [int(c[0]), int(c[1])] for b, c in counts.items()}

    def _cell(self, b: int) -> list[int]:
        own = self.counts.get(b, [0, 0])
        seed = self.seed.get(b, [0, 0])
        return [own[0] + seed[0], own[1] + seed[1]]

    def safe_slack(self, target: float) -> float | None:
        """Lowest predicted slack from which every observed bucket upwards has a
        violation bound <= target (None: not observed yet)."""
        with self._lock:
            buckets = sorted(set(self.counts) | set(self.seed))
            observed = [b for b in buckets if self._cell(b)[0] >= self.min_samples]
            safe = None
            for b in sorted(observed, reverse=True):
                n, k = self._cell(b)
                if wilson_ucb(k, n) > target:
                    break
                safe = b
        if safe is None or safe == 0:
            return None
        return self.edges[safe - 1]

    def bucket(self, slack_ms: float) -> int:
        return sum(slack_ms >= e for e in self.edges)

    def record(self, slack_ms: float, violated: bool) -> None:
        with self._lock:
            cell = self.counts.setdefault(self.bucket(slack_ms), [0, 0])
            cell[0] += 1
            cell[1] += int(violated)

    def estimate(self, slack_ms: float) -> SlackEstimate:
        b = self.bucket(slack_ms)
        with self._lock:
            n, k = self._cell(b)
            if n >= self.min_samples:
                return SlackEstimate(b, wilson_ucb(k, n), n, "observed")
            lower_buckets = {c for c in set(self.counts) | set(self.seed) if c < b}
            for lower in sorted(lower_buckets, reverse=True):
                cn, ck = self._cell(lower)
                if cn >= self.min_samples:
                    return SlackEstimate(b, wilson_ucb(ck, cn), cn, "bounded")
        return SlackEstimate(b, 1.0, n, "unknown")

    def to_json(self) -> dict[str, Any]:
        with self._lock:
            return {
                "edges_ms": self.edges,
                "counts": {str(b): list(c) for b, c in sorted(self.counts.items())},
                "seed": {str(b): list(c) for b, c in sorted(self.seed.items())},
            }

    def load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        if [float(e) for e in raw.get("edges_ms", [])] != self.edges:
            return  # other buckets: start empty rather than mix
        with self._lock:
            self.counts = {int(b): [int(v[0]), int(v[1])] for b, v in raw["counts"].items()}
            self.seed = {int(b): [int(v[0]), int(v[1])] for b, v in raw.get("seed", {}).items()}

    def save(self) -> None:
        if self.path is None:
            return
        text = json.dumps(self.to_json(), indent=2) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp = tempfile.mkstemp(dir=self.path.parent, prefix=".slack-")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, self.path)
        except BaseException:
            Path(temp).unlink(missing_ok=True)
            raise
