"""Request length statistics for probe generation and shift detection.

The Router records (prompt tokens, generated tokens) of every finished
production request. The Canary draws probe lengths from these pairs (joint, so
the prompt/output correlation is kept) instead of copying requests, and a
change of the median or p90 against the distribution used for the last
calibration triggers a new one.
"""

import random
import threading
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class LengthSummary:
    samples: int
    prompt_p50: float
    prompt_p90: float
    output_p50: float
    output_p90: float
    prompt_mean: float

    def shifted(self, other: "LengthSummary", threshold: float) -> bool:
        """Relative change of any median or p90 above `threshold`."""
        pairs = (
            (self.prompt_p50, other.prompt_p50),
            (self.prompt_p90, other.prompt_p90),
            (self.output_p50, other.output_p50),
            (self.output_p90, other.output_p90),
        )
        return any(abs(a - b) / max(b, 1.0) > threshold for a, b in pairs)


def _quantile(values: Sequence[int], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return float(ordered[index])


class LengthStats:
    """Bounded window of recent (prompt, output) lengths, thread safe.

    `default` pairs are used while fewer than `min_samples` real requests have
    finished (cold start before production traffic arrives).
    """

    def __init__(
        self,
        default: Iterable[tuple[int, int]],
        *,
        keep: int = 2000,
        min_samples: int = 50,
    ) -> None:
        self.default = [(int(p), int(o)) for p, o in default]
        if not self.default or any(p < 1 or o < 1 for p, o in self.default):
            raise ValueError("default lengths must be nonempty positive (prompt, output) pairs")
        self.min_samples = min_samples
        self._pairs: deque[tuple[int, int]] = deque(maxlen=keep)
        self._lock = threading.Lock()

    def record(self, prompt_tokens: int, output_tokens: int) -> None:
        if prompt_tokens < 1 or output_tokens < 1:
            return
        with self._lock:
            self._pairs.append((int(prompt_tokens), int(output_tokens)))

    def pairs(self) -> list[tuple[int, int]]:
        with self._lock:
            real = list(self._pairs)
        return real if len(real) >= self.min_samples else list(self.default)

    @property
    def using_default(self) -> bool:
        with self._lock:
            return len(self._pairs) < self.min_samples

    def pairs_upto(self, max_prompt: int | None) -> list[tuple[int, int]]:
        """Pairs with prompt <= max_prompt (all when None); never empty."""
        pairs = self.pairs()
        if max_prompt is None:
            return pairs
        kept = [(p, o) for p, o in pairs if p <= max_prompt]
        if kept:
            return kept
        outputs = sorted(o for _, o in pairs)
        return [(max_prompt, outputs[len(outputs) // 2])]

    def sample(
        self, rng: random.Random, n: int, max_prompt: int | None = None
    ) -> list[tuple[int, int]]:
        pairs = self.pairs_upto(max_prompt)
        return [pairs[rng.randrange(len(pairs))] for _ in range(n)]

    def mean_prompt(self, max_prompt: int | None = None) -> float:
        pairs = self.pairs_upto(max_prompt)
        return sum(p for p, _ in pairs) / len(pairs)

    def summary(self) -> LengthSummary:
        pairs = self.pairs()
        prompts = [p for p, _ in pairs]
        outputs = [o for _, o in pairs]
        return LengthSummary(
            samples=len(pairs),
            prompt_p50=_quantile(prompts, 0.5),
            prompt_p90=_quantile(prompts, 0.9),
            output_p50=_quantile(outputs, 0.5),
            output_p90=_quantile(outputs, 0.9),
            prompt_mean=sum(prompts) / len(prompts),
        )
