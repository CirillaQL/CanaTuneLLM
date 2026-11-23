"""Five prompt distributions and nearest-distribution P capacity lookup."""

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any


def p_profiles(max_prompt: int = 2048) -> dict[str, tuple[int, ...]]:
    """The experimental bank, clipped to the deployment's measured prompt range."""
    short, medium, large, long = (min(n, max_prompt) for n in (128, 512, 1024, 2048))
    return {
        "short": (short,),
        "short_heavy": (short,) * 7 + (medium, large, long),
        "mixed": (short, medium, large, long),
        "long_heavy": (short, medium, large) + (long,) * 7,
        "long": (long,),
    }


def nearest_p_rho(
    lengths: Sequence[int], profiles: Mapping[str, Mapping[str, Any]],
) -> tuple[str, float] | None:
    """Nearest histogram by total variation; exact ties choose the lower measured rho.

    Real prompt lengths map to the nearest calibrated length (ties to the shorter).
    No distance cutoff, hold time or minimum-capacity fallback is imposed.
    """
    measured = {name: row for name, row in profiles.items() if row["p_rho"] is not None}
    if not lengths or not measured:
        return None
    anchors = sorted({n for row in measured.values() for n in row["prompt_lengths"]})

    def histogram(values: Sequence[int]) -> list[float]:
        counts = Counter(min(anchors, key=lambda n: (abs(n - value), n)) for value in values)
        return [counts[n] / len(values) for n in anchors]

    current = histogram(lengths)
    distances = {
        name: 0.5 * sum(abs(a - b) for a, b in zip(current, histogram(row["prompt_lengths"])))
        for name, row in measured.items()
    }
    name = min(measured, key=lambda n: (distances[n], measured[n]["p_rho"], n))
    return name, measured[name]["p_rho"]
