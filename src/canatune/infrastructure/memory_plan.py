"""D GPU memory utilization from an absolute reserve.

vLLM pre-allocates gpu_memory_utilization x the GPU's memory for itself (weights,
activations, KV cache). What it does not plan for lives in the rest: the P2P
connector's receive buffer (kv_transfer.kv_buffer_bytes), received tensors past that
buffer and the allocator's cached blocks (overflow), one NCCL communicator per P that
sends to this D, and a safety margin. A fixed fraction leaves very different headroom
per GPU (0.82: 4.3 GB on an L4, 8.7 GB on an L40S; job K's L4s ran out of it under
overload), so the reserve is set in bytes and the fraction follows from the GPU:

    utilization = 1 - (kv_buffer + overflow + peers x per_peer + safety) / GPU memory

peers: the P a D receives from (fixed pairing: 1; dynamic: every P).
topology.decode_nodegroup.gpu_memory_utilization: a number keeps a fixed fraction;
auto uses the reserve (topology.decode_nodegroup.memory_reserve).

  python -m canatune.infrastructure.memory_plan --config CONFIG --total-mib MIB
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from canatune.config import load_config

DEFAULT_RESERVE = {"overflow_bytes": 3.0e9, "per_peer_bytes": 0.25e9, "safety_bytes": 1.0e9}
MIN_UTILIZATION, MAX_UTILIZATION = 0.5, 0.95


class MemoryPlanError(ValueError):
    pass


def peers(config: Mapping[str, Any]) -> int:
    """P endpoints one D receives from."""
    if config.get("router", {}).get("pairing", "fixed") != "dynamic":
        return 1
    endpoints = config.get("topology", {}).get("endpoints") or {}
    return max(1, sum(1 for e in endpoints.values() if e.get("role") == "prefill"))


def _bytes(value: Any, name: str) -> float:
    """A non-negative number (YAML 1.1 reads 3.0e9 as a string, so strings parse too)."""
    try:
        number = float(value) if not isinstance(value, bool) else math.nan
    except (TypeError, ValueError):
        number = math.nan
    if not number >= 0:
        raise MemoryPlanError(f"{name} must be a non-negative number")
    return number


def reserve_bytes(config: Mapping[str, Any]) -> float:
    given = config["topology"]["decode_nodegroup"].get("memory_reserve") or {}
    raw = {k: _bytes(v, f"memory_reserve.{k}") for k, v in {**DEFAULT_RESERVE, **given}.items()}
    kv_buffer = _bytes(config.get("kv_transfer", {}).get("kv_buffer_bytes", 1e9), "kv_buffer_bytes")
    return (
        kv_buffer + raw["overflow_bytes"] + peers(config) * raw["per_peer_bytes"]
        + raw["safety_bytes"]
    )  # fmt: skip


def utilization(config: Mapping[str, Any], total_bytes: float) -> float:
    """The decode endpoints' --gpu-memory-utilization on a GPU of total_bytes."""
    value = config["topology"]["decode_nodegroup"].get("gpu_memory_utilization", "auto")
    if value != "auto":
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < 1:
            raise MemoryPlanError("gpu_memory_utilization must be auto or in (0, 1)")
        return float(value)
    if total_bytes <= 0:
        raise MemoryPlanError("the GPU memory must be positive")
    fraction = 1.0 - reserve_bytes(config) / total_bytes
    if not MIN_UTILIZATION <= fraction <= MAX_UTILIZATION:
        raise MemoryPlanError(
            f"the reserve leaves utilization {fraction:.3f} on a {total_bytes / 1e9:.1f} GB "
            f"GPU, outside [{MIN_UTILIZATION}, {MAX_UTILIZATION}]"
        )
    return fraction


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--total-mib", type=float, required=True, help="nvidia-smi memory.total")
    args = ap.parse_args(argv)
    config = load_config(args.config)
    total = args.total_mib * 1024 * 1024
    try:
        fraction = utilization(config, total)
    except MemoryPlanError as error:
        print(f"memory_plan: {error}", file=sys.stderr)
        return 2
    reserve = reserve_bytes(config)
    print(
        f"memory_plan: GPU {total / 1e9:.2f} GB, reserve {reserve / 1e9:.2f} GB "
        f"({peers(config)} P), utilization {fraction:.4f}",
        file=sys.stderr,
    )
    print(f"{fraction:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
