"""Priors from what is known before deployment: the model's config.json, the GPU
models (datasheet numbers) and the KV connector settings. No measurement goes in.

They only have to be of the right scale: they drive admission until the Canary has
measured the cluster (cold start), and they anchor the ridge fits of the Canary's
and production's own samples. Roofline estimates:

  prefill  plateau  = weight bytes / (P memory bandwidth x BW_EFF) + overhead
           slope    = 2 x parameters FLOP per token / (P dense BF16 FLOPS x MFU)
  decode   iteration = weight bytes / (D bandwidth x BW_EFF) + overhead
           per sequence = KV bytes of a mean context / (D bandwidth x BW_EFF)
           KV capacity = (D memory x utilization - weights - reserve) / KV bytes per token
  TTFT     = fixed + own prefill + pending prefill + KV in flight / link + D first token

(Mistral-7B on L40S/L4: plateau 34 ms, slope 80 us/token, D iteration 66 ms, 32k KV
tokens; the r6b measurements were 30-45 ms, 68-113 us/token, 55-60 ms, ~26-33k.)
"""

import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BW_EFF = 0.7  # achievable share of the memory bandwidth (streaming the weights)
MFU = 0.5  # achievable share of dense tensor FLOPS in prefill GEMMs
FIXED_OVERHEAD_MS = 10.0  # kernel launches, scheduling, HTTP within a step
PROXY_OVERHEAD_MS = 20.0  # proxy -> P -> proxy -> D round trips
ACTIVATION_RESERVE_BYTES = 1.0e9  # vLLM activations / CUDA graphs outside the KV cache


@dataclass(frozen=True)
class GpuSpec:
    name: str
    memory_gb: float
    bandwidth_gbs: float
    bf16_tflops: float  # dense


# Public datasheet numbers (dense BF16 tensor TFLOPS, no sparsity).
GPU_SPECS = {
    "H100 SXM": GpuSpec("H100 SXM", 80, 3350, 989),
    "H100 PCIE": GpuSpec("H100 PCIe", 80, 2000, 756),
    "H100": GpuSpec("H100 SXM", 80, 3350, 989),
    "A100 80GB": GpuSpec("A100 80GB", 80, 2039, 312),
    "A100": GpuSpec("A100 40GB", 40, 1555, 312),
    "L40S": GpuSpec("L40S", 48, 864, 362),
    "L40": GpuSpec("L40", 48, 864, 181),
    "L4": GpuSpec("L4", 24, 300, 121),
    "A10G": GpuSpec("A10G", 24, 600, 70),
    "A10": GpuSpec("A10", 24, 600, 125),
    "RTX 6000 ADA": GpuSpec("RTX 6000 Ada", 48, 960, 365),
    "A6000": GpuSpec("RTX A6000", 48, 768, 155),
    "RTX 4090": GpuSpec("RTX 4090", 24, 1008, 165),
}


def gpu_spec(name: str, overrides: Mapping[str, Any] | None = None) -> GpuSpec | None:
    """Datasheet entry for a GPU name such as "NVIDIA L40S" or "A100-SXM4-80GB": the
    key whose words all occur in the name, the most specific one first ("L40S" is
    not "L40" or "L4"; "A100 80GB" beats "A100"); `overrides` may set memory_gb /
    bandwidth_gbs / bf16_tflops for GPUs not in the table."""
    words = set(re.split(r"[\s\-_/]+", name.upper()))
    hits = [key for key in GPU_SPECS if set(key.split()) <= words]
    base = GPU_SPECS[max(hits, key=lambda k: (len(k.split()), len(k)))] if hits else None
    raw = dict(overrides or {})
    if base is None and not {"memory_gb", "bandwidth_gbs", "bf16_tflops"} <= set(raw):
        return None
    values = {} if base is None else base.__dict__.copy()
    values.update({k: float(v) for k, v in raw.items() if k != "name"})
    values["name"] = values.get("name") or name
    return GpuSpec(**values)


@dataclass(frozen=True)
class ModelSpec:
    parameters: float
    weight_bytes: float
    kv_bytes_per_token: int
    max_model_len: int


def model_spec(raw: Mapping[str, Any], dtype_bytes: int = 2) -> ModelSpec:
    """Parameter count of a decoder-only transformer from its HF config.json:
    attention (q, k, v, o), gated MLP (3 matrices) per layer, embeddings."""
    hidden = int(raw["hidden_size"])
    layers = int(raw["num_hidden_layers"])
    heads = int(raw["num_attention_heads"])
    kv_heads = int(raw.get("num_key_value_heads", heads))
    head_dim = int(raw.get("head_dim") or hidden // heads)
    inter = int(raw.get("intermediate_size", 4 * hidden))
    vocab = int(raw.get("vocab_size", 32000))
    attention = hidden * heads * head_dim * 2 + hidden * kv_heads * head_dim * 2
    mlp = 3 * hidden * inter
    embeddings = vocab * hidden * (1 if raw.get("tie_word_embeddings") else 2)
    params = layers * (attention + mlp) + embeddings
    experts = int(raw.get("num_local_experts") or raw.get("n_routed_experts") or 1)
    if experts > 1:  # MoE: all experts are in memory
        params += layers * mlp * (experts - 1)
    return ModelSpec(
        parameters=float(params),
        weight_bytes=float(params * dtype_bytes),
        kv_bytes_per_token=2 * layers * kv_heads * head_dim * dtype_bytes,
        max_model_len=int(raw.get("max_position_embeddings", 4096)),
    )


@dataclass(frozen=True)
class Priors:
    prefill: GpuSpec
    decode: GpuSpec
    model: ModelSpec
    kv_buffer_bytes: float
    link_bytes_s: float
    decode_memory_utilization: float = 0.9
    mean_context_tokens: float = 1024.0

    @property
    def prefill_plateau_ms(self) -> float:
        return self.model.weight_bytes / (self.prefill.bandwidth_gbs * 1e9 * BW_EFF) * 1e3 + (
            FIXED_OVERHEAD_MS
        )

    @property
    def prefill_ms_per_token(self) -> float:
        return 2 * self.model.parameters / (self.prefill.bf16_tflops * 1e12 * MFU) * 1e3

    def prefill_ms(self, tokens: int, clock_ratio: float = 1.0) -> float:
        """Roofline S(L) at `clock_ratio` = clock / highest clock (the compute-bound
        slope scales with the clock, the bandwidth-bound plateau does not)."""
        slope = self.prefill_ms_per_token / max(clock_ratio, 0.1)
        return max(self.prefill_plateau_ms, FIXED_OVERHEAD_MS + slope * tokens)

    def prefill_table(self, clock_ratio: float = 1.0) -> dict[int, float]:
        top = max(256, self.model.max_model_len)
        lengths = sorted({16, 64, 256, 1024, top // 2, top})
        return {n: self.prefill_ms(n, clock_ratio) for n in lengths}

    @property
    def decode_iteration_ms(self) -> float:
        return self.model.weight_bytes / (self.decode.bandwidth_gbs * 1e9 * BW_EFF) * 1e3 + (
            FIXED_OVERHEAD_MS / 2
        )

    @property
    def decode_ms_per_sequence(self) -> float:
        kv = self.mean_context_tokens * self.model.kv_bytes_per_token
        return kv / (self.decode.bandwidth_gbs * 1e9 * BW_EFF) * 1e3

    @property
    def decode_kv_tokens(self) -> int:
        free = (
            self.decode.memory_gb * 1e9 * self.decode_memory_utilization
            - self.model.weight_bytes
            - ACTIVATION_RESERVE_BYTES
        )
        return max(0, int(free / self.model.kv_bytes_per_token))

    def predictor_coef(self) -> tuple[float, float, float, float, float]:
        """Prior of the admission TTFT model over (intercept, own S(L), pending P
        work, KV in flight GB, requests decoding): fixed proxy cost plus D's first
        token (join at the next iteration + one iteration), own and pending prefill
        at face value, the in-flight KV drained over the link, and each decoding
        sequence lengthening the two iterations before the first token."""
        return (
            PROXY_OVERHEAD_MS + 2 * self.decode_iteration_ms,
            1.0,
            1.0,
            1e3 * 1e9 / self.link_bytes_s,
            2 * self.decode_ms_per_sequence,
        )

    def summary(self) -> dict[str, Any]:
        return {
            "prefill_gpu": self.prefill.name,
            "decode_gpu": self.decode.name,
            "parameters_b": round(self.model.parameters / 1e9, 2),
            "kv_bytes_per_token": self.model.kv_bytes_per_token,
            "prefill_plateau_ms": round(self.prefill_plateau_ms, 1),
            "prefill_us_per_token": round(self.prefill_ms_per_token * 1e3, 1),
            "decode_iteration_ms": round(self.decode_iteration_ms, 1),
            "decode_ms_per_sequence": round(self.decode_ms_per_sequence, 2),
            "decode_kv_tokens": self.decode_kv_tokens,
            "link_gbps": round(self.link_bytes_s * 8 / 1e9, 1),
            "kv_buffer_gb": round(self.kv_buffer_bytes / 1e9, 2),
            "predictor_coef": [round(c, 2) for c in self.predictor_coef()],
        }


def model_dir(config: Mapping[str, Any]) -> str | None:
    for env in ("CANATUNE_MODEL_PATH", "MODEL_PATH"):
        if os.environ.get(env):
            return os.environ[env]
    return config.get("model", {}).get("path")


def cluster_priors(config: Mapping[str, Any]) -> Priors | None:
    """Priors for this deployment, or None when the model's config.json or a GPU
    spec is unavailable (admission then uses its generic cold-start values)."""
    path = model_dir(config)
    if not path:
        return None
    try:
        raw = json.loads((Path(path) / "config.json").read_text(encoding="utf-8"))
        model = model_spec(raw)
    except (OSError, ValueError, KeyError):
        return None
    topology = config.get("topology", {})
    specs = config.get("cluster", {}).get("gpu_specs", {})
    prefill = gpu_spec(
        str(topology.get("prefill_nodegroup", {}).get("gpu_type", "")), specs.get("prefill")
    )
    decode = gpu_spec(
        str(topology.get("decode_nodegroup", {}).get("gpu_type", "")), specs.get("decode")
    )
    if prefill is None or decode is None:
        return None
    max_len = int(config.get("model", {}).get("max_model_len") or model.max_model_len)
    model = ModelSpec(model.parameters, model.weight_bytes, model.kv_bytes_per_token, max_len)
    kv = config.get("kv_transfer", {})
    lengths = config.get("canary", {}).get("default_lengths") or [[512, 64]]
    mean_context = sum(p + o for p, o in lengths) / len(lengths)
    return Priors(
        prefill=prefill,
        decode=decode,
        model=model,
        kv_buffer_bytes=float(kv.get("kv_buffer_bytes", 1e9)),
        link_bytes_s=float(kv.get("link_gbps", 10.0)) * 1e9 / 8,
        decode_memory_utilization=float(
            topology.get("decode_nodegroup", {}).get("gpu_memory_utilization", 0.9)
        ),
        mean_context_tokens=mean_context,
    )
