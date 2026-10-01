"""Self-calibration on unseen hardware (E6, simulated): deploy with only the model's
config.json, the GPU models (datasheet priors) and the connector settings; the real
Canary tier locator measures the simulated cluster through a probe backend, then
production runs the smoke-2 profile with the published table.

Each variant is a different `Physics` (what the cluster really is) and the
deployment inputs an operator would know (GPU models, NIC speed, kv_buffer_size):

  reference     L40S / L4 as measured in r6b / r7
  slow_link     2 Gb/s between P and D (KV transfer dominates)
  small_buffer  kv_buffer_size 0.3 GB (the overflow cliff comes much earlier)
  fast_pair     H100-class P, A100-class D, 25 Gb/s, no P power cap

Per variant: what the Canary published (H, C_H, B*, KV gate, predictor) and the
production outcome of three policies on the same trace:
  baseline     round robin at MAX, no admission (the default system)
  priors_only  CanaTune with the Canary's tiers but admission from the priors
               (evidence["admission"] removed: predictor prior, 0.5 gate, no seed)
  calibrated   CanaTune with everything the Canary measured

  python scripts/analysis/selfcal_sim.py [--variants reference,slow_link,...] [--out DIR]
"""

import argparse
import asyncio
import json
import random
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import overload_sim as sim  # noqa: E402

from canatune import loadgen  # noqa: E402
from canatune.config import load_config  # noqa: E402
from canatune.controller.locator import (  # noqa: E402
    AdmissionInputs,
    Hardware,
    LocatorSettings,
    TierLocator,
    WindowResult,
)
from canatune.domain.groups import ClockPoint, TierTable  # noqa: E402
from canatune.domain.priors import ModelSpec, Priors, gpu_spec, model_spec  # noqa: E402

MISTRAL = model_spec(
    {
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "intermediate_size": 14336,
        "vocab_size": 32000,
    }
)
MODEL = ModelSpec(MISTRAL.parameters, MISTRAL.weight_bytes, MISTRAL.kv_bytes_per_token, 4096)
LENGTHS = sim.LENGTHS
PROMPTS = sorted({p for p, _ in LENGTHS})

VARIANTS = {
    # (physics, P GPU, D GPU, operator-known NIC Gb/s)
    "reference": (sim.REFERENCE, "NVIDIA L40S", "NVIDIA L4", 10.0),
    "slow_link": (
        replace(sim.REFERENCE, name="slow_link", link_bytes_s=2e9 / 8),
        "NVIDIA L40S",
        "NVIDIA L4",
        2.5,
    ),
    "small_buffer": (
        replace(sim.REFERENCE, name="small_buffer", kv_buffer_bytes=0.3e9),
        "NVIDIA L40S",
        "NVIDIA L4",
        10.0,
    ),
    "fast_pair": (
        replace(
            sim.REFERENCE,
            name="fast_pair",
            p_t0_ms=14.0,
            p_beta_top_ms=0.022,
            p_f_eff=1980,
            p_clocks=(1095, 1305, 1500, 1755, 1980),
            d_alpha_ms=13.0,
            d_delta_ms=0.45,
            d_f_knee=900,
            d_clocks=(705, 900, 1110, 1305, 1410),
            d_kv_tokens=75000,
            d_first_extra_ms=25.0,
            overhead_ms=25.0,
            link_bytes_s=25e9 / 8,
            p_idle_w=70.0,
            p_dyn_w=600.0,
            d_idle_w=50.0,
            d_dyn_w=330.0,
        ),
        "NVIDIA H100 80GB HBM3",
        "NVIDIA A100-SXM4-80GB",
        25.0,
    ),
}


class SimBackend:
    """`ProbeBackend` over the simulated pair: the locator cannot tell it from the
    Canary probe (same windows, energies, clocks, per-request state samples)."""

    service_source = "metrics"

    def __init__(self, phys: sim.Physics, seed: int = 7) -> None:
        self.phys = phys
        self.rng = random.Random(seed)
        self.limit: int | None = None
        self.windows = 0
        self.sim_seconds = 0.0

    def _pairs(self) -> list[tuple[int, int]]:
        pairs = [p for p in LENGTHS if self.limit is None or p[0] <= self.limit]
        return pairs or LENGTHS[:1]

    async def hardware(self) -> Hardware:
        return Hardware(self.phys.p_clocks, self.phys.d_clocks)

    async def idle_power(self, clock: ClockPoint, seconds: float) -> tuple[float, float]:
        noise = 1 + self.rng.gauss(0, 0.005)
        return (
            self.phys.p_power(clock.prefill_mhz, 0.0) * noise,
            self.phys.d_power(clock.decode_mhz, 0.0) * noise,
        )

    async def service_times(self, clock, prompts):
        out = []
        for n in prompts:
            pre = self.phys.prefill_ms(clock.prefill_mhz, n) * (1 + self.rng.gauss(0, 0.01))
            ttft = (
                self.phys.overhead_ms
                + pre
                + n * self.phys.kv_bytes_per_token / self.phys.link_bytes_s * 1000
                + self.phys.d_iter_ms(clock.decode_mhz, 1)
                + self.phys.d_first_extra_ms
            )
            out.append((n, pre, ttft))
        return out

    def set_prompt_limit(self, max_prompt: int | None) -> float:
        self.limit = max_prompt
        pairs = self._pairs()
        return sum(p for p, _ in pairs) / len(pairs)

    def _result(self, s: sim.Sim, clock: ClockPoint, kind: str, load: float) -> WindowResult:
        pair = s.pairs[0]
        reqs = [r for r in s.requests if r.status == "ok"]
        end = max((r.done for r in reqs), default=1.0)
        start = min((r.at for r in s.requests), default=0.0)
        duration = max(end - start, 1e-3)
        f_p, f_d = clock.prefill_mhz, clock.decode_mhz
        p_j = self.phys.p_power(f_p, 0) * duration + pair.p_busy_s * (
            self.phys.p_power(f_p, 1) - self.phys.p_power(f_p, 0)
        )
        d_j = self.phys.d_power(f_d, 0) * duration + pair.d_busy_s * (
            self.phys.d_power(f_d, 1) - self.phys.d_power(f_d, 0)
        )
        tokens = sum(r.output for r in reqs)
        viol = [sim.violated(r) for r in reqs]
        self.windows += 1
        self.sim_seconds += duration
        capped = f_p > self.phys.p_f_eff
        return WindowResult(
            clock=clock,
            kind=kind,
            load=load,
            duration_s=duration,
            requests=len(reqs),
            violations=sum(viol),
            ttft_p95_ms=sim.pct([(r.first - r.at) * 1000 for r in reqs], 0.95),
            tpot_p95_ms=sim.pct([s.tpot(r) for r in reqs if s.tpot(r) is not None], 0.95),
            prefill_j_per_request=p_j / len(reqs) if reqs else None,
            decode_j_per_token=d_j / tokens if tokens else None,
            prefill_mhz_median=float(min(f_p, self.phys.p_f_eff)),
            prefill_limited_fraction=1.0 if capped else 0.0,
            decode_waiting_max=float(pair.d_waiting_max),
            decode_kv_max=pair.d_kv_max,
            decode_running_max=float(pair.d_running_max),
            samples=[s.probe_sample(r) for r in reqs],
        )

    async def open_window(self, clock, eq_tps, alpha, seconds, abort_above):
        trace = random.Random(int(eq_tps * 10) + 17)
        pairs = self._pairs()
        mean = sum(p for p, _ in pairs) / len(pairs)
        count = max(1, round(eq_tps / (mean + alpha) * seconds))
        arrivals = [
            loadgen.Arrival(i, t, "w", *pairs[trace.randrange(len(pairs))])
            for i, t in enumerate(sorted(trace.uniform(0, seconds) for _ in range(count)))
        ]
        s = sim.Sim(1, "baseline", clock, phys=self.phys)
        s.run(arrivals, seconds)
        return self._result(s, clock, "open", eq_tps)

    async def closed_window(self, clock, concurrency, seconds, rep=0):
        trace = random.Random(concurrency * 101 + rep)
        pairs = self._pairs()
        s = sim.Sim(1, "baseline", clock, phys=self.phys)
        ids = iter(range(10**9))
        stagger = min(15.0, seconds / 3)

        def send(worker: int, at: float) -> None:
            if at >= seconds:
                return
            prompt = pairs[trace.randrange(len(pairs))][0]
            r = sim.Req(next(ids), at, "w", prompt, 256, worker=worker)
            s.at(at, s.arrive, r)

        s.on_done = lambda r: send(r.worker, s.now)
        for w in range(concurrency):
            send(w, stagger * w / max(concurrency, 1))
        s.loop(seconds)
        return self._result(s, clock, "closed", concurrency)

    async def kv_capacity(self) -> int:
        return self.phys.d_kv_tokens


def deployment_priors(phys: sim.Physics, p_gpu: str, d_gpu: str, link_gbps: float) -> Priors:
    """Only what an operator knows: GPU models, model config, NIC, kv_buffer_size."""
    mean_context = sum(p + o for p, o in LENGTHS) / len(LENGTHS)
    return Priors(
        prefill=gpu_spec(p_gpu),
        decode=gpu_spec(d_gpu),
        model=MODEL,
        kv_buffer_bytes=phys.kv_buffer_bytes,
        link_bytes_s=link_gbps * 1e9 / 8,
        decode_memory_utilization=0.82,
        mean_context_tokens=mean_context,
    )


def canary(phys: sim.Physics, priors: Priors) -> tuple[TierTable, SimBackend]:
    config = load_config()
    settings = LocatorSettings.from_config(config)
    backend = SimBackend(phys)
    locator = TierLocator(
        backend,
        settings,
        admission=AdmissionInputs(
            kv_bytes_per_token=phys.kv_bytes_per_token,
            kv_buffer_bytes=phys.kv_buffer_bytes,
            gate_prior=0.5,
            predictor_prior=priors.predictor_coef(),
        ),
    )
    mean_prompt = sum(PROMPTS) / len(PROMPTS)
    table = asyncio.run(locator.locate(mean_prompt, PROMPTS))
    return table, backend


def production(phys, table, priors, policy, seed, profile):
    meta, arrivals = loadgen.build_trace(
        loadgen.parse_profile(profile),
        LENGTHS,
        capacity_h=table.capacity_h,
        alpha=table.alpha_tokens,
        seed=seed,
    )
    if policy == "baseline":
        s = sim.Sim(2, "baseline", phys.max_point, phys=phys)
    else:
        used = table
        if policy == "priors_only":
            used = TierTable.from_json(table.to_json())
            used.evidence = {k: v for k, v in table.evidence.items() if k != "admission"}
        s = sim.Sim(2, "serve", table.h, phys=phys, table=used, priors=priors)
    s.run(arrivals, meta["duration_s"])
    res = sim.summarize(s, meta["phases"])
    res["energy_kj"] = s.energy_j / 1000
    res["rate_rps_peak"] = max(p["rate_rps"] for p in meta["phases"])
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--profile", default=loadgen.DEFAULT_PROFILE)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    report = {}
    for name in args.variants.split(","):
        phys, p_gpu, d_gpu, link = VARIANTS[name]
        priors = deployment_priors(phys, p_gpu, d_gpu, link)
        started = time.monotonic()
        table, backend = canary(phys, priors)
        ev = table.evidence
        adm = ev.get("admission") or {}
        mean_prompt = sum(p for p, _ in LENGTHS) / len(LENGTHS)
        c_h_rps = table.capacity_h / (mean_prompt + table.alpha_tokens)
        h_fit = (ev.get("prefill_fit") or {}).get(table.h.prefill_mhz) or {}
        print(f"\n=== {name}: P {p_gpu}, D {d_gpu}, link {phys.link_bytes_s * 8 / 1e9:g} Gb/s")
        print(f"  priors: {json.dumps(priors.summary())}")
        print(
            f"  Canary: {backend.windows} windows, {backend.sim_seconds / 60:.0f} min simulated;"
            f" H {table.h.key()}  C_H {c_h_rps:.2f} req/s  alpha {table.alpha_tokens:.0f}"
            f"  B* {table.decode_max_running}"
        )
        print(f"  S(L) at H: {h_fit}")
        print(
            f"  admission: {adm.get('samples')} samples, coef {adm.get('predictor_coef')},"
            f" rms {adm.get('residual_ms_rms')} ms, KV gate {adm.get('kv_gate_fraction')}"
            f" ({adm.get('kv_gate_source')}; prior 0.5)"
        )
        rows = {}
        for policy in ("baseline", "priors_only", "calibrated"):
            runs = [
                production(phys, table, priors, policy, 1000 + k, args.profile)
                for k in range(args.seeds)
            ]
            rows[policy] = {
                key: statistics.mean(r["total"][key] for r in runs)
                for key in ("goodput", "served", "offered", "rejected", "served_late")
            }
            rows[policy]["overload_goodput"] = statistics.mean(
                r["overload"]["goodput"] for r in runs
            )
            rows[policy]["energy_kj"] = statistics.mean(r["energy_kj"] for r in runs)
            rows[policy]["kv_gate"] = (runs[0].get("router") or {}).get("kv_gate_fraction")
            rows[policy]["backfill_ms"] = (runs[0].get("router") or {}).get("backfill_slack_ms")
        base_e = rows["baseline"]["energy_kj"]
        print(
            f"  {'policy':>12} {'goodput':>8} {'overload':>9} {'served%':>8} {'rejected':>8}"
            f" {'energy kJ':>10} {'vs base':>8} {'gate':>5} {'backfill':>8}"
        )
        for policy, r in rows.items():
            print(
                f"  {policy:>12} {r['goodput']:>8.3f} {r['overload_goodput']:>9.3f}"
                f" {r['served'] / r['offered']:>8.3f} {r['rejected']:>8.0f}"
                f" {r['energy_kj']:>10.0f} {r['energy_kj'] / base_e - 1:>+8.1%}"
                f" {str(r['kv_gate']):>5} {str(r['backfill_ms']):>8}"
            )
        print(f"  ({time.monotonic() - started:.0f} s)")
        report[name] = {
            "priors": priors.summary(),
            "table": {
                "h": table.h.key(),
                "park": table.park.key(),
                "capacity_h_rps": c_h_rps,
                "alpha": table.alpha_tokens,
                "b_star": table.decode_max_running,
                "prefill_fit": ev.get("prefill_fit"),
                "admission": {k: v for k, v in adm.items() if k != "slack_counts"},
            },
            "production": rows,
        }
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "selfcal.json").write_text(json.dumps(report, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
