"""Self-calibration on unseen hardware (E6, simulated). The system gets only what a
deployment knows (the model's KV geometry and kv_buffer_size); the real Canary
tier locator measures each simulated cluster through `overload_sim.SimBackend`,
and production runs with what it published.

Variants (what the cluster really is, `overload_sim.Physics`):
  reference     the example environment of overload_sim
  slow_link     2 Gb/s between P and D (KV transfer dominates)
  small_buffer  kv_buffer_size 0.3 GB
  fast_pair     a faster P and D, 25 Gb/s, no P power cap

Policies on the same trace (load in units of the Canary's measured C_H):
  baseline   round robin at MAX, no admission (the default system)
  static     CanaTune with the Canary's H for every group, no solver (model removed)
  solver     CanaTune with the solver over the Canary's cluster model
and a length-shift scenario (short prompts, then long ones) on the reference.

The simulated power is part of the environment (`Physics.p_power/d_power`); the
system only sees it through the Canary's energy readings, so absolute energies
are illustrative.

  python scripts/analysis/selfcal_sim.py [--variants ...] [--seeds N] [--out DIR]
"""

import argparse
import json
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import overload_sim as sim  # noqa: E402

from canatune import loadgen  # noqa: E402
from canatune.domain.groups import TierTable  # noqa: E402

VARIANTS = {
    "reference": sim.REFERENCE,
    "slow_link": replace(sim.REFERENCE, name="slow_link", link_bytes_s=2e9 / 8),
    "small_buffer": replace(sim.REFERENCE, name="small_buffer", kv_buffer_bytes=0.3e9),
    "fast_pair": replace(
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
}
SHORT = [(128, 64), (256, 64)]
LONG = [(1024, 64), (2048, 64)]


def static(table: TierTable) -> TierTable:
    out = TierTable.from_json(table.to_json())
    out.evidence = {k: v for k, v in table.evidence.items() if k != "model"}
    return out


def shift_trace(table: TierTable, seed: int):
    """Short prompts for the first half, long ones after (same per-phase loads)."""
    half = "low:120:0.3,mid:120:0.8,high:120:1.4"
    meta_a, a = sim.trace(table, half, seed, SHORT)
    meta_b, b = sim.trace(table, half, seed + 1, LONG)
    offset = meta_a["duration_s"]
    arrivals = list(a) + [
        loadgen.Arrival(
            len(a) + i, x.at_s + offset, f"long_{x.phase}", x.prompt_tokens, x.output_tokens
        )
        for i, x in enumerate(b)
    ]
    phases = [dict(p, name=f"short_{p['name']}") for p in meta_a["phases"]] + [
        dict(p, name=f"long_{p['name']}", start_s=p["start_s"] + offset, end_s=p["end_s"] + offset)
        for p in meta_b["phases"]
    ]
    arrivals = [
        loadgen.Arrival(x.index, x.at_s, f"short_{x.phase}", x.prompt_tokens, x.output_tokens)
        if x.index < len(a)
        else x
        for x in arrivals
    ]
    return {"phases": phases, "duration_s": offset + meta_b["duration_s"]}, arrivals


def run(phys, table, policy, meta, arrivals):
    used = static(table) if policy == "static" else table
    s = sim.Sim(
        2,
        "baseline" if policy == "baseline" else "serve",
        phys.max_point if policy == "baseline" else table.h,
        phys=phys,
        table=None if policy == "baseline" else used,
    )
    s.run(arrivals, meta["duration_s"])
    res = sim.summarize(s, meta["phases"])
    res["energy_kj"] = s.energy_j / 1000
    plans = (
        []
        if s.controller is None
        else [e for e in s.controller.log.recent if e.get("event") == "plan"]
    )
    res["plans"] = len(plans)
    res["points"] = sorted({e["to"]["point"] for e in plans})
    return res


def report(name, phys, table, backend, seeds, scenario):
    ev = table.evidence
    model = ev.get("model") or {}
    adm = ev.get("admission") or {}
    print(
        f"\n=== {name} ({scenario}): Canary {backend.windows} windows,"
        f" {backend.sim_seconds / 60:.0f} min simulated"
    )
    print(
        f"  H {table.h.key()}  C_H {ev.get('capacity_rps', 0):.2f} req/s"
        f"  rho_P {model.get('rho_prefill', 0):.2f}  D fits {len(model.get('decode', {}))}"
        f"  KV gate {adm.get('kv_gate_fraction')} ({adm.get('kv_gate_source')})"
        f"  predictor rms {adm.get('residual_ms_rms')} ms"
    )
    rows = {}
    for policy in ("baseline", "static", "solver"):
        runs = []
        for k in range(seeds):
            if scenario == "shift":
                meta, arrivals = shift_trace(table, 1000 + k)
            else:
                meta, arrivals = sim.trace(table, loadgen.DEFAULT_PROFILE, 1000 + k)
            runs.append(run(phys, table, policy, meta, arrivals))
        rows[policy] = {
            "goodput": statistics.mean(r["total"]["goodput"] for r in runs),
            "served": statistics.mean(r["total"]["served"] / r["total"]["offered"] for r in runs),
            "rejected": statistics.mean(r["total"]["rejected"] for r in runs),
            "energy_kj": statistics.mean(r["energy_kj"] for r in runs),
            "plans": statistics.mean(r["plans"] for r in runs),
            "points": runs[0]["points"],
        }
    base = rows["baseline"]["energy_kj"]
    print(
        f"  {'policy':>9} {'goodput':>8} {'served':>7} {'rejected':>8} {'energy kJ':>10}"
        f" {'vs base':>8} {'plans':>6}  points"
    )
    for policy, r in rows.items():
        print(
            f"  {policy:>9} {r['goodput']:>8.3f} {r['served']:>7.3f} {r['rejected']:>8.0f}"
            f" {r['energy_kj']:>10.0f} {r['energy_kj'] / base - 1:>+8.1%} {r['plans']:>6.0f}"
            f"  {' '.join(r['points'])}"
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--no-shift", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = {}
    for name in args.variants.split(","):
        started = time.monotonic()
        phys = VARIANTS[name]
        table, backend = sim.run_canary(phys)
        out[name] = report(name, phys, table, backend, args.seeds, "profile")
        print(f"  ({time.monotonic() - started:.0f} s)")
    if not args.no_shift:
        phys = sim.REFERENCE
        table, backend = sim.run_canary(phys, SHORT + LONG)
        out["length_shift"] = report("reference", phys, table, backend, args.seeds, "shift")
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "selfcal.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
