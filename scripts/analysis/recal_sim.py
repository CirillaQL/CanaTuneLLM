"""Re-calibration after a length shift (simulated). The real Canary scheduler and tier
locator run on the simulated cluster (`overload_sim`, physics l4g: L40S P, L4 D,
Mistral-7B) while production serves a trace whose length mix changes halfway:

  cold start   the Canary calibrates with the default probe lengths (as on the cluster)
  phase A      short requests: BurstGPT prompts <= 512 tokens, A_RATE req/s
  phase B      long requests: BurstGPT prompts 1536-3584 tokens, B_RATE req/s
  (lengths and output lengths from the BurstGPT trace, Poisson arrivals)

Runs on the same arrivals:
  baseline   round robin over all pairs at MAX (the default system)
  frozen     CanaTune keeping the cold-start table (canary.recalibrate=false)
  adaptive   CanaTune with re-calibration (a length shift relocates the P side)

Reported per 30-minute window: energy against the baseline, goodput, groups at MAX,
and every Canary experiment (trigger, simulated duration, H before / after). The
simulated power is the environment's; absolute energies are illustrative.

  python scripts/analysis/recal_sim.py --burstgpt BurstGPT_1.csv [--hours-a 2 --hours-b 2]
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import overload_sim as sim  # noqa: E402

from canatune import loadgen  # noqa: E402

COLD = [(p, 256) for p in (128, 512, 1024, 2048)]  # cold-start probes (output cap 256)


def phase_pairs(path: str, max_len: int) -> tuple[list, list]:
    """(prompt, output) pairs of the short and the long phase, from BurstGPT."""
    _, prompt, output, _ = sim.burstgpt_rows([Path(path)], "all", max_len)
    pairs = list(zip(prompt.tolist(), output.tolist()))
    short = [(p, o) for p, o in pairs if p <= 512]
    long = [(p, o) for p, o in pairs if 1536 <= p <= 3584]
    return short, long


def two_phase_trace(short, long, rate_a, rate_b, hours_a, hours_b, window_s, seed):
    rng = random.Random(seed)
    rows, t = [], 0.0
    for pairs, rate, hours, tag in ((short, rate_a, hours_a, "A"), (long, rate_b, hours_b, "B")):
        start = t
        while True:
            t += rng.expovariate(rate)
            if t >= start + hours * 3600:
                break
            k = int((t - start) // window_s)
            rows.append((t, f"{tag}{k}", *pairs[rng.randrange(len(pairs))]))
        t = start + hours * 3600
    end = t
    names = sorted({r[1] for r in rows}, key=lambda n: (n[0], int(n[1:])))
    phases = []
    for n in names:
        ts = [r[0] for r in rows if r[1] == n]
        phases.append({"name": n, "start_s": min(ts), "end_s": max(ts)})
    arrivals = [loadgen.Arrival(i, *row) for i, row in enumerate(rows)]
    return {"phases": phases, "duration_s": end}, arrivals


def run(phys, table, policy, meta, arrivals, groups, seed):
    canary = {"frozen": {"recalibrate": False}, "adaptive": {}}.get(policy)
    s = sim.Sim(
        groups,
        "baseline" if policy == "baseline" else "best_effort",
        phys.max_point if policy == "baseline" else table.h,
        phys=phys,
        table=None if policy == "baseline" else table,
        solver=False,
        lengths=COLD,
        seed=seed,
        canary=canary,
        router_overrides={"group_choice": "fill"},
    )
    started = time.monotonic()
    s.run(arrivals, meta["duration_s"])
    res = sim.summarize(s, meta["phases"])
    res["experiments"] = s.experiments
    res["wall_s"] = round(time.monotonic() - started)
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--burstgpt", required=True, help="BurstGPT CSV (lengths only)")
    ap.add_argument("--physics", default="l4g", choices=sorted(sim.PHYSICS))
    ap.add_argument("--groups", type=int, default=4)
    ap.add_argument("--rate-a", type=float, default=2.0)
    ap.add_argument("--rate-b", type=float, default=1.0)
    ap.add_argument("--hours-a", type=float, default=2.0)
    ap.add_argument("--hours-b", type=float, default=2.0)
    ap.add_argument("--window-s", type=float, default=1800.0)
    ap.add_argument("--slo", default="600,100")
    ap.add_argument("--seed", type=int, default=20261030)
    ap.add_argument("--policies", default="baseline,frozen,adaptive")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    sim.configure_slo(*(float(x) for x in args.slo.split(",")))
    sim.CONFIG_OVERRIDES.update({"router.group_choice": "fill", "controller.solver": False})
    phys = sim.PHYSICS[args.physics]

    short, long = phase_pairs(args.burstgpt, 4096)
    mean = lambda ps, i: sum(p[i] for p in ps) / len(ps)  # noqa: E731
    print(f"phase A: {len(short)} BurstGPT pairs, prompt {mean(short, 0):.0f} / output "
          f"{mean(short, 1):.0f}; phase B: {len(long)} pairs, prompt {mean(long, 0):.0f} / "
          f"output {mean(long, 1):.0f}")  # fmt: skip
    table, backend = sim.run_canary(phys, COLD)
    print(f"cold-start Canary: {backend.windows} windows, {backend.sim_seconds / 60:.0f} min "
          f"simulated; H {table.h.key()}, C_H {table.evidence.get('capacity_rps', 0):.2f} req/s "
          f"(default lengths)")  # fmt: skip
    meta, arrivals = two_phase_trace(short, long, args.rate_a, args.rate_b, args.hours_a,
                                     args.hours_b, args.window_s, args.seed)  # fmt: skip
    results = {}
    for policy in args.policies.split(","):
        results[policy] = run(phys, table, policy, meta, arrivals, args.groups, args.seed)
        print(f"  {policy}: {results[policy]['wall_s']} s wall")

    names = [p["name"] for p in meta["phases"]]
    base = results.get("baseline")
    print(f"\n{'window':>7} " + " ".join(f"{p:>26}" for p in results if p != "baseline")
          + f" {'baseline goodput':>17}")  # fmt: skip
    for n in names + ["total"]:
        cells = []
        for policy, res in results.items():
            if policy == "baseline":
                continue
            e = res[n].get("energy_kj") if n != "total" else sum(
                res[m].get("energy_kj") or 0 for m in names)
            b = (base[n].get("energy_kj") if n != "total" else sum(
                base[m].get("energy_kj") or 0 for m in names)) if base else None
            de = f"{(e / b - 1) * 100:+6.1f}%" if e and b else "     -"
            cells.append(f"dE {de} gp {res[n]['goodput']:.3f} MAX {res[n]['max_group_s']:5d}s")
        print(f"{n:>7} " + " ".join(f"{c:>26}" for c in cells)
              + (f" {base[n]['goodput']:>17.3f}" if base else ""))  # fmt: skip
    for policy, res in results.items():
        for e in res["experiments"]:
            print(f"{policy}: t={e['start_s'] / 60:6.1f} min {e['kind']}/{e['reason']} "
                  f"{e['duration_s'] / 60:.0f} min ({e['windows']} windows) {e['outcome']}: "
                  f"H {e['h_before']} -> {e['h_after']}, C_H {e['c_h_rps_after']}, "
                  f"lengths {e['lengths']}")  # fmt: skip
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "recal_sim.json").write_text(json.dumps(
            {"cold_h": table.h.key(), "results": results}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
