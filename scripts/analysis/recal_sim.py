"""Re-calibration after a length shift (simulated). The real Canary scheduler and tier
locator run on the simulated cluster (`overload_sim`, physics l4g: L40S P, L4 D,
Mistral-7B) while production serves a trace whose length mix changes between phases:

  cold start   the Canary calibrates with the default probe lengths (as on the cluster)
  phases       `--phases kind:rate:hours,...`; kind short (BurstGPT prompts <= 512
               tokens) or long (BurstGPT prompts 1536-3584), rate in req/s
  (lengths and output lengths from the BurstGPT trace, Poisson arrivals)

The default is short then long (2 req/s, then 1 req/s). Long then short at a long-phase
rate below one group's long-request capacity re-calibrates twice without full effort:
  --phases long:0.4:5.5,short:2:2

Runs on the same arrivals:
  baseline   round robin over all pairs at MAX (the default system)
  frozen     CanaTune keeping the cold-start table (canary.recalibrate=false)
  adaptive   CanaTune with re-calibration (a length shift relocates the P side)

Reported per 30-minute window: energy against the baseline, goodput, groups at MAX,
and every Canary experiment (trigger, simulated duration, H before / after); per run
the time in full effort and the Canary aborts. The simulated power is the
environment's; absolute energies are illustrative.

  python scripts/analysis/recal_sim.py --burstgpt BurstGPT_1.csv [--phases ...] [--seeds 3]
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


def parse_phases(spec: str) -> list[tuple[str, float, float]]:
    """ "long:0.4:5.5,short:2:2" -> [(kind, rate req/s, hours), ...]"""
    phases = []
    for item in spec.split(","):
        kind, rate, hours = item.split(":")
        if kind not in ("short", "long"):
            raise ValueError(f"phase kind {kind!r}: short or long")
        phases.append((kind, float(rate), float(hours)))
    return phases


def phased_trace(pools, phases, window_s, seed):
    """Poisson arrivals phase after phase; windows named <phase letter><index>, the
    phase letter A, B, ... in order."""
    rng = random.Random(seed)
    rows, t = [], 0.0
    for i, (kind, rate, hours) in enumerate(phases):
        pairs, tag = pools[kind], chr(ord("A") + i)
        start = t
        while True:
            t += rng.expovariate(rate)
            if t >= start + hours * 3600:
                break
            k = int((t - start) // window_s)
            rows.append((t, f"{tag}{k}", *pairs[rng.randrange(len(pairs))]))
        t = start + hours * 3600
    end = t
    names = list(dict.fromkeys(r[1] for r in rows))  # in arrival order
    phases = []
    for n in names:
        ts = [r[0] for r in rows if r[1] == n]
        phases.append({"name": n, "start_s": min(ts), "end_s": max(ts)})
    arrivals = [loadgen.Arrival(i, *row) for i, row in enumerate(rows)]
    return {"phases": phases, "duration_s": end}, arrivals


def full_effort(control: list[dict], end_s: float) -> tuple[int, float]:
    """(entries, seconds) in full effort, from the Controller's mode changes."""
    entries, seconds, since = 0, 0.0, None
    for c in control:
        if c.get("event") != "control_mode":
            continue
        if c["mode"] == "full_effort" and since is None:
            entries, since = entries + 1, c["t"]
        elif c["mode"] != "full_effort" and since is not None:
            seconds, since = seconds + c["t"] - since, None
    if since is not None:
        seconds += end_s - since
    return entries, seconds


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
    res["full_effort"] = full_effort(res["control"], meta["duration_s"])
    res["aborts"] = sum(c.get("event") == "canary_abort" for c in res["control"])
    res["wall_s"] = round(time.monotonic() - started)
    return res


def report(seed, meta, results) -> None:
    print(f"\n== seed {seed} ==")
    names = [p["name"] for p in meta["phases"]]
    base = results.get("baseline")
    print(f"{'window':>7} " + " ".join(f"{p:>32}" for p in results if p != "baseline")
          + f" {'baseline goodput':>17}")  # fmt: skip
    for n in names + ["total"]:
        cells = []
        for policy, res in results.items():
            if policy == "baseline":
                continue
            e = (
                res[n].get("energy_kj")
                if n != "total"
                else sum(res[m].get("energy_kj") or 0 for m in names)
            )
            b = (
                (
                    base[n].get("energy_kj")
                    if n != "total"
                    else sum(base[m].get("energy_kj") or 0 for m in names)
                )
                if base
                else None
            )
            de = f"{(e / b - 1) * 100:+6.1f}%" if e and b else "     -"
            cells.append(f"dE {de} gp {res[n]['goodput']:.3f} MAX {res[n]['max_group_s']:5d}s")
        print(f"{n:>7} " + " ".join(f"{c:>32}" for c in cells)
              + (f" {base[n]['goodput']:>17.3f}" if base else ""))  # fmt: skip
    for policy, res in results.items():
        entries, seconds = res["full_effort"]
        print(f"{policy}: full effort {entries}x {seconds / 60:.1f} min, Canary aborts "
              f"{res['aborts']}, {res['wall_s']} s wall")  # fmt: skip
        for e in res["experiments"]:
            print(f"  t={e['start_s'] / 60:6.1f} min {e['kind']}/{e['reason']} "
                  f"{e['duration_s'] / 60:.0f} min ({e['windows']} windows) {e['outcome']}: "
                  f"H {e['h_before']} -> {e['h_after']}, C_H {e['c_h_rps_after']}, "
                  f"lengths {e['lengths']}")  # fmt: skip


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--burstgpt", required=True, help="BurstGPT CSV (lengths only)")
    ap.add_argument("--physics", default="l4g", choices=sorted(sim.PHYSICS))
    ap.add_argument("--groups", type=int, default=4)
    ap.add_argument("--phases", default="short:2:2,long:1:2", help="kind:rate:hours,...")
    ap.add_argument("--window-s", type=float, default=1800.0)
    ap.add_argument("--slo", default="600,100")
    ap.add_argument("--seed", type=int, default=20261030)
    ap.add_argument("--seeds", type=int, default=1, help="traces seed, seed+1, ...")
    ap.add_argument("--policies", default="baseline,frozen,adaptive")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    sim.configure_slo(*(float(x) for x in args.slo.split(",")))
    sim.CONFIG_OVERRIDES.update({"router.group_choice": "fill", "controller.solver": False})
    phys = sim.PHYSICS[args.physics]

    short, long = phase_pairs(args.burstgpt, 4096)
    mean = lambda ps, i: sum(p[i] for p in ps) / len(ps)  # noqa: E731
    print(f"short: {len(short)} BurstGPT pairs, prompt {mean(short, 0):.0f} / output "
          f"{mean(short, 1):.0f}; long: {len(long)} pairs, prompt {mean(long, 0):.0f} / "
          f"output {mean(long, 1):.0f}")  # fmt: skip
    table, backend = sim.run_canary(phys, COLD)
    print(f"cold-start Canary: {backend.windows} windows, {backend.sim_seconds / 60:.0f} min "
          f"simulated; H {table.h.key()}, C_H {table.evidence.get('capacity_rps', 0):.2f} req/s "
          f"(default lengths)")  # fmt: skip
    phases = parse_phases(args.phases)
    print("phases: " + ", ".join(f"{chr(ord('A') + i)} {k} {r} req/s {h} h"
                                 for i, (k, r, h) in enumerate(phases)))  # fmt: skip
    runs = []
    for seed in range(args.seed, args.seed + args.seeds):
        meta, arrivals = phased_trace({"short": short, "long": long}, phases, args.window_s, seed)
        results = {}
        for policy in args.policies.split(","):
            results[policy] = run(phys, table, policy, meta, arrivals, args.groups, seed)
        report(seed, meta, results)
        runs.append({"seed": seed, "results": results})
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "recal_sim.json").write_text(
            json.dumps(
                {"cold_h": table.h.key(), "phases": phases, "runs": runs}, indent=1, default=str
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
