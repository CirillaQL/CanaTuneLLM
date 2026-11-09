"""Replay a Vera job's `vllm bench serve` stages through the real Router and Controller
on the simulated clock of overload_sim.py, with the A100 physics taken from that job's
own Canary table. Routing and control policies can be compared offline this way; the
result is a simulation, not a measurement: run the policy on the GPUs before using its
numbers.

  python scripts/analysis/a100_replay.py JOB_DIR [--variant NAME=key=json,key=json ...]
      [--stages sharegpt_r5_18rps,...] [--out FILE]

Physics (sources: JOB_DIR/tier_table.json evidence unless noted):
  P  one prompt at a time, S(L) = plateau + slope x max(0, L - knee) per clock
     (prefill_fit), power idle + utilization x dynamic (model.power_prefill)
  KV NIXL, no receive buffer; residence 20 ms + 12 ms per 2048 tokens (10645497
     residence_ms_by_length) gives the link rate
  D  steps of alpha + beta X + gamma K (length_fit without its r95), sequences limited
     to 256 (job 10645497 peaks at 255-256 and TTFT then rises: assumed vLLM's limit)
  First token 1.15 ms per running sequence later (10645497: TTFT p50 152 ms at 25
  sequences, 412 ms at 245) on top of the fixed D-first overhead (--d-first-extra-ms)
  Proxy overhead 10 ms + exponential(5 ms) (request records: admit + to_prefill)
"""

import argparse
import asyncio
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

from canatune.controller.locator import LocatorRun
from canatune.domain.groups import TierTable

_spec = importlib.util.spec_from_file_location(
    "overload_sim", Path(__file__).with_name("overload_sim.py")
)
sim_mod = importlib.util.module_from_spec(_spec)
sys.modules["overload_sim"] = sim_mod
_spec.loader.exec_module(sim_mod)

COLD_LENGTHS = [(p, 256) for p in (128, 512, 1024, 2048)]


def a100_physics(table: TierTable, *, d_first_extra_ms: float, overhead_ms: float):
    evidence = table.evidence
    fit = evidence["prefill_fit"]
    p_model = tuple(
        (int(f), r["plateau_ms"], r["slope_us_per_token"] / 1000.0, r["knee_tokens"])
        for f, r in sorted(fit.items(), key=lambda kv: int(kv[0]))
    )
    d_model = tuple(
        (int(f), c[0], c[1], c[2])
        for f, c in sorted(table.decode_length.items(), key=lambda kv: int(kv[0]))
    )
    model = evidence["model"]

    def rows(raw):
        return tuple((int(f), v[0], v[1]) for f, v in sorted(raw.items(), key=lambda x: int(x[0])))

    p_clocks = tuple(int(f) for f in sorted(fit, key=int))
    d_clocks = tuple(sorted({int(f) for f in model["power_decode"]}))
    return replace(
        sim_mod.Physics(),
        name="a100",
        kv_buffer_bytes=1e13,
        link_bytes_s=2048 * 131072 / 0.012,
        d_first_extra_ms=d_first_extra_ms,
        d_first_per_running_ms=1.15,
        d_kv_tokens=int(table.decode_kv_tokens),
        d_max_running=256,
        overhead_ms=overhead_ms,
        overhead_jitter_ms=5.0,
        p_batch_tokens=1,  # always one prompt: the first of a batch is taken regardless
        p_f_eff=p_clocks[-1],
        p_clocks=p_clocks,
        d_clocks=d_clocks,
        p_model=p_model,
        d_model=d_model,
        p_power_model=rows(model["power_prefill"]),
        d_power_model=rows(model["power_decode"]),
        d_admit="vllm",
    )


def load_job_config(job: Path) -> None:
    """The job's router and controller settings for every simulated config."""
    import yaml

    raw = yaml.safe_load((job / "config.yaml").read_text())
    for section in ("router", "controller"):
        for key, value in (raw.get(section) or {}).items():
            sim_mod.CONFIG_OVERRIDES[f"{section}.{key}"] = value
    sim_mod.configure_slo(raw["experiment"]["slo"]["ttft_ms"], raw["experiment"]["slo"]["tpot_ms"])


def measure_room(table: TierTable, phys, lengths) -> dict:
    """evidence["room"] by the real Canary step on the simulated pair (tables published
    before that step have none)."""
    prompts = sorted({p for p, _ in lengths})
    backend = sim_mod.SimBackend(phys, lengths, seed=7, salt=0)
    settings = sim_mod.LocatorSettings.from_config(sim_mod.sim_config())
    locator = sim_mod.TierLocator(backend, settings)
    locator.mean_prompt = sum(prompts) / len(prompts)
    locator.run = LocatorRun(started_at=0.0)
    locator.run.evidence.update(
        {k: table.evidence[k] for k in ("model", "admission", "decode", "prefill_ms_by_clock")}
    )
    asyncio.run(locator.room_limits(table.h, table.alpha_tokens, table.capacity_h, None))
    return locator.run.evidence["room"]


def run_variant(job, table, phys, meta, arrivals, overrides, seed, n_groups=2):
    sim = sim_mod.Sim(
        n_groups, "best_effort", table.h, router_overrides=overrides, phys=phys, table=table,
        solver=False, lengths=COLD_LENGTHS, seed=seed,
    )  # fmt: skip
    sim.groups["G0"].canary = True  # production first: G1, then the Canary
    sim.run(arrivals, meta["duration_s"])
    result = sim_mod.summarize(sim, meta["phases"])
    stages = {}
    for phase in meta["phases"]:
        name = phase["name"]
        rows = [r for r in sim.requests if r.phase == name and r.status == "ok"]
        row = result[name]
        span = max(r.done for r in rows) - min(r.at for r in rows) if rows else 0.0
        timeline = [t for t in sim.timeline if phase["start_s"] <= t["t"] < phase["end_s"]]
        running = sorted(
            max(t[p.name]["running"] for t in timeline) if timeline else 0 for p in sim.pairs
        )
        stages[name] = {
            "goodput": row["goodput"],
            "ttft_p95": row["ttft_p95"],
            "ttft_p99": row["ttft_p99"],
            "tpot_p95": row["tpot_p95"],
            "power_w": round(1000 * row.get("energy_kj", 0.0) / span) if span else None,
            "max_s": row["max_group_s"] / 2,
            "stage_s": len(timeline),
            "g0_share": round(sum(r.pair.name == "G0" for r in rows) / max(len(rows), 1), 3),
            "d_running_max": running,
        }
    control = [
        {"t": c["t"], "mode": c.get("mode"), "reason": c.get("reason")}
        for c in result.get("control", [])
    ]
    return {"stages": stages, "control": control}


def parse_variant(text: str) -> tuple[str, dict]:
    name, _, body = text.partition("=")
    overrides = {}
    for item in filter(None, body.split(",")):
        key, _, value = item.partition("=")
        overrides[key] = json.loads(value)
    return name, overrides


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("job", type=Path)
    ap.add_argument("--variant", action="append", default=[])
    ap.add_argument("--stages", default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--d-first-extra-ms", type=float, default=60.0)
    ap.add_argument("--overhead-ms", type=float, default=10.0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--step-jitter-ms", type=float, default=0.0)
    ap.add_argument("--stall-prob", type=float, default=0.0)
    ap.add_argument("--stall-ms", type=float, default=0.0)
    ap.add_argument("--groups", type=int, default=2)
    ap.add_argument("--table", type=Path, default=None, help="another job: its Canary table")
    ap.add_argument("--rates", default=None, help="Poisson stages (req/s) of the job's lengths")
    args = ap.parse_args()
    table_job = args.table or args.job
    table = TierTable.from_json(json.loads((table_job / "tier_table.json").read_text())["table"])
    load_job_config(args.job)
    phys = a100_physics(
        table, d_first_extra_ms=args.d_first_extra_ms, overhead_ms=args.overhead_ms
    )
    phys = replace(phys, d_step_jitter_ms=args.step_jitter_ms, d_stall_prob=args.stall_prob,
                   d_stall_ms=args.stall_ms)  # fmt: skip
    if "room" not in table.evidence:
        table.evidence["room"] = measure_room(table, phys, COLD_LENGTHS)
    room = table.evidence["room"]
    print(f"room limits: P busy {room['p_rho']} ({room['p_source']}), "
          f"D sequences {room['d_sequences']} ({room['d_source']}, reached {room['d_reached']})")
    if args.rates:
        rates = [float(x) for x in args.rates.split(",")]
        meta, arrivals = sim_mod.synthetic_trace(args.job, rates, 180.0, 30.0, args.seed)
    else:
        meta, arrivals = sim_mod.replay_trace(args.job)
    if args.stages:
        keep = args.stages.split(",")
        arrivals = [a for a in arrivals if a.phase in keep]
        meta["phases"] = [p for p in meta["phases"] if p["name"] in keep]
        t0 = min(a.at_s for a in arrivals)
        arrivals = [replace(a, at_s=a.at_s - t0) for a in arrivals]
        meta["phases"] = [
            {**p, "start_s": p["start_s"] - t0, "end_s": p["end_s"] - t0} for p in meta["phases"]
        ]
        meta["duration_s"] = meta["phases"][-1]["end_s"]
    variants = dict(parse_variant(v) for v in args.variant) or {"job": {}}
    measured = json.loads((args.job / "bench_summary.json").read_text())["stages"]
    out = {"measured": measured, "variants": {}}
    for name, overrides in variants.items():
        out["variants"][name] = run_variant(args.job, table, phys, meta, arrivals, overrides,
                                             args.seed, args.groups)  # fmt: skip
        print(f"== {name} {overrides}")
        print(f"{'stage':>20} {'good':>6} {'p95':>6} {'p99':>6} {'tpot95':>6} {'W':>5} {'maxs':>5} "
              f"{'G0':>5} {'Dmax':>10} | measured good p95 p99 tpot95 W")  # fmt: skip
        for stage, s in out["variants"][name]["stages"].items():
            m = measured.get(stage, {})
            print(f"{stage:>20} {s['goodput']:>6.3f} {s['ttft_p95']:>6} {s['ttft_p99']:>6} "
                  f"{s['tpot_p95']:>6} {s['power_w']:>5} {s['max_s']:>5.0f} {s['g0_share']:>5.2f} "
                  f"{str(s['d_running_max']):>14} | {m.get('goodput', 0):.3f} "
                  f"{m.get('ttft_p95_ms', 0):.0f} {m.get('ttft_p99_ms', 0):.0f} "
                  f"{m.get('tpot_p95_ms', 0):.0f} {m.get('power_w', 0):.0f}")  # fmt: skip
        print("control:", [(c["t"], c["mode"], c["reason"])
                           for c in out["variants"][name]["control"]][:14])  # fmt: skip
    if args.out:
        args.out.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
