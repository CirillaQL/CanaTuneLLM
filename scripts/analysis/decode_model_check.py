"""Offline check of the D iteration model: count-only `alpha + delta X` against the
length-aware `alpha + beta X + gamma K` (Ramani & Tantawi, arXiv 2609.20957: a
decode step costs beta + gamma x context; summed over the batch, K = context tokens
held by the running sequences). Research only: the fitted numbers decide whether
the model is worth adopting; a deployment still measures its own in the Canary.

Each request is one sample. From its first-token and finish times it occupies its D
for [first, done]; its context grows from prompt + 1 to prompt + output. Summing all
requests per D on a fine grid gives X(t) (sequences decoding) and K(t) (their
context tokens); a request's TPOT is regressed on the time averages of X and K over
its own decode interval. Sources:

  r6b    canatune.measure (one pair, locked clocks): decode windows (closed loop,
         concurrency 1..32, output 256, D 2040/1170/735) and load/mix windows
         (Poisson, output 64, D at the base clock; short and long prompt mixes)
  job    a CanaTune run directory (requests.jsonl per run; cantune clocks from
         events.jsonl, every run's actual SM clocks from energy.jsonl.gz)

  python scripts/analysis/decode_model_check.py --r6b DIR --job DIR [--out DIR]
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

GRID_S = 0.05


@dataclass
class Req:
    source: str  # "r6b:decode", "E:cantune", ...
    decode: str  # D endpoint (one occupancy timeline per D and source family)
    clock: int  # nominal D clock (MHz)
    first: float  # first token (s, source clock)
    done: float
    prompt: int
    output: int
    tpot_ms: float
    sm_mhz: float | None = None  # measured mean SM clock over the decode interval
    x_mean: float = 0.0
    k_mean: float = 0.0
    tags: dict[str, Any] = field(default_factory=dict)

    @property
    def ctx_per_seq(self) -> float:
        return self.k_mean / self.x_mean if self.x_mean else 0.0


# ---- loading ------------------------------------------------------------------------


def load_r6b(data: Path) -> list[Req]:
    plan = json.loads((data / "plan.json").read_text())
    base_d = int(plan["base_decode_clock"])
    out = []
    for line in (data / "requests.jsonl").open():
        r = json.loads(line)
        stage = r["stage"]
        if stage == "lut" or r["status"] != "ok" or not r.get("tpot_ms"):
            continue
        n = int(r["output_tokens"])
        if n < 8:
            continue
        parts = r["key"].split("|")
        clock = int(parts[1]) if stage == "decode" else base_d
        done = float(r["done_s"])
        first = done - r["tpot_ms"] * (n - 1) / 1000.0
        tags = {"key": r["key"]}
        if stage == "mix":
            tags["mix"] = parts[2]
        out.append(
            Req(
                f"r6b:{stage}",
                "D0",
                clock,
                first,
                done,
                int(r["prompt_tokens"]),
                n,
                float(r["tpot_ms"]),
                tags=tags,
            )
        )
    return out


def _sm_series(run_dir: Path, decode_ip: str | None) -> dict[str, list[tuple[float, float]]]:
    """D endpoint -> [(wall time, SM MHz)] from the run's energy samples."""
    path = run_dir / "energy.jsonl.gz"
    if not path.exists():
        return {}
    series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    with gzip.open(path, "rt") as f:
        for line in f:
            s = json.loads(line)
            if decode_ip is not None and decode_ip not in s.get("agent", ""):
                continue
            for g in s.get("gpus", []):
                if g.get("sm_mhz") is not None:
                    series[f"D{g['index']}"].append((float(s["wall_time"]), float(g["sm_mhz"])))
    return series


def _cantune_clocks(run_dir: Path) -> dict[str, list[tuple[float, int]]]:
    clocks: dict[str, list[tuple[float, int]]] = defaultdict(list)
    path = run_dir / "events.jsonl"
    if path.exists():
        for line in path.open():
            e = json.loads(line)
            if e.get("event") == "clock" and str(e.get("endpoint", "")).startswith("D"):
                clocks[e["endpoint"]].append((float(e["wall_time"]), int(e["mhz"])))
    return clocks


def _at(series: list[tuple[float, Any]], t: float) -> Any:
    value = None
    for ts, v in series:
        if ts > t:
            break
        value = v
    return value


def node_ip(job: Path, node: str) -> str | None:
    """The node's address from the job's gpu_allocation.txt ("ips name=addr ...")."""
    path = job / "gpu_allocation.txt"
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        if line.startswith("ips "):
            for item in line.split()[1:]:
                name, _, addr = item.partition("=")
                if name == node:
                    return addr
    return None


def load_job(job: Path, label: str, decode_node: str, default_clock: int) -> list[Req]:
    decode_ip = node_ip(job, decode_node)
    if decode_ip is None:
        raise SystemExit(f"{job}: no address for decode node {decode_node!r} in gpu_allocation.txt")
    out = []
    for run in ("cantune", "baseline"):
        run_dir = job / run
        path = run_dir / "requests.jsonl"
        if not path.exists():
            continue
        sm = _sm_series(run_dir, decode_ip)
        clocks = _cantune_clocks(run_dir)
        for line in path.open():
            r = json.loads(line)
            if r.get("event") != "request" or r.get("status") != "ok" or not r.get("tpot_ms"):
                continue
            n = int(r["output_tokens"])
            if n < 8:
                continue
            done = float(r["wall_time"])
            first = done - r["tpot_ms"] * (n - 1) / 1000.0
            d = r["decode"]
            clock = _at(clocks.get(d, []), first) or default_clock
            pts = [mhz for ts, mhz in sm.get(d, []) if first <= ts <= done]
            out.append(
                Req(
                    f"{label}:{run}",
                    d,
                    int(clock),
                    first,
                    done,
                    int(r["prompt_tokens"]),
                    n,
                    float(r["tpot_ms"]),
                    sm_mhz=float(np.mean(pts)) if pts else None,
                )
            )
    return out


# ---- occupancy ----------------------------------------------------------------------


def occupancy(reqs: list[Req]) -> None:
    """Fill x_mean / k_mean: time averages over each request's decode interval of the
    sequences decoding on its D and their context tokens (itself included)."""
    by_timeline: dict[tuple[str, str], list[Req]] = defaultdict(list)
    for r in reqs:
        by_timeline[(r.source, r.decode)].append(r)
    for rs in by_timeline.values():
        t0 = min(r.first for r in rs)
        t1 = max(r.done for r in rs)
        n = int(math.ceil((t1 - t0) / GRID_S)) + 2
        x = np.zeros(n)
        k = np.zeros(n)
        spans = []
        for r in rs:
            i0 = int((r.first - t0) / GRID_S)
            i1 = max(i0 + 1, int((r.done - t0) / GRID_S))
            ts = t0 + (np.arange(i0, i1) + 0.5) * GRID_S
            tokens = 1 + (ts - r.first) * 1000.0 / r.tpot_ms
            x[i0:i1] += 1
            k[i0:i1] += r.prompt + np.clip(tokens, 1, r.output)
            spans.append((r, i0, i1))
        for r, i0, i1 in spans:
            r.x_mean = float(x[i0:i1].mean())
            r.k_mean = float(k[i0:i1].mean())


# ---- models -------------------------------------------------------------------------


def design(reqs: list[Req], model: str) -> np.ndarray:
    x = np.array([r.x_mean for r in reqs])
    one = np.ones_like(x)
    if model == "count":
        return np.column_stack([one, x])
    k = np.array([r.k_mean for r in reqs])
    return np.column_stack([one, x, k])


def fit(reqs: list[Req], model: str) -> list[float] | None:
    if len(reqs) < 8:
        return None
    a = design(reqs, model)
    y = np.array([r.tpot_ms for r in reqs])
    coef, *_ = np.linalg.lstsq(a, y, rcond=None)
    return [float(c) for c in coef]


def bootstrap(reqs: list[Req], model: str, n: int = 300, seed: int = 0) -> list[list[float]] | None:
    """5th / 95th percentile of each coefficient over request resamples: wide
    intervals mean X and K are too collinear to separate beta from gamma."""
    if len(reqs) < 8:
        return None
    rng = np.random.default_rng(seed)
    a = design(reqs, model)
    y = np.array([r.tpot_ms for r in reqs])
    coefs = []
    for _ in range(n):
        idx = rng.integers(0, len(reqs), len(reqs))
        coefs.append(np.linalg.lstsq(a[idx], y[idx], rcond=None)[0])
    c = np.array(coefs)
    return [
        [float(np.percentile(c[:, j], 5)), float(np.percentile(c[:, j], 95))]
        for j in range(c.shape[1])
    ]


def interpolated(
    fits: dict[str, dict[str, Any]], prefix: str, model: str, mhz: float
) -> list[float] | None:
    """Coefficients at an unmeasured clock: linear between the nearest fitted clocks."""
    pts = sorted(
        (int(k.split("@")[1]), e[model])
        for k, e in fits.items()
        if k.startswith(prefix + "@") and e[model] is not None
    )
    if not pts:
        return None
    if mhz <= pts[0][0]:
        return pts[0][1]
    for (f0, c0), (f1, c1) in zip(pts, pts[1:]):
        if mhz <= f1:
            w = (mhz - f0) / (f1 - f0)
            return [a + w * (b - a) for a, b in zip(c0, c1)]
    return pts[-1][1]


def predict(reqs: list[Req], model: str, coef: list[float]) -> np.ndarray:
    return design(reqs, model) @ np.array(coef)


def score(reqs: list[Req], pred: np.ndarray) -> dict[str, float]:
    y = np.array([r.tpot_ms for r in reqs])
    err = pred - y
    ss = float(((y - y.mean()) ** 2).sum())
    return {
        "n": len(reqs),
        "mae_ms": float(np.abs(err).mean()),
        "mape": float((np.abs(err) / y).mean()),
        "bias_ms": float(err.mean()),
        "r2": 1 - float((err**2).sum()) / ss if ss > 0 else float("nan"),
        "measured_mean_ms": float(y.mean()),
    }


def by_length(reqs: list[Req], preds: dict[str, np.ndarray], edges: list[float]) -> list[dict]:
    """Bias per bin of context per sequence (K/X): a count model drifts with length."""
    ctx = np.array([r.ctx_per_seq for r in reqs])
    rows = []
    for lo, hi in zip(edges, edges[1:]):
        mask = (ctx >= lo) & (ctx < hi)
        if mask.sum() < 5:
            continue
        sub = [r for r, m in zip(reqs, mask) if m]
        row = {
            "ctx_per_seq": [lo, hi],
            "n": int(mask.sum()),
            "measured_ms": float(np.mean([r.tpot_ms for r in sub])),
        }
        for name, p in preds.items():
            row[f"{name}_bias_ms"] = float((p[mask] - row["measured_ms"]).mean())
            row[f"{name}_mae_ms"] = float(
                np.abs(p[mask] - np.array([r.tpot_ms for r in sub])).mean()
            )
        rows.append(row)
    return rows


def capacity(model: str, coef: list[float], slo_ms: float, ctx: float) -> float | None:
    """Sequences at which the predicted TPOT reaches the SLO, context per sequence ctx."""
    per_seq = coef[1] + (coef[2] * ctx if model == "length" else 0.0)
    return (slo_ms - coef[0]) / per_seq if per_seq > 0 else None


# ---- report -------------------------------------------------------------------------


def describe(reqs: list[Req]) -> dict[str, Any]:
    x = np.array([r.x_mean for r in reqs])
    k = np.array([r.k_mean for r in reqs])
    ctx = np.array([r.ctx_per_seq for r in reqs])
    return {
        "n": len(reqs),
        "x_mean": [float(x.min()), float(np.median(x)), float(x.max())],
        "ctx_per_seq": [
            float(np.percentile(ctx, 5)),
            float(np.median(ctx)),
            float(np.percentile(ctx, 95)),
        ],
        "corr_x_k": float(np.corrcoef(x, k)[0, 1]) if len(reqs) > 2 and x.std() > 0 else None,
        "tpot_ms": [float(np.percentile([r.tpot_ms for r in reqs], q)) for q in (5, 50, 95)],
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--r6b", type=Path, required=True, help="r6b measure/data directory")
    ap.add_argument(
        "--job", type=Path, action="append", default=[], help="CanaTune results/<id> dir"
    )
    ap.add_argument(
        "--decode-node", default="ganymede", help="D node (address from gpu_allocation.txt)"
    )
    ap.add_argument(
        "--baseline-clock", type=int, default=2040, help="nominal D clock of unlocked runs"
    )
    ap.add_argument("--slo-ms", type=float, default=200.0)
    ap.add_argument("--kv-bytes-per-token", type=float, default=131072.0)  # Mistral-7B bf16
    ap.add_argument("--bandwidth-gbs", type=float, default=300.0)  # L4 nominal
    ap.add_argument("--kv-tokens", type=float, default=33648.0, help="D KV cache size (vLLM log)")
    ap.add_argument("--kv-limit", type=float, default=0.90, help="decode KV wall fraction")
    ap.add_argument("--out", type=Path, default=Path("decode_model_check"))
    args = ap.parse_args()

    reqs = load_r6b(args.r6b)
    for i, job in enumerate(args.job):
        reqs += load_job(job, chr(ord("E") + i), args.decode_node, args.baseline_clock)
    occupancy(reqs)

    groups: dict[tuple[str, int], list[Req]] = defaultdict(list)
    for r in reqs:
        groups[(r.source, r.clock)].append(r)

    report: dict[str, Any] = {
        "data": {f"{s}@{c}": describe(rs) for (s, c), rs in sorted(groups.items())}
    }

    # Fits per D clock on r6b (every r6b window at that clock), and on the decode
    # windows only (to test transfer to the load and mix windows' lengths).
    fits: dict[str, dict[str, Any]] = {}
    clocks = sorted({r.clock for r in reqs if r.source.startswith("r6b")})
    for f in clocks:
        train_all = [r for r in reqs if r.source.startswith("r6b") and r.clock == f]
        train_dec = [r for r in train_all if r.source == "r6b:decode"]
        for name, train in (("r6b_all", train_all), ("r6b_decode", train_dec)):
            entry = {
                "n": len(train),
                "corr_x_k": describe(train)["corr_x_k"] if len(train) > 2 else None,
            }
            for model in ("count", "length"):
                entry[model] = fit(train, model)
                entry[f"{model}_ci90"] = bootstrap(train, model)
            fits[f"{name}@{f}"] = entry
    report["fits"] = fits

    # Physical check of gamma: KV bytes per token over the memory bandwidth.
    gamma_hw = args.kv_bytes_per_token / (args.bandwidth_gbs * 1e9) * 1000.0
    report["gamma_hw_ms_per_token"] = gamma_hw

    edges = [0, 300, 600, 900, 1200, 1600, 2200, 4000]
    tests: dict[str, Any] = {}

    def evaluate(label: str, test: list[Req], fit_key: str) -> None:
        entry = fits.get(fit_key)
        if not test or entry is None or entry["count"] is None or entry["length"] is None:
            return
        preds = {m: predict(test, m, entry[m]) for m in ("count", "length")}
        tests[label] = {
            "fit": fit_key,
            "count": score(test, preds["count"]),
            "length": score(test, preds["length"]),
            "by_length": by_length(test, preds, edges),
        }

    for f in clocks:
        r6b_f = [r for r in reqs if r.source.startswith("r6b") and r.clock == f]
        dec = [r for r in r6b_f if r.source == "r6b:decode"]
        evaluate(f"r6b_all@{f} (in-sample)", r6b_f, f"r6b_all@{f}")
        evaluate(f"r6b_decode@{f} (in-sample)", dec, f"r6b_decode@{f}")
        other = [r for r in r6b_f if r.source != "r6b:decode"]
        evaluate(f"r6b_load_mix@{f} <- decode fit", other, f"r6b_decode@{f}")
        for mix in ("short", "long"):
            sub = [r for r in other if r.tags.get("mix") == mix]
            evaluate(f"r6b_mix_{mix}@{f} <- decode fit", sub, f"r6b_decode@{f}")
    for (source, f), rs in sorted(groups.items()):
        if source.startswith("r6b"):
            continue
        target = min(clocks, key=lambda c: abs(c - f))
        if source.endswith("baseline"):
            # Unlocked clocks (power capped under load): coefficients interpolated at
            # each request's measured mean SM clock.
            measured = [r for r in rs if r.sm_mhz is not None]
            if measured:
                preds = {
                    m: np.array(
                        [
                            predict([r], m, interpolated(fits, "r6b_all", m, r.sm_mhz))[0]
                            for r in measured
                        ]
                    )
                    for m in ("count", "length")
                }
                sm = [r.sm_mhz for r in measured]
                tests[f"{source} sm={min(sm):.0f}-{max(sm):.0f} <- r6b_all interp"] = {
                    "fit": "r6b_all interpolated at measured SM clock",
                    "count": score(measured, preds["count"]),
                    "length": score(measured, preds["length"]),
                    "by_length": by_length(measured, preds, edges),
                }
        else:
            evaluate(f"{source}@{f} <- r6b_all", rs, f"r6b_all@{target}")
            evaluate(f"{source}@{f} <- r6b_decode", rs, f"r6b_decode@{target}")
    for i, job in enumerate(args.job):
        label = chr(ord("E") + i)
        tiers = job / "cantune" / "tiers.json"
        if not tiers.exists():
            continue
        table = json.loads(tiers.read_text()).get("table") or {}
        decode = ((table.get("evidence") or {}).get("model") or {}).get("decode") or {}
        rs = [r for r in reqs if r.source == f"{label}:cantune" and str(r.clock) in decode]
        if rs:
            pred = np.array(
                [decode[str(r.clock)][0] + decode[str(r.clock)][1] * r.x_mean for r in rs]
            )
            tests[f"{label}:cantune <- own Canary count model"] = {
                "fit": "tiers.json evidence.model.decode",
                "count": score(rs, pred),
                "length": score(rs, pred),
                "by_length": by_length(rs, {"count": pred}, edges),
            }
        report.setdefault("deployed", {})[label] = {
            "decode_model": decode,
            "b_star": table.get("decode_max_running"),
            "kv_wall": ((table.get("evidence") or {}).get("decode") or {}).get("kv_wall"),
        }
    report["tests"] = tests

    # Reverse direction and self-fit on job data: does gamma come out the same?
    self_fits = {}
    for (source, f), rs in sorted(groups.items()):
        if source.startswith("r6b") or len(rs) < 50:
            continue
        sel = rs
        if source.endswith("baseline"):
            sel = [r for r in rs if r.sm_mhz is not None and abs(r.sm_mhz - f) <= 100]
        self_fits[f"{source}@{f}"] = {
            "n": len(sel),
            "count": fit(sel, "count"),
            "length": fit(sel, "length"),
            "corr_x_k": describe(sel)["corr_x_k"] if len(sel) > 2 else None,
        }
    report["self_fits"] = self_fits

    caps = {}
    for key, entry in fits.items():
        if not key.startswith("r6b_all") or entry["length"] is None:
            continue
        caps[key] = {
            "count": capacity("count", entry["count"], args.slo_ms, 0),
            **{
                f"length@ctx{c}": capacity("length", entry["length"], args.slo_ms, c)
                for c in (300, 600, 1200, 2400)
            },
        }
    caps["kv_wall"] = {
        f"length@ctx{c}": args.kv_limit * args.kv_tokens / c for c in (300, 600, 1200, 2400)
    }
    report["capacity_at_slo"] = caps

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(json.dumps(report, indent=1))
    with (args.out / "samples.jsonl").open("w") as f:
        for r in reqs:
            f.write(
                json.dumps(
                    {
                        "source": r.source,
                        "decode": r.decode,
                        "clock": r.clock,
                        "sm_mhz": r.sm_mhz,
                        "prompt": r.prompt,
                        "output": r.output,
                        "tpot_ms": r.tpot_ms,
                        "x_mean": r.x_mean,
                        "k_mean": r.k_mean,
                        **r.tags,
                    }
                )
                + "\n"
            )
    print_report(report)


def print_report(report: dict[str, Any]) -> None:
    print("== data (x_mean min/med/max; ctx/seq p5/med/p95; corr(X,K); TPOT p5/50/95)")
    for key, d in report["data"].items():
        print(
            f"  {key:24s} n={d['n']:5d} X={_f(d['x_mean'])} ctx={_f(d['ctx_per_seq'], 0)}"
            f" corr={_f(d['corr_x_k'], 2)} tpot={_f(d['tpot_ms'])}"
        )
    print(
        "== fits (count: alpha, delta | length: alpha, beta, gamma)"
        f"   gamma_hw={report['gamma_hw_ms_per_token']:.2e}"
    )
    for key, e in report["fits"].items():
        c, ln = e["count"], e["length"]
        print(
            f"  {key:18s} n={e['n']:4d} corr={_f(e['corr_x_k'], 2)}  count={_f(c, 3)}  "
            f"length=[{ln[0]:.2f}, {ln[1]:.3f}, {ln[2]:.2e}]"
            if ln
            else ""
        )
        ci = e.get("length_ci90")
        if ci:
            print(
                f"  {'':18s} 90% CI beta=[{ci[1][0]:.3f}, {ci[1][1]:.3f}]"
                f" gamma=[{ci[2][0]:.2e}, {ci[2][1]:.2e}]"
            )
    print("== self fits on job data")
    for key, e in report["self_fits"].items():
        c, ln = e["count"], e["length"]
        if ln:
            print(
                f"  {key:22s} n={e['n']:4d} corr={_f(e['corr_x_k'], 2)}  count={_f(c, 3)}  "
                f"length=[{ln[0]:.2f}, {ln[1]:.3f}, {ln[2]:.2e}]"
            )
    print("== tests (MAE ms / MAPE / bias ms / R2)")
    for key, t in report["tests"].items():
        c, ln = t["count"], t["length"]
        print(
            f"  {key:44s} n={c['n']:5d} meas={c['measured_mean_ms']:6.1f}  "
            f"count {c['mae_ms']:6.2f} {c['mape']:6.1%} {c['bias_ms']:+7.2f} {c['r2']:+.2f}  |  "
            f"length {ln['mae_ms']:6.2f} {ln['mape']:6.1%} {ln['bias_ms']:+7.2f} {ln['r2']:+.2f}"
        )
    for label, d in report.get("deployed", {}).items():
        print(
            f"== deployed ({label}): B*={d['b_star']} kv_wall={d['kv_wall']}"
            f" decode@1170={_f(d['decode_model'].get('1170'), 3)}"
        )
    print("== capacity at the TPOT SLO (sequences)")
    for key, c in report["capacity_at_slo"].items():
        print(f"  {key:18s} " + "  ".join(f"{k}={_f(v, 1)}" for k, v in c.items()))


def _f(v: Any, digits: int = 1) -> str:
    if v is None:
        return "-"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_f(x, digits) for x in v) + "]"
    return f"{v:.{digits}f}"


if __name__ == "__main__":
    main()
