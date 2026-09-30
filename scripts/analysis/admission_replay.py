"""Offline admission replay on measured requests (r6b / r7 `canatune.measure` output).

For every request, rebuild the state the proxy could see when it arrived (from the
requests of the same window: P send / P done / first token / done times):
  own S_f(L), pending P work (sum S_f of requests still at P), KV bytes in flight
  (P done, first token not yet), requests decoding on D, D clock.
Then compare admission rules, trained on one run and tested on the other:
  all        admit everything
  count      the v1 risk table: cells (queued-at-P bucket, prompt bucket), risk =
             Wilson UCB of the training violations, admit if <= theta
  slack      predicted TTFT (linear model on the state) -> slack = SLO - prediction,
             cells = slack buckets, same UCB rule; plus a hard gate on KV bytes in
             flight <= fraction x kv_buffer_size
Open loop: a rejection does not change later requests' state, so the replay counts
more violations than a closed-loop system would see after rejecting.

  python scripts/analysis/admission_replay.py --r6b DIR --r7 DIR
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

KV_BYTES_PER_TOKEN = 2 * 32 * 8 * 128 * 2  # Mistral-7B, bf16: K and V x layers x kv heads x dim
TTFT_SLO_MS = 1000.0
TPOT_SLO_MS = 200.0

# Single-request P prefill time (ms) by prompt length, per P clock (r6b minima / r7 medians).
LUT = {
    2520: {16: 30, 128: 33, 256: 42, 512: 55, 1024: 82, 2048: 150, 3072: 228},
    1545: {16: 45, 128: 52, 256: 58, 512: 75, 1024: 98, 2048: 177, 3072: 264},
    1080: {16: 29, 128: 33, 256: 41, 512: 71, 1024: 131, 2048: 240, 3072: 362},
}


def s_of(mhz: int, length: int) -> float:
    table = LUT[min(LUT, key=lambda f: abs(f - mhz))]
    keys = sorted(table)
    if length <= keys[0]:
        return float(table[keys[0]])
    for a, b in zip(keys, keys[1:]):
        if length <= b:
            return table[a] + (table[b] - table[a]) * (length - a) / (b - a)
    return float(table[keys[-1]]) * length / keys[-1]


def wilson_ucb(k: int, n: int, z: float = 1.28) -> float:
    if n == 0:
        return 1.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c + s) / d


def window_clocks(key: str) -> tuple[int, int] | None:
    parts = key.split("|")
    if parts[0] == "load" and len(parts) == 4:  # r6b: load|P|mix|rate, D 1170
        return int(parts[1]), 1170
    if parts[0] == "scan":  # r7: scan|tag|name|P|D|mix|pairs|rate
        return int(parts[3]), int(parts[4])
    return None


def load(run_dir: Path, run: str, exclude: set[str]) -> list[dict]:
    rows = [json.loads(line) for line in (run_dir / "requests.jsonl").read_text().splitlines()]
    out = []
    for key in dict.fromkeys(r["key"] for r in rows):
        clocks = window_clocks(key)
        if clocks is None or key in exclude or "|put|" in key:
            continue
        window = [r for r in rows if r["key"] == key]
        for r in window:
            ok = r["status"] == "ok" and r["ttft_ms"] is not None and r["prefill_ms"] is not None
            r["_ok"] = ok
            r["_pdone"] = r["sent_s"] + (r["prefill_ms"] or 0) / 1000
            r["_first"] = r["sent_s"] + (r["ttft_ms"] or 1e9) / 1000
        for r in window:
            pair = r.get("pair", "G0")
            t = r["sent_s"]
            same = [o for o in window if o.get("pair", "G0") == pair and o is not r]
            at_p = [o for o in same if o["sent_s"] < t < o["_pdone"]]
            flying = [o for o in same if o["_pdone"] <= t < o["_first"]]
            decoding = [o for o in same if o["_first"] <= t < o["done_s"]]
            violated = (
                (not r["_ok"]) or r["ttft_ms"] > TTFT_SLO_MS or ((r["tpot_ms"] or 0) > TPOT_SLO_MS)
            )
            out.append(
                {
                    "run": run,
                    "key": key,
                    "rate": float(key.split("|")[-1]),
                    "p_mhz": clocks[0],
                    "d_mhz": clocks[1],
                    "s_own": s_of(clocks[0], r["prompt_tokens"]),
                    "prompt": r["prompt_tokens"],
                    "pending_ms": sum(s_of(clocks[0], o["prompt_tokens"]) for o in at_p),
                    "n_at_p": len(at_p),
                    "inflight_gb": sum(o["prompt_tokens"] for o in flying)
                    * KV_BYTES_PER_TOKEN
                    / 1e9,
                    "decoding": len(decoding),
                    "violated": violated,
                    "ttft": r["ttft_ms"] if r["_ok"] else 60000.0,
                }
            )
    return out


def features(rows: list[dict]) -> np.ndarray:
    return np.array(
        [
            [
                1.0,
                r["s_own"],
                r["pending_ms"],
                r["inflight_gb"],
                r["decoding"],
                1.0 if r["d_mhz"] <= 800 else 0.0,
                1.0 if r["d_mhz"] >= 1800 else 0.0,
            ]
            for r in rows
        ]
    )


def count_cell(r: dict) -> tuple[int, int]:
    n = r["n_at_p"]
    queue = 0 if n == 0 else 1 if n == 1 else 2 if n == 2 else 3 if n == 3 else 4 if n <= 5 else 5
    prompt = sum(r["prompt"] > edge for edge in (128, 512, 1024, 2048))
    return queue, prompt


def evaluate(train: list[dict], test: list[dict], theta: float, gate_gb: float) -> dict:
    # v1: count x prompt cells
    cells: dict = {}
    for r in train:
        cells.setdefault(count_cell(r), []).append(r["violated"])
    count_risk = {c: wilson_ucb(sum(v), len(v)) for c, v in cells.items()}
    # new: linear TTFT prediction -> slack buckets (clipped ttft keeps the fit sane)
    x_train = features(train)
    y_train = np.minimum([r["ttft"] for r in train], 3000.0)
    coef, *_ = np.linalg.lstsq(x_train, y_train, rcond=None)
    edges = np.arange(-400, 900, 100)

    def bucket(pred: float) -> int:
        return int(np.searchsorted(edges, TTFT_SLO_MS - pred))

    slack_cells: dict = {}
    for r, p in zip(train, x_train @ coef):
        slack_cells.setdefault(bucket(p), []).append(r["violated"])
    slack_risk = {b: wilson_ucb(sum(v), len(v)) for b, v in slack_cells.items()}
    pred_test = features(test) @ coef

    rules = {
        "all": [True] * len(test),
        "count": [count_risk.get(count_cell(r), 1.0) <= theta for r in test],
        "slack": [
            slack_risk.get(bucket(p), 1.0) <= theta and r["inflight_gb"] <= gate_gb
            for r, p in zip(test, pred_test)
        ],
    }
    out = {
        "coef": dict(
            zip(
                ["1", "S_own", "pending", "inflight_gb", "decoding", "D735", "D2040"],
                np.round(coef, 2).tolist(),
            )
        )
    }
    total_viol = sum(r["violated"] for r in test)
    for name, admit in rules.items():
        adm = [r for r, a in zip(test, admit) if a]
        rej = [r for r, a in zip(test, admit) if not a]
        out[name] = {
            "admitted": len(adm) / len(test),
            "viol_admitted": sum(r["violated"] for r in adm) / max(len(adm), 1),
            "violations_removed": 1 - sum(r["violated"] for r in adm) / max(total_viol, 1),
            "rejected_ok": sum(not r["violated"] for r in rej) / len(test),
        }
    out["by_rate"] = {}
    for rate in sorted({r["rate"] for r in test}):
        idx = [i for i, r in enumerate(test) if r["rate"] == rate]
        row = {}
        for name, admit in rules.items():
            adm = [test[i] for i in idx if admit[i]]
            row[name] = (
                len(adm) / len(idx),
                sum(r["violated"] for r in adm) / max(len(adm), 1),
            )
        out["by_rate"][rate] = row
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--r6b", type=Path, required=True, help="r6b measure/data directory")
    ap.add_argument("--r7", type=Path, required=True, help="r7 async/data directory")
    ap.add_argument("--theta", type=float, default=0.10)
    ap.add_argument("--kv-buffer-gb", type=float, default=1.0, help="D kv_buffer_size")
    ap.add_argument("--gate-fraction", type=float, default=0.8)
    args = ap.parse_args()
    r6b_bad = {"mix|2520|short|0.5", "mix|2520|long|2", "mix|2520|long|3", "mix|2520|short|0.5"}
    r7_bad = {"scan|async|mix|1545|1170|long|1|3.0"}
    runs = {
        "r6b": load(args.r6b, "r6b", r6b_bad),
        "r7": load(args.r7, "r7", r7_bad),
    }
    gate = args.gate_fraction * args.kv_buffer_gb
    for train, test in (("r6b", "r7"), ("r7", "r6b")):
        result = evaluate(runs[train], runs[test], args.theta, gate)
        n = len(runs[test])
        v = sum(r["violated"] for r in runs[test])
        print(f"\n=== train {train} -> test {test}: {n} requests, {v} violations ({v / n:.1%})")
        print(f"  TTFT model coef: {result['coef']}")
        for name in ("all", "count", "slack"):
            m = result[name]
            print(
                f"  {name:6s} admitted {m['admitted']:6.1%}  violation among admitted "
                f"{m['viol_admitted']:6.2%}  violations removed {m['violations_removed']:6.1%}  "
                f"rejected-but-ok {m['rejected_ok']:6.1%}"
            )
        print("  by offered rate (admitted share / violation among admitted):")
        for rate, row in result["by_rate"].items():
            cells = "  ".join(f"{k} {a:5.1%}/{b:5.1%}" for k, (a, b) in row.items())
            print(f"    {rate:>4} req/s  {cells}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
