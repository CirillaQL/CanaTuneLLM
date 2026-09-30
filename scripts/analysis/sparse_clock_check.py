"""E3 (offline, P and D energy curves): does a sparse set of clocks plus a smooth fit
find the lowest-energy clock that a dense sweep finds?

Input: tiers.json files of calibration runs. Each holds the energy per request (P)
or per token (D) measured at every clock the locator visited. For each curve and
each sparse design (k clocks spread over the measured range, MAX included), fit a
quadratic in f to the k points, take its minimum over the measured range, snap it
to the nearest measured clock and report the regret: E(chosen) / E(best) - 1.
A design passes when the regret is within eps (the locator's energy band, 2 %).

  python scripts/analysis/sparse_clock_check.py path/to/tiers.json ... [--eps 0.02]
"""

import argparse
import json
from pathlib import Path


def fit_quadratic(points: list[tuple[float, float]]) -> tuple[float, float, float]:
    """Least squares y = c0 + c1 x + c2 x^2 (x scaled to GHz for conditioning)."""
    rows = [(1.0, x / 1000.0, (x / 1000.0) ** 2) for x, _ in points]
    ys = [y for _, y in points]
    ata = [[sum(r[i] * r[j] for r in rows) for j in range(3)] for i in range(3)]
    aty = [sum(r[i] * y for r, y in zip(rows, ys)) for i in range(3)]
    # Gaussian elimination (3x3)
    m = [ata[i] + [aty[i]] for i in range(3)]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda r: abs(m[r][col]))
        m[col], m[pivot] = m[pivot], m[col]
        for r in range(3):
            if r != col and m[col][col]:
                f = m[r][col] / m[col][col]
                m[r] = [a - f * b for a, b in zip(m[r], m[col])]
    return tuple(m[i][3] / m[i][i] for i in range(3))


def sparse_design(clocks: list[int], k: int) -> list[int]:
    """k clocks spread evenly over [min, max], snapped to measured ones; MAX included."""
    lo, hi = min(clocks), max(clocks)
    targets = [lo + (hi - lo) * i / (k - 1) for i in range(k)]
    chosen = []
    for t in targets:
        best = min((c for c in clocks if c not in chosen), key=lambda c: abs(c - t))
        chosen.append(best)
    return sorted(chosen)


def choose(curve: dict[int, float], design: list[int]) -> int:
    c0, c1, c2 = fit_quadratic([(f, curve[f]) for f in design])
    lo, hi = min(curve), max(curve)
    if c2 > 0:
        x = -c1 / (2 * c2) * 1000.0
        x = min(max(x, lo), hi)
    else:  # no interior minimum: the cheaper end of the range
        x = (
            lo
            if c0 + c1 * lo / 1e3 + c2 * (lo / 1e3) ** 2 < c0 + c1 * hi / 1e3 + c2 * (hi / 1e3) ** 2
            else hi
        )
    return min(curve, key=lambda f: abs(f - x))


def refine(curve: dict[int, float], design: list[int]) -> tuple[int, int]:
    """Coarse design, then one measured clock halfway to each neighbour of the best
    design point (what a locator does with 2 extra windows); -> (chosen, windows)."""
    measured = set(design)
    best = min(design, key=curve.get)
    i = design.index(best)
    for j in (i - 1, i + 1):
        if 0 <= j < len(design):
            mid = (best + design[j]) / 2
            inside = [c for c in curve if min(best, design[j]) < c < max(best, design[j])]
            if inside:
                measured.add(min(inside, key=lambda c: abs(c - mid)))
    return min(measured, key=curve.get), len(measured)


def curves(path: Path) -> dict[str, dict[int, float]]:
    raw = json.loads(path.read_text())
    table = raw.get("table") or raw
    ev = table.get("evidence") or {}
    out = {}
    for name in ("prefill_energy_target", "prefill_energy_low"):
        if len(ev.get(name) or {}) >= 4:
            out[name] = {int(f): float(e) for f, e in ev[name].items()}
    d = (ev.get("decode") or {}).get("decode_j_per_token") or {}
    if len(d) >= 4:
        out["decode_j_per_token"] = {int(f): float(e) for f, e in d.items()}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("tiers", nargs="+", type=Path)
    ap.add_argument("--eps", type=float, default=0.02)
    ap.add_argument("--json", type=Path, help="write the rows here")
    args = ap.parse_args()
    rows = []
    for path in args.tiers:
        for name, curve in curves(path).items():
            best = min(curve, key=curve.get)
            for k in (3, 4):
                if len(curve) <= k:
                    continue
                design = sparse_design(sorted(curve), k)
                for method in ("quadratic", "refine"):
                    if method == "quadratic":
                        chosen, windows = choose(curve, design), k
                    else:
                        chosen, windows = refine(curve, design)
                    regret = curve[chosen] / curve[best] - 1
                    rows.append(
                        {
                            "run": str(path),
                            "curve": name,
                            "measured": len(curve),
                            "k": k,
                            "method": method,
                            "windows": windows,
                            "design": design,
                            "best": best,
                            "chosen": chosen,
                            "regret": regret,
                            "pass": regret <= args.eps,
                        }
                    )
    for r in rows:
        run = (
            Path(r["run"]).parent.name
            if Path(r["run"]).parent.name not in ("run_a", "run_b", "cantune")
            else "/".join(Path(r["run"]).parts[-3:-1])
        )
        print(
            f"{run:15s} {r['curve']:22s} n={r['measured']} k={r['k']} {r['method']:9s} "
            f"w={r['windows']} "
            f"best={r['best']} chosen={r['chosen']} regret={r['regret']:+.1%} "
            f"{'PASS' if r['pass'] else 'FAIL'}"
        )
    for method in ("quadratic", "refine"):
        for k in (3, 4):
            sel = [r for r in rows if r["method"] == method and r["k"] == k]
            if sel:
                ok = sum(r["pass"] for r in sel)
                worst = max(r["regret"] for r in sel)
                share = f"{ok}/{len(sel)} within eps={args.eps:.0%}"
                print(f"{method:9s} k={k}: {share}, worst {worst:+.1%}")
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
