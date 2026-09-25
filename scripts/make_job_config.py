"""Write a job-specific CanaTune config from the repository config.

The repository `config.yaml` keeps placeholders and four P/D pairs. A job knows
its nodes, the GPUs Slurm allocated (index and UUID) and its paths only at run
time; this script fills them in:

  python scripts/make_job_config.py --base config.yaml --out job.yaml \
      --prefill-node neptune --decode-node ganymede \
      --prefill-gpus 0,1 --prefill-uuids GPU-a,GPU-b \
      --decode-gpus 0,1 --decode-uuids GPU-c,GPU-d \
      --python /env/bin/python --work-dir /data/.../123 \
      --set canary.locator.window_s=20 --set clock_control.enabled=true

Pair i is (P_i, D_i); pair 0 is the Canary, the others form the production pool.
`--set a.b.c=value` overrides any key (the value is parsed as YAML).
"""

import argparse
import sys
from pathlib import Path

import yaml


def ids(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def uuids(text: str | None) -> list[str] | None:
    return None if not text else [part.strip() for part in text.split(",")]


def set_path(config: dict, dotted: str, raw: str) -> None:
    *parents, leaf = dotted.split(".")
    node = config
    for key in parents:
        node = node.setdefault(key, {})
    node[leaf] = yaml.safe_load(raw)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefill-node", required=True)
    ap.add_argument("--decode-node", required=True)
    ap.add_argument("--prefill-gpus", required=True, type=ids)
    ap.add_argument("--decode-gpus", required=True, type=ids)
    ap.add_argument("--prefill-uuids")
    ap.add_argument("--decode-uuids")
    ap.add_argument("--python", required=True, help="vLLM environment's Python")
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args(argv)

    n = len(args.prefill_gpus)
    if n < 2 or len(args.decode_gpus) != n:
        ap.error("need the same number (>= 2) of prefill and decode GPUs: pair 0 is the Canary")
    config = yaml.safe_load(Path(args.base).read_text(encoding="utf-8"))
    topology = config["topology"]
    groups = {"prefill": topology["prefill_nodegroup"], "decode": topology["decode_nodegroup"]}
    groups["prefill"].update(node=args.prefill_node, gpu_ids=args.prefill_gpus)
    groups["decode"].update(node=args.decode_node, gpu_ids=args.decode_gpus)
    for role, raw in (("prefill", args.prefill_uuids), ("decode", args.decode_uuids)):
        values = uuids(raw)
        if values is not None and len(values) != n:
            ap.error(f"--{role}-uuids must list one UUID per GPU")
        groups[role]["gpu_uuids"] = values
        groups[role]["memory_frequency_mhz"] = None  # SM clocks only (as in K1-K4a)
    endpoints = {}
    for role, prefix, gpus in (
        ("prefill", "P", args.prefill_gpus),
        ("decode", "D", args.decode_gpus),
    ):
        group = groups[role]
        for index, gpu in enumerate(gpus):
            endpoints[f"{prefix}{index}"] = {
                "role": role,
                "gpu_id": gpu,
                "http_port": group["http_port_base"] + index,
                "kv_port": group["kv_port_base"] + index,
            }
    topology["endpoints"] = endpoints
    config["routing"]["canary_pair"] = ["P0", "D0"]
    config["routing"]["production_pairs"] = [[f"P{i}", f"D{i}"] for i in range(1, n)]
    config["runtime"]["python"] = args.python
    config["project"]["work_dir_root"] = str(Path(args.work_dir).parent)
    for item in args.set:
        key, sep, value = item.partition("=")
        if not sep:
            ap.error(f"--set needs KEY=VALUE: {item}")
        set_path(config, key, value)
    Path(args.out).write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(f"wrote {args.out}: {n} pairs, canary P0/D0", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
