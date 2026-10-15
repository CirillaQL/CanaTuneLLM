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

Tensor parallelism and several nodes per role: `--tp 2` makes each endpoint span two
consecutive GPUs of one node (rank r's P2pNccl/NIXL port is the endpoint's kv_port + r,
so kv ports step by the TP size); a role on several nodes lists them comma-separated
and their GPUs (and UUIDs) per node separated by ";", in the same order:

  --prefill-node node-a,node-b --prefill-gpus "0,1,2,3;0,1,2,3" --tp 2
  --decode-node node-c --decode-gpus 0,1,2,3,4,5,6,7

Endpoints are numbered across the nodes in order (P0, P1 on node-a, P2, P3 on node-b);
ports restart from the base on every node.
"""

import argparse
import sys
from pathlib import Path

import yaml


def ids(text: str) -> list[list[int]]:
    """GPU indices per node: "0,1;0,1" -> [[0, 1], [0, 1]]."""
    return [[int(part) for part in node.split(",") if part.strip()] for node in text.split(";")]


def uuids(text: str | None) -> list[list[str]] | None:
    if not text:
        return None
    return [[part.strip() for part in node.split(",")] for node in text.split(";")]


def nodes(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


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
    ap.add_argument("--tp", type=int, help="GPUs per endpoint on both roles (default: config)")
    ap.add_argument("--python", required=True, help="vLLM environment's Python")
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args(argv)

    config = yaml.safe_load(Path(args.base).read_text(encoding="utf-8"))
    overrides = []
    for item in args.set:
        key, sep, value = item.partition("=")
        if not sep:
            ap.error(f"--set needs KEY=VALUE: {item}")
        overrides.append((key, value))
    # Node-group settings (ports, GPU type) must be in place before the endpoints are
    # derived from them; the overrides are applied again at the end, so an explicit
    # topology.endpoints.* value still wins.
    for key, value in overrides:
        set_path(config, key, value)
    topology = config["topology"]
    groups = {"prefill": topology["prefill_nodegroup"], "decode": topology["decode_nodegroup"]}
    placed = {
        "prefill": (nodes(args.prefill_node), args.prefill_gpus, uuids(args.prefill_uuids)),
        "decode": (nodes(args.decode_node), args.decode_gpus, uuids(args.decode_uuids)),
    }
    endpoints = {}
    counts = {}
    for role, prefix in (("prefill", "P"), ("decode", "D")):
        group = groups[role]
        names, gpus, ids_ = placed[role]
        tp = args.tp or int(group.get("tensor_parallel_size", 1))
        if len(gpus) != len(names) or (ids_ is not None and len(ids_) != len(names)):
            ap.error(f"--{role}-gpus (and -uuids) need one ';'-separated list per node")
        if ids_ is not None and any(len(u) != len(g) for u, g in zip(ids_, gpus)):
            ap.error(f"--{role}-uuids must list one UUID per GPU")
        if any(len(g) % tp for g in gpus):
            ap.error(f"every {role} node needs a multiple of --tp {tp} GPUs")
        spread = len(names) > 1 or tp > 1
        group.update(node=names[0], gpu_ids=gpus[0], gpu_uuids=None if ids_ is None else ids_[0])
        group["tensor_parallel_size"] = tp
        if len(names) > 1:
            group["nodes"] = [
                {"node": name, "gpu_ids": g, "gpu_uuids": None if ids_ is None else u}
                for name, g, u in zip(names, gpus, ids_ or [None] * len(names))
            ]
        else:
            group.pop("nodes", None)
        group["memory_frequency_mhz"] = None  # SM clocks only (as in K1-K4a)
        index = 0
        for name, g in zip(names, gpus):
            for local in range(len(g) // tp):
                endpoint = {
                    "role": role,
                    "gpu_id": g[local * tp],
                    "http_port": group["http_port_base"] + local,
                    "kv_port": group["kv_port_base"] + local * tp,
                }
                if spread:
                    endpoint.update(node=name, gpu_ids=g[local * tp : (local + 1) * tp])
                endpoints[f"{prefix}{index}"] = endpoint
                index += 1
        counts[role] = index
    n = counts["prefill"]
    if n < 2 or counts["decode"] != n:
        ap.error("need the same number (>= 2) of P and D endpoints: pair 0 is the Canary")
    topology["endpoints"] = endpoints
    config["routing"]["canary_pair"] = ["P0", "D0"]
    config["routing"]["production_pairs"] = [[f"P{i}", f"D{i}"] for i in range(1, n)]
    config["runtime"]["python"] = args.python
    config["project"]["work_dir_root"] = str(Path(args.work_dir).parent)
    for key, value in overrides:
        set_path(config, key, value)
    Path(args.out).write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(f"wrote {args.out}: {n} pairs, canary P0/D0", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
