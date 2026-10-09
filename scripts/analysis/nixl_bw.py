"""NIXL transfer bandwidth between two processes, without vLLM.

vLLM 0.15.1's NixlConnector pulls KV (D issues READ on P's blocks). On one node
without RDMA, UCX's cuda_ipc is reported to serve WRITE at peer-copy speed but READ
at ~0.5 GB/s (vllm-project/vllm#52607); smoke job 274777 measured ~0.23 GB/s per
transfer. This measures READ and WRITE for VRAM (kv_buffer_device cuda) and DRAM
(kv_buffer_device cpu) buffers, one or many descriptors, several sizes, on whatever
pair of processes it is started as (same node or two nodes).

Two processes share a directory (e.g. on NFS) for the agent metadata:
  python nixl_bw.py --role target    --dir D --mem VRAM
  python nixl_bw.py --role initiator --dir D --mem VRAM --out result.json
Each sees one GPU (CUDA_VISIBLE_DEVICES), as vLLM does. UCX_PROTO_INFO=y in the
environment makes UCX print the protocol (tcp / cuda_ipc / rc ...) it selects.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

MB = 1 << 20


def wait_for(path: Path, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while not path.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(f"no {path} after {timeout_s:.0f} s")
        time.sleep(0.1)


def write_atomic(path: Path, data: bytes) -> None:
    part = path.with_name(path.name + ".part")
    part.write_bytes(data)
    part.rename(path)


def buffer(torch, mem: str, nbytes: int):
    if mem == "VRAM":
        return torch.ones(nbytes, dtype=torch.uint8, device="cuda:0")
    return torch.ones(nbytes, dtype=torch.uint8).pin_memory()


def descs(agent, base: int, dev: int, mem: str, nbytes: int, pieces: int):
    """nbytes from base as `pieces` equal descriptors (vLLM uses one per block/layer)."""
    step = nbytes // pieces
    return agent.get_xfer_descs([(base + i * step, step, dev) for i in range(pieces)], mem)


def run_target(args, torch, nixl_api) -> int:
    agent = nixl_api.nixl_agent(f"target-{args.tag}", nixl_api.nixl_agent_config())
    buf = buffer(torch, args.mem, max(args.sizes_mb) * MB)
    agent.register_memory([buf])
    dev = buf.get_device() if args.mem == "VRAM" else 0
    d = Path(args.dir)
    write_atomic(d / "target.meta", agent.get_agent_metadata())
    write_atomic(d / "target.json", json.dumps({"addr": buf.data_ptr(), "dev": dev}).encode())
    print(f"target ready: {args.mem} {buf.numel() // MB} MB", flush=True)
    wait_for(d / "done", args.timeout_s)
    return 0


def run_initiator(args, torch, nixl_api) -> int:
    agent = nixl_api.nixl_agent(f"initiator-{args.tag}", nixl_api.nixl_agent_config())
    buf = buffer(torch, args.mem, max(args.sizes_mb) * MB)
    agent.register_memory([buf])
    dev = buf.get_device() if args.mem == "VRAM" else 0
    d = Path(args.dir)
    rows = []
    try:
        wait_for(d / "target.json", args.timeout_s)
        remote_name = agent.add_remote_agent((d / "target.meta").read_bytes())
        remote = json.loads((d / "target.json").read_text())
        for op in args.ops:
            for pieces in args.pieces:
                for size_mb in args.sizes_mb:
                    nbytes = size_mb * MB
                    local = descs(agent, buf.data_ptr(), dev, args.mem, nbytes, pieces)
                    far = descs(agent, remote["addr"], remote["dev"], args.mem, nbytes, pieces)
                    handle = agent.initialize_xfer(op, local, far, remote_name)
                    times = []
                    status = "ok"
                    for rep in range(args.warmup + args.reps):
                        t0 = time.perf_counter()
                        state = agent.transfer(handle)
                        while state == "PROC":
                            if time.perf_counter() - t0 > args.xfer_timeout_s:
                                state = "TIMEOUT"
                                break
                            state = agent.check_xfer_state(handle)
                        if state != "DONE":
                            status = state
                            break
                        if rep >= args.warmup:
                            times.append(time.perf_counter() - t0)
                    agent.release_xfer_handle(handle)
                    row = {
                        "op": op, "mem": args.mem, "pieces": pieces, "size_mb": size_mb,
                        "status": status, "n": len(times),
                        "median_ms": statistics.median(times) * 1e3 if times else None,
                        "gb_s": nbytes / statistics.median(times) / 1e9 if times else None,
                    }  # fmt: skip
                    rows.append(row)
                    print(json.dumps(row), flush=True)
                    if status != "ok":
                        break
    finally:
        (d / "done").touch()
    if args.out:
        Path(args.out).write_text(json.dumps({"tag": args.tag, "rows": rows}, indent=1))
    return 0 if rows and all(r["status"] == "ok" for r in rows) else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--role", choices=("target", "initiator"), required=True)
    ap.add_argument("--dir", required=True, help="directory both processes see")
    ap.add_argument("--mem", choices=("VRAM", "DRAM"), default="VRAM")
    ap.add_argument("--tag", default="bw")
    ap.add_argument("--ops", nargs="+", default=["READ", "WRITE"])
    ap.add_argument("--sizes-mb", type=int, nargs="+", default=[2, 16, 64, 256])
    ap.add_argument("--pieces", type=int, nargs="+", default=[1, 64])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--timeout-s", type=float, default=180.0)
    ap.add_argument("--xfer-timeout-s", type=float, default=30.0)
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    os.makedirs(args.dir, exist_ok=True)

    import torch
    from nixl import _api as nixl_api

    print(f"{args.role} pid {os.getpid()} CUDA_VISIBLE_DEVICES="
          f"{os.environ.get('CUDA_VISIBLE_DEVICES')} cuda devices {torch.cuda.device_count()}",
          flush=True)  # fmt: skip
    run = run_target if args.role == "target" else run_initiator
    return run(args, torch, nixl_api)


if __name__ == "__main__":
    sys.exit(main())
