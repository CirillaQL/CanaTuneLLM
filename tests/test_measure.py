"""Model measurements against mock vLLM endpoints and agents (no GPUs)."""

import asyncio
import json
import time

import httpx
import pytest

from canatune.measure import Measure, Pair, Plan
from canatune.proxy.proxy import Endpoint

PAIRS = [
    Pair(
        "G0",
        Endpoint("P0", "prefill", "p0", "p0", 8100, 14579),
        Endpoint("D0", "decode", "d0", "d0", 8200, 14579),
        ("http://pa:9300", 0),
        ("http://da:9300", 0),
    ),
    Pair(
        "G1",
        Endpoint("P1", "prefill", "p1", "p1", 8101, 14580),
        Endpoint("D1", "decode", "d1", "d1", 8201, 14580),
        ("http://pa:9300", 1),
        ("http://da:9300", 1),
    ),
]

# NixlConnector: P names the blocks D reads.
KV_PARAMS = {"do_remote_prefill": True, "remote_block_ids": [1], "remote_engine_id": "p"}


def backend(stall_every: int = 0, stall_s: float = 0.0, growing_s: float = 0.0):
    state = {"prefill_sum": 0.0, "count": 0, "gen": 0, "energy": 0.0, "net": 0, "locks": []}
    state["hosts"] = []
    token = b'data: {"choices":[{"text":"a"}]}\n\n'

    async def handler(request: httpx.Request) -> httpx.Response:
        host, path = request.url.host, request.url.path
        if path == "/v1/completions" and host.startswith("p"):
            body = json.loads(request.read())
            state["prefill_sum"] += 0.02 + 0.00005 * len(body["prompt"])
            state["count"] += 1
            state["hosts"].append(host)
            if stall_every and state["count"] % stall_every == 0:
                await asyncio.sleep(stall_s)
            if growing_s:  # overload: every request waits longer than the one before
                await asyncio.sleep(growing_s * state["count"])
            return httpx.Response(
                200, json={"choices": [{"text": "x"}], "kv_transfer_params": KV_PARAMS}
            )
        if path == "/v1/completions" and host.startswith("d"):
            body = json.loads(request.read())
            state["gen"] += body["max_tokens"]
            content = token * body["max_tokens"] + b"data: [DONE]\n\n"
            return httpx.Response(
                200, content=content, headers={"content-type": "text/event-stream"}
            )
        if path == "/metrics":
            text = (
                f"vllm:request_prefill_time_seconds_sum {state['prefill_sum']}\n"
                f"vllm:request_prefill_time_seconds_count {state['count']}\n"
                f"vllm:generation_tokens_total {state['gen']}\n"
                "vllm:num_requests_running 1\nvllm:kv_cache_usage_perc 0.25\n"
            )
            return httpx.Response(200, text=text)
        if path.endswith("/lock"):
            gpu = int(path.split("/")[2])
            state["locks"].append((host, gpu, json.loads(request.read())["mhz"]))
            return httpx.Response(200, json={"ok": True})
        if path == "/gpus":
            state["energy"] += 1000.0
            return httpx.Response(
                200,
                json=[{"index": i, "energy_mj": state["energy"] + i} for i in (0, 1)],
            )
        if path == "/net":
            state["net"] += 5000
            return httpx.Response(
                200,
                json={"time": 0, "interfaces": {"eth0": {"rx_bytes": 1, "tx_bytes": state["net"]}}},
            )
        return httpx.Response(404)

    return state, httpx.MockTransport(handler)


def tiny_plan(**kwargs) -> Plan:
    values = dict(
        prefill_clocks=[2520, 1080],
        decode_clocks=[1170],
        lut_lengths=[16, 512],
        lut_repeats=2,
        warmup_requests=1,
        warmup_s=0.2,
        warmup_rate=10,
        power_s=0.01,
        settle_s=0.0,
        decode_concurrency=[2],
        decode_output=4,
        decode_window_s=0.2,
        decode_stagger_s=0.0,
        load_rates=[20, 40],
        load_window_s=0.2,
        load_output=2,
        mix_names=["short"],
    )
    values.update(kwargs)
    return Plan(**values)


def make(tmp_path, plan, transport, pairs=PAIRS, **kwargs) -> Measure:
    return Measure(
        plan,
        pairs,
        "m",
        tmp_path,
        client_factory=lambda: httpx.AsyncClient(transport=transport),
        sample_period_s=0.05,
        **kwargs,
    )


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_all_stages_write_windows_and_requests(tmp_path) -> None:
    state, transport = backend()
    result = asyncio.run(make(tmp_path, tiny_plan(), transport).run())
    assert result["status"] == "done"
    windows = {w["key"]: w for w in rows(tmp_path / "windows.jsonl")}
    assert {"lut|async|2520", "lut|async|1080", "power|async|2520|1170"} <= set(windows)
    assert {"decode|async|1170|2", "load|async|load|2520|1170|default|1|20"} <= set(windows)
    assert "mix|async|mix|2520|1170|short|1|40" in windows
    lut = [r for r in rows(tmp_path / "requests.jsonl") if r["key"] == "lut|async|2520"]
    assert len(lut) == 4 and all(r["status"] == "ok" and r["pair"] == "G0" for r in lut)
    # P's own prefill time comes from the counter delta around each sequential request.
    long = [r for r in lut if r["prompt_tokens"] == 512]
    assert all(r["p_engine_ms"] == pytest.approx(45.6) for r in long)
    load = windows["load|async|load|2520|1170|default|1|20"]
    assert load["requests"] == 4 and load["energy_j"]["prefill"] > 0
    assert set(load["energy_j"]) >= {"P0", "D0", "P1", "D1", "prefill", "decode"}
    assert load["sampled"]["decode_kv_usage"] == pytest.approx(0.25)
    assert load["net_bytes"]["http://pa:9300"]["eth0"]["tx_bytes"] > 0
    assert ("pa", 0, 1080) in state["locks"] and ("da", 0, 1170) in state["locks"]
    assert set(state["hosts"]) == {"p0"}  # single-pair stages never touch G1


def test_two_pair_scan_loads_both_pairs(tmp_path) -> None:
    state, transport = backend()
    plan = tiny_plan(
        stages=["scans"], scans=[{"name": "link", "p": 1545, "d": 1170, "rates": [20], "pairs": 2}]
    )
    asyncio.run(make(tmp_path, plan, transport).run())
    (window,) = rows(tmp_path / "windows.jsonl")
    assert window["pairs"] == ["G0", "G1"] and window["requests"] == 8
    assert {r["pair"] for r in rows(tmp_path / "requests.jsonl")} == {"G0", "G1"}
    assert {("pa", 1, 1545), ("da", 1, 1170)} <= set(state["locks"])
    assert window["sampled"]["G1_decode_running"] == 1


def test_stalled_window_is_repeated_and_does_not_stop_the_scan(tmp_path) -> None:
    _, transport = backend(stall_every=3, stall_s=0.15)
    plan = tiny_plan(stages=["load"], prefill_clocks=[2520], stall_ms=100, saturation_ttft_ms=50)
    asyncio.run(make(tmp_path, plan, transport).run())
    keys = [w["key"] for w in rows(tmp_path / "windows.jsonl")]
    base = "load|async|load|2520|1170|default|1"
    assert keys == [f"{base}|20", f"{base}|20#retry", f"{base}|40", f"{base}|40#retry"]


def test_overload_is_saturation_not_a_stall(tmp_path) -> None:
    _, transport = backend(growing_s=0.02)
    plan = tiny_plan(stages=["load"], prefill_clocks=[2520], stall_ms=60, saturation_ttft_ms=50)
    asyncio.run(make(tmp_path, plan, transport).run())
    (window,) = rows(tmp_path / "windows.jsonl")  # stopped after the first rate, no retry
    assert window["stalls"] == 0 and window["ttft_ms"]["p95"] > 50


def test_warmup_runs_every_time_and_resume_skips_done(tmp_path) -> None:
    _, transport = backend()
    plan = tiny_plan(stages=["warmup", "lut"])
    asyncio.run(make(tmp_path, plan, transport).run())
    first = rows(tmp_path / "windows.jsonl")
    assert [w["key"] for w in first] == ["lut|async|2520", "lut|async|1080"]  # warmup unrecorded
    asyncio.run(make(tmp_path, tiny_plan(stages=["warmup", "lut", "power"]), transport).run())
    keys = [w["key"] for w in rows(tmp_path / "windows.jsonl")]
    assert len(keys) == len(set(keys)) == 4
    # another tag is another configuration: nothing is skipped
    asyncio.run(make(tmp_path, tiny_plan(stages=["lut"], tag="put"), transport).run())
    assert len(rows(tmp_path / "windows.jsonl")) == 6


def test_deadline_stops_before_a_window(tmp_path) -> None:
    _, transport = backend()
    measure = make(tmp_path, tiny_plan(), transport, deadline=time.monotonic() + 1.0)
    result = asyncio.run(measure.run())
    assert result["status"].startswith("out_of_time")


def test_plan_validation() -> None:
    assert Plan.from_json({"lut_repeats": 5}).lut_repeats == 5
    with pytest.raises(ValueError):
        Plan.from_json({"lut_repeat": 5})
    with pytest.raises(ValueError):
        Plan.from_json({"scans": [{"name": "x", "p": 1}]})
