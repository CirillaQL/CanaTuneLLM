"""Model measurements against mock vLLM endpoints and agents (no GPUs)."""

import asyncio
import json

import httpx
import pytest

from canatune.measure import Measure, Plan
from canatune.proxy.proxy import Endpoint

P = Endpoint("P0", "prefill", "p", "p", 8100, 14579)
D = Endpoint("D0", "decode", "d", "d", 8200, 14579)
AGENTS = {"prefill": ("http://pa:9300", 0), "decode": ("http://da:9300", 1)}


def backend():
    state = {"prefill_sum": 0.0, "count": 0, "gen": 0, "energy": 0.0, "locks": []}
    token = b'data: {"choices":[{"text":"a"}]}\n\n'

    def handler(request: httpx.Request) -> httpx.Response:
        host, path = request.url.host, request.url.path
        if path == "/v1/completions" and host == "p":
            body = json.loads(request.read())
            state["prefill_sum"] += 0.02 + 0.00005 * len(body["prompt"])
            state["count"] += 1
            return httpx.Response(200, json={"choices": [{"text": "x"}]})
        if path == "/v1/completions" and host == "d":
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
            state["locks"].append((host, json.loads(request.read())["mhz"]))
            return httpx.Response(200, json={"ok": True})
        if path == "/gpus":
            state["energy"] += 1000.0
            gpu = 0 if host == "pa" else 1
            return httpx.Response(200, json=[{"index": gpu, "energy_mj": state["energy"]}])
        return httpx.Response(404)

    return state, httpx.MockTransport(handler)


def tiny_plan(**kwargs) -> Plan:
    values = dict(
        prefill_clocks=[2520, 1080],
        decode_clocks=[1170],
        lut_lengths=[16, 512],
        lut_repeats=2,
        warmup_requests=1,
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


def make(tmp_path, plan, transport, **kwargs) -> Measure:
    return Measure(
        plan,
        P,
        D,
        AGENTS,
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
    assert {"lut|2520", "lut|1080", "power|2520|1170", "power|1080|1170"} <= set(windows)
    assert {"decode|1170|2", "load|2520|default|20", "mix|2520|short|40"} <= set(windows)
    lut = [r for r in rows(tmp_path / "requests.jsonl") if r["key"] == "lut|2520"]
    assert len(lut) == 4 and all(r["status"] == "ok" for r in lut)
    # P's own prefill time comes from the counter delta around each sequential request.
    long = [r for r in lut if r["prompt_tokens"] == 512]
    assert all(r["p_engine_ms"] == pytest.approx(45.6) for r in long)
    assert all(r["decode_first_ms"] is not None and r["ttft_ms"] is not None for r in lut)
    load = windows["load|2520|default|20"]
    assert load["requests"] == 4 and load["energy_j"]["prefill"] > 0
    assert load["sampled"]["decode_kv_usage"] == pytest.approx(0.25)
    assert ("pa", 1080) in state["locks"] and ("da", 1170) in state["locks"]


def test_rerun_resumes_after_finished_windows(tmp_path) -> None:
    _, transport = backend()
    asyncio.run(make(tmp_path, tiny_plan(stages=["lut"]), transport).run())
    first = len(rows(tmp_path / "windows.jsonl"))
    result = asyncio.run(make(tmp_path, tiny_plan(stages=["lut", "power"]), transport).run())
    keys = [w["key"] for w in rows(tmp_path / "windows.jsonl")]
    assert first == 2 and len(keys) == len(set(keys)) == 4
    assert result["windows_done"] == 4


def test_deadline_stops_before_a_window(tmp_path) -> None:
    import time

    _, transport = backend()
    measure = make(tmp_path, tiny_plan(), transport, deadline=time.monotonic() + 1.0)
    result = asyncio.run(measure.run())
    assert result["status"].startswith("out_of_time")


def test_plan_from_json_rejects_unknown_keys() -> None:
    assert Plan.from_json({"lut_repeats": 5}).lut_repeats == 5
    with pytest.raises(ValueError):
        Plan.from_json({"lut_repeat": 5})
