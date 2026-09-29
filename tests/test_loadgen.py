"""Load generator: profiles, traces, one run against a mock proxy, and the summary."""

import asyncio
import json

import httpx
import pytest

from canatune import loadgen


def test_parse_profile_and_errors() -> None:
    phases = loadgen.parse_profile("low:10:0.3, canary:5:0.4:canary_recheck")
    assert phases == [
        loadgen.Phase("low", 10.0, 0.3),
        loadgen.Phase("canary", 5.0, 0.4, "canary_recheck"),
    ]
    assert len(loadgen.parse_profile(loadgen.DEFAULT_PROFILE)) == 8
    for bad in ("", "a:0:1", "a:1", "a:1:1:jump", "a:1:1,a:2:1", "a:1:-1"):
        with pytest.raises(loadgen.ProfileError):
            loadgen.parse_profile(bad)


def test_trace_rate_follows_capacity_and_is_deterministic() -> None:
    phases = loadgen.parse_profile("a:2000:0.5,b:2000:1.0,c:10:0")
    pairs = [(128, 64), (2048, 64)]
    meta, trace = loadgen.build_trace(phases, pairs, capacity_h=6000, alpha=900, seed=3)
    expected = 0.5 * 6000 / (1088 + 900)
    assert meta["phases"][0]["rate_rps"] == pytest.approx(expected)
    counts = {name: sum(a.phase == name for a in trace) for name in "abc"}
    assert counts["a"] == pytest.approx(expected * 2000, rel=0.1)
    assert counts["b"] == pytest.approx(2 * expected * 2000, rel=0.1)
    assert counts["c"] == 0
    assert all(meta["phases"][1]["start_s"] <= a.at_s < 4000 for a in trace if a.phase == "b")
    assert [a.index for a in trace] == list(range(len(trace)))
    again = loadgen.build_trace(phases, pairs, capacity_h=6000, alpha=900, seed=3)[1]
    assert again == trace
    assert loadgen.prompt_ids(3, 5, 16) == loadgen.prompt_ids(3, 5, 16)
    assert loadgen.prompt_ids(3, 5, 16) != loadgen.prompt_ids(3, 6, 16)


def test_trace_round_trip(tmp_path) -> None:
    meta, trace = loadgen.build_trace(
        loadgen.parse_profile("a:30:1"), [(512, 8)], capacity_h=3000, alpha=500, seed=1
    )
    loadgen.save_trace(tmp_path / "t.json", meta, trace)
    meta2, trace2 = loadgen.load_trace(tmp_path / "t.json")
    assert meta2 == json.loads(json.dumps(meta))
    assert trace2 == trace


def test_energy_between_interpolates() -> None:
    points = [(0.0, 0.0), (1.0, 100_000.0), (3.0, 300_000.0)]
    assert loadgen.energy_between(points, 0.5, 2.0) == pytest.approx(150.0)
    assert loadgen.energy_between(points, -1.0, 2.0) == pytest.approx(200.0)
    assert loadgen.energy_between(points, -3.0, 2.0) is None
    assert loadgen.energy_between(points[:1], 0.0, 0.0) is None


def test_run_and_summary_against_mock_proxy(tmp_path) -> None:
    token = b'data: {"choices":[{"text":"a"}]}\n\n'
    sse = token + token + b"data: [DONE]\n\n"
    energy = {"mj": 0.0}
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/completions":
            body = json.loads(request.read())
            seen.append(body)
            if len(body["prompt"]) > 1000:
                return httpx.Response(503, headers={"X-CanaTune-Rejected": "1"})
            return httpx.Response(
                200,
                content=sse,
                headers={"content-type": "text/event-stream", "X-CanaTune-Group": "G1"},
            )
        if request.url.path == "/canatune/state":
            return httpx.Response(
                200,
                json={"router": {"groups": [{"name": "G1", "state": "active", "tier": "H"}]}},
            )
        if request.url.path == "/canatune/canary":
            return httpx.Response(200, json={"ok": True, "reason": "ok"})
        if request.url.path == "/gpus":
            energy["mj"] += 50_000.0
            return httpx.Response(200, json=[{"index": 0, "energy_mj": energy["mj"]}])
        return httpx.Response(404)

    meta, trace = loadgen.build_trace(
        loadgen.parse_profile("a:0.6:2:canary_recheck,b:0.6:2"),
        [(64, 2), (2000, 2)],
        capacity_h=20000.0,
        alpha=0.0,
        seed=11,
    )
    transport = httpx.MockTransport(handler)
    run = loadgen.LoadRun(
        meta,
        trace,
        base_url="http://proxy",
        model="m",
        out_dir=tmp_path,
        agents=["http://agent"],
        poll_period_s=0.1,
        client_factory=lambda: httpx.AsyncClient(transport=transport),
    )
    result = asyncio.run(run.run())
    assert result["unfinished"] == 0
    assert len(seen) == len(trace) > 0
    assert all(b["stream"] and b["ignore_eos"] for b in seen)

    summary = loadgen.summarize(
        tmp_path, meta, ttft_slo_ms=1000, tpot_slo_ms=200, gpu_names={"http://agent|0": "P0"}
    )
    total = summary["phases"]["all"]
    long = sum(a.prompt_tokens > 1000 for a in trace)
    assert total["offered"] == len(trace)
    assert total["rejected"] == long
    assert total["ok"] == len(trace) - long
    assert total["violated"] == 0
    assert set(summary["phases"]) == {"a", "b", "all"}
    assert summary["phases"]["a"]["energy_by_gpu_j"]["P0"] > 0
    actions = (tmp_path / "actions.jsonl").read_text().splitlines()
    assert json.loads(actions[0])["ok"] is True
    assert (tmp_path / "timeline.jsonl").read_text()
