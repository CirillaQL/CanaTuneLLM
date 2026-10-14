"""The benchmark driver (scripts/bench_production.py) without GPUs: command lines,
SLO outcomes from vllm's detailed results, energy windows, and the stage loop
driven by a stand-in for `vllm bench serve`."""

import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import bench_production as bench  # noqa: E402

FAKE_BENCH = """
import json, sys, time
args = sys.argv[1:]
value = lambda flag: args[args.index(flag) + 1]
n = int(value("--num-prompts"))
time.sleep(0.4)  # dataset loading, then the load itself
result = {
    "duration": 0.2, "completed": n, "failed": 0, "total_output_tokens": 10 * n,
    "request_goodput": 1.0, "ttfts": [0.3] * (n - 1) + [1.5],
    "itls": [[0.05, 0.07]] * n, "errors": [""] * n,
}
with open(value("--result-dir") + "/" + value("--result-filename"), "w") as f:
    json.dump(result, f)
"""


def settings(**overrides) -> bench.BenchSettings:
    env = {"BENCH_DATASET": "/data/sharegpt.json", "BENCH_RATES": "1,2.5", "MODEL_PATH": "/m"}
    env.update(overrides)
    return bench.BenchSettings.from_env(env)


def test_settings_and_stages_from_the_environment() -> None:
    s = settings(BENCH_STAGE_S="100", BENCH_OUTPUT_LEN="256")
    assert [(st.rate_rps, st.num_prompts) for st in s.stages()] == [(1.0, 100), (2.5, 250)]
    assert s.output_len == 256 and s.ignore_eos and s.tokenizer == "/m"
    assert s.command == ("vllm", "bench", "serve")
    with pytest.raises(SystemExit):
        settings(BENCH_RATES="1,0")
    with pytest.raises(SystemExit):
        settings(BENCH_ENDPOINT="/v1/embeddings")


def test_bench_command_matches_the_slo_and_the_dataset_sample() -> None:
    s = settings(BENCH_OUTPUT_LEN="256", BENCH_EXTRA_ARGS="--max-concurrency 64")
    stage = s.stages()[1]
    cmd = bench.bench_command(
        s, stage, base_url="http://p:1", model="m", result_dir=Path("/r"),
        ttft_slo_ms=1000, tpot_slo_ms=200,
    )  # fmt: skip
    joined = " ".join(cmd)
    assert cmd[:3] == ["vllm", "bench", "serve"]
    assert "--backend openai --base-url http://p:1 --endpoint /v1/completions" in joined
    assert "--dataset-name sharegpt --dataset-path /data/sharegpt.json" in joined
    assert "--num-prompts 450 --request-rate 2.5" in joined
    assert "--seed 20261002" in joined and "--temperature 0" in joined
    assert "--goodput ttft:1000 tpot:200" in joined
    assert "--sharegpt-output-len 256" in joined and "--ignore-eos" in cmd
    assert "--tokenizer /m" in joined and cmd[-2:] == ["--max-concurrency", "64"]
    chat = settings(BENCH_ENDPOINT="/v1/chat/completions", BENCH_IGNORE_EOS="0")
    cmd = bench.bench_command(
        chat, stage, base_url="u", model="m", result_dir=Path("/r"),
        ttft_slo_ms=1000, tpot_slo_ms=200,
    )  # fmt: skip
    assert "openai-chat" in cmd and "--ignore-eos" not in cmd
    assert "--sharegpt-output-len" not in cmd


def test_request_outcomes_follow_the_slo() -> None:
    result = {
        "ttfts": [0.4, 1.2, 0.5, 0.0, 0.3, 0.2],
        "itls": [[0.05, 0.06], [0.05], [0.25, 0.25], [], [], [0.5]],
        "errors": ["", "", "", "boom", "", ""],
        # the last one: one SSE chunk of 4 tokens after the first, 0.5 s / 4 = 125 ms
        "output_lens": [3, 2, 3, 0, 1, 5],
    }
    o = bench.request_outcomes(result, ttft_slo_ms=1000, tpot_slo_ms=200)
    # ok: 5 (one error); good: 1st, 5th (one output token: no TPOT) and 6th
    assert (o["offered"], o["ok"], o["good"]) == (6, 5, 3)
    assert o["goodput"] == pytest.approx(0.5)
    assert o["ttft_p50_ms"] == pytest.approx(400)
    assert o["tpot_p95_ms"] == pytest.approx(250)


def test_stage_energy_covers_only_the_benchmark_window() -> None:
    stage = bench.Stage("r0_1rps", 1.0, 10)
    series = {"a|0": [(0.0, 0.0), (10.0, 100_000.0)], "a|1": [(0.0, 0.0), (10.0, 50_000.0)]}
    result = {"duration": 4.0, "completed": 10, "total_output_tokens": 300,
              "ttfts": [0.2] * 10, "itls": [[0.05]] * 10, "errors": [""] * 10}  # fmt: skip
    timeline = [{"t_s": 3.0, "mode": "energy"}, {"t_s": 5.0, "mode": "full_effort"},
                {"t_s": 9.5, "mode": "energy"}]  # fmt: skip
    s = bench.stage_summary(
        stage, (4.0, 8.0), result, series, {"a|0": "P0", "a|1": "D0"}, timeline,
        ttft_slo_ms=1000, tpot_slo_ms=200,
    )  # fmt: skip
    assert s["energy_by_gpu_j"] == {"P0": pytest.approx(40.0), "D0": pytest.approx(20.0)}
    assert s["energy_j"] == pytest.approx(60.0) and s["power_w"] == pytest.approx(15.0)
    assert s["j_per_good_request"] == pytest.approx(6.0)
    assert s["j_per_output_token"] == pytest.approx(0.2)
    assert s["modes"] == {"full_effort": 1}


def test_stage_loop_runs_each_rate_and_windows_the_load(tmp_path) -> None:
    fake = tmp_path / "fake_bench.py"
    fake.write_text(FAKE_BENCH)
    s = settings(BENCH_CMD=f"{sys.executable} {fake}", BENCH_STAGE_S="4", BENCH_GAP_S="0")
    out = tmp_path / "out"
    stages, finished = asyncio.run(
        bench.run_stages(
            s, base_url="http://127.0.0.1:9", model="m", out=out, agents=[],
            state_poll=False, ttft_slo_ms=1000, tpot_slo_ms=200,
        )
    )  # fmt: skip
    assert [(st.name, code) for st, _, _, code in stages] == [("r0_1rps", 0), ("r1_2.5rps", 0)]
    for _, (start, end), result, _ in stages:
        assert end - start == pytest.approx(0.2, abs=0.05)  # the load, not the loading
    assert finished >= stages[-1][1][1]
    assert "--num-prompts 10" in (out / "stages/r1_2.5rps/command.txt").read_text()
    summary = bench.summarize(out, stages, finished, {}, ttft_slo_ms=1000, tpot_slo_ms=200)
    first = summary["stages"]["r0_1rps"]
    assert first["exit"] == 0 and first["completed"] == 4
    assert (first["ok"], first["good"]) == (4, 3)  # the last TTFT (1.5 s) misses the SLO
    assert first["energy_j"] is None  # no agents in this test
    json.dumps(summary)


def test_a_failed_benchmark_stops_the_run(tmp_path) -> None:
    fake = tmp_path / "fail.py"
    fake.write_text("import sys; sys.exit(3)\n")
    s = settings(BENCH_CMD=f"{sys.executable} {fake}", BENCH_GAP_S="0")
    stages, _ = asyncio.run(
        bench.run_stages(
            s, base_url="u", model="m", out=tmp_path, agents=[], state_poll=False,
            ttft_slo_ms=1000, tpot_slo_ms=200,
        )
    )  # fmt: skip
    assert len(stages) == 1 and stages[0][3] == 3 and stages[0][2] is None


def test_a_hung_benchmark_is_stopped_and_ends_the_run(tmp_path) -> None:
    fake = tmp_path / "hang.py"
    fake.write_text("import time; time.sleep(60)\n")
    s = settings(BENCH_CMD=f"{sys.executable} {fake}", BENCH_GAP_S="0", BENCH_STAGE_TIMEOUT_S="0.5")
    stages, _ = asyncio.run(
        bench.run_stages(
            s, base_url="u", model="m", out=tmp_path, agents=[], state_poll=False,
            ttft_slo_ms=1000, tpot_slo_ms=200,
        )
    )  # fmt: skip
    assert len(stages) == 1 and stages[0][3] == -9 and stages[0][2] is None


def test_workloads_run_in_turn_with_their_own_datasets_and_rates(tmp_path) -> None:
    workloads = json.dumps([
        {"name": "sharegpt", "dataset": "/d/sg.json", "rates": "1,2"},
        {"name": "burstgpt", "dataset": "/d/bg.jsonl", "dataset_name": "custom",
         "rates": [0.5], "ignore_eos": True, "output_len": None},
    ])  # fmt: skip
    s = bench.BenchSettings.from_env({
        "BENCH_WORKLOADS": workloads, "BENCH_OUTPUT_LEN": "256", "BENCH_IGNORE_EOS": "0",
        "BENCH_STAGE_S": "10", "MODEL_PATH": "/m", "BENCH_SCORE_SLOS": "600:65,1000:100",
    })  # fmt: skip
    assert [(st.name, st.num_prompts) for st in s.stages()] == [
        ("sharegpt_r0_1rps", 10), ("sharegpt_r1_2rps", 20), ("burstgpt_r0_0.5rps", 5)
    ]  # fmt: skip
    assert s.score_slos == ((600.0, 65.0), (1000.0, 100.0))
    (sg, stage), (bg, bstage) = s.plan()[0], s.plan()[2]
    kw = dict(base_url="u", model="m", result_dir=Path("/r"), ttft_slo_ms=600, tpot_slo_ms=65)
    cmd = " ".join(bench.bench_command(sg, stage, **kw))
    assert "--dataset-name sharegpt --dataset-path /d/sg.json" in cmd
    assert "--sharegpt-output-len 256" in cmd and "--ignore-eos" not in cmd
    cmd = bench.bench_command(bg, bstage, **kw)
    joined = " ".join(cmd)
    assert "--dataset-name custom --dataset-path /d/bg.jsonl" in joined
    assert "--custom-output-len -1" in joined and "--sharegpt-output-len" not in joined
    assert {"--ignore-eos", "--skip-chat-template", "--disable-shuffle"} <= set(cmd)
    with pytest.raises(SystemExit):
        bench.BenchSettings.from_env({"BENCH_WORKLOADS": '[{"name": "a"}]'})
    with pytest.raises(SystemExit):
        bench.BenchSettings.from_env({})


def test_a_failed_stage_skips_only_the_rest_of_its_workload(tmp_path) -> None:
    fake = tmp_path / "fake_bench.py"
    fake.write_text(FAKE_BENCH.replace("args = sys.argv[1:]",
                    "args = sys.argv[1:]\nif 'bad' in args: sys.exit(3)"))  # fmt: skip
    workloads = json.dumps([
        {"name": "a", "dataset": "bad", "rates": "1,2"},
        {"name": "b", "dataset": "good", "rates": "1"},
    ])  # fmt: skip
    s = bench.BenchSettings.from_env({
        "BENCH_WORKLOADS": workloads, "BENCH_CMD": f"{sys.executable} {fake}",
        "BENCH_STAGE_S": "4", "BENCH_GAP_S": "0", "BENCH_SCORE_SLOS": "1000:100,200:50",
    })  # fmt: skip
    stages, finished = asyncio.run(
        bench.run_stages(
            s, base_url="u", model="m", out=tmp_path, agents=[], state_poll=False,
            ttft_slo_ms=1000, tpot_slo_ms=200,
        )
    )  # fmt: skip
    assert [(st.name, code) for st, _, _, code in stages] == [("a_r0_1rps", 3), ("b_r0_1rps", 0)]
    summary = bench.summarize(
        tmp_path, stages, finished, {}, ttft_slo_ms=1000, tpot_slo_ms=200,
        score_slos=s.score_slos,
    )  # fmt: skip
    b = summary["stages"]["b_r0_1rps"]
    assert b["workload"] == "b" and b["good"] == 3
    assert b["by_slo"]["1000/100"]["good"] == 3 and b["by_slo"]["200/50"]["good"] == 0


# ---- replay workloads --------------------------------------------------------------------


def _replay_dataset(tmp_path, arrivals):
    path = tmp_path / "replay.jsonl"
    with open(path, "w") as handle:
        for i, at in enumerate(arrivals):
            handle.write(json.dumps({"arrival_s": at, "prompt": f"p{i}", "output_tokens": 3,
                                     "request_tokens": 5}) + "\n")  # fmt: skip
    bench.replay_rows.cache_clear()
    return path


def test_a_replay_workload_is_planned_as_segments_and_a_total(tmp_path) -> None:
    path = _replay_dataset(tmp_path, [0.0, 1.0, 650.0, 1300.0])
    env = {"BENCH_WORKLOADS": json.dumps([
        {"name": "bgpt", "dataset": str(path), "replay": True, "segment_s": 600}])}  # fmt: skip
    stages = bench.BenchSettings.from_env(env).stages()
    assert [s.name for s in stages] == ["bgpt_w00", "bgpt_w01", "bgpt_w02", "bgpt_total"]
    assert [s.num_prompts for s in stages] == [2, 1, 1, 4]
    assert stages[0].segment == (0.0, 600.0) and stages[-1].segment[1] == float("inf")


def test_replay_sends_at_the_arrival_times_and_measures_the_stream(tmp_path) -> None:
    import httpx

    sent = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        sent.append((body["prompt"], body["max_tokens"], body.get("ignore_eos"), clock.now))
        if body["prompt"] == "p1":
            return httpx.Response(503, text="busy")
        lines = [f'data: {json.dumps({"choices": [{"text": "a"}]})}' for _ in range(3)]
        lines += [f'data: {json.dumps({"choices": [], "usage": {"completion_tokens": 3}})}',
                  "data: [DONE]"]  # fmt: skip
        return httpx.Response(200, text="\n\n".join(lines) + "\n\n")

    class Clock:
        now = 0.0

        def __call__(self) -> float:
            return self.now

    clock = Clock()
    real_sleep = asyncio.sleep

    async def fake_sleep(delay):  # the replay waits on its own clock; started tasks run first
        await real_sleep(0)
        clock.now += delay

    rows = bench.replay_rows(str(_replay_dataset(tmp_path, [0.0, 2.5, 7.0])))
    settings = bench.BenchSettings(dataset="x", rates=(1.0,), replay=True)
    bench.asyncio.sleep = fake_sleep
    try:
        t0, records, end = asyncio.run(bench.replay_requests(
            settings, rows, base_url="http://proxy", model="m", clock=clock,
            transport=httpx.MockTransport(handler)))  # fmt: skip
    finally:
        bench.asyncio.sleep = real_sleep
    assert [s[3] for s in sent] == [0.0, 2.5, 7.0]  # each at its own arrival
    assert all(s[1] == 3 and s[2] is True for s in sent)
    assert records[1]["error"] == "HTTP 503" and records[1]["ttft"] == 0.0
    assert records[0]["output_len"] == 3 and len(records[0]["itl"]) == 2
    result = bench.replay_result(records)
    assert result["completed"] == 2 and result["failed"] == 1
    outcome = bench.request_outcomes(result, 600, 100)
    assert outcome["offered"] == 3 and outcome["ok"] == 2
