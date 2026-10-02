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
