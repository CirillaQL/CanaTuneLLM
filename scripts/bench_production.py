"""Benchmark through the production path with `vllm bench serve` (e.g. ShareGPT).

The service starts as in smoke_production.py (BENCH_MODE):
  cantune   the `cantune` policy from a cold start: the Canary calibrates and
            publishes, the production groups move to H (controller.solver decides
            between the static default and the optional solver)
  static    the `cantune` policy from a stored tier table (CANATUNE_TIER_TABLE):
            no calibration
  baseline  the `round_robin` policy, clocks left to the driver
Then `vllm bench serve` runs once per request rate (BENCH_RATES) against the proxy,
with the same dataset sample (fixed seed) in every mode, while the node agents'
NVML energy counters of every GPU and (cantune/static) the service state are read
every second. A stage's energy window is the benchmark itself: from its first
request to its end (vllm's `duration`), not the dataset loading before it.

Environment: everything `canatune.controller.process_controller` needs, plus
  BENCH_OUT          directory for the results (required)
  BENCH_MODE         cantune | static | baseline
  BENCH_DATASET      the ShareGPT json (required)
  BENCH_RATES        request rates in req/s, comma separated (default 1,2,4)
  BENCH_STAGE_S      seconds of arrivals per rate: num_prompts = rate x this (default 180)
  BENCH_GAP_S        idle seconds after each stage (default 30)
  BENCH_OUTPUT_LEN   fixed output length (--sharegpt-output-len); unset: the dataset's
  BENCH_IGNORE_EOS   1 (default): generate the full output length; 0: stop at EOS
  BENCH_SEED         dataset sampling seed (default 20261002)
  BENCH_BURSTINESS   arrival burstiness, 1 = Poisson (default 1)
  BENCH_ENDPOINT     /v1/completions (default) or /v1/chat/completions
  BENCH_TOKENIZER    tokenizer for vllm bench (default MODEL_PATH)
  BENCH_CMD          the benchmark command (default "vllm bench serve")
  BENCH_EXTRA_ARGS   more arguments for it
  BENCH_DEADLINE_S   cantune: seconds to wait for the publish (default 3000)

Exit status: 0 done, 2 no table before the deadline, 3 the service stopped early,
4 the service never became ready, 5 calibration kept failing, 6 a benchmark failed.
"""

import asyncio
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))

import smoke_production as smoke  # noqa: E402

from canatune import loadgen  # noqa: E402
from canatune.config import load_config  # noqa: E402
from canatune.proxy.proxy import parse_endpoints  # noqa: E402


def log(message: str) -> None:
    print(f"bench: {message}", flush=True)


@dataclass(frozen=True)
class Stage:
    name: str
    rate_rps: float
    num_prompts: int


@dataclass(frozen=True)
class BenchSettings:
    dataset: str
    rates: tuple[float, ...]
    stage_s: float = 180.0
    gap_s: float = 30.0
    output_len: int | None = None
    ignore_eos: bool = True
    seed: int = 20261002
    burstiness: float = 1.0
    endpoint: str = "/v1/completions"
    tokenizer: str | None = None
    command: tuple[str, ...] = ("vllm", "bench", "serve")
    extra_args: tuple[str, ...] = ()

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "BenchSettings":
        rates = tuple(float(r) for r in env.get("BENCH_RATES", "1,2,4").split(",") if r.strip())
        if not rates or any(r <= 0 for r in rates):
            raise SystemExit("BENCH_RATES must be positive request rates")
        endpoint = env.get("BENCH_ENDPOINT", "/v1/completions")
        if endpoint not in ("/v1/completions", "/v1/chat/completions"):
            raise SystemExit("BENCH_ENDPOINT must be /v1/completions or /v1/chat/completions")
        output = env.get("BENCH_OUTPUT_LEN")
        return cls(
            dataset=env["BENCH_DATASET"],
            rates=rates,
            stage_s=float(env.get("BENCH_STAGE_S", "180")),
            gap_s=float(env.get("BENCH_GAP_S", "30")),
            output_len=int(output) if output else None,
            ignore_eos=env.get("BENCH_IGNORE_EOS", "1") != "0",
            seed=int(env.get("BENCH_SEED", "20261002")),
            burstiness=float(env.get("BENCH_BURSTINESS", "1")),
            endpoint=endpoint,
            tokenizer=env.get("BENCH_TOKENIZER") or env.get("MODEL_PATH") or None,
            command=tuple(shlex.split(env.get("BENCH_CMD", "vllm bench serve"))),
            extra_args=tuple(shlex.split(env.get("BENCH_EXTRA_ARGS", ""))),
        )

    def stages(self) -> list[Stage]:
        return [
            Stage(f"r{i}_{rate:g}rps", rate, max(1, round(rate * self.stage_s)))
            for i, rate in enumerate(self.rates)
        ]


def bench_command(
    settings: BenchSettings,
    stage: Stage,
    *,
    base_url: str,
    model: str,
    result_dir: Path,
    ttft_slo_ms: float,
    tpot_slo_ms: float,
) -> list[str]:
    """One `vllm bench serve` run of a stage. The seed fixes the dataset sample, so
    every mode sends the same prompts at a given rate."""
    chat = settings.endpoint == "/v1/chat/completions"
    cmd = [
        *settings.command,
        "--backend", "openai-chat" if chat else "openai",
        "--base-url", base_url,
        "--endpoint", settings.endpoint,
        "--model", model,
        "--dataset-name", "sharegpt",
        "--dataset-path", settings.dataset,
        "--num-prompts", str(stage.num_prompts),
        "--request-rate", f"{stage.rate_rps:g}",
        "--burstiness", f"{settings.burstiness:g}",
        "--seed", str(settings.seed),
        "--temperature", "0",
        "--percentile-metrics", "ttft,tpot,itl,e2el",
        "--metric-percentiles", "50,90,95,99",
        "--goodput", f"ttft:{ttft_slo_ms:g}", f"tpot:{tpot_slo_ms:g}",
        "--save-result", "--save-detailed",
        "--result-dir", str(result_dir),
        "--result-filename", "bench.json",
        "--disable-tqdm",
    ]  # fmt: skip
    if settings.tokenizer:
        cmd += ["--tokenizer", settings.tokenizer]
    if settings.output_len is not None:
        cmd += ["--sharegpt-output-len", str(settings.output_len)]
    if settings.ignore_eos:
        cmd.append("--ignore-eos")
    return cmd + list(settings.extra_args)


def _quantile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1) + 0.5))]


def request_outcomes(
    result: Mapping[str, Any], ttft_slo_ms: float, tpot_slo_ms: float
) -> dict[str, Any]:
    """Per-request SLO outcomes from vllm's detailed result (ttfts in s, itls per
    request in s). A request without a first token failed. TPOT as vllm defines it:
    the time after the first token (the sum of its ITLs) over output tokens - 1, so
    an SSE chunk carrying several tokens does not inflate it."""
    ttfts = result.get("ttfts") or []
    itls = result.get("itls") or [[] for _ in ttfts]
    errors = result.get("errors") or ["" for _ in ttfts]
    outputs = result.get("output_lens") or [len(i) + 1 for i in itls]
    ok, good, ttft_ms, tpot_ms = 0, 0, [], []
    for ttft, itl, error, output in zip(ttfts, itls, errors, outputs):
        if error or not ttft:
            continue
        ok += 1
        ttft_ms.append(ttft * 1000.0)
        tpot = sum(itl) / (output - 1) * 1000.0 if itl and output and output > 1 else None
        if tpot is not None:
            tpot_ms.append(tpot)
        if ttft * 1000.0 <= ttft_slo_ms and (tpot is None or tpot <= tpot_slo_ms):
            good += 1
    offered = len(ttfts)
    return {
        "offered": offered,
        "ok": ok,
        "good": good,
        "goodput": good / offered if offered else None,
        "ttft_p50_ms": _quantile(ttft_ms, 0.5),
        "ttft_p95_ms": _quantile(ttft_ms, 0.95),
        "ttft_p99_ms": _quantile(ttft_ms, 0.99),
        "tpot_p50_ms": _quantile(tpot_ms, 0.5),
        "tpot_p95_ms": _quantile(tpot_ms, 0.95),
    }


def stage_summary(
    stage: Stage,
    window: tuple[float, float],
    result: Mapping[str, Any] | None,
    series: Mapping[str, Sequence[tuple[float, float]]],
    gpu_names: Mapping[str, str],
    timeline: Sequence[Mapping[str, Any]],
    *,
    ttft_slo_ms: float,
    tpot_slo_ms: float,
) -> dict[str, Any]:
    start, end = window
    per_gpu = {
        gpu_names.get(k, k): loadgen.energy_between(v, start, end) for k, v in series.items()
    }
    known = [e for e in per_gpu.values() if e is not None]
    energy = sum(known) if known and len(known) == len(per_gpu) else None
    out: dict[str, Any] = {
        "rate_rps": stage.rate_rps,
        "num_prompts": stage.num_prompts,
        "window_s": [round(start, 2), round(end, 2)],
        "energy_j": energy,
        "power_w": energy / (end - start) if energy is not None and end > start else None,
        "energy_by_gpu_j": per_gpu,
        "modes": dict(Counter(r.get("mode") for r in timeline if start <= r.get("t_s", -1) < end)),
    }
    if result is None:
        return out
    outcomes = request_outcomes(result, ttft_slo_ms, tpot_slo_ms)
    output_tokens = result.get("total_output_tokens")
    out.update(
        {
            "completed": result.get("completed"),
            "failed": result.get("failed"),
            "duration_s": result.get("duration"),
            "request_throughput": result.get("request_throughput"),
            "output_throughput": result.get("output_throughput"),
            "total_input_tokens": result.get("total_input_tokens"),
            "total_output_tokens": output_tokens,
            "vllm_request_goodput": result.get("request_goodput"),
            **outcomes,
            "j_per_good_request": (
                energy / outcomes["good"] if energy is not None and outcomes["good"] else None
            ),
            "j_per_output_token": (
                energy / output_tokens if energy is not None and output_tokens else None
            ),
        }
    )
    return out


class Sampler:
    """The load generator's poller (energy of every GPU, the service state) on its
    own clock, while the benchmark subprocesses run."""

    def __init__(self, base_url: str, agents: Sequence[str], out: Path, state_poll: bool):
        meta = {"phases": [], "duration_s": 0.0, "seed": 0}
        self.run = loadgen.LoadRun(
            meta, [], base_url=base_url, model="", out_dir=out, agents=agents,
            state_poll=state_poll, actions=False,
        )  # fmt: skip
        self.stop = asyncio.Event()
        self.task: asyncio.Task | None = None
        self._client: httpx.AsyncClient | None = None

    def now_s(self) -> float:
        return self.run.now_s()

    async def __aenter__(self) -> "Sampler":
        self.run._t0 = time.monotonic()
        self._client = httpx.AsyncClient(timeout=10, trust_env=False)
        self.task = asyncio.create_task(self.run._poll(self._client, self.stop))
        return self

    async def __aexit__(self, *_: object) -> None:
        self.stop.set()
        if self.task is not None:
            await self.task
        if self._client is not None:
            await self._client.aclose()


async def run_stages(
    settings: BenchSettings,
    *,
    base_url: str,
    model: str,
    out: Path,
    agents: Sequence[str],
    state_poll: bool,
    ttft_slo_ms: float,
    tpot_slo_ms: float,
) -> tuple[list[tuple[Stage, tuple[float, float], dict | None, int]], float]:
    """-> per stage (stage, energy window, vllm result or None, exit code), and the
    end of the run on the sampler's clock."""
    done = []
    async with Sampler(base_url, agents, out, state_poll) as sampler:
        for stage in settings.stages():
            stage_dir = out / "stages" / stage.name
            stage_dir.mkdir(parents=True, exist_ok=True)
            cmd = bench_command(
                settings, stage, base_url=base_url, model=model, result_dir=stage_dir,
                ttft_slo_ms=ttft_slo_ms, tpot_slo_ms=tpot_slo_ms,
            )  # fmt: skip
            (stage_dir / "command.txt").write_text(shlex.join(cmd) + "\n")
            log(f"{stage.name}: {stage.num_prompts} prompts at {stage.rate_rps:g} req/s")
            launched = sampler.now_s()
            with open(stage_dir / "bench.log", "wb") as handle:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=handle, stderr=subprocess.STDOUT
                )
                code = await proc.wait()
            ended = sampler.now_s()
            path = stage_dir / "bench.json"
            result = json.loads(path.read_text()) if code == 0 and path.exists() else None
            duration = float((result or {}).get("duration") or 0.0)
            # The benchmark's own span; the dataset loading before it is not load.
            start = max(launched, ended - duration) if duration > 0 else launched
            done.append((stage, (start, ended), result, code))
            log(f"{stage.name}: exit {code}, {duration:.0f} s of load")
            if code != 0:
                break
            if settings.gap_s > 0:
                await asyncio.sleep(settings.gap_s)
        finished = sampler.now_s()
    return done, finished


def summarize(
    out: Path,
    stages: Sequence[tuple[Stage, tuple[float, float], dict | None, int]],
    finished_s: float,
    gpu_names: Mapping[str, str],
    *,
    ttft_slo_ms: float,
    tpot_slo_ms: float,
) -> dict[str, Any]:
    series = loadgen.energy_series(loadgen._read_jsonl(out / "energy.jsonl"))
    timeline = loadgen._read_jsonl(out / "timeline.jsonl")
    result: dict[str, Any] = {"stages": {}}
    for stage, window, bench, code in stages:
        summary = stage_summary(
            stage, window, bench, series, gpu_names, timeline,
            ttft_slo_ms=ttft_slo_ms, tpot_slo_ms=tpot_slo_ms,
        )  # fmt: skip
        summary["exit"] = code
        result["stages"][stage.name] = summary
    if stages:
        span = (stages[0][1][0], finished_s)  # the whole run, gaps included
        energies = [loadgen.energy_between(v, *span) for v in series.values()]
        known = [e for e in energies if e is not None]
        result["all"] = {
            "window_s": [round(span[0], 2), round(span[1], 2)],
            "energy_j": sum(known) if known and len(known) == len(energies) else None,
        }
    return result


def main() -> int:
    out = Path(os.environ["BENCH_OUT"])
    out.mkdir(parents=True, exist_ok=True)
    mode = os.environ.get("BENCH_MODE", "cantune")
    if mode not in ("cantune", "static", "baseline"):
        raise SystemExit("BENCH_MODE must be cantune, static or baseline")
    settings = BenchSettings.from_env(os.environ)
    config = load_config(os.environ["CANATUNE_CONFIG"])
    policy = config["routing"]["policy"]
    if (mode != "baseline") != (policy == "cantune"):
        raise SystemExit(f"BENCH_MODE={mode} does not match routing.policy={policy}")
    if not Path(settings.dataset).is_file():
        raise SystemExit(f"BENCH_DATASET {settings.dataset} is not a file")
    cap = config.get("canary", {}).get("max_output_tokens")
    if mode == "cantune" and settings.output_len and cap and settings.output_len > cap:
        log(f"warning: output length {settings.output_len} exceeds the Canary's cap {cap}")
    proxy = config["proxy"]
    host = "127.0.0.1" if proxy["host"] == "0.0.0.0" else proxy["host"]
    base = f"http://{host}:{proxy['port']}"
    slo = config["experiment"]["slo"]
    ttft_slo, tpot_slo = float(slo["ttft_ms"]), float(slo["tpot_ms"])
    agents, gpu_names = smoke.agents_and_names(config)
    (out / "bench_plan.json").write_text(
        json.dumps(
            {"mode": mode, "stages": [s.__dict__ for s in settings.stages()],
             **{k: v for k, v in settings.__dict__.items() if k != "rates"}},
            indent=2, default=list,
        ) + "\n"
    )  # fmt: skip

    started = time.monotonic()
    controller = subprocess.Popen(
        [sys.executable, "-m", "canatune.controller.process_controller"],
        start_new_session=True,
    )
    status = 4
    summary: dict[str, Any] = {"mode": mode}
    try:
        with httpx.Client(trust_env=False) as client:
            failed = smoke.wait_ready(client, base, mode, controller)
            if failed is not None:
                status = failed
                return status
            summary["ready_s"] = round(time.monotonic() - started, 1)
            if mode == "cantune":
                deadline = float(os.environ.get("BENCH_DEADLINE_S", "3000"))
                failed, table = smoke.wait_publish(client, base, controller, deadline)
                if failed is not None:
                    status = failed
                    return status
                summary["published_s"] = round(time.monotonic() - started, 1)
                summary["reached_h"] = smoke.wait_h(client, base, table)
            elif mode == "static":
                failed, table = smoke.wait_publish(client, base, controller, 60.0)
                if failed is not None:
                    log("the stored tier table was not loaded")
                    status = failed
                    return status
                summary["reached_h"] = smoke.wait_h(client, base, table)
            else:
                summary["clocks_at_start"] = smoke.reset_clocks(client, agents)
            endpoints = parse_endpoints(config)
            model = smoke.model_name(client, endpoints)
            smoke.warm_pairs(client, config, endpoints, model)

            status = 6
            stages, finished = asyncio.run(
                run_stages(
                    settings, base_url=base, model=model, out=out, agents=agents,
                    state_poll=mode != "baseline", ttft_slo_ms=ttft_slo, tpot_slo_ms=tpot_slo,
                )
            )  # fmt: skip
            result = summarize(
                out, stages, finished, gpu_names, ttft_slo_ms=ttft_slo, tpot_slo_ms=tpot_slo
            )
            (out / "bench_summary.json").write_text(json.dumps(result, indent=2) + "\n")
            for name, s in result["stages"].items():
                log(
                    f"{name}: exit={s['exit']} ok={s.get('ok')}/{s.get('offered')} "
                    f"goodput={s.get('goodput')} ttft_p95={s.get('ttft_p95_ms')} "
                    f"energy_j={s['energy_j']} J/tok={s.get('j_per_output_token')} "
                    f"modes={s['modes']}"
                )
            smoke.save_service_state(client, base, out, config)
            if all(code == 0 for *_, code in stages) and len(stages) == len(settings.rates):
                status = 0
            return status
    except Exception as error:
        log(f"error: {error!r}")
        summary["error"] = repr(error)
        return status
    finally:
        if controller.poll() is None:
            controller.send_signal(signal.SIGTERM)  # ordered shutdown, agents reset clocks
            try:
                controller.wait(timeout=180)
            except subprocess.TimeoutExpired:
                os.killpg(controller.pid, signal.SIGKILL)
                controller.wait()
        summary["status"] = {
            0: "done",
            2: "deadline",
            3: "stopped_early",
            4: "never_ready",
            5: "calibration_failed",
            6: "bench_failed",
        }[status]
        summary["controller_exit"] = controller.returncode
        (out / "bench_run.json").write_text(json.dumps(summary, indent=2) + "\n")
        log(json.dumps(summary))


if __name__ == "__main__":
    raise SystemExit(main())
