import asyncio
import sys
import time

import pytest

from canatune.infrastructure.stack_sampler import StackSampler, collapse, start_from_env


def busy_leaf() -> None:
    sorted(range(2000))


async def busy_loop(seconds: float) -> None:
    end = time.process_time() + seconds
    while time.process_time() < end:
        for _ in range(20):
            busy_leaf()
        await asyncio.sleep(0)  # the loop's selector releases the GIL every round


@pytest.mark.skipif(
    sys.platform != "linux",
    reason="macOS delivers SIGPROF at the next system call (the selector), Linux on the tick",
)
def test_sampler_sees_the_busy_coroutine_not_the_selector(tmp_path):
    out = tmp_path / "service.collapsed"
    sampler = StackSampler(out, interval_s=0.002, flush_s=0.1).start()
    try:
        asyncio.run(busy_loop(0.5))
        time.sleep(0.15)
        assert out.exists()  # flushed while running: a killed process keeps its samples
    finally:
        sampler.stop()
    counts = {}
    for line in out.read_text().splitlines():
        stack, _, n = line.rpartition(" ")
        counts[stack] = int(n)
    total = sum(counts.values())
    in_busy = sum(n for stack, n in counts.items() if "busy_loop (" in stack)
    assert total >= 50 and in_busy / total > 0.8
    assert any(stack.split(";")[-1].startswith("busy_leaf (") for stack in counts)


def test_collapse_names_function_and_file():
    leaf = collapse(sys._getframe()).split(";")[-1]
    assert leaf.startswith("test_collapse_names_function_and_file (")
    assert "test_stack_sampler.py" in leaf


def test_sampler_starts_only_when_configured(tmp_path):
    assert start_from_env({}) is None
    sampler = start_from_env(
        {"CANATUNE_PROFILE_OUT": str(tmp_path / "p.collapsed"), "CANATUNE_PROFILE_INTERVAL_MS": "2"}
    )
    try:
        assert sampler is not None and sampler.interval_s == 0.002
    finally:
        sampler.stop()
    assert (tmp_path / "p.collapsed").exists()
