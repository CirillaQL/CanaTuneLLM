"""A saturated D must not look safe just because its queued requests were cut off."""

import asyncio
import dataclasses

import httpx
import pytest
from test_probe import FakeAgent, make_probe, vllm_handler

from canatune.controller.locator import LocatorRun, LocatorSettings, TierLocator, WindowResult
from canatune.domain.calibration import d_sequence_cap
from canatune.domain.groups import ClockPoint

TOKEN = b'data: {"choices":[{"text":"t"}]}\n\n'


def install_stream(probe, stream_factory):
    upstream = vllm_handler([])

    async def handler(request):
        if request.url.port == 8200 and request.url.path == "/v1/completions":
            return httpx.Response(200, stream=stream_factory(),
                                  headers={"content-type": "text/event-stream"})
        return upstream(request)

    probe.client_factory = lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_closed_window_keeps_late_queued_requests_and_finds_the_engine_knee():
    async def run():
        probe, risk = make_probe([], FakeAgent())
        probe.ttft_slo_ms = 40.0
        slots = asyncio.Semaphore(16)

        class DecodeStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                async with slots:
                    yield TOKEN
                    await asyncio.sleep(0.08)
                    yield TOKEN + b"data: [DONE]\n\n"

        install_stream(probe, DecodeStream)
        result = await probe.closed_window(ClockPoint(1815, 1050), 80, 0.04,
                                           stagger_s=0.02, drain_s=0.08, output_tokens=2)
        assert result.aborted and result.requests_censored > 0
        assert result.requests_completed >= 16
        assert result.requests == result.requests_sent == 80
        assert result.requests_failed == result.requests_unknown == 0
        assert result.violations == 80 - result.requests_completed
        assert len(result.samples) == 80
        assert d_sequence_cap(result.samples, theta=0.1)[:2] == (16, "measured")
        # Missing first tokens are represented by known violations with no fake TTFT.
        assert any(s.violated and s.ttft_ms is None for s in result.samples)
        assert any(s.violated and s.ttft_ms is not None for s in result.samples)
        assert sum(c["violated"] for c in risk.to_json()["cells"].values()) > 0
        assert probe._n_await == probe._decoding == 0
        assert not probe._active and not probe._at_prefill

    asyncio.run(run())


@pytest.mark.parametrize("first_token", [True, False])
def test_censored_but_not_late_is_unknown_instead_of_a_success_or_violation(first_token):
    async def run():
        probe, risk = make_probe([], FakeAgent())
        probe.ttft_slo_ms = 1000.0

        class Unfinished(httpx.AsyncByteStream):
            async def __aiter__(self):
                if first_token:
                    yield TOKEN
                await asyncio.Event().wait()

        install_stream(probe, Unfinished)
        result = await probe.closed_window(ClockPoint(1815, 1050), 1, 0.01,
                                           stagger_s=0, drain_s=0.01)
        assert result.requests_sent == result.requests_censored == result.requests_unknown == 1
        assert result.requests == result.violations == result.requests_completed == 0
        assert result.aborted and not result.samples
        assert not risk.to_json()["cells"]
        assert not probe._active

    asyncio.run(run())


def test_late_first_token_is_recorded_even_if_generation_never_finishes():
    async def run():
        probe, _ = make_probe([], FakeAgent())
        probe.ttft_slo_ms = 5.0

        class LateToken(httpx.AsyncByteStream):
            async def __aiter__(self):
                await asyncio.sleep(0.02)
                yield TOKEN
                await asyncio.Event().wait()

        install_stream(probe, LateToken)
        result = await probe.closed_window(ClockPoint(1815, 1050), 1, 0.01,
                                           stagger_s=0, drain_s=0.04)
        assert result.requests == result.violations == result.requests_censored == 1
        assert result.requests_completed == 0
        assert result.samples[0].ttft_ms >= 20
        assert result.samples[0].violated

    asyncio.run(run())


def test_open_window_also_keeps_requests_waiting_past_ttft_slo():
    async def run():
        probe, _ = make_probe([], FakeAgent())
        probe.s = dataclasses.replace(probe.s, drain_timeout_s=0.03)
        probe.ttft_slo_ms = 5.0

        class NoToken(httpx.AsyncByteStream):
            async def __aiter__(self):
                await asyncio.Event().wait()
                yield TOKEN

        install_stream(probe, NoToken)
        # 160 eq tokens/request, 100 req/s for 0.02 s -> two requests.
        result = await probe.open_window(ClockPoint(1815, 1050), 16000, 0, 0.02, 1.0)
        assert result.requests == result.violations == result.requests_censored == 2
        assert result.aborted and all(s.ttft_ms is None for s in result.samples)

    asyncio.run(run())


def test_censoring_cancels_before_final_metrics_io_can_complete_the_same_request():
    async def run():
        probe, _ = make_probe([], FakeAgent())
        counter = probe._counter
        calls = 0

        async def delayed_counter(*args):
            nonlocal calls
            calls += 1
            if calls == 2:
                await asyncio.sleep(0.08)
            return await counter(*args)

        class FinishesAfterDeadline(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield TOKEN
                await asyncio.sleep(0.06)
                yield TOKEN + b"data: [DONE]\n\n"

        install_stream(probe, FinishesAfterDeadline)
        probe._counter = delayed_counter
        result = await probe.closed_window(ClockPoint(1815, 1050), 1, 0.01,
                                           stagger_s=0, drain_s=0.01)
        assert result.requests_sent == result.requests_censored == 1
        assert result.requests_completed == 0  # no duplicate successful outcome after cutoff
        assert result.requests_unknown == 1

    asyncio.run(run())


def test_a_truncated_repeat_cannot_validate_a_clock_or_capacity():
    class Backend:
        calls = 0

        async def closed_window(self, clock, concurrency, seconds, rep=0):
            self.calls += 1
            return WindowResult(clock, "closed", concurrency, seconds, requests=100,
                                tpot_p95_ms=10, decode_j_per_token=1.0,
                                aborted=self.calls == 2, requests_sent=100,
                                requests_completed=100 if self.calls == 1 else 90,
                                requests_censored=0 if self.calls == 1 else 10)

    locator = TierLocator(Backend(), LocatorSettings(decode_clock_repeats=2))
    locator.run = LocatorRun(started_at=0)
    result = asyncio.run(locator.closed_median(ClockPoint(1815, 1050), 16))
    assert result.aborted and result.requests_censored == 10
    assert result.requests_sent == 200
    assert not locator.feasible(result)
    assert not locator.meets_target(result)
    assert not locator.decode_clean(result)
