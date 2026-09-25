"""Live vLLM state from each endpoint's Prometheus `/metrics`."""

import asyncio
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import httpx

RUNNING = ("vllm:num_requests_running",)
WAITING = ("vllm:num_requests_waiting",)
KV_USAGE = ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc")
PREEMPTIONS = ("vllm:num_preemptions_total",)


def prom_value(text: str, names: tuple[str, ...]) -> float | None:
    """Sum all samples of the first metric name present (label sets are summed)."""
    for name in names:
        total, seen = 0.0, False
        for line in text.splitlines():
            if not line.startswith(name) or line.startswith("#"):
                continue
            if line[len(name) : len(name) + 1] not in ("{", " "):
                continue
            try:
                total += float(line.rsplit(" ", 1)[1])
            except (IndexError, ValueError):
                continue
            seen = True
        if seen:
            return total
    return None


@dataclass(frozen=True)
class EndpointSnapshot:
    taken_at: float  # monotonic seconds
    running: float | None
    waiting: float | None
    kv_usage: float | None
    preemptions_total: float | None
    ok: bool

    def age(self, now: float) -> float:
        return now - self.taken_at


def parse_snapshot(text: str, taken_at: float) -> EndpointSnapshot:
    return EndpointSnapshot(
        taken_at=taken_at,
        running=prom_value(text, RUNNING),
        waiting=prom_value(text, WAITING),
        kv_usage=prom_value(text, KV_USAGE),
        preemptions_total=prom_value(text, PREEMPTIONS),
        ok=True,
    )


class Telemetry:
    """Polls `/metrics` of every endpoint; keeps the latest snapshot and the
    preemption delta between the two most recent successful scrapes."""

    def __init__(
        self,
        metrics_urls: Mapping[str, str],
        *,
        period_s: float,
        client_factory: Callable[[], httpx.AsyncClient],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if period_s <= 0:
            raise ValueError("telemetry period must be positive")
        self.metrics_urls = dict(metrics_urls)
        self.period_s = period_s
        self._client_factory = client_factory
        self._clock = clock
        self.latest: dict[str, EndpointSnapshot] = {}
        self.preemption_delta: dict[str, float] = {}

    def update(self, endpoint: str, snapshot: EndpointSnapshot) -> None:
        previous = self.latest.get(endpoint)
        if (
            snapshot.ok
            and previous is not None
            and previous.ok
            and previous.preemptions_total is not None
            and snapshot.preemptions_total is not None
        ):
            self.preemption_delta[endpoint] = max(
                0.0, snapshot.preemptions_total - previous.preemptions_total
            )
        if snapshot.ok or previous is None:
            self.latest[endpoint] = snapshot

    async def scrape_once(self, client: httpx.AsyncClient) -> None:
        async def one(endpoint: str, url: str) -> None:
            try:
                response = await client.get(url)
                response.raise_for_status()
                self.update(endpoint, parse_snapshot(response.text, self._clock()))
            except httpx.HTTPError:
                # A failed scrape leaves the old snapshot; it ages out and turns unsafe.
                pass

        await asyncio.gather(*(one(e, u) for e, u in self.metrics_urls.items()))

    async def run(self, stop: asyncio.Event) -> None:
        async with self._client_factory() as client:
            while not stop.is_set():
                await self.scrape_once(client)
                try:
                    await asyncio.wait_for(stop.wait(), timeout=self.period_s)
                except TimeoutError:
                    pass

    def fresh(self, endpoint: str, max_age_s: float) -> EndpointSnapshot | None:
        snapshot = self.latest.get(endpoint)
        if snapshot is None or not snapshot.ok or snapshot.age(self._clock()) > max_age_s:
            return None
        return snapshot
