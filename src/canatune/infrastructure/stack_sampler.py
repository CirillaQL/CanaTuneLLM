"""In-process CPU profiler: where the service's event loop spends its CPU time.

A SIGPROF interval timer (ITIMER_PROF: process CPU time) fires every `interval_s` of
CPU; the handler runs in the main thread and counts the Python stack that was running.
No ptrace, so it works where py-spy cannot attach (Vera: kernel.yama.ptrace_scope=2).
A sampling thread would not do: it only gets the GIL when the event loop releases it,
i.e. in the selector, so it would see an idle loop however busy the loop is.

Every `flush_s` a daemon thread rewrites the counts atomically to `path` in the
collapsed format that flame graph tools and https://www.speedscope.app read: one line
per stack, `root;...;leaf count`, so a killed process keeps all but the last `flush_s`.
An idle loop uses no CPU and is not sampled; CPU spent in C code (uvloop, httptools,
json, tokenizers) is counted in the Python frame that called it. Enabled for the
service by CANATUNE_PROFILE_OUT.
"""

from __future__ import annotations

import os
import signal
import threading
from collections import Counter
from pathlib import Path
from types import FrameType


def _where(filename: str) -> str:
    parts = filename.replace("\\", "/").split("/")
    for anchor in ("site-packages", "dist-packages", "src"):
        if anchor in parts:
            return "/".join(parts[len(parts) - parts[::-1].index(anchor) :])
    return "/".join(parts[-2:])


def collapse(frame: FrameType | None) -> str:
    """The stack ending in `frame`, root first: `func (file);...` (no line numbers, so
    one function is one node however many lines of it are sampled)."""
    names = []
    while frame is not None:
        code = frame.f_code
        name = getattr(code, "co_qualname", code.co_name)
        names.append(f"{name} ({_where(code.co_filename)})".replace(";", ","))
        frame = frame.f_back
    return ";".join(reversed(names))


class StackSampler:
    """Start from the main thread (signal handlers can only be set there)."""

    def __init__(self, path: str | Path, *, interval_s: float = 0.005, flush_s: float = 10.0):
        self.path = Path(path)
        self.interval_s = interval_s
        self.flush_s = flush_s
        self.counts: Counter[str] = Counter()
        self._halt = threading.Event()
        self._flusher = threading.Thread(target=self._flush_loop, daemon=True,
                                         name="canatune-profile-flush")  # fmt: skip
        self._previous = None

    def _on_sample(self, _signum: int, frame: FrameType | None) -> None:
        self.counts[collapse(frame)] += 1

    def start(self) -> StackSampler:
        self._previous = signal.signal(signal.SIGPROF, self._on_sample)
        signal.setitimer(signal.ITIMER_PROF, self.interval_s, self.interval_s)
        self._flusher.start()
        return self

    def flush(self) -> None:
        counts = dict(self.counts)  # one C-level copy under the GIL: no handler runs inside
        lines = [f"{stack} {n}\n" for stack, n in sorted(counts.items(), key=lambda x: -x[1])]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text("".join(lines))
        os.replace(tmp, self.path)

    def _flush_loop(self) -> None:
        while not self._halt.wait(self.flush_s):
            self.flush()

    def stop(self) -> None:
        signal.setitimer(signal.ITIMER_PROF, 0, 0)
        signal.signal(signal.SIGPROF, self._previous or signal.SIG_DFL)
        self._halt.set()
        self._flusher.join()
        self.flush()


def start_from_env(env: dict[str, str] | None = None) -> StackSampler | None:
    """Profile this process if CANATUNE_PROFILE_OUT is set: one sample per
    CANATUNE_PROFILE_INTERVAL_MS (default 5) ms of CPU. Call from the main thread."""
    env = os.environ if env is None else env
    path = env.get("CANATUNE_PROFILE_OUT")
    if not path:
        return None
    interval = float(env.get("CANATUNE_PROFILE_INTERVAL_MS", "5")) / 1000.0
    sampler = StackSampler(path, interval_s=interval).start()
    print(f"CanaTune: CPU profile, a sample per {interval * 1000:g} ms CPU -> {path}", flush=True)
    return sampler
