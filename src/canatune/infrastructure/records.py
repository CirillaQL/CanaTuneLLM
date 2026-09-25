"""Append-only JSONL logs for request outcomes and controller events."""

import json
import threading
import time
from pathlib import Path
from typing import Any


class JsonlLog:
    """Thread-safe JSONL appender; each line is flushed so a killed job keeps its data.
    With `path=None` records are only kept in memory (tests, dry runs)."""

    def __init__(self, path: str | Path | None, *, keep: int = 1000) -> None:
        self.path = None if path is None else Path(path)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.recent: list[dict[str, Any]] = []
        self._keep = keep
        self._lock = threading.Lock()

    def write(self, record: dict[str, Any]) -> None:
        record = {"wall_time": time.time(), **record}
        line = json.dumps(record, sort_keys=True, default=str)
        with self._lock:
            self.recent.append(record)
            if len(self.recent) > self._keep:
                del self.recent[: len(self.recent) - self._keep]
            if self.path is not None:
                with self.path.open("a", encoding="utf-8") as stream:
                    stream.write(line + "\n")
