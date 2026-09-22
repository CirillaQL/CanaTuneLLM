"""Versioned Canary frequency decisions with locked, atomic JSON persistence."""

import json
import math
import os
import tempfile
import threading
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, ClassVar

SCHEMA_VERSION = 1
WORKLOAD_IDS = (
    "small_light",
    "prefill_medium",
    "prefill_heavy",
    "decode_medium",
    "decode_heavy",
    "balanced_medium",
    "both_heavy",
)
FALLBACK_SOURCES = frozenset(
    {"safe_high_after_slo_exhausted", "safe_high_after_measurement_exhausted"}
)
_VALUE_FIELDS = frozenset(
    {
        "prefill_frequency_mhz",
        "decode_frequency_mhz",
        "measured_power_w",
        "measured_energy_j",
        "ttft_ms",
        "tpot_ms",
        "prefill_endpoint_id",
        "decode_endpoint_id",
        "sample_count",
        "updated_unix_s",
        "source",
        "slo_met",
        "fallback_reason",
    }
)


class FrequencyTableError(ValueError):
    """The table schema, entry, or update is invalid."""


class RevisionConflict(FrequencyTableError):
    """A conditional write used an outdated revision."""


def _integer(value: Any, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise FrequencyTableError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: Any, name: str, *, minimum: float, strict: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FrequencyTableError(f"{name} must be a finite number")
    try:
        result = float(value)
    except OverflowError as error:
        raise FrequencyTableError(f"{name} must be a finite number") from error
    if not math.isfinite(result) or (result <= minimum if strict else result < minimum):
        boundary = ">" if strict else ">="
        raise FrequencyTableError(f"{name} must be finite and {boundary} {minimum}")
    return result


def _optional_number(value: Any, name: str, *, minimum: float, strict: bool) -> float | None:
    return None if value is None else _number(value, name, minimum=minimum, strict=strict)


@dataclass(frozen=True)
class FrequencyDecision:
    """One confirmed measurement or explicitly marked safe-high fallback."""

    prefill_frequency_mhz: int
    decode_frequency_mhz: int
    measured_power_w: float | None
    measured_energy_j: float | None
    ttft_ms: float | None
    tpot_ms: float | None
    prefill_endpoint_id: str
    decode_endpoint_id: str
    sample_count: int
    updated_unix_s: float
    source: str
    slo_met: bool
    fallback_reason: str | None

    TTFT_SLO_MS: ClassVar[float] = 500.0
    TPOT_SLO_MS: ClassVar[float] = 200.0

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "FrequencyDecision":
        if not isinstance(raw, Mapping) or set(raw) != _VALUE_FIELDS:
            raise FrequencyTableError("decision must contain exactly the v1 decision fields")
        value = cls(
            prefill_frequency_mhz=_integer(
                raw["prefill_frequency_mhz"], "prefill_frequency_mhz", minimum=1
            ),
            decode_frequency_mhz=_integer(
                raw["decode_frequency_mhz"], "decode_frequency_mhz", minimum=1
            ),
            measured_power_w=_optional_number(
                raw["measured_power_w"], "measured_power_w", minimum=0, strict=True
            ),
            measured_energy_j=_optional_number(
                raw["measured_energy_j"], "measured_energy_j", minimum=0, strict=True
            ),
            ttft_ms=_optional_number(raw["ttft_ms"], "ttft_ms", minimum=0, strict=False),
            tpot_ms=_optional_number(raw["tpot_ms"], "tpot_ms", minimum=0, strict=False),
            prefill_endpoint_id=raw["prefill_endpoint_id"],
            decode_endpoint_id=raw["decode_endpoint_id"],
            sample_count=_integer(raw["sample_count"], "sample_count", minimum=0),
            updated_unix_s=_number(raw["updated_unix_s"], "updated_unix_s", minimum=0, strict=True),
            source=raw["source"],
            slo_met=raw["slo_met"],
            fallback_reason=raw["fallback_reason"],
        )
        value.validate()
        return value

    def validate(self) -> None:
        # Revalidate dataclass inputs: callers may bypass from_mapping().
        _integer(self.prefill_frequency_mhz, "prefill_frequency_mhz", minimum=1)
        _integer(self.decode_frequency_mhz, "decode_frequency_mhz", minimum=1)
        _integer(self.sample_count, "sample_count", minimum=0)
        _number(self.updated_unix_s, "updated_unix_s", minimum=0, strict=True)
        if self.prefill_endpoint_id != "P0" or self.decode_endpoint_id != "D0":
            raise FrequencyTableError("v1 evidence must come from Canary pair P0-D0")
        if not isinstance(self.source, str) or not self.source.strip():
            raise FrequencyTableError("source must be a nonempty string")
        if type(self.slo_met) is not bool:
            raise FrequencyTableError("slo_met must be a boolean")
        if self.fallback_reason is not None and not isinstance(self.fallback_reason, str):
            raise FrequencyTableError("fallback_reason must be a string or null")

        metrics = (
            _optional_number(self.measured_power_w, "measured_power_w", minimum=0, strict=True),
            _optional_number(self.measured_energy_j, "measured_energy_j", minimum=0, strict=True),
            _optional_number(self.ttft_ms, "ttft_ms", minimum=0, strict=False),
            _optional_number(self.tpot_ms, "tpot_ms", minimum=0, strict=False),
        )
        if self.slo_met:
            if self.source in FALLBACK_SOURCES:
                raise FrequencyTableError("a measured decision cannot use a fallback source")
            if self.sample_count == 0 or any(metric is None for metric in metrics):
                raise FrequencyTableError("a measured decision requires samples and all metrics")
            if self.fallback_reason is not None:
                raise FrequencyTableError("a measured decision must not have a fallback reason")
            if self.ttft_ms >= self.TTFT_SLO_MS or self.tpot_ms > self.TPOT_SLO_MS:
                raise FrequencyTableError("measured decision exceeds the TTFT or TPOT SLO")
        elif (
            self.source not in FALLBACK_SOURCES
            or self.sample_count != 0
            or any(metric is not None for metric in metrics)
            or not self.fallback_reason
            or not self.fallback_reason.strip()
        ):
            raise FrequencyTableError("invalid safe-high fallback decision")


@dataclass(frozen=True)
class FrequencyTableEntry:
    workload_id: str
    revision: int
    value: FrequencyDecision | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "workload_id": self.workload_id,
            "revision": self.revision,
            "value": asdict(self.value) if self.value is not None else None,
        }


class WorkloadFrequencyTable:
    """Single-process owner of the fixed v1 workload table."""

    def __init__(self, persistence_path: str | Path) -> None:
        self._path = Path(persistence_path)
        self._lock = threading.RLock()
        self._entries = {key: FrequencyTableEntry(key, 0, None) for key in WORKLOAD_IDS}
        with self._lock:
            if self._path.exists():
                self._entries = self._load()
            else:
                self._persist_locked()

    def _document_locked(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "entries": {key: entry.as_dict() for key, entry in self._entries.items()},
        }

    def _load(self) -> dict[str, FrequencyTableEntry]:
        try:
            document = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise FrequencyTableError(f"cannot load frequency table: {self._path}") from error
        if not isinstance(document, dict) or set(document) != {"schema_version", "entries"}:
            raise FrequencyTableError("invalid frequency table document")
        if (
            type(document["schema_version"]) is not int
            or document["schema_version"] != SCHEMA_VERSION
        ):
            raise FrequencyTableError("unsupported frequency table schema version")
        raw_entries = document["entries"]
        if not isinstance(raw_entries, dict) or set(raw_entries) != set(WORKLOAD_IDS):
            raise FrequencyTableError("frequency table must contain the seven v1 workloads")
        entries = {}
        for key in WORKLOAD_IDS:
            raw = raw_entries[key]
            if not isinstance(raw, dict) or set(raw) != {"workload_id", "revision", "value"}:
                raise FrequencyTableError(f"invalid entry for {key}")
            if raw["workload_id"] != key:
                raise FrequencyTableError(f"entry workload_id mismatch for {key}")
            revision = _integer(raw["revision"], f"{key}.revision", minimum=0)
            raw_value = raw["value"]
            if (revision == 0) != (raw_value is None):
                raise FrequencyTableError(f"entry revision/value mismatch for {key}")
            value = None if raw_value is None else FrequencyDecision.from_mapping(raw_value)
            entries[key] = FrequencyTableEntry(key, revision, value)
        return entries

    def _persist_locked(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self._path.parent,
                prefix=f".{self._path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                json.dump(
                    self._document_locked(),
                    temporary,
                    indent=2,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                temporary.write("\n")
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, self._path)
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def read(self, workload_id: str) -> FrequencyTableEntry:
        """Return one immutable entry, including its revision."""
        with self._lock:
            return self._entries[workload_id]

    def snapshot(self) -> dict[str, Any]:
        """Return the full JSON document as a detached dictionary."""
        with self._lock:
            return self._document_locked()

    def write(
        self,
        workload_id: str,
        value: FrequencyDecision | Mapping[str, Any],
        *,
        expected_revision: int | None = None,
    ) -> FrequencyTableEntry:
        """Validate and publish a decision; no-op when it cannot improve an existing one."""
        raw_value = asdict(value) if isinstance(value, FrequencyDecision) else value
        candidate = FrequencyDecision.from_mapping(raw_value)
        if expected_revision is not None:
            _integer(expected_revision, "expected_revision", minimum=0)

        with self._lock:
            current = self._entries[workload_id]
            if expected_revision is not None and current.revision != expected_revision:
                raise RevisionConflict(
                    f"revision conflict for {workload_id}: "
                    f"expected {expected_revision}, actual {current.revision}"
                )
            if current.value == candidate:
                return current
            if current.value is not None and current.value.slo_met:
                if (
                    not candidate.slo_met
                    or candidate.measured_energy_j >= current.value.measured_energy_j
                ):
                    return current

            updated = FrequencyTableEntry(workload_id, current.revision + 1, candidate)
            self._entries[workload_id] = updated
            try:
                self._persist_locked()
            except Exception:
                self._entries[workload_id] = current
                raise
            return updated
