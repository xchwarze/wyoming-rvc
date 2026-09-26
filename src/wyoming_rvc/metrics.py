"""Per-request timing metrics and small statistics helpers."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass


@dataclass
class SynthesisMetrics:
    """Timings for one synthesis request, in milliseconds."""

    text_chars: int = 0
    sentences: int = 0
    mode: str = ""
    rvc: bool = True
    source_ms: float = 0.0
    rvc_ms: float = 0.0
    encode_ms: float = 0.0
    queue_ms: float = 0.0
    total_ms: float = 0.0
    ttfa_ms: float | None = None
    audio_duration_ms: float = 0.0
    sample_rate: int = 0

    @property
    def rtf(self) -> float:
        """Processing time divided by generated audio time."""
        if self.audio_duration_ms <= 0:
            return math.inf
        return self.total_ms / self.audio_duration_ms

    def as_dict(self) -> dict[str, float | int | str | bool | None]:
        data = asdict(self)
        data["rtf"] = round(self.rtf, 4) if math.isfinite(self.rtf) else None
        for key in ("source_ms", "rvc_ms", "encode_ms", "queue_ms", "total_ms", "audio_duration_ms", "ttfa_ms"):
            if data[key] is not None:
                data[key] = round(data[key], 1)
        return data

    def log(self, logger: logging.Logger, source: str) -> None:
        fields = self.as_dict()
        logger.info(
            "synthesis source=%s " + " ".join(f"{k}=%s" for k in fields),
            source,
            *fields.values(),
            extra={"fields": {"source": source, **fields}},
        )


class Stopwatch:
    """Accumulating monotonic timer: ``with sw: ...`` adds elapsed time."""

    def __init__(self) -> None:
        self.elapsed_ms = 0.0
        self._start = 0.0

    def __enter__(self) -> Stopwatch:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed_ms += (time.perf_counter() - self._start) * 1000.0


def now_ms() -> float:
    return time.perf_counter() * 1000.0


def percentile(values: Sequence[float], pct: float) -> float:
    """Linear-interpolated percentile; ``pct`` in [0, 100]."""
    if not values:
        return math.nan
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    return float(ordered[low] + (ordered[high] - ordered[low]) * (rank - low))


def median(values: Sequence[float]) -> float:
    return percentile(values, 50.0)
