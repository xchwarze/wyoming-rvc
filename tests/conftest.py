"""Shared fakes: no model downloads, no torch."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator

import numpy as np
import pytest

from wyoming_rvc.audio import resample
from wyoming_rvc.pipeline import SynthesisOptions, TtsPipeline


class FakePiper:
    """Fake TTS source: one 0.25 s tone per '.'-separated sentence; records timing."""

    sample_rate = 22050
    device = "cpu"

    def __init__(self, seconds_per_sentence: float = 0.25, delay: float = 0.0) -> None:
        self.seconds = seconds_per_sentence
        self.delay = delay
        self.calls: list[str] = []
        self.started: list[float] = []
        self.closed = 0

    def sentences(self, text: str) -> Iterator[np.ndarray]:
        self.calls.append(text)
        try:
            for _part in [p for p in text.split(".") if p.strip()]:
                self.started.append(time.perf_counter())
                time.sleep(self.delay)
                n = int(self.sample_rate * self.seconds)
                t = np.arange(n) / self.sample_rate
                yield (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        finally:
            self.closed += 1


class FakeRvc:
    output_sample_rate = 32000

    def __init__(self, delay: float = 0.02, fail: bool = False, sample_rate: int = 32000) -> None:
        self.delay = delay
        self.fail = fail
        self.output_sample_rate = sample_rate
        self.unloaded = False
        self.pitches: list[int] = []
        self.active = 0
        self.max_active = 0
        self.calls = 0
        self.finished: list[float] = []
        self._lock = threading.Lock()

    def _enter(self) -> None:
        with self._lock:
            self.active += 1
            self.calls += 1
            self.max_active = max(self.max_active, self.active)

    def _exit(self) -> None:
        with self._lock:
            self.active -= 1
            self.finished.append(time.perf_counter())

    def unload(self) -> None:
        self.unloaded = True

    def convert(self, audio, sample_rate, pitch, f0_method, index_rate, protect):
        self.pitches.append(pitch)
        self._enter()
        try:
            time.sleep(self.delay)
            if self.fail:
                raise RuntimeError("boom")
            return resample(audio, sample_rate, self.output_sample_rate), self.output_sample_rate
        finally:
            self._exit()


@pytest.fixture
def piper() -> FakePiper:
    return FakePiper()


@pytest.fixture
def rvc() -> FakeRvc:
    return FakeRvc()


@pytest.fixture
def pipeline(piper: FakePiper, rvc: FakeRvc) -> TtsPipeline:
    return TtsPipeline(source=piper, rvc=rvc, defaults=SynthesisOptions(), rvc_concurrency=1)
