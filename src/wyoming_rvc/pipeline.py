"""Text -> TTS source -> (RVC) -> PCM16, with per-request metrics.

Modes (``RVC_MODE``):
  sentence  each sentence is converted as soon as the source produces it (default)
  whole     all sentences are synthesized, concatenated and converted in one RVC pass
  stream    each sentence is converted in overlapping chunks (experimental)

Sentences are converted one at a time on purpose: running the TTS for sentence
N+1 concurrently with RVC for sentence N was measured to add ~55 ms to the time
to first audio (GPU and GIL contention) while barely changing the total.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np

from .audio import float_to_pcm16, prevent_clipping, silence
from .config import F0_METHODS, RVC_MODES, Settings
from .metrics import Stopwatch, SynthesisMetrics, now_ms

_LOGGER = logging.getLogger(__name__)


class SourceLike(Protocol):
    @property
    def sample_rate(self) -> int: ...

    def sentences(self, text: str) -> Iterator[np.ndarray]: ...


class RvcLike(Protocol):
    @property
    def output_sample_rate(self) -> int: ...

    def convert(
        self, audio: np.ndarray, sample_rate: int, pitch: int, f0_method: str, index_rate: float, protect: float
    ) -> tuple[np.ndarray, int]: ...

    def convert_stream(
        self,
        audio: np.ndarray,
        sample_rate: int,
        pitch: int,
        f0_method: str,
        index_rate: float,
        protect: float,
        params: object,
    ) -> Iterator[np.ndarray]: ...


@dataclass(frozen=True)
class SynthesisOptions:
    pitch: int = 0
    f0_method: str = "rmvpe"
    index_rate: float = 0.6
    protect: float = 0.33
    mode: str = "sentence"
    rvc: bool = True

    def validate(self) -> SynthesisOptions:
        if not -24 <= self.pitch <= 24:
            raise ValueError("pitch must be within [-24, 24]")
        if self.f0_method not in F0_METHODS:
            raise ValueError(f"f0_method must be one of {', '.join(F0_METHODS)}")
        if not 0.0 <= self.index_rate <= 1.0:
            raise ValueError("index_rate must be within [0, 1]")
        if not 0.0 <= self.protect <= 0.5:
            raise ValueError("protect must be within [0, 0.5]")
        if self.mode not in RVC_MODES:
            raise ValueError(f"mode must be one of {', '.join(RVC_MODES)}")
        return self

    @classmethod
    def from_settings(cls, settings: Settings) -> SynthesisOptions:
        return cls(
            pitch=settings.rvc_pitch,
            f0_method=settings.rvc_f0_method,
            index_rate=settings.rvc_index_rate,
            protect=settings.rvc_protect,
            mode=settings.rvc_mode,
            rvc=settings.rvc_enabled,
        )


@dataclass
class SynthesisResult:
    pcm: bytes
    sample_rate: int
    metrics: SynthesisMetrics


async def iterate_in_thread[T](gen: Iterator[T], sw: Stopwatch | None = None) -> AsyncIterator[T]:
    """Drive a blocking generator from asyncio, one ``next()`` per worker-thread call.

    Always closes ``gen``; if the consumer goes away mid-step, waits for the
    in-flight step first (closing a running generator raises).
    """
    sentinel = object()

    def step() -> object:
        if sw is None:
            return next(gen, sentinel)
        with sw:
            return next(gen, sentinel)

    pending: asyncio.Future | None = None
    try:
        while True:
            pending = asyncio.ensure_future(asyncio.to_thread(step))
            item = await asyncio.shield(pending)
            pending = None
            if item is sentinel:
                return
            yield item  # type: ignore[misc]
    finally:
        if pending is not None:
            with contextlib.suppress(BaseException):
                await pending
        await asyncio.to_thread(gen.close)


class TtsPipeline:
    """Shared by the HTTP and Wyoming front-ends; one instance per process."""

    def __init__(
        self,
        source: SourceLike,
        rvc: RvcLike | None,
        defaults: SynthesisOptions,
        stream_params: object = None,
        rvc_concurrency: int = 1,
        sentence_silence_ms: int = 0,
    ) -> None:
        self.source = source
        self.rvc = rvc
        self.defaults = defaults.validate()
        self.stream_params = stream_params
        self.sentence_silence_ms = sentence_silence_ms
        self._rvc_slots = asyncio.Semaphore(rvc_concurrency)

    def options(self, **overrides: object) -> SynthesisOptions:
        """Defaults with non-None overrides applied, validated."""
        values = {k: v for k, v in overrides.items() if v is not None}
        opts = replace(self.defaults, **values).validate()
        if opts.rvc and self.rvc is None:
            raise ValueError("RVC is disabled on this server")
        return opts

    def sample_rate(self, options: SynthesisOptions) -> int:
        """Output sample rate for a request with these options."""
        if options.rvc and self.rvc is not None:
            return self.rvc.output_sample_rate
        return self.source.sample_rate

    async def synthesize(self, text: str, options: SynthesisOptions | None = None) -> SynthesisResult:
        """Whole response in memory."""
        options = options or self.defaults
        metrics = SynthesisMetrics()
        chunks = [chunk async for chunk in self.stream(text, options, metrics)]
        return SynthesisResult(pcm=b"".join(chunks), sample_rate=self.sample_rate(options), metrics=metrics)

    async def stream(
        self, text: str, options: SynthesisOptions | None = None, metrics: SynthesisMetrics | None = None
    ) -> AsyncIterator[bytes]:
        """Yield PCM16 mono chunks at ``sample_rate(options)``; fills ``metrics``."""
        options = options or self.defaults
        metrics = metrics if metrics is not None else SynthesisMetrics()
        start = now_ms()
        use_rvc = options.rvc and self.rvc is not None
        whole = use_rvc and options.mode == "whole"
        out_rate = self.sample_rate(options)
        in_rate = self.source.sample_rate
        metrics.text_chars = len(text)
        metrics.mode = options.mode if use_rvc else "tts-only"
        metrics.rvc = use_rvc
        metrics.sample_rate = out_rate
        source_sw, rvc_sw, encode_sw = Stopwatch(), Stopwatch(), Stopwatch()
        samples_out = 0
        gap = silence(in_rate, self.sentence_silence_ms)

        def encode(audio: np.ndarray) -> bytes:
            nonlocal samples_out
            with encode_sw:
                samples_out += audio.size
                return float_to_pcm16(audio)

        def mark_first() -> None:
            if metrics.ttfa_ms is None:
                metrics.ttfa_ms = now_ms() - start

        sentences = self._sentences(text, gap, source_sw, metrics)
        try:
            if not use_rvc:
                async for audio in sentences:
                    mark_first()
                    yield encode(prevent_clipping(audio))
            elif whole:
                parts = [audio async for audio in sentences]
                if parts:
                    converted = await self._rvc_call(
                        lambda: self._convert(np.concatenate(parts), in_rate, options), rvc_sw, metrics
                    )
                    mark_first()
                    yield encode(converted)
            elif options.mode == "sentence":
                async for audio in sentences:
                    converted = await self._rvc_call(
                        lambda a=audio: self._convert(a, in_rate, options), rvc_sw, metrics
                    )
                    mark_first()
                    yield encode(converted)
            else:
                async for audio in sentences:
                    async for piece in self._stream_rvc(audio, in_rate, options, rvc_sw, metrics):
                        mark_first()
                        yield encode(piece)
        finally:
            await sentences.aclose()
            metrics.source_ms = source_sw.elapsed_ms
            metrics.rvc_ms = rvc_sw.elapsed_ms
            metrics.encode_ms = encode_sw.elapsed_ms
            metrics.audio_duration_ms = 1000.0 * samples_out / out_rate if out_rate else 0.0
            metrics.total_ms = now_ms() - start

    # ------------------------------------------------------------------ helpers

    async def _sentences(
        self, text: str, gap: np.ndarray, sw: Stopwatch, metrics: SynthesisMetrics
    ) -> AsyncIterator[np.ndarray]:
        """Source audio per sentence, with inter-sentence silence before all but the first."""
        async with contextlib.aclosing(iterate_in_thread(self.source.sentences(text), sw)) as produced:
            async for audio in produced:
                metrics.sentences += 1
                if metrics.sentences > 1 and gap.size:
                    audio = np.concatenate([gap, audio])
                yield audio

    def _convert(self, audio: np.ndarray, rate: int, options: SynthesisOptions) -> np.ndarray:
        assert self.rvc is not None
        converted, _ = self.rvc.convert(
            audio, rate, options.pitch, options.f0_method, options.index_rate, options.protect
        )
        return converted

    async def _rvc_call(self, fn: Callable[[], np.ndarray], sw: Stopwatch, metrics: SynthesisMetrics) -> np.ndarray:
        queued = now_ms()
        async with self._rvc_slots:
            metrics.queue_ms += now_ms() - queued

            def work() -> np.ndarray:
                with sw:
                    return fn()

            # A cancelled request cannot stop the GPU thread; keep the slot until it ends.
            job = asyncio.ensure_future(asyncio.to_thread(work))
            try:
                return await asyncio.shield(job)
            finally:
                if not job.done():
                    with contextlib.suppress(BaseException):
                        await job

    async def _stream_rvc(
        self, audio: np.ndarray, rate: int, options: SynthesisOptions, sw: Stopwatch, metrics: SynthesisMetrics
    ) -> AsyncIterator[np.ndarray]:
        assert self.rvc is not None
        queued = now_ms()
        async with self._rvc_slots:
            metrics.queue_ms += now_ms() - queued
            gen = self.rvc.convert_stream(
                audio, rate, options.pitch, options.f0_method, options.index_rate, options.protect, self.stream_params
            )
            async with contextlib.aclosing(iterate_in_thread(gen, sw)) as pieces:
                async for piece in pieces:
                    yield piece

    async def warmup(self, text: str) -> float:
        """Exercise every code path once so kernels and allocators are ready."""
        start = now_ms()
        modes = ["whole", "sentence", "stream"] if self.rvc is not None and self.defaults.rvc else [self.defaults.mode]
        for mode in modes:
            result = await self.synthesize(text, replace(self.defaults, mode=mode))
            if not result.pcm:
                raise RuntimeError(f"Warmup produced no audio (mode={mode})")
        return now_ms() - start
