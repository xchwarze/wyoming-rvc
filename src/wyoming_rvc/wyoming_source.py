"""Audio source that proxies an upstream Wyoming TTS server (``SOURCE=wyoming``).

Blocking sockets on purpose: the pipeline runs sources in worker threads, one
sentence per step, exactly like the in-process Piper source.
"""

from __future__ import annotations

import logging
import socket
from collections.abc import Iterator
from contextlib import closing

import numpy as np
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.error import Error
from wyoming.event import read_event, write_event
from wyoming.info import Describe, Info
from wyoming.tts import Synthesize, SynthesizeVoice

from .audio import resample
from .text import split_sentences

_LOGGER = logging.getLogger(__name__)

PROBE_TEXT = "Hola."


class UpstreamError(RuntimeError):
    """The upstream Wyoming TTS is unreachable, misconfigured, or returned an error."""


class WyomingSource:
    """``sentences(text)`` yields float32 mono audio per sentence at ``sample_rate``."""

    def __init__(
        self, host: str, port: int, voice: str | None = None, speaker: str | None = None, timeout: float = 30.0
    ) -> None:
        self.host, self.port = host, port
        self.voice = SynthesizeVoice(name=voice, speaker=speaker) if voice else None
        self.timeout = timeout
        self.program: str | None = None
        self._rate = 0

    @property
    def sample_rate(self) -> int:
        if not self._rate:
            raise RuntimeError("Upstream source is not loaded")
        return self._rate

    @property
    def device(self) -> str:
        return f"wyoming://{self.host}:{self.port}"

    def load(self) -> None:
        """Check the upstream offers TTS and learn its sample rate with one probe."""
        with self._connect() as (_, stream):
            self._write(stream, Describe().event())
            event = self._read(stream)
            if not Info.is_type(event.type):
                raise UpstreamError(f"Upstream replied {event.type!r} to describe")
            programs = [p for p in Info.from_event(event).tts if p.installed]
            if not programs:
                raise UpstreamError(f"{self.device} has no installed TTS program")
            self.program = programs[0].name
            voices = [v.name for p in programs for v in p.voices if v.installed]
            if self.voice and self.voice.name not in voices:
                raise UpstreamError(f"UPSTREAM_VOICE={self.voice.name!r} not offered by upstream (voices: {voices})")
            audio, rate = self._synthesize(stream, PROBE_TEXT)
        if audio.size == 0:
            raise UpstreamError("Upstream probe synthesis returned no audio")
        self._rate = rate
        _LOGGER.info("Upstream TTS ready: %s program=%s sample_rate=%d", self.device, self.program, rate)

    def sentences(self, text: str) -> Iterator[np.ndarray]:
        parts = split_sentences(text)
        if not parts:
            return
        with self._connect() as (_, stream):
            for part in parts:
                audio, rate = self._synthesize(stream, part)
                yield resample(audio, rate, self.sample_rate) if rate != self.sample_rate else audio

    # ------------------------------------------------------------------ internals

    def _connect(self):
        try:
            sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        except OSError as err:
            raise UpstreamError(f"Cannot connect to upstream TTS {self.device}: {err}") from err
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return _Connection(sock)

    @staticmethod
    def _write(stream, event) -> None:
        try:
            write_event(event, stream)
        except OSError as err:
            raise UpstreamError(f"Upstream connection failed: {err}") from err

    def _read(self, stream):
        try:
            event = read_event(stream)
        except (OSError, ValueError) as err:
            raise UpstreamError(f"Upstream connection failed: {err}") from err
        if event is None:
            raise UpstreamError("Upstream closed the connection")
        if Error.is_type(event.type):
            raise UpstreamError(f"Upstream error: {Error.from_event(event).text}")
        return event

    def _synthesize(self, stream, text: str) -> tuple[np.ndarray, int]:
        self._write(stream, Synthesize(text=text, voice=self.voice).event())
        rate, width, channels = 0, 2, 1
        buffer = bytearray()
        while True:
            event = self._read(stream)
            if AudioStart.is_type(event.type):
                start = AudioStart.from_event(event)
                rate, width, channels = start.rate, start.width, start.channels
            elif AudioChunk.is_type(event.type):
                chunk = AudioChunk.from_event(event)
                rate, width, channels = chunk.rate, chunk.width, chunk.channels
                buffer += chunk.audio
            elif AudioStop.is_type(event.type):
                break
        if width != 2:
            raise UpstreamError(f"Upstream sent {8 * width}-bit audio; only 16-bit PCM is supported")
        audio = np.frombuffer(bytes(buffer), dtype="<i2").astype(np.float32) / 32768.0
        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1)
        return audio, rate


class _Connection:
    """Context manager yielding (socket, buffered binary stream)."""

    def __init__(self, sock: socket.socket) -> None:
        self._sock = sock
        self._stream = sock.makefile("rwb")

    def __enter__(self):
        return self._sock, self._stream

    def __exit__(self, *exc: object) -> None:
        with closing(self._sock):
            try:
                self._stream.close()
            except OSError:
                pass
