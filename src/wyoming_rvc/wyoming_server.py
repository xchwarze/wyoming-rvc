"""Wyoming TTS server (the interface Home Assistant uses).

Event flows handled (matching homeassistant/components/wyoming/tts.py):

  describe                                  -> info
  synthesize                                -> audio-start, audio-chunk*, audio-stop
  synthesize-start, synthesize-chunk*,
  synthesize (compat copy, ignored),
  synthesize-stop                           -> per sentence: audio-start, audio-chunk*, audio-stop;
                                               then synthesize-stopped
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable, Iterable
from functools import partial
from typing import Any

from sentence_stream import SentenceBoundaryDetector
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.error import Error
from wyoming.event import Event
from wyoming.info import Attribution, Describe, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncEventHandler, AsyncTcpServer
from wyoming.tts import Synthesize, SynthesizeChunk, SynthesizeStart, SynthesizeStop, SynthesizeStopped

from . import PROJECT_URL, __version__
from .audio import iter_pcm_chunks
from .config import Settings
from .http_server import ServiceState
from .metrics import SynthesisMetrics

_LOGGER = logging.getLogger(__name__)

WIDTH, CHANNELS = 2, 1


def build_info(settings: Settings, voices: Iterable[Any]) -> Info:
    """Wyoming ``info``: one TTS program listing every installed voice.

    ``voices`` are objects with ``id``, ``name`` and ``language`` (``VoiceConfig``).
    """
    attribution = Attribution(name="wyoming-rvc", url=PROJECT_URL)
    tts_voices = [
        TtsVoice(
            name=voice.id,
            description=voice.name,
            attribution=attribution,
            installed=True,
            version=None,
            languages=[voice.language],
        )
        for voice in voices
    ]
    program = TtsProgram(
        name=settings.program_name,
        description="Piper TTS + RVC voice conversion",
        attribution=attribution,
        installed=True,
        version=__version__,
        voices=tts_voices,
        supports_synthesize_streaming=settings.wyoming_streaming,
    )
    return Info(tts=[program])


class RvcEventHandler(AsyncEventHandler):
    """One instance per client connection."""

    def __init__(
        self,
        info_factory: Callable[[], Info],
        state: ServiceState,
        settings: Settings,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        super().__init__(reader, writer)
        self._info_factory = info_factory
        self._state = state
        self._settings = settings
        self._streaming = False
        self._stream_voice: str | None = None
        self._sbd = SentenceBoundaryDetector()

    async def handle_event(self, event: Event) -> bool:
        try:
            if Describe.is_type(event.type):
                await self.write_event(self._info_factory().event())
                return True

            if Synthesize.is_type(event.type):
                if self._streaming:
                    return True  # compatibility copy of the streamed text
                synthesize = Synthesize.from_event(event)
                await self._speak(synthesize.text, "wyoming", _voice_id(synthesize.voice))
                return True

            if not self._settings.wyoming_streaming:
                return True

            if SynthesizeStart.is_type(event.type):
                start = SynthesizeStart.from_event(event)
                self._stream_voice = _voice_id(start.voice)
                self._pipeline().options(voice=self._stream_voice)  # unknown voice -> error now
                self._streaming = True
                self._sbd = SentenceBoundaryDetector()
                return True

            if SynthesizeChunk.is_type(event.type) and self._streaming:
                for sentence in self._sbd.add_chunk(SynthesizeChunk.from_event(event).text):
                    await self._speak(sentence, "wyoming-stream", self._stream_voice)
                return True

            if SynthesizeStop.is_type(event.type) and self._streaming:
                remaining = self._sbd.finish()
                if remaining.strip():
                    await self._speak(remaining, "wyoming-stream", self._stream_voice)
                await self.write_event(SynthesizeStopped().event())
                self._streaming = False
                return True
        except (ConnectionError, OSError) as err:
            _LOGGER.debug("Wyoming client went away: %s", err)
            return False
        except Exception as err:
            _LOGGER.exception("Wyoming synthesis failed")
            with contextlib.suppress(Exception):  # the client may already be gone
                await self.write_event(Error(text=str(err), code=err.__class__.__name__).event())
            return False

        return True

    def _pipeline(self):
        pipeline = self._state.pipeline
        if not self._state.ready or pipeline is None:
            raise RuntimeError(f"Service not ready (stage={self._state.stage})")
        return pipeline

    async def _speak(self, raw_text: str, source: str, voice: str | None) -> None:
        """Synthesize one utterance and send audio-start / chunks / audio-stop."""
        pipeline = self._pipeline()
        text = " ".join(raw_text.split())
        options = pipeline.options(voice=voice)  # UnknownVoiceError -> Wyoming error event
        rate = pipeline.sample_rate(options)

        await self.write_event(AudioStart(rate=rate, width=WIDTH, channels=CHANNELS).event())
        if text:
            metrics = SynthesisMetrics()
            spc = self._settings.wyoming_samples_per_chunk
            # aclosing: a failed write must release the source and RVC slot right away, not at GC.
            async with contextlib.aclosing(pipeline.stream(text, options, metrics)) as pcm_stream:
                async for pcm in pcm_stream:
                    for chunk in iter_pcm_chunks(pcm, spc, WIDTH, CHANNELS):
                        event = AudioChunk(rate=rate, width=WIDTH, channels=CHANNELS, audio=chunk).event()
                        await self.write_event(event)
            metrics.log(_LOGGER, source)
        await self.write_event(AudioStop().event())


class WyomingService:
    """Owns the TCP server (and optional zeroconf registration)."""

    def __init__(self, settings: Settings, state: ServiceState, info_factory: Callable[[], Info]) -> None:
        self._settings = settings
        self._server = AsyncTcpServer(settings.wyoming_host, settings.wyoming_port)
        self._factory = partial(RvcEventHandler, info_factory, state, settings)
        self._zeroconf = None

    async def start(self) -> None:
        await self._server.start(self._factory)
        _LOGGER.info("Wyoming listening on %s:%s", self._settings.wyoming_host, self._settings.wyoming_port)
        if self._settings.wyoming_zeroconf:
            from wyoming.zeroconf import HomeAssistantZeroconf

            self._zeroconf = HomeAssistantZeroconf(
                port=self._settings.wyoming_port, name=self._settings.wyoming_zeroconf_name
            )
            await self._zeroconf.register_server()
            _LOGGER.info("Zeroconf discovery registered (_wyoming._tcp.local.)")

    async def stop(self) -> None:
        await self._server.stop()
        if self._zeroconf is not None:
            # wyoming.zeroconf has no public close(); tolerate the private attribute changing.
            with contextlib.suppress(AttributeError):
                await self._zeroconf._aiozc.async_close()


def _voice_id(voice: Any) -> str | None:
    """Requested voice name; ``None`` means the default voice (language-only requests too)."""
    return voice.name if voice is not None and voice.name else None
