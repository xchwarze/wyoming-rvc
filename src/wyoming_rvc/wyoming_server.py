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
from collections.abc import Callable
from functools import partial

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


def build_info(settings: Settings, voice_description: str) -> Info:
    """Wyoming ``info`` advertising one installed TTS program with one voice."""
    attribution = Attribution(name="wyoming-rvc", url=PROJECT_URL)
    voice = TtsVoice(
        name=settings.voice_name,
        description=voice_description,
        attribution=attribution,
        installed=True,
        version=None,
        languages=[settings.voice_language],
    )
    program = TtsProgram(
        name=settings.program_name,
        description="Piper TTS + RVC voice conversion",
        attribution=attribution,
        installed=True,
        version=__version__,
        voices=[voice],
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
        self._sbd = SentenceBoundaryDetector()
        self._stream_metrics: list[SynthesisMetrics] = []

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self._info_factory().event())
            return True

        try:
            if Synthesize.is_type(event.type):
                if self._streaming:
                    return True  # compatibility copy of the streamed text
                synthesize = Synthesize.from_event(event)
                self._check_voice(synthesize.voice)
                await self._speak(synthesize.text, source="wyoming")
                return True

            if not self._settings.wyoming_streaming:
                return True

            if SynthesizeStart.is_type(event.type):
                start = SynthesizeStart.from_event(event)
                self._check_voice(start.voice)
                self._streaming = True
                self._sbd = SentenceBoundaryDetector()
                self._stream_metrics = []
                return True

            if SynthesizeChunk.is_type(event.type) and self._streaming:
                for sentence in self._sbd.add_chunk(SynthesizeChunk.from_event(event).text):
                    await self._speak(sentence, source="wyoming-stream")
                return True

            if SynthesizeStop.is_type(event.type) and self._streaming:
                remaining = self._sbd.finish()
                if remaining.strip():
                    await self._speak(remaining, source="wyoming-stream")
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

    def _check_voice(self, voice) -> None:
        if voice is not None and voice.name and voice.name != self._settings.voice_name:
            _LOGGER.warning("Unknown voice %r requested; using %r", voice.name, self._settings.voice_name)

    async def _speak(self, raw_text: str, source: str) -> None:
        """Synthesize one utterance and send audio-start / chunks / audio-stop."""
        pipeline = self._state.pipeline
        if not self._state.ready or pipeline is None:
            raise RuntimeError(f"Service not ready (stage={self._state.stage})")
        text = " ".join(raw_text.split())
        options = pipeline.defaults
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
            await self._zeroconf._aiozc.async_close()  # no public close() in wyoming.zeroconf
