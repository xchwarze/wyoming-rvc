"""Drive the server with the same event sequences Home Assistant's wyoming/tts.py sends."""

import asyncio
import socket

import pytest
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.error import Error
from wyoming.info import Describe, Info
from wyoming.tts import (
    Synthesize,
    SynthesizeChunk,
    SynthesizeStart,
    SynthesizeStop,
    SynthesizeStopped,
    SynthesizeVoice,
)

from tests.conftest import FakePiper, FakeRvc
from wyoming_rvc.config import Settings
from wyoming_rvc.http_server import ServiceState
from wyoming_rvc.pipeline import SynthesisOptions, TtsPipeline
from wyoming_rvc.wyoming_server import WyomingService, build_info


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
async def server(request):
    params = getattr(request, "param", {})
    port = free_port()
    settings = Settings.from_env({"WYOMING_HOST": "127.0.0.1", "WYOMING_PORT": str(port), **params.get("env", {})})
    rvc = params.get("rvc") or FakeRvc()
    state = ServiceState(pipeline=TtsPipeline(FakePiper(), rvc, SynthesisOptions()), ready=True, stage="ready")
    info = build_info(settings, "Teto (test)")
    service = WyomingService(settings, state, lambda: info)
    await service.start()
    yield port
    await service.stop()


async def test_describe(server):
    async with AsyncTcpClient("127.0.0.1", server) as client:
        await client.write_event(Describe().event())
        event = await client.read_event()
    info = Info.from_event(event)
    assert len(info.tts) == 1
    program = info.tts[0]
    assert program.name == "Wyoming RVC" and program.installed
    assert program.supports_synthesize_streaming is True
    voice = program.voices[0]
    assert voice.name == "teto" and voice.languages == ["es"] and voice.installed


async def test_synthesize_like_home_assistant(server):
    """homeassistant.components.wyoming.tts.WyomingTtsProvider.async_get_tts_audio."""
    async with AsyncTcpClient("127.0.0.1", server) as client:
        await client.write_event(Synthesize(text="Hola. Soy Teto.", voice=SynthesizeVoice(name="teto")).event())
        start, chunks = None, []
        while True:
            event = await asyncio.wait_for(client.read_event(), timeout=5)
            assert event is not None
            assert not Error.is_type(event.type)
            if AudioStart.is_type(event.type):
                start = AudioStart.from_event(event)
            elif AudioChunk.is_type(event.type):
                chunks.append(AudioChunk.from_event(event))
            elif AudioStop.is_type(event.type):
                break
    assert start is not None and (start.rate, start.width, start.channels) == (32000, 2, 1)
    assert chunks and all(c.rate == 32000 for c in chunks)
    assert sum(len(c.audio) for c in chunks) // 2 == pytest.approx(2 * 0.25 * 32000, abs=50)


async def test_streaming_like_home_assistant(server):
    """WyomingTtsProvider.async_stream_tts_audio: start, chunks, full synthesize, stop."""
    async with AsyncTcpClient("127.0.0.1", server) as client:
        voice = SynthesizeVoice(name="teto")
        await client.write_event(SynthesizeStart(voice=voice).event())
        text = ""
        for piece in ["Hola, ", "soy Teto. ", "¿Cómo ", "estás?"]:
            text += piece
            await client.write_event(SynthesizeChunk(text=piece).event())
        await client.write_event(Synthesize(text=text, voice=voice).event())
        await client.write_event(SynthesizeStop().event())

        starts, audio_bytes = 0, 0
        while True:
            event = await asyncio.wait_for(client.read_event(), timeout=5)
            assert event is not None and not Error.is_type(event.type)
            if AudioStart.is_type(event.type):
                starts += 1
            elif AudioChunk.is_type(event.type):
                audio_bytes += len(AudioChunk.from_event(event).audio)
            elif SynthesizeStopped.is_type(event.type):
                break
    assert starts >= 1
    # the compatibility Synthesize event must not produce a second copy of the audio
    assert audio_bytes // 2 == pytest.approx(2 * 0.25 * 32000, abs=100)


@pytest.mark.parametrize("server", [{"rvc": FakeRvc(fail=True)}], indirect=True)
async def test_error_event_on_failure(server):
    async with AsyncTcpClient("127.0.0.1", server) as client:
        await client.write_event(Synthesize(text="Hola.").event())
        while True:
            event = await asyncio.wait_for(client.read_event(), timeout=5)
            assert event is not None
            if Error.is_type(event.type):
                assert "boom" in Error.from_event(event).text
                break


@pytest.mark.parametrize("server", [{"env": {"WYOMING_STREAMING": "false"}}], indirect=True)
async def test_streaming_disabled_advertised(server):
    async with AsyncTcpClient("127.0.0.1", server) as client:
        await client.write_event(Describe().event())
        info = Info.from_event(await client.read_event())
    assert info.tts[0].supports_synthesize_streaming is False
