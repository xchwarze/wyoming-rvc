"""WyomingSource against a real Wyoming server (this project's, running TTS-only)."""

import asyncio
import socket
from types import SimpleNamespace

import numpy as np
import pytest

from tests.conftest import FakePiper
from wyoming_rvc.config import Settings
from wyoming_rvc.http_server import ServiceState
from wyoming_rvc.main import Service
from wyoming_rvc.pipeline import SynthesisOptions, TtsPipeline
from wyoming_rvc.sources import SourceRegistry
from wyoming_rvc.voices import SourceSpec
from wyoming_rvc.wyoming_server import WyomingService, build_info
from wyoming_rvc.wyoming_source import UpstreamError, WyomingSource


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def upstream_piper():
    return FakePiper()


@pytest.fixture
async def upstream(upstream_piper):
    """A Wyoming TTS at 22050 Hz (fake Piper, no RVC) playing the upstream role."""
    port = free_port()
    settings = Settings.from_env(
        {"WYOMING_HOST": "127.0.0.1", "WYOMING_PORT": str(port), "VOICE_NAME": "daniela", "RVC_ENABLED": "false"}
    )
    pipeline = TtsPipeline(upstream_piper, None, SynthesisOptions(rvc=False))
    state = ServiceState(pipeline=pipeline, ready=True, stage="ready")
    info = build_info(settings, [SimpleNamespace(id="daniela", name="Fake Piper", language="es")])
    service = WyomingService(settings, state, lambda: info)
    await service.start()
    yield port
    await service.stop()


async def test_load_learns_program_and_rate(upstream):
    source = WyomingSource("127.0.0.1", upstream)
    await asyncio.to_thread(source.load)
    assert source.sample_rate == 22050
    assert source.program == "Wyoming RVC"


async def test_load_uses_default_probe_text(upstream, upstream_piper):
    source = WyomingSource("127.0.0.1", upstream)
    await asyncio.to_thread(source.load)
    assert upstream_piper.calls == ["La casa es del tamaño del mundo; mejor dicho, es el mundo."]


async def test_load_uses_configured_probe_text(upstream, upstream_piper):
    source = WyomingSource("127.0.0.1", upstream, probe_text="Comprobación.")
    await asyncio.to_thread(source.load)
    assert upstream_piper.calls == ["Comprobación."]


async def test_service_uses_configured_probe_text(upstream, upstream_piper):
    settings = Settings.from_env(
        {"SOURCE": "wyoming", "WYOMING_UPSTREAM": f"tcp://127.0.0.1:{upstream}", "PROBE_TEXT": "Comprobación."}
    )
    service = Service(settings)
    await asyncio.to_thread(service.load_source)
    assert service.source.sample_rate == 22050
    assert upstream_piper.calls == ["Comprobación."]


@pytest.mark.parametrize("own_upstream", [False, True])
async def test_per_voice_upstream_uses_configured_probe_text(upstream, upstream_piper, own_upstream):
    address = f"tcp://127.0.0.1:{upstream}"
    settings = Settings.from_env({"WYOMING_UPSTREAM": address, "PROBE_TEXT": "Comprobación."})
    registry = SourceRegistry(settings, object())
    spec = SourceSpec(kind="wyoming", upstream=address if own_upstream else None)
    await asyncio.to_thread(registry.prepare, spec)
    source = await asyncio.to_thread(registry.get, spec)
    assert source.sample_rate == 22050
    assert upstream_piper.calls == ["Comprobación."]


async def test_sentences_one_audio_per_sentence(upstream):
    source = WyomingSource("127.0.0.1", upstream, voice="daniela")
    await asyncio.to_thread(source.load)
    parts = await asyncio.to_thread(lambda: list(source.sentences("Hola, soy Teto. ¿Cómo estás? Bien.")))
    assert len(parts) == 3
    assert all(p.dtype == np.float32 and abs(p.size - 0.25 * 22050) < 5 for p in parts)
    assert all(np.max(np.abs(p)) > 0.1 for p in parts)


async def test_unknown_upstream_voice_fails_fast(upstream):
    source = WyomingSource("127.0.0.1", upstream, voice="nope")
    with pytest.raises(UpstreamError, match="not offered"):
        await asyncio.to_thread(source.load)


async def test_unreachable_upstream_fails_fast():
    source = WyomingSource("127.0.0.1", free_port(), timeout=2)
    with pytest.raises(UpstreamError, match="Cannot connect"):
        await asyncio.to_thread(source.load)


async def test_pipeline_converts_upstream_audio(upstream):
    """End to end: upstream Wyoming TTS -> WyomingSource -> fake RVC at 32 kHz."""
    from tests.conftest import FakeRvc

    source = WyomingSource("127.0.0.1", upstream)
    await asyncio.to_thread(source.load)
    p = TtsPipeline(source, FakeRvc(), SynthesisOptions(mode="sentence"))
    result = await p.synthesize("Uno. Dos.")
    assert result.sample_rate == 32000 and result.metrics.sentences == 2
    assert abs(len(result.pcm) // 2 - 0.5 * 32000) < 100


async def test_upstream_write_failure_is_an_upstream_error(upstream, monkeypatch):
    """A dead upstream must surface as UpstreamError (sent to HA), not as a client disconnect."""
    from wyoming_rvc import wyoming_source

    source = WyomingSource("127.0.0.1", upstream)
    await asyncio.to_thread(source.load)

    def broken(*_args, **_kwargs):
        raise BrokenPipeError("upstream went away")

    monkeypatch.setattr(wyoming_source, "write_event", broken)
    with pytest.raises(UpstreamError, match="upstream went away"):
        await asyncio.to_thread(lambda: list(source.sentences("Hola.")))
