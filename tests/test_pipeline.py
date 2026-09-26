import asyncio
from dataclasses import replace

import numpy as np
import pytest

from tests.conftest import FakePiper, FakeRvc
from wyoming_rvc.metrics import Stopwatch
from wyoming_rvc.pipeline import SynthesisOptions, TtsPipeline, iterate_in_thread
from wyoming_rvc.text import split_sentences


@pytest.mark.parametrize("mode", ["whole", "sentence"])
async def test_modes_produce_rvc_rate_audio(pipeline, rvc, mode):
    result = await pipeline.synthesize("Hola. Soy Teto.", replace(pipeline.defaults, mode=mode))
    assert result.sample_rate == 32000
    samples = len(result.pcm) // 2
    assert abs(samples - 2 * 0.25 * 32000) < 50
    m = result.metrics
    assert m.sentences == 2
    assert m.mode == mode and m.rvc
    assert m.ttfa_ms is not None and m.total_ms >= m.ttfa_ms
    assert m.rvc_ms > 0 and m.source_ms >= 0
    assert np.isclose(m.audio_duration_ms, 1000 * samples / 32000)
    assert m.rtf == pytest.approx(m.total_ms / m.audio_duration_ms)
    assert rvc.calls == (1 if mode == "whole" else 2)


async def test_disable_rvc_uses_piper_rate(pipeline, rvc):
    opts = pipeline.options(rvc=False)
    result = await pipeline.synthesize("Hola.", opts)
    assert result.sample_rate == 22050
    assert rvc.calls == 0
    assert result.metrics.mode == "tts-only"


async def test_sentence_silence_inserted(piper, rvc):
    p = TtsPipeline(piper, rvc, SynthesisOptions(mode="sentence"), sentence_silence_ms=100)
    result = await p.synthesize("A. B.")
    expected = (2 * 0.25 + 0.1) * 32000
    assert abs(len(result.pcm) // 2 - expected) < 50


async def test_rvc_concurrency_is_serialized(piper):
    rvc = FakeRvc(delay=0.05)
    p = TtsPipeline(piper, rvc, SynthesisOptions(mode="sentence"), rvc_concurrency=1)
    results = await asyncio.gather(*(p.synthesize("Uno. Dos.") for _ in range(4)))
    assert rvc.max_active == 1
    assert rvc.calls == 8
    assert any(r.metrics.queue_ms > 0 for r in results)


async def test_rvc_concurrency_two(piper):
    rvc = FakeRvc(delay=0.1)
    p = TtsPipeline(piper, rvc, SynthesisOptions(), rvc_concurrency=2)
    await asyncio.gather(*(p.synthesize("Uno.") for _ in range(4)))
    assert rvc.max_active == 2


async def test_empty_text_yields_nothing(pipeline):
    result = await pipeline.synthesize("   ")
    assert result.pcm == b""
    assert result.metrics.sentences == 0


async def test_rvc_error_propagates(piper):
    p = TtsPipeline(piper, FakeRvc(fail=True), SynthesisOptions())
    with pytest.raises(RuntimeError, match="boom"):
        await p.synthesize("Hola.")


async def test_semaphore_released_after_error(piper):
    rvc = FakeRvc(fail=True)
    p = TtsPipeline(piper, rvc, SynthesisOptions(), rvc_concurrency=1)
    for _ in range(2):
        with pytest.raises(RuntimeError):
            await p.synthesize("Hola.")
    rvc.fail = False
    assert (await p.synthesize("Hola.")).pcm


async def test_abandoned_stream_releases_slot(piper):
    rvc = FakeRvc()
    p = TtsPipeline(piper, rvc, SynthesisOptions(mode="sentence"), rvc_concurrency=1)
    agen = p.stream("Uno. Dos. Tres.")
    await agen.__anext__()
    await agen.aclose()
    result = await asyncio.wait_for(p.synthesize("Hola."), timeout=2)
    assert result.pcm


def test_options_validation(pipeline):
    assert pipeline.options(pitch=3).pitch == 3
    assert pipeline.options(pitch=None).pitch == 0
    for bad in ({"pitch": 30}, {"index_rate": 2.0}, {"protect": -1.0}, {"mode": "x"}, {"f0_method": "crepe"}):
        with pytest.raises(ValueError):
            pipeline.options(**bad)


def test_rvc_required_when_disabled_engine():
    p = TtsPipeline(FakePiper(), None, SynthesisOptions(rvc=False))
    with pytest.raises(ValueError):
        p.options(rvc=True)


async def test_warmup(pipeline, rvc):
    ms = await pipeline.warmup("Sistema iniciado.")
    assert ms > 0 and rvc.calls == 2  # whole + sentence


async def test_sentences_are_converted_in_order(piper, rvc):
    p = TtsPipeline(piper, rvc, SynthesisOptions(mode="sentence"))
    chunks = [c async for c in p.stream("Uno. Dos. Tres.")]
    assert len(chunks) == 3 and rvc.max_active == 1


async def test_abandoned_stream_closes_source(piper):
    p = TtsPipeline(piper, FakeRvc(), SynthesisOptions(mode="sentence"))
    agen = p.stream("Uno. Dos. Tres. Cuatro.")
    await agen.__anext__()
    await agen.aclose()
    assert piper.closed == 1


async def test_iterate_in_thread_cancel_mid_step():
    import time

    closed = []

    def slow():
        try:
            yield 1
            time.sleep(0.2)
            yield 2
        finally:
            closed.append(True)

    async def consume():
        async for _ in iterate_in_thread(slow(), Stopwatch()):
            pass

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed == [True]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Uno. Dos.", ["Uno.", "Dos."]),
        ("Hola, soy Teto. ¿Cómo estás? Bien.", ["Hola, soy Teto.", "¿Cómo estás?", "Bien."]),
        ("El Sr. Pérez llegó. Luego se fue.", ["El Sr. Pérez llegó.", "Luego se fue."]),
        ("Vale 3.5 euros. J. R. R. Tolkien escribió.", ["Vale 3.5 euros.", "J. R. R. Tolkien escribió."]),
        ('Dijo "hola." Después, nada.', ['Dijo "hola."', "Después, nada."]),
        ("sin punto final", ["sin punto final"]),
        ("  ", []),
    ],
)
def test_split_sentences(text, expected):
    assert split_sentences(text) == expected


async def test_cancelled_request_keeps_rvc_slot_until_conversion_ends(piper):
    """A client disconnect must not let the next request run a second conversion on the GPU."""
    rvc = FakeRvc(delay=0.2)
    p = TtsPipeline(piper, rvc, SynthesisOptions(mode="sentence"), rvc_concurrency=1)
    first = asyncio.create_task(p.synthesize("Uno."))
    await asyncio.sleep(0.05)  # first conversion is running in its worker thread
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    await p.synthesize("Dos.")
    assert rvc.max_active == 1


async def test_run_in_daemon_thread_result_error_and_daemon():
    import threading

    from wyoming_rvc.main import run_in_daemon_thread

    seen = {}

    def work():
        seen["daemon"] = threading.current_thread().daemon
        return 42

    assert await run_in_daemon_thread(work) == 42 and seen["daemon"] is True

    def fail():
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        await run_in_daemon_thread(fail)
