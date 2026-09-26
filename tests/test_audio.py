import io
import wave

import numpy as np
import pytest

from wyoming_rvc import audio


def test_pcm16_roundtrip():
    x = np.array([0.0, 0.5, -0.5, 0.999], dtype=np.float32)
    pcm = audio.float_to_pcm16(x)
    assert len(pcm) == 8
    back = audio.pcm16_to_float(pcm)
    assert np.allclose(back, x, atol=1 / 32767)


def test_pcm16_hard_clips_out_of_range():
    pcm = audio.float_to_pcm16(np.array([2.0, -2.0], dtype=np.float32))
    assert np.frombuffer(pcm, dtype="<i2").tolist() == [32767, -32767]


def test_to_float32_from_int16():
    x = np.array([0, 16384, -32768], dtype=np.int16)
    assert np.allclose(audio.to_float32(x), [0.0, 0.5, -1.0])


def test_prevent_clipping_scales_only_when_needed():
    quiet = np.array([0.1, -0.2], dtype=np.float32)
    assert audio.prevent_clipping(quiet) is quiet
    loud = np.array([0.5, -2.0], dtype=np.float32)
    out = audio.prevent_clipping(loud, ceiling=0.99)
    assert np.isclose(np.max(np.abs(out)), 0.99)
    assert np.isclose(out[0] / out[1], loud[0] / loud[1])


def test_prevent_clipping_handles_nan():
    out = audio.prevent_clipping(np.array([np.nan, 0.5], dtype=np.float32))
    assert np.all(np.isfinite(out))


@pytest.mark.parametrize("src,dst", [(22050, 16000), (16000, 32000), (24000, 48000)])
def test_resample_length(src, dst):
    x = np.random.default_rng(0).uniform(-0.5, 0.5, src).astype(np.float32)
    y = audio.resample(x, src, dst)
    assert y.dtype == np.float32
    assert abs(y.size - dst) <= 2


def test_resample_noop_same_rate():
    x = np.ones(100, dtype=np.float32)
    assert audio.resample(x, 16000, 16000) is x


def test_resample_rejects_bad_rate():
    with pytest.raises(ValueError):
        audio.resample(np.zeros(10, dtype=np.float32), 0, 16000)


def test_wav_bytes_header():
    pcm = audio.float_to_pcm16(np.zeros(320, dtype=np.float32))
    data = audio.wav_bytes(pcm, 32000)
    with wave.open(io.BytesIO(data)) as wav:
        assert wav.getframerate() == 32000
        assert wav.getsampwidth() == 2
        assert wav.getnchannels() == 1
        assert wav.getnframes() == 320


def test_streaming_header_has_zero_frames():
    with wave.open(io.BytesIO(audio.streaming_wav_header(22050))) as wav:
        assert wav.getnframes() == 0
        assert wav.getframerate() == 22050


def test_iter_pcm_chunks():
    pcm = bytes(2 * 1000)
    chunks = list(audio.iter_pcm_chunks(pcm, 256))
    assert [len(c) for c in chunks] == [512, 512, 512, 464]
    assert b"".join(chunks) == pcm


def test_crossfade_is_continuous():
    tail = np.ones(100, dtype=np.float32)
    head = np.zeros(100, dtype=np.float32)
    mixed = audio.crossfade(tail, head)
    assert np.isclose(mixed[0], 1.0) and np.isclose(mixed[-1], 0.0, atol=1e-6)
    assert np.all(np.diff(mixed) <= 1e-6)


def test_crossfade_length_mismatch():
    with pytest.raises(ValueError):
        audio.crossfade(np.zeros(3), np.zeros(4))


@pytest.mark.parametrize("n", [1, 159, 16000, 16001, 16000 * 3 + 500, 16000 * 3 + 9000, 123457])
def test_chunk_bounds_cover_input_and_last_exceeds_overlap(n):
    from wyoming_rvc.rvc_engine import chunk_bounds

    chunk, overlap = 16000, 960
    bounds = chunk_bounds(n, chunk, overlap)
    assert bounds[0][0] == 0 and bounds[-1][1] == n
    assert all(a[1] == b[0] for a, b in zip(bounds, bounds[1:], strict=False))
    if len(bounds) > 1:
        assert bounds[-1][1] - bounds[-1][0] >= 2 * overlap
