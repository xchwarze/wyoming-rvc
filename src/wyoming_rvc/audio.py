"""Audio helpers: dtype conversion, resampling, clipping protection, WAV."""

from __future__ import annotations

import io
import wave
from collections.abc import Iterator

import numpy as np

INT16_MAX = 32767
DEFAULT_CEILING = 0.99


def validate_sample_rate(sample_rate: int) -> int:
    """Return ``sample_rate`` or raise ``ValueError`` if it is not plausible."""
    if not isinstance(sample_rate, (int, np.integer)) or not (8000 <= int(sample_rate) <= 192000):
        raise ValueError(f"Invalid sample rate: {sample_rate!r}")
    return int(sample_rate)


def to_float32(audio: np.ndarray) -> np.ndarray:
    """Convert int16 or float audio to mono float32 in [-1, 1] (no rescaling of floats)."""
    audio = np.asarray(audio)
    if audio.ndim == 2:
        audio = audio.mean(axis=1) if audio.shape[1] <= 8 else audio.mean(axis=0)
    if audio.ndim != 1:
        raise ValueError(f"Expected mono audio, got shape {audio.shape}")
    if audio.dtype == np.int16:
        return audio.astype(np.float32) / 32768.0
    return audio.astype(np.float32, copy=False)


def prevent_clipping(audio: np.ndarray, ceiling: float = DEFAULT_CEILING) -> np.ndarray:
    """Scale down only if the peak exceeds ``ceiling``; never amplify."""
    if audio.size == 0:
        return audio
    peak = float(np.max(np.abs(audio)))
    if not np.isfinite(peak):
        audio = np.nan_to_num(audio, nan=0.0, posinf=ceiling, neginf=-ceiling)
        peak = float(np.max(np.abs(audio)))
    if peak > ceiling:
        audio = audio * (ceiling / peak)
    return audio


def float_to_pcm16(audio: np.ndarray) -> bytes:
    """Float32 [-1, 1] to signed 16-bit little-endian PCM bytes (hard clip as last resort)."""
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * INT16_MAX).astype("<i2").tobytes()


def pcm16_to_float(pcm: bytes) -> np.ndarray:
    """Signed 16-bit little-endian PCM bytes to float32."""
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """High-quality resampling with soxr; no-op when the rates match."""
    validate_sample_rate(src_rate)
    validate_sample_rate(dst_rate)
    if src_rate == dst_rate or audio.size == 0:
        return audio.astype(np.float32, copy=False)
    import soxr

    return soxr.resample(audio.astype(np.float32, copy=False), src_rate, dst_rate, quality="HQ").astype(
        np.float32, copy=False
    )


def silence(sample_rate: int, milliseconds: int) -> np.ndarray:
    """Zero-valued float32 audio."""
    return np.zeros(int(sample_rate * milliseconds / 1000), dtype=np.float32)


def wav_bytes(pcm: bytes, sample_rate: int, width: int = 2, channels: int = 1) -> bytes:
    """Wrap raw PCM in a WAV container."""
    validate_sample_rate(sample_rate)
    with io.BytesIO() as buffer:
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(channels)
            wav.setsampwidth(width)
            wav.setframerate(sample_rate)
            wav.writeframes(pcm)
        return buffer.getvalue()


def streaming_wav_header(sample_rate: int, width: int = 2, channels: int = 1) -> bytes:
    """WAV header with zero frames, for streamed responses of unknown length."""
    return wav_bytes(b"", sample_rate, width, channels)


def iter_pcm_chunks(pcm: bytes, samples_per_chunk: int, width: int = 2, channels: int = 1) -> Iterator[bytes]:
    """Split PCM bytes into fixed-size chunks (last one may be shorter)."""
    step = samples_per_chunk * width * channels
    for offset in range(0, len(pcm), step):
        yield pcm[offset : offset + step]
