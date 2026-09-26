"""Home Assistant -> Wyoming -> Piper -> RVC -> Wyoming -> Home Assistant.

Run against a live server:
  WYOMING_RVC_HOST=127.0.0.1 WYOMING_RVC_PORT=10200 WYOMING_RVC_HA_OUT=./ha-out pytest tests_ha -v
"""

from __future__ import annotations

import struct

import numpy as np
from homeassistant import config_entries
from homeassistant.components import tts
from homeassistant.components.tts.const import DATA_COMPONENT
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType

ENTITY_ID = "tts.wyoming_rvc"
TEXT = "Hola. Soy Teto, tu asistente. Todos los sistemas están funcionando correctamente."


def pcm_stats(pcm: bytes) -> tuple[int, float]:
    """(sample count, rms) for 16-bit mono PCM."""
    audio = np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype="<i2").astype(np.float32) / 32768
    return audio.size, float(np.sqrt(np.mean(audio**2))) if audio.size else 0.0


def read_wav(data: bytes) -> tuple[int, int, int, bytes]:
    """Parse WAV bytes by hand: streamed WAVs carry 0 or 0xFFFFFFFF sizes in their headers."""
    assert data[:4] == b"RIFF" and data[8:12] == b"WAVE", data[:16]
    pos, fmt = 12, None
    while pos + 8 <= len(data):
        chunk_id, size = data[pos : pos + 4], struct.unpack("<I", data[pos + 4 : pos + 8])[0]
        if chunk_id == b"fmt ":
            _, channels, rate, _, _, bits = struct.unpack("<HHIIHH", data[pos + 8 : pos + 24])
            fmt = (rate, bits // 8, channels)
        elif chunk_id == b"data":
            assert fmt is not None
            return (*fmt, data[pos + 8 :])
        pos += 8 + size + (size & 1)
    raise AssertionError("no data chunk")


async def add_wyoming_entry(hass: HomeAssistant, host: str, port: int) -> None:
    """Settings -> Devices & Services -> Add Integration -> Wyoming Protocol -> host/port."""
    result = await hass.config_entries.flow.async_init("wyoming", context={"source": config_entries.SOURCE_USER})
    assert result["type"] is FlowResultType.FORM and result["step_id"] == "user"
    result = await hass.config_entries.flow.async_configure(result["flow_id"], {"host": host, "port": port})
    assert result["type"] is FlowResultType.CREATE_ENTRY, result
    assert result["title"] == "Wyoming RVC"
    await hass.async_block_till_done()


async def test_config_flow_exposes_tts_entity(hass: HomeAssistant, server) -> None:
    await add_wyoming_entry(hass, *server)
    state = hass.states.get(ENTITY_ID)
    assert state is not None, [s.entity_id for s in hass.states.async_all()]
    entity = hass.data[DATA_COMPONENT].get_entity(ENTITY_ID)
    assert entity.supported_languages == ["es"]
    voices = entity.async_get_supported_voices("es")
    assert voices and voices[0].voice_id == "teto"
    assert entity.async_supports_streaming_input() is True


async def test_entity_get_tts_audio(hass: HomeAssistant, server, save_audio) -> None:
    """WyomingTtsProvider.async_get_tts_audio (non-streaming request)."""
    await add_wyoming_entry(hass, *server)
    entity = hass.data[DATA_COMPONENT].get_entity(ENTITY_ID)
    extension, data = await entity.async_get_tts_audio(TEXT, "es", {tts.ATTR_VOICE: "teto"})
    assert extension == "wav" and data
    save_audio("ha_entity.wav", data)
    rate, width, channels, frames = read_wav(data)
    assert (rate, width, channels) == (32000, 2, 1)
    samples, rms = pcm_stats(frames)
    assert samples / rate > 3.0 and rms > 0.01


async def test_tts_manager_with_ffmpeg_conversion(hass: HomeAssistant, server, save_audio) -> None:
    """Full HA media path: TTS manager + cache + ffmpeg conversion (as used for satellites)."""
    await add_wyoming_entry(hass, *server)
    stream = tts.async_create_stream(
        hass,
        ENTITY_ID,
        "es",
        {
            tts.ATTR_VOICE: "teto",
            tts.ATTR_PREFERRED_FORMAT: "wav",
            tts.ATTR_PREFERRED_SAMPLE_RATE: 16000,
            tts.ATTR_PREFERRED_SAMPLE_CHANNELS: 1,
            tts.ATTR_PREFERRED_SAMPLE_BYTES: 2,
        },
    )
    stream.async_set_message(TEXT)
    data = b"".join([chunk async for chunk in stream.async_stream_result()])
    assert stream.extension == "wav"
    save_audio("ha_manager_16k.wav", data)
    rate, width, channels, frames = read_wav(data)
    assert (rate, width, channels) == (16000, 2, 1)
    samples, rms = pcm_stats(frames)
    assert samples / rate > 3.0 and rms > 0.01


async def test_streaming_text_input(hass: HomeAssistant, server, save_audio) -> None:
    """Assist with an LLM: text arrives in pieces (synthesize-start/chunk/stop)."""
    await add_wyoming_entry(hass, *server)

    async def message_gen():
        for piece in ["Hola, ", "soy Teto. ", "Encendí las luces ", "de la cocina. ", "¿Algo más?"]:
            yield piece

    # Without preferred_* options the TTS manager transcodes to MP3; ask for WAV to inspect it.
    stream = tts.async_create_stream(hass, ENTITY_ID, "es", {tts.ATTR_VOICE: "teto", tts.ATTR_PREFERRED_FORMAT: "wav"})
    stream.async_set_message_stream(message_gen())
    data = b"".join([chunk async for chunk in stream.async_stream_result()])
    save_audio("ha_streaming.wav", data)
    rate, width, channels, frames = read_wav(data)
    assert (width, channels) == (2, 1)
    samples, rms = pcm_stats(frames)
    assert samples / rate > 2.5 and rms > 0.01


async def test_default_playback_format_is_mp3(hass: HomeAssistant, server, save_audio) -> None:
    """What a media player receives by default: HA transcodes the Wyoming WAV to MP3 with ffmpeg."""
    await add_wyoming_entry(hass, *server)
    stream = tts.async_create_stream(hass, ENTITY_ID, "es", {tts.ATTR_VOICE: "teto"})
    stream.async_set_message("Hola, soy Teto.")
    data = b"".join([chunk async for chunk in stream.async_stream_result()])
    save_audio("ha_default.mp3", data)
    assert stream.extension == "mp3"
    assert data[:3] == b"ID3" or data[0] == 0xFF  # ID3 tag or MPEG frame sync
    assert len(data) > 2000
