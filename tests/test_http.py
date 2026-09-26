import io
import wave

import pytest
from fastapi.testclient import TestClient

from tests.conftest import FakePiper, FakeRvc
from wyoming_rvc.http_server import ServiceState, create_app
from wyoming_rvc.pipeline import SynthesisOptions, TtsPipeline


def make_client(ready: bool = True, rvc: FakeRvc | None = None) -> TestClient:
    pipeline = TtsPipeline(FakePiper(), rvc or FakeRvc(), SynthesisOptions(mode="whole"))
    state = ServiceState(pipeline=pipeline, ready=ready, stage="ready" if ready else "loading-rvc", max_text_chars=100)
    state.info = lambda: {"version": "test"}
    return TestClient(create_app(state))


def test_health_and_ready():
    client = make_client(ready=False)
    assert client.get("/healthz").status_code == 200
    r = client.get("/readyz")
    assert r.status_code == 503 and r.json()["stage"] == "loading-rvc"
    assert make_client().get("/readyz").status_code == 200


def test_info():
    body = make_client().get("/info").json()
    assert body["ready"] is True and body["version"] == "test"


def test_tts_returns_wav_and_metrics():
    r = make_client().post("/v1/tts", json={"text": "Hola, soy Teto."})
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/wav"
    with wave.open(io.BytesIO(r.content)) as wav:
        assert wav.getframerate() == 32000 and wav.getnchannels() == 1 and wav.getsampwidth() == 2
        assert wav.getnframes() > 0
    for key in ("X-TTS-Source-Ms", "X-TTS-Rvc-Ms", "X-TTS-Encode-Ms", "X-TTS-Total-Ms", "X-TTS-Rtf"):
        assert key in r.headers


def test_tts_disable_rvc():
    r = make_client().post("/v1/tts", json={"text": "Hola.", "disable_rvc": True})
    with wave.open(io.BytesIO(r.content)) as wav:
        assert wav.getframerate() == 22050
    assert r.headers["X-TTS-Mode"] == "tts-only"


def test_tts_stream():
    r = make_client().post("/v1/tts/stream", json={"text": "Hola. Chau.", "mode": "sentence"})
    assert r.status_code == 200
    assert r.content[:4] == b"RIFF" and len(r.content) > 44


@pytest.mark.parametrize(
    "payload,status",
    [
        ({"text": ""}, 422),
        ({"text": "   "}, 422),
        ({"text": "x" * 101}, 413),
        ({"text": "hola", "pitch": 99}, 422),
        ({"text": "hola", "protect": 0.9}, 422),
        ({"text": "hola", "f0_method": "crepe"}, 422),
        ({"text": "hola", "mode": "turbo"}, 422),
        ({"text": "hola", "mode": "stream"}, 422),
        ({"text": "hola", "unknown": 1}, 422),
        ({}, 422),
    ],
)
def test_tts_validation(payload, status):
    assert make_client().post("/v1/tts", json=payload).status_code == status


def test_tts_not_ready():
    r = make_client(ready=False).post("/v1/tts", json={"text": "hola"})
    assert r.status_code == 503 and "not ready" in r.json()["error"]


def test_tts_engine_error_is_500():
    r = make_client(rvc=FakeRvc(fail=True)).post("/v1/tts", json={"text": "hola"})
    assert r.status_code == 500 and "boom" in r.json()["error"]


def test_tts_accepts_cp1252_body_from_windows_shells():
    body = '{"text": "Todos los sistemas están bien."}'.encode("cp1252")
    r = make_client().post("/v1/tts", content=body, headers={"Content-Type": "application/json"})
    assert r.status_code == 200


def test_tts_invalid_json():
    r = make_client().post("/v1/tts", content=b"{nope", headers={"Content-Type": "application/json"})
    assert r.status_code == 422 and "invalid JSON" in r.json()["error"]


def test_tts_rejects_oversized_body():
    big = b'{"text": "' + b"a" * (2 << 20) + b'"}'
    r = make_client().post("/v1/tts", content=big, headers={"Content-Type": "application/json"})
    assert r.status_code == 413


def test_internal_value_error_is_500_not_422():
    rvc = FakeRvc()

    def broken(*_args, **_kwargs):
        raise ValueError("shape mismatch deep inside the engine")

    rvc.convert = broken
    r = make_client(rvc=rvc).post("/v1/tts", json={"text": "hola"})
    assert r.status_code == 500
