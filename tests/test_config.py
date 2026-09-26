from pathlib import Path

import pytest

from wyoming_rvc.config import ConfigError, Settings


def test_defaults():
    s = Settings.from_env({})
    assert s.device == "cuda"
    assert s.http_port == 8080 and s.wyoming_port == 10200
    assert s.piper_voice == "es_AR-daniela-high"
    assert s.piper_device == "auto" and s.piper_use_cuda is True  # auto follows DEVICE=cuda
    assert s.source == "piper"
    assert s.rvc_repo_id == "Slichi/KasaneTeto"
    assert (s.rvc_f0_method, s.rvc_pitch, s.rvc_index_rate, s.rvc_protect) == ("rmvpe", 0, 0.6, 0.33)
    assert s.rvc_mode == "sentence" and s.rvc_concurrency == 1
    assert (s.program_name, s.voice_name, s.voice_language) == ("Wyoming RVC", "teto", "es")
    assert s.hf_home == Path("/models/huggingface")


def test_overrides_and_empty_values():
    s = Settings.from_env(
        {
            "DEVICE": "cpu",
            "RVC_PITCH": "-3",
            "RVC_INDEX_RATE": "0.75",
            "RVC_MODE": "STREAM",
            "RVC_MODEL_FILE": "",
            "PIPER_DEVICE": "cuda",
            "MODELS_DIR": "/data",
        }
    )
    assert s.device == "cpu"
    assert s.rvc_pitch == -3 and s.rvc_index_rate == 0.75
    assert s.rvc_mode == "stream"
    assert s.rvc_model_file is None
    assert s.piper_use_cuda is True
    assert s.piper_data_dir == Path("/data/piper") and s.hf_home == Path("/data/huggingface")


@pytest.mark.parametrize(
    "env",
    [
        {"DEVICE": "tpu"},
        {"RVC_PITCH": "25"},
        {"RVC_PITCH": "abc"},
        {"RVC_INDEX_RATE": "1.5"},
        {"RVC_PROTECT": "0.6"},
        {"RVC_MODE": "fast"},
        {"RVC_F0_METHOD": "crepe"},
        {"HTTP_PORT": "70000"},
        {"PIPER_DEVICE": "maybe"},
        {"SOURCE": "wyoming"},
        {"SOURCE": "wyoming", "WYOMING_UPSTREAM": "http://x:1"},
        {"SOURCE": "wyoming", "WYOMING_UPSTREAM": "tcp://x:abc"},
        {"SOURCE": "espeak"},
        {"STREAM_CHUNK_MS": "300", "STREAM_OVERLAP_MS": "300"},
        {"PIPER_CONFIG": "/x.json"},
        {"HTTP_PORT": "10200"},
    ],
)
def test_invalid(env):
    with pytest.raises(ConfigError):
        Settings.from_env(env)


def test_summary_is_json_friendly():
    import json

    json.dumps(Settings.from_env({}).summary())


def test_piper_device_auto_follows_device():
    assert Settings.from_env({"DEVICE": "cpu"}).piper_use_cuda is False
    assert Settings.from_env({"DEVICE": "cuda", "PIPER_DEVICE": "cpu"}).piper_use_cuda is False


@pytest.mark.parametrize("raw", ["tcp://piper:10200", "piper:10200"])
def test_upstream_address(raw):
    s = Settings.from_env({"SOURCE": "wyoming", "WYOMING_UPSTREAM": raw})
    assert s.upstream_address() == ("piper", 10200)


def test_port_clash_detected_across_different_hosts():
    with pytest.raises(ConfigError):
        Settings.from_env({"HTTP_HOST": "0.0.0.0", "WYOMING_HOST": "127.0.0.1", "HTTP_PORT": "9000", "WYOMING_PORT": "9000"})
