"""Per-voice TTS sources: voices.yaml keys, SourceRegistry, pipeline routing."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from wyoming_rvc import model_loader, piper_engine
from wyoming_rvc.config import ConfigError, Settings
from wyoming_rvc.pipeline import SynthesisOptions, TtsPipeline
from wyoming_rvc.sources import SourceRegistry
from wyoming_rvc.voices import VoiceManager, parse_voices

SETTINGS = Settings.from_env({})

JARVIS = """
voices:
  jarvis:
    name: "jarvis"
    language: "en"
    piper_voice: en_GB-northern_english_male-medium
    repo_id: a/j
    pitch: -6
  plain:
    repo_id: a/b
  es:
    repo_id: a/c
    piper_voice: es_MX-claude-high
    piper_length_scale: 1
  kokoro:
    repo_id: a/d
    upstream: tcp://kokoro:10200
    upstream_voice: af_heart
"""


def voices():
    return {c.id: c for c in parse_voices(JARVIS, SETTINGS)}


def test_voice_without_source_keys_uses_global_source():
    assert voices()["plain"].source is None


def test_piper_voice_key_selects_piper_and_language():
    v = voices()
    assert v["jarvis"].source.kind == "piper"
    assert v["jarvis"].source.piper_voice == "en_GB-northern_english_male-medium"
    assert v["jarvis"].pitch == -6
    assert v["es"].language == "es"  # derived from the Piper voice when 'language' is not given
    assert v["es"].source.piper_length_scale == 1.0


def test_upstream_keys_select_wyoming():
    spec = voices()["kokoro"].source
    assert spec.kind == "wyoming" and spec.describe() == "wyoming:tcp://kokoro:10200/af_heart"


def test_wyoming_voice_can_reuse_the_global_upstream():
    text = "voices:\n  x:\n    repo_id: a/b\n    source: wyoming\n    upstream_voice: v\n"
    [voice] = parse_voices(text, replace(SETTINGS, wyoming_upstream="tcp://h:1"))
    assert voice.source.kind == "wyoming" and voice.source.upstream is None


@pytest.mark.parametrize(
    ("fragment", "message"),
    [
        ("piper_voice: en_US-lessac-medium\n    upstream: tcp://h:1", "cannot be mixed"),
        ("piper_voice: nonsense", "does not look like"),
        ("piper_voice: en_US-lessac-medium\n    piper_model: /m.onnx", "not both"),
        ("source: piper\n    piper_config: /m.onnx.json", "requires piper_model"),
        ("source: wyoming", "needs 'upstream"),
        ("source: nope", "source must be one of"),
        ("piper_length_scale: 9", "piper_length_scale"),
        ("source: wyoming\n    piper_speaker_id: 1", "does not take piper_"),
        ("source: piper\n    upstream_voice: v", "does not take upstream"),
    ],
)
def test_invalid_source_keys_are_rejected(fragment, message):
    with pytest.raises(ConfigError, match=message):
        parse_voices(f"voices:\n  x:\n    repo_id: a/b\n    {fragment}\n", SETTINGS)


# --------------------------------------------------------------------------- registry


class FakePiper:
    built: list[str] = []

    def __init__(self, model, config, options):
        self.sample_rate = 16000 if "northern" in str(model) else 22050
        FakePiper.built.append(str(model))

    def load(self):
        pass

    device = "cpu"

    def sentences(self, text):
        level = 0.4 if self.sample_rate == 16000 else 0.1
        for _ in (p for p in text.split(".") if p.strip()):
            yield np.full(int(self.sample_rate * 0.25), level, np.float32)


@pytest.fixture
def fake_piper(monkeypatch):
    FakePiper.built = []
    resolved = []

    def resolve(settings):
        resolved.append(settings)
        name = settings.piper_model.name if settings.piper_model else settings.piper_voice
        return model_loader.PiperFiles(Path(name), Path(f"{name}.json"), name)

    monkeypatch.setattr(model_loader, "resolve_piper", resolve)
    monkeypatch.setattr(piper_engine, "PiperEngine", FakePiper)
    return resolved


def test_registry_shares_and_evicts(fake_piper):
    default = object()
    reg = SourceRegistry(replace(SETTINGS, piper_model=Path("/global.onnx")), default, max_loaded=1)
    v = voices()
    assert reg.get(None) is default
    first = reg.get(v["jarvis"].source)
    assert reg.get(v["jarvis"].source) is first and len(FakePiper.built) == 1
    # a named voice is not shadowed by a global PIPER_MODEL
    assert fake_piper[-1].piper_model is None and fake_piper[-1].piper_voice == "en_GB-northern_english_male-medium"
    reg.get(v["es"].source)  # max_loaded=1: jarvis is evicted
    assert reg.loaded() == [v["es"].source]
    assert fake_piper[-1].piper_length_scale == 1.0


def test_registry_unload_all_drops_piper_voices(fake_piper):
    reg = SourceRegistry(SETTINGS, object())
    v = voices()
    first = reg.get(v["jarvis"].source)
    assert reg.unload_all() == [v["jarvis"].source] and reg.loaded() == []
    assert reg.get(v["jarvis"].source) is not first  # idle sleep: loaded again on its next use


# --------------------------------------------------------------------------- pipeline


class FakeRvc:
    output_sample_rate = 32000

    def __init__(self):
        self.rates, self.levels = [], []

    def convert(self, audio, rate, pitch, f0_method, index_rate, protect):
        self.rates.append(rate)
        self.levels.append(round(float(audio[0]), 1))
        return np.zeros(int(audio.size * 32000 / rate), np.float32), 32000

    def unload(self):
        pass


async def test_pipeline_uses_the_voices_own_source(fake_piper):
    default = FakePiper(Path("default"), Path("d.json"), None)
    rvc = FakeRvc()
    configs = [c for c in voices().values() if c.id in ("jarvis", "plain")]
    manager = VoiceManager(configs, "plain", lambda c: ("files", 32000), lambda c, f: rvc, 2)
    manager.provision_all()
    pipeline = TtsPipeline(default, manager, SynthesisOptions(), sources=SourceRegistry(SETTINGS, default))

    await pipeline.synthesize("One.", pipeline.options(voice="plain"))
    await pipeline.synthesize("One.", pipeline.options(voice="jarvis"))
    tts_only = await pipeline.synthesize("One.", pipeline.options(rvc=False))

    assert rvc.rates == [22050, 16000]  # global source for 'plain', jarvis's own Piper voice for 'jarvis'
    assert rvc.levels == [0.1, 0.4]
    assert tts_only.sample_rate == 22050  # without RVC the global source is used


def test_piper_engines_share_one_synthesis_lock():
    # espeak-ng is process-global: two voices' engines must not phonemize at the same time.
    options = piper_engine.PiperOptions()
    first = piper_engine.PiperEngine(Path("a.onnx"), Path("a.json"), options)
    second = piper_engine.PiperEngine(Path("b.onnx"), Path("b.json"), options)
    assert first._lock is second._lock
