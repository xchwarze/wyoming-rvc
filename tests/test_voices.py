"""Voice configuration and VoiceManager lifecycle (no torch: fake engines)."""

import asyncio

import pytest

from tests.conftest import FakePiper, FakeRvc
from wyoming_rvc.config import ConfigError, Settings
from wyoming_rvc.pipeline import SynthesisOptions, TtsPipeline
from wyoming_rvc.voices import (
    UnknownVoiceError,
    VoiceConfig,
    VoiceManager,
    load_voice_configs,
    parse_voices,
)

YAML = """
voices:
  teto:
    name: "Kasane Teto"
    preload: true
    repo_id: "Slichi/KasaneTeto"
  miku:
    name: "Hatsune Miku"
    language: "es"
    repo_id: "someone/miku"
    model_file: "miku.pth"
    pitch: 4
    index_rate: 0.5
  retired:
    enabled: false
    repo_id: "x/y"
"""


def settings(**env: str) -> Settings:
    return Settings.from_env(env)


# --------------------------------------------------------------------------- config


def test_parse_voices_defaults_and_overrides():
    voices = {v.id: v for v in parse_voices(YAML, settings(RVC_PITCH="1"))}
    assert list(voices) == ["teto", "miku", "retired"]
    teto, miku = voices["teto"], voices["miku"]
    assert (teto.name, teto.language, teto.preload, teto.pitch) == ("Kasane Teto", "en", True, 1)
    assert (miku.language, miku.pitch, miku.index_rate, miku.model_file, miku.preload) == (
        "es",
        4,
        0.5,
        "miku.pth",
        False,
    )
    assert voices["retired"].enabled is False


@pytest.mark.parametrize(
    "text,match",
    [
        ("voices:\n  Teto:\n    repo_id: a/b\n", "lowercase"),
        ("voices:\n  teto:\n    repo_id: a/b\n    colour: red\n", "unknown keys"),
        ("voices:\n  teto:\n    repo_id: a/b\n    pitch: 40\n", "pitch"),
        ("voices:\n  teto:\n    repo_id: a/b\n    pitch: high\n", "pitch must be int"),
        ("voices:\n  teto:\n    name: x\n", "repo_id"),
        ("voices:\n  teto:\n    repo_id: a/b\n    f0_method: crepe\n", "f0_method"),
        ("voices: {}\n", "non-empty"),
        ("voices: [\n", "invalid YAML"),
    ],
)
def test_parse_voices_rejects_bad_config(text, match):
    with pytest.raises(ConfigError, match=match):
        parse_voices(text, settings())


def test_env_fallback_is_a_single_preloaded_voice(tmp_path):
    s = settings(VOICE_NAME="teto", RVC_REPO_ID="a/b", RVC_PITCH="2")
    voices, default = load_voice_configs(s)  # /config/voices.yaml does not exist here
    assert default == "teto" and len(voices) == 1
    assert (voices[0].repo_id, voices[0].pitch, voices[0].preload) == ("a/b", 2, True)


def test_voices_file_default_voice_and_errors(tmp_path):
    path = tmp_path / "voices.yaml"
    path.write_text(YAML)
    voices, default = load_voice_configs(settings(VOICES_FILE=str(path)))
    assert default == "teto" and [v.id for v in voices] == ["teto", "miku", "retired"]
    assert load_voice_configs(settings(VOICES_FILE=str(path), DEFAULT_VOICE="miku"))[1] == "miku"
    with pytest.raises(ConfigError, match="DEFAULT_VOICE"):
        load_voice_configs(settings(VOICES_FILE=str(path), DEFAULT_VOICE="retired"))
    with pytest.raises(ConfigError, match="not found"):
        load_voice_configs(settings(VOICES_FILE=str(tmp_path / "missing.yaml")))


# --------------------------------------------------------------------------- manager


class Registry:
    """Fake provision/load that records what happened."""

    def __init__(self, broken=(), rates=None, load_delay=0.0, fail_loads=0):
        self.broken, self.rates = set(broken), rates or {}
        self.load_delay, self.fail_loads = load_delay, fail_loads
        self.loads: list[str] = []
        self.engines: dict[str, FakeRvc] = {}

    def provision(self, voice: VoiceConfig):
        if voice.id in self.broken:
            raise RuntimeError("download failed")
        return f"files-{voice.id}", self.rates.get(voice.id, 32000)

    def load(self, voice: VoiceConfig, files):
        import time

        time.sleep(self.load_delay)
        if self.fail_loads:
            self.fail_loads -= 1
            raise RuntimeError("cuda oom")
        self.loads.append(voice.id)
        engine = self.engines[voice.id] = FakeRvc(delay=0.0, sample_rate=self.rates.get(voice.id, 32000))
        return engine


def configs(*ids, preload=()):
    return [VoiceConfig(id=i, name=i.title(), language="en", repo_id=f"r/{i}", preload=i in preload) for i in ids]


def manager(reg, ids=("teto", "miku"), max_loaded=1, preload=()):
    m = VoiceManager(configs(*ids, preload=preload), ids[0], reg.provision, reg.load, max_loaded)
    m.provision_all()
    return m


async def test_broken_voice_is_not_installed_nor_substituted():
    m = manager(Registry(broken={"miku"}))
    assert m.installed_ids() == ["teto"]
    with pytest.raises(UnknownVoiceError, match="not installed"):
        m.resolve("miku")
    with pytest.raises(UnknownVoiceError, match="Unknown voice"):
        m.resolve("zundamon")
    assert m.resolve(None).config.id == "teto"


async def test_lru_single_slot_switches_and_reuses():
    reg = Registry()
    m = manager(reg)
    async with m.acquire("teto") as lease:
        assert lease.load_ms > 0
    async with m.acquire("teto") as lease:
        assert lease.load_ms == 0  # warm
    async with m.acquire("miku") as lease:
        assert lease.load_ms > 0
    assert reg.loads == ["teto", "miku"] and reg.engines["teto"].unloaded
    assert [v["id"] for v in m.list_voices() if v["loaded"]] == ["miku"]


async def test_lru_evicts_least_recently_used():
    reg = Registry()
    m = manager(reg, ("a", "b", "c"), max_loaded=2)
    for vid in ("a", "b", "a", "c"):
        async with m.acquire(vid):
            pass
    assert reg.engines["b"].unloaded and not reg.engines["a"].unloaded
    assert sorted(m.loaded_engines()) == ["a", "c"]


async def test_voice_in_use_is_never_evicted():
    reg = Registry()
    m = manager(reg)
    release = asyncio.Event()

    async def hold_teto():
        async with m.acquire("teto"):
            await release.wait()

    holder = asyncio.create_task(hold_teto())
    await asyncio.sleep(0.05)

    async def use_miku():
        async with m.acquire("miku"):
            return "done"

    waiter = asyncio.create_task(use_miku())
    await asyncio.sleep(0.1)
    assert not waiter.done() and not reg.engines["teto"].unloaded
    release.set()
    assert await asyncio.wait_for(waiter, 2) == "done"
    await holder
    assert reg.engines["teto"].unloaded


async def test_concurrent_requests_load_a_voice_once():
    reg = Registry(load_delay=0.1)
    m = manager(reg)

    async def use():
        async with m.acquire("teto") as lease:
            return lease.engine

    engines = await asyncio.gather(*(use() for _ in range(5)))
    assert reg.loads == ["teto"] and len({id(e) for e in engines}) == 1


async def test_failed_load_is_retried_next_time():
    reg = Registry(fail_loads=1)
    m = manager(reg)
    with pytest.raises(RuntimeError, match="oom"):
        async with m.acquire("teto"):
            pass
    assert not m.is_loaded("teto")
    async with m.acquire("teto") as lease:
        assert lease.engine is reg.engines["teto"]


async def test_cancelled_load_keeps_the_model():
    reg = Registry(load_delay=0.2)
    m = manager(reg)

    async def use():
        async with m.acquire("teto"):
            pass

    task = asyncio.create_task(use())
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert m.is_loaded("teto")  # the load finished and was kept
    async with m.acquire("teto") as lease:
        assert lease.load_ms == 0
    assert reg.loads == ["teto"]  # loaded once, nothing orphaned
    async with asyncio.timeout(2), m.acquire("miku"):  # teto is idle, so it can be evicted
        pass
    assert reg.engines["teto"].unloaded


async def test_preload_and_unload():
    reg = Registry()
    m = manager(reg, ("teto", "miku", "yui"), max_loaded=1, preload=("teto", "miku"))
    assert await m.preload() == ["teto"]  # capped by max_loaded
    assert reg.loads == ["teto"]
    async with m.acquire("teto"):
        assert await m.unload_voice("teto") is False  # busy
    assert await m.unload_voice("teto") is True and reg.engines["teto"].unloaded
    assert await m.unload_voice("teto") is False  # not loaded


# --------------------------------------------------------------------------- pipeline routing


async def test_pipeline_routes_by_voice_with_per_voice_settings():
    reg = Registry(rates={"miku": 40000})
    cfgs = [
        VoiceConfig(id="teto", name="Teto", language="en", repo_id="r/t", pitch=0),
        VoiceConfig(id="miku", name="Miku", language="en", repo_id="r/m", pitch=5),
    ]
    m = VoiceManager(cfgs, "teto", reg.provision, reg.load, 1)
    m.provision_all()
    p = TtsPipeline(FakePiper(), m, SynthesisOptions())

    default = await p.synthesize("Hola.")
    assert default.sample_rate == 32000 and default.metrics.voice == "teto"
    miku = await p.synthesize("Hola.", p.options(voice="miku"))
    assert miku.sample_rate == 40000 and miku.metrics.voice == "miku" and miku.metrics.rvc_load_ms > 0
    assert reg.engines["miku"].pitches == [5]  # voice default
    await p.synthesize("Hola.", p.options(voice="miku", pitch=-2))
    assert reg.engines["miku"].pitches == [5, -2]  # request override
    again = await p.synthesize("Hola.", p.options(voice="miku"))
    assert again.metrics.rvc_load_ms == 0
    with pytest.raises(ValueError, match="Unknown voice"):
        p.options(voice="zundamon")


def test_yaml_boolean_voice_id_gets_a_clear_error():
    with pytest.raises(ConfigError, match="quote it"):
        parse_voices("voices:\n  off:\n    repo_id: a/b\n", settings())


def test_example_voices_file_is_valid():
    from pathlib import Path

    example = Path(__file__).parent.parent / "examples" / "voices.yaml"
    voices = {v.id: v for v in parse_voices(example.read_text(encoding="utf-8"), settings())}
    assert voices["teto"].preload and voices["myvoice"].enabled is False
