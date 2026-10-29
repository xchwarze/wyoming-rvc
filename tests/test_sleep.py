"""Idle sleep: IdleSleeper, the pipeline guard, and the unload/reload hooks (fakes: no torch)."""

import asyncio
import contextlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.conftest import FakePiper, FakeRvc
from wyoming_rvc import main
from wyoming_rvc.config import ConfigError, Settings
from wyoming_rvc.pipeline import SynthesisOptions, TtsPipeline
from wyoming_rvc.piper_engine import PiperEngine, PiperOptions
from wyoming_rvc.rvc_engine import SharedModels
from wyoming_rvc.sleep import IdleSleeper
from wyoming_rvc.voices import VoiceConfig, VoiceManager


class Clock:
    t = 0.0

    def __call__(self) -> float:
        return self.t


def make(idle=10.0, fail=None):
    clock, log, fail = Clock(), [], fail if fail is not None else {"wake": False}

    async def sleep():
        log.append("sleep")

    async def wake():
        await asyncio.sleep(0.01)  # lets concurrent requests overlap
        if fail["wake"]:
            raise RuntimeError("no VRAM")
        log.append("wake")

    return IdleSleeper(idle, sleep, wake, clock), clock, log


def manager() -> VoiceManager:
    config = VoiceConfig(id="v", name="v", language="en", repo_id="-")
    m = VoiceManager([config], "v", provision=lambda c: ("f", 32000), load=lambda c, f: FakeRvc())
    m.provision_all()
    return m


# --------------------------------------------------------------------------- config


@pytest.mark.parametrize("env,expected", [({}, 0.0), ({"IDLE_SLEEP_S": "0"}, 0.0), ({"IDLE_SLEEP_S": "90"}, 90.0)])
def test_idle_sleep_setting(env, expected):
    assert Settings.from_env(env).idle_sleep_s == expected


@pytest.mark.parametrize("value", ["-1", "soon", "100000"])
def test_idle_sleep_setting_rejects_bad_values(value):
    with pytest.raises(ConfigError):
        Settings.from_env({"IDLE_SLEEP_S": value})


def test_sleeper_needs_a_positive_time():
    with pytest.raises(ValueError):
        IdleSleeper(0, None, None)


# --------------------------------------------------------------------------- IdleSleeper


async def test_sleeps_only_after_the_idle_time():
    s, clock, log = make()
    clock.t = 9.9
    assert not await s.check()
    clock.t = 10
    assert await s.check() and s.asleep and not await s.check()
    assert log == ["sleep"]


async def test_never_sleeps_during_a_request_and_idle_time_starts_when_it_ends():
    s, clock, log = make()
    async with s.request():
        clock.t = 100
        assert not await s.check() and s.active == 1
    clock.t = 105
    assert not await s.check()
    clock.t = 110
    assert await s.check()


async def test_concurrent_requests_wake_the_models_once():
    s, clock, log = make()
    clock.t = 10
    await s.check()

    async def one():
        async with s.request():
            assert not s.asleep

    await asyncio.gather(one(), one(), one())
    assert log == ["sleep", "wake"] and s.wakes == 1 and s.active == 0


async def test_failed_wake_stays_asleep_and_the_next_request_retries():
    fail = {"wake": True}
    s, clock, log = make(fail=fail)
    clock.t = 10
    await s.check()
    with pytest.raises(RuntimeError):
        async with s.request():
            pass
    assert s.asleep and s.active == 0
    fail["wake"] = False
    async with s.request():
        assert not s.asleep
    assert log == ["sleep", "wake"]


async def test_cancelled_request_still_finishes_the_wake():
    s, clock, log = make()
    clock.t = 10
    await s.check()

    async def one():
        async with s.request():
            pass

    task = asyncio.create_task(one())
    await asyncio.sleep(0.001)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    assert not s.asleep and s.active == 0 and log == ["sleep", "wake"]


async def test_monitor_puts_an_idle_service_to_sleep():
    log = []

    async def sleep():
        log.append("sleep")

    async def wake():
        log.append("wake")

    s = IdleSleeper(0.05, sleep, wake)
    s.start()
    await asyncio.sleep(0.3)
    await s.stop()
    assert s.asleep and log == ["sleep"]


# --------------------------------------------------------------------------- pipeline


class SleepyPiper(FakePiper):
    awake = True

    def sentences(self, text):
        assert self.awake, "source used while the models were unloaded"
        yield from super().sentences(text)


async def test_pipeline_wakes_the_models_before_serving_and_releases_on_close():
    piper = SleepyPiper()
    clock = Clock()

    async def sleep():
        piper.awake = False

    async def wake():
        piper.awake = True

    sleeper = IdleSleeper(10, sleep, wake, clock)
    pipeline = TtsPipeline(piper, FakeRvc(), SynthesisOptions(), sleeper=sleeper)
    clock.t = 10
    assert await sleeper.check() and not piper.awake
    result = await pipeline.synthesize("Hello. World.")
    assert piper.awake and result.pcm and sleeper.active == 0 and not sleeper.asleep
    async with contextlib.aclosing(pipeline.stream("One. Two.")) as stream:
        await anext(stream)
        assert sleeper.active == 1
    assert sleeper.active == 0


# --------------------------------------------------------------------------- unload hooks


async def test_unload_all_unloads_idle_voices_only_and_they_reload_on_demand():
    m = manager()
    async with m.acquire("v") as lease:
        first = lease.engine
        assert await m.unload_all() == []  # in use: stays
    assert await m.unload_all() == ["v"] and first.unloaded and not m.is_loaded("v")
    async with m.acquire("v") as lease:
        assert lease.engine is not first


def test_piper_unload_keeps_the_sample_rate_and_ensure_loaded_reloads(monkeypatch):
    engine = PiperEngine(Path("m.onnx"), Path("m.json"), PiperOptions())
    with pytest.raises(RuntimeError):
        _ = engine.sample_rate
    engine._voice, engine._sample_rate = object(), 22050
    engine.unload()
    assert not engine.loaded and engine.sample_rate == 22050
    with pytest.raises(RuntimeError):
        list(engine.sentences("hi"))
    monkeypatch.setattr(engine, "load", lambda: setattr(engine, "_voice", object()))
    engine.ensure_loaded()
    assert engine.loaded


def test_shared_models_unload_and_load_only_what_is_missing():
    shared = SharedModels.__new__(SharedModels)  # no torch here: skip __init__
    shared._assets = SimpleNamespace(contentvec_dir="cv", rmvpe="r")
    shared.hubert, shared.rmvpe = None, object()
    shared._load_hubert = lambda directory: f"hubert:{directory}"
    shared.load()
    assert shared.hubert == "hubert:cv" and shared.loaded
    shared.unload()
    assert not shared.loaded and shared.hubert is None and shared.rmvpe is None


async def test_service_sleep_and_wake_unload_and_reload_everything(monkeypatch):
    freed = []
    monkeypatch.setattr(main, "release_memory", lambda: freed.append(1))

    class Shared:
        loads = unloads = 0

        def load(self):
            self.loads += 1

        def unload(self):
            self.unloads += 1

    class Source:
        loaded = True

        def unload(self):
            self.loaded = False

        def ensure_loaded(self):
            self.loaded = True

    service = main.Service(Settings.from_env({}))
    service.shared, service.source, service.voices = Shared(), Source(), manager()
    async with service.voices.acquire("v"):
        pass
    assert service._can_sleep()
    await service._sleep_models()
    assert service.shared.unloads == 1 and not service.source.loaded and freed
    assert not service.voices.is_loaded("v")
    await service._wake_models()
    assert service.shared.loads == 1 and service.source.loaded
