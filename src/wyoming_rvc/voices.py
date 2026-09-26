"""RVC voices: declarative configuration and the manager that owns model lifecycle.

Two separate states per voice:

* installed: configured, enabled, files provisioned on disk and the checkpoint validated.
  Installed voices are advertised to Home Assistant.
* loaded: the RVC generator and index are resident in RAM/VRAM, ready for inference.

``VoiceManager`` keeps at most ``RVC_MAX_LOADED_MODELS`` voices loaded (LRU). A voice in
use holds a lease and is never evicted; requests needing a slot wait for one to free up.
ContentVec, RMVPE and Piper are shared by all voices and are not managed here.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import F0_METHODS, ConfigError, Settings
from .metrics import now_ms

_LOGGER = logging.getLogger(__name__)

DEFAULT_VOICES_FILE = Path("/config/voices.yaml")
_VOICE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_VOICE_KEYS = {
    "name",
    "language",
    "enabled",
    "preload",
    "repo_id",
    "revision",
    "model_file",
    "index_file",
    "pitch",
    "index_rate",
    "protect",
    "f0_method",
}


class UnknownVoiceError(ValueError):
    """The requested voice is not configured, not enabled, or not installed."""


@dataclass(frozen=True)
class VoiceConfig:
    id: str
    name: str
    language: str
    repo_id: str
    enabled: bool = True
    preload: bool = False
    revision: str | None = None
    model_file: str | None = None
    index_file: str | None = None
    pitch: int = 0
    index_rate: float = 0.6
    protect: float = 0.33
    f0_method: str = "rmvpe"


def load_voice_configs(settings: Settings) -> tuple[list[VoiceConfig], str]:
    """Voices from VOICES_FILE (default /config/voices.yaml), else the single RVC_* voice.

    Returns (voices, default voice id).
    """
    path = settings.voices_file or DEFAULT_VOICES_FILE
    if not path.is_file():
        if settings.voices_file is not None:
            raise ConfigError(f"VOICES_FILE not found: {path}")
        voice = VoiceConfig(
            id=settings.voice_name,
            name=settings.voice_name,
            language=settings.voice_language,
            repo_id=settings.rvc_repo_id,
            preload=True,
            revision=settings.rvc_revision,
            model_file=settings.rvc_model_file,
            index_file=settings.rvc_index_file,
            pitch=settings.rvc_pitch,
            index_rate=settings.rvc_index_rate,
            protect=settings.rvc_protect,
            f0_method=settings.rvc_f0_method,
        )
        _check_default(settings.default_voice, [voice])
        return [voice], settings.default_voice or voice.id

    voices = parse_voices(path.read_text(encoding="utf-8"), settings, str(path))
    enabled = [v for v in voices if v.enabled]
    if not enabled:
        raise ConfigError(f"{path}: no enabled voices")
    default = settings.default_voice or enabled[0].id
    _check_default(default, enabled)
    return voices, default


def parse_voices(text: str, settings: Settings, source: str = "voices.yaml") -> list[VoiceConfig]:
    """Parse and validate the ``voices:`` mapping of a voices file."""
    import yaml

    try:
        data = yaml.safe_load(text) or {}
    except yaml.YAMLError as err:
        raise ConfigError(f"{source}: invalid YAML: {err}") from err
    entries = data.get("voices") if isinstance(data, dict) else None
    if not isinstance(entries, dict) or not entries:
        raise ConfigError(f"{source}: expected a non-empty 'voices:' mapping")
    for vid in entries:
        if not isinstance(vid, str):
            raise ConfigError(
                f"{source}: voice id {vid!r} is not text; quote it (YAML reads on/off/yes/no as booleans)"
            )
    return [_voice(vid, raw or {}, settings, source) for vid, raw in entries.items()]


def _voice(vid: str, raw: Any, settings: Settings, source: str) -> VoiceConfig:
    where = f"{source}: voice {vid!r}"
    if not _VOICE_ID.match(vid):
        raise ConfigError(f"{where}: id must be lowercase letters, digits, '-' or '_'")
    if not isinstance(raw, dict):
        raise ConfigError(f"{where}: expected a mapping")
    unknown = set(raw) - _VOICE_KEYS
    if unknown:
        raise ConfigError(f"{where}: unknown keys {sorted(unknown)}")

    def get(key: str, kind: type, default: Any) -> Any:
        value = raw.get(key)
        if value is None:
            return default
        if kind is float and isinstance(value, int) and not isinstance(value, bool):
            value = float(value)
        if not isinstance(value, kind) or (kind is int and isinstance(value, bool)):
            raise ConfigError(f"{where}: {key} must be {kind.__name__}")
        return value

    model_file = get("model_file", str, None)
    repo_id = get("repo_id", str, "")
    if not repo_id and not (model_file and Path(model_file).is_absolute()):
        raise ConfigError(f"{where}: needs repo_id (or an absolute model_file)")
    voice = VoiceConfig(
        id=vid,
        name=get("name", str, vid),
        language=get("language", str, settings.voice_language),
        repo_id=repo_id,
        enabled=get("enabled", bool, True),
        preload=get("preload", bool, False),
        revision=get("revision", str, None),
        model_file=model_file,
        index_file=get("index_file", str, None),
        pitch=get("pitch", int, settings.rvc_pitch),
        index_rate=get("index_rate", float, settings.rvc_index_rate),
        protect=get("protect", float, settings.rvc_protect),
        f0_method=get("f0_method", str, settings.rvc_f0_method),
    )
    if not -24 <= voice.pitch <= 24:
        raise ConfigError(f"{where}: pitch must be within [-24, 24]")
    if not 0.0 <= voice.index_rate <= 1.0:
        raise ConfigError(f"{where}: index_rate must be within [0, 1]")
    if not 0.0 <= voice.protect <= 0.5:
        raise ConfigError(f"{where}: protect must be within [0, 0.5]")
    if voice.f0_method not in F0_METHODS:
        raise ConfigError(f"{where}: f0_method must be one of {', '.join(F0_METHODS)}")
    return voice


def _check_default(default: str | None, voices: list[VoiceConfig]) -> None:
    if default is not None and default not in {v.id for v in voices if v.enabled}:
        raise ConfigError(f"DEFAULT_VOICE={default!r} is not an enabled voice ({[v.id for v in voices]})")


# --------------------------------------------------------------------------- manager


@dataclass
class VoiceState:
    config: VoiceConfig
    files: Any = None
    sample_rate: int | None = None
    error: str | None = None

    @property
    def installed(self) -> bool:
        return self.config.enabled and self.files is not None and self.error is None


@dataclass
class Lease:
    """A loaded voice, reserved for the duration of an ``acquire`` block."""

    voice: VoiceConfig
    engine: Any
    load_ms: float = 0.0  # > 0 only when this request had to load the model


@dataclass
class _Slot:
    users: int = 0
    engine: Any = None
    ready: bool = False
    last_load_ms: float = 0.0


class VoiceManager:
    """Owns provisioning and the LRU of loaded RVC voices."""

    def __init__(
        self,
        configs: list[VoiceConfig],
        default_voice: str,
        provision: Callable[[VoiceConfig], tuple[Any, int]],
        load: Callable[[VoiceConfig, Any], Any],
        max_loaded: int = 1,
    ) -> None:
        if max_loaded < 1:
            raise ValueError("max_loaded must be >= 1")
        self._states = {c.id: VoiceState(c) for c in configs}
        self.default_voice = default_voice
        self._provision = provision
        self._load = load
        self.max_loaded = max_loaded
        self._loaded: OrderedDict[str, _Slot] = OrderedDict()
        self._cond = asyncio.Condition()

    @classmethod
    def single(cls, engine: Any, voice_id: str = "default", language: str = "en") -> VoiceManager:
        """Wrap one already-loaded engine (tests, simple embedding)."""
        config = VoiceConfig(id=voice_id, name=voice_id, language=language, repo_id="-")
        manager = cls([config], voice_id, provision=lambda c: (None, 0), load=lambda c, f: engine)
        state = manager._states[voice_id]
        state.files, state.sample_rate = "-", int(engine.output_sample_rate)
        manager._loaded[voice_id] = _Slot(engine=engine, ready=True)
        return manager

    # ------------------------------------------------------------------ provisioning

    def provision_all(self) -> None:
        """Download/validate every enabled voice (blocking). Failures are logged, not raised."""
        for state in self._states.values():
            if not state.config.enabled:
                continue
            try:
                state.files, state.sample_rate = self._provision(state.config)
                state.error = None
                _LOGGER.info("Voice %s installed (%d Hz)", state.config.id, state.sample_rate)
            except Exception as err:  # a broken voice must not take the others down
                state.files, state.error = None, str(err)
                _LOGGER.error("Voice %s is not available: %s", state.config.id, err)

    # ------------------------------------------------------------------ queries

    def resolve(self, voice_id: str | None) -> VoiceState:
        """The installed voice for ``voice_id`` (default when None); never a substitute."""
        vid = voice_id or self.default_voice
        state = self._states.get(vid)
        if state is None or not state.config.enabled:
            raise UnknownVoiceError(f"Unknown voice {vid!r}; available: {self.installed_ids()}")
        if not state.installed:
            raise UnknownVoiceError(f"Voice {vid!r} is not installed: {state.error or 'not provisioned'}")
        return state

    def installed_ids(self) -> list[str]:
        return [vid for vid, s in self._states.items() if s.installed]

    def installed_voices(self) -> list[VoiceConfig]:
        return [s.config for s in self._states.values() if s.installed]

    def is_loaded(self, voice_id: str) -> bool:
        slot = self._loaded.get(voice_id)
        return bool(slot and slot.ready)

    def list_voices(self) -> list[dict[str, Any]]:
        return [
            {
                "id": vid,
                "name": s.config.name,
                "language": s.config.language,
                "enabled": s.config.enabled,
                "installed": s.installed,
                "loaded": self.is_loaded(vid),
                "default": vid == self.default_voice,
                "sample_rate": s.sample_rate,
                "error": s.error,
            }
            for vid, s in self._states.items()
        ]

    def loaded_engines(self) -> dict[str, Any]:
        return {vid: slot.engine for vid, slot in self._loaded.items() if slot.ready}

    # ------------------------------------------------------------------ lifecycle

    @contextlib.asynccontextmanager
    async def acquire(self, voice_id: str | None = None) -> AsyncIterator[Lease]:
        """Reserve a loaded voice, loading it (and evicting an idle one) if needed."""
        state = self.resolve(voice_id)
        vid = state.config.id
        to_unload: list[Any] = []
        must_load = False
        async with self._cond:
            while True:
                slot = self._loaded.get(vid)
                if slot is not None:
                    if slot.ready:
                        slot.users += 1
                        self._loaded.move_to_end(vid)
                        break
                    await self._cond.wait()  # another request is loading it
                    continue
                if len(self._loaded) < self.max_loaded:
                    slot = self._loaded[vid] = _Slot(users=1)
                    must_load = True
                    break
                victim = next((k for k, s in self._loaded.items() if s.ready and s.users == 0), None)
                if victim is None:
                    await self._cond.wait()  # every loaded voice is busy
                    continue
                to_unload.append(self._loaded.pop(victim).engine)
                _LOGGER.info("Evicting voice %s to make room for %s", victim, vid)

        load_ms = 0.0
        try:
            for engine in to_unload:
                await asyncio.to_thread(engine.unload)
            if must_load:
                start = now_ms()
                engine = await asyncio.to_thread(self._load, state.config, state.files)
                load_ms = now_ms() - start
                async with self._cond:
                    slot.engine, slot.ready, slot.last_load_ms = engine, True, load_ms
                    self._cond.notify_all()
                _LOGGER.info("Voice %s loaded in %.0f ms", vid, load_ms)
        except BaseException:
            async with self._cond:
                if must_load and self._loaded.get(vid) is slot:
                    del self._loaded[vid]
                self._cond.notify_all()
            raise

        try:
            yield Lease(state.config, slot.engine, load_ms)
        finally:
            async with self._cond:
                slot.users -= 1
                self._cond.notify_all()

    async def load_voice(self, voice_id: str) -> float:
        """Make sure a voice is loaded; returns the load time (0 if it already was)."""
        async with self.acquire(voice_id) as lease:
            return lease.load_ms

    async def unload_voice(self, voice_id: str) -> bool:
        """Unload an idle voice. Returns False if it is not loaded or in use."""
        async with self._cond:
            slot = self._loaded.get(voice_id)
            if slot is None or not slot.ready or slot.users:
                return False
            del self._loaded[voice_id]
            self._cond.notify_all()
        await asyncio.to_thread(slot.engine.unload)
        return True

    async def preload(self) -> list[str]:
        """Load voices marked ``preload`` (config order), up to ``max_loaded``."""
        wanted = [c.id for c in self.installed_voices() if c.preload][: self.max_loaded]
        for vid in wanted:
            await self.load_voice(vid)
        return wanted
