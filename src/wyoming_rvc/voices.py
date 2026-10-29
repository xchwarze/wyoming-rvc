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
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .config import F0_METHODS, SOURCES, ConfigError, Settings, piper_voice_language
from .metrics import now_ms
from .model_loader import PIPER_VOICE_PATTERN

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
    # Per-voice TTS source (all optional; a voice without any of these uses the global source).
    "source",
    "piper_voice",
    "piper_model",
    "piper_config",
    "piper_speaker_id",
    "piper_length_scale",
    "piper_noise_scale",
    "piper_noise_w_scale",
    "upstream",
    "upstream_voice",
    "upstream_speaker",
}
_PIPER_KEYS = (
    "piper_voice",
    "piper_model",
    "piper_config",
    "piper_speaker_id",
    "piper_length_scale",
    "piper_noise_scale",
    "piper_noise_w_scale",
)
_UPSTREAM_KEYS = ("upstream", "upstream_voice", "upstream_speaker")


class UnknownVoiceError(ValueError):
    """The requested voice is not configured, not enabled, or not installed."""


@dataclass(frozen=True)
class SourceSpec:
    """Where one voice's source audio comes from (``None`` fields inherit the global settings).

    Hashable on purpose: ``SourceRegistry`` shares one loaded source between voices with equal specs.
    """

    kind: str  # "piper" | "wyoming"
    piper_voice: str | None = None
    piper_model: str | None = None
    piper_config: str | None = None
    piper_speaker_id: int | None = None
    piper_length_scale: float | None = None
    piper_noise_scale: float | None = None
    piper_noise_w_scale: float | None = None
    upstream: str | None = None
    upstream_voice: str | None = None
    upstream_speaker: str | None = None

    def describe(self) -> str:
        if self.kind == "wyoming":
            target = self.upstream or "default-upstream"
            return f"wyoming:{target}" + (f"/{self.upstream_voice}" if self.upstream_voice else "")
        return f"piper:{self.piper_model or self.piper_voice or 'default'}"


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
    source: SourceSpec | None = None  # None: the global source (SOURCE / PIPER_VOICE / WYOMING_UPSTREAM)


# Served when there is no voices file and the RVC_* voice was not customized, so a fresh
# install already offers a choice in Home Assistant. Languages follow VOICE_LANGUAGE.
BUILTIN_VOICES = """
voices:
  teto:
    name: "Kasane Teto"
    repo_id: "Slichi/KasaneTeto"
    preload: true
  miku:
    name: "Hatsune Miku"
    repo_id: "binant/Hatsune_Miku__RVC_v2_"
    revision: "b05cba64eaa30f3bac39ff8ce7caff14a9af122b"
    pitch: 2
"""


def load_voice_configs(settings: Settings) -> tuple[list[VoiceConfig], str]:
    """Voices from VOICES_FILE (default /config/voices.yaml); else the RVC_* voice if it was
    customized; else the built-in voices.

    Returns (voices, default voice id).
    """
    path = settings.voices_file or DEFAULT_VOICES_FILE
    if not path.is_file():
        if settings.voices_file is not None:
            raise ConfigError(f"VOICES_FILE not found: {path}")
        if not _customized(settings):
            voices = parse_voices(BUILTIN_VOICES, settings, "built-in voices")
            default = settings.default_voice or voices[0].id
            _check_default(default, voices)
            return voices, default
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


def _customized(settings: Settings) -> bool:
    """Whether the RVC_* / VOICE_NAME settings describe the user's own single voice."""
    defaults = Settings()
    return (
        settings.rvc_repo_id != defaults.rvc_repo_id
        or settings.voice_name != defaults.voice_name
        or any((settings.rvc_revision, settings.rvc_model_file, settings.rvc_index_file))
    )


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

    source = _source_spec(raw, get, settings, where)
    model_file = get("model_file", str, None)
    repo_id = get("repo_id", str, "")
    if not repo_id and not (model_file and Path(model_file).is_absolute()):
        raise ConfigError(f"{where}: needs repo_id (or an absolute model_file)")
    voice = VoiceConfig(
        id=vid,
        name=get("name", str, vid),
        language=get("language", str, _source_language(source, settings.voice_language)),
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
        source=source,
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


def _source_language(spec: SourceSpec | None, default: str) -> str:
    """A named Piper voice implies its language (``en_GB-...`` -> ``en``), like the global PIPER_VOICE."""
    if spec is not None and spec.kind == "piper" and spec.piper_voice:
        return piper_voice_language(spec.piper_voice) or default
    return default


def _source_spec(raw: dict, get: Callable[[str, type, Any], Any], settings: Settings, where: str) -> SourceSpec | None:
    """Parse the per-voice source keys; ``None`` when the voice uses the global source."""
    kind = get("source", str, None)
    if kind is not None and kind not in SOURCES:
        raise ConfigError(f"{where}: source must be one of {', '.join(SOURCES)}; got {kind!r}")
    has_piper = any(raw.get(k) is not None for k in _PIPER_KEYS)
    has_upstream = any(raw.get(k) is not None for k in _UPSTREAM_KEYS)
    if has_piper and has_upstream:
        raise ConfigError(f"{where}: piper_* and upstream* keys cannot be mixed")
    if kind is None:
        if has_piper:
            kind = "piper"
        elif has_upstream:
            kind = "wyoming"
        else:
            return None
    if kind == "piper" and has_upstream:
        raise ConfigError(f"{where}: source 'piper' does not take upstream* keys")
    if kind == "wyoming" and has_piper:
        raise ConfigError(f"{where}: source 'wyoming' does not take piper_* keys")

    if kind == "wyoming":
        upstream = get("upstream", str, None)
        if not (upstream or settings.wyoming_upstream):
            raise ConfigError(f"{where}: source 'wyoming' needs 'upstream: tcp://host:port' (or WYOMING_UPSTREAM)")
        try:
            replace(settings, wyoming_upstream=upstream or settings.wyoming_upstream).upstream_address()
        except ConfigError as err:
            raise ConfigError(f"{where}: {err}") from err
        return SourceSpec(
            kind="wyoming",
            upstream=upstream,
            upstream_voice=get("upstream_voice", str, None),
            upstream_speaker=get("upstream_speaker", str, None),
        )

    piper_voice = get("piper_voice", str, None)
    piper_model = get("piper_model", str, None)
    piper_config = get("piper_config", str, None)
    if piper_voice and piper_model:
        raise ConfigError(f"{where}: set piper_voice or piper_model, not both")
    if piper_config and not piper_model:
        raise ConfigError(f"{where}: piper_config requires piper_model")
    if piper_voice and not PIPER_VOICE_PATTERN.match(piper_voice):
        raise ConfigError(
            f"{where}: piper_voice {piper_voice!r} does not look like '<lang>_<REGION>-<name>-<quality>', "
            "e.g. en_GB-northern_english_male-medium"
        )
    speaker = get("piper_speaker_id", int, None)
    length = get("piper_length_scale", float, None)
    noise = get("piper_noise_scale", float, None)
    noise_w = get("piper_noise_w_scale", float, None)
    for key, value, lo, hi in (
        ("piper_speaker_id", speaker, 0, None),
        ("piper_length_scale", length, 0.1, 5.0),
        ("piper_noise_scale", noise, 0.0, 2.0),
        ("piper_noise_w_scale", noise_w, 0.0, 2.0),
    ):
        if value is not None and ((lo is not None and value < lo) or (hi is not None and value > hi)):
            raise ConfigError(f"{where}: {key} must be within [{lo}, {hi if hi is not None else 'inf'}]")
    return SourceSpec(
        kind="piper",
        piper_voice=piper_voice,
        piper_model=piper_model,
        piper_config=piper_config,
        piper_speaker_id=speaker,
        piper_length_scale=length,
        piper_noise_scale=noise,
        piper_noise_w_scale=noise_w,
    )


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
                "source": s.config.source.describe() if s.config.source else "default",
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
                # A cancelled request cannot stop the loading thread: wait for it and keep
                # the model, so it is neither orphaned on the GPU nor loaded a second time.
                job = asyncio.ensure_future(asyncio.to_thread(self._load, state.config, state.files))
                try:
                    await asyncio.shield(job)
                finally:
                    if not job.done():
                        with contextlib.suppress(BaseException):
                            await job
                    if job.done() and not job.cancelled() and job.exception() is None:
                        load_ms = now_ms() - start
                        async with self._cond:
                            slot.engine, slot.ready, slot.last_load_ms = job.result(), True, load_ms
                            self._cond.notify_all()
                        _LOGGER.info("Voice %s loaded in %.0f ms", vid, load_ms)
        except BaseException:
            async with self._cond:
                if must_load and self._loaded.get(vid) is slot:
                    if slot.ready:  # loaded, but this request was cancelled: release it
                        slot.users -= 1
                    else:
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
        """Load voices marked ``preload`` (the default voice if none is), up to ``max_loaded``.

        Another voice that fails to load is dropped from the installed list; the default must load.
        """
        wanted = [c.id for c in self.installed_voices() if c.preload] or [self.default_voice]
        loaded = []
        for vid in wanted[: self.max_loaded]:
            try:
                await self.load_voice(vid)
            except Exception as err:
                if vid == self.default_voice:
                    raise
                _LOGGER.error("Voice %s failed to load and will not be offered: %s", vid, err)
                self._states[vid].error = str(err)
                continue
            loaded.append(vid)
        return loaded
