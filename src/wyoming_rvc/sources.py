"""Per-voice TTS sources.

Every voice can name its own source in ``voices.yaml`` (``piper_voice``, ``piper_model``,
``upstream`` ...). ``SourceRegistry`` turns a voice's ``SourceSpec`` into a loaded source and
shares it between voices with an equal spec. Voices without a spec use the global source that
``main.Service`` loads from ``SOURCE`` / ``PIPER_VOICE`` / ``WYOMING_UPSTREAM``.

Piper engines are small and are loaded on first use; at most ``SOURCE_MAX_LOADED`` per-voice
sources stay resident (LRU). The global source is never evicted. A request that already holds a
source keeps it alive even if it is evicted meanwhile.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path
from typing import Any

from .config import Settings
from .voices import SourceSpec

_LOGGER = logging.getLogger(__name__)


def _first(voice_value: Any, global_value: Any) -> Any:
    """The voice's own setting when it has one, else the global one."""
    return global_value if voice_value is None else voice_value


class SourceRegistry:
    """Blocking ``get``/``prepare``; call them from a worker thread."""

    def __init__(self, settings: Settings, default: Any, max_loaded: int | None = None) -> None:
        self._settings = settings
        self._default = default
        self.max_loaded = max_loaded or settings.source_max_loaded
        self._cache: OrderedDict[SourceSpec, Any] = OrderedDict()
        self._lock = threading.Lock()  # guards _cache only
        self._build_lock = threading.Lock()  # one load at a time (onnxruntime / espeak are not re-entrant-friendly)

    @property
    def default(self) -> Any:
        return self._default

    def loaded(self) -> list[SourceSpec]:
        with self._lock:
            return list(self._cache)

    # ------------------------------------------------------------------ provisioning

    def prepare(self, spec: SourceSpec) -> None:
        """Startup check for a voice: download/validate Piper files, or probe the upstream."""
        if spec.kind == "wyoming":
            self.get(spec)  # probes the upstream and caches the connection details
            return
        from .model_loader import resolve_piper

        resolve_piper(self._piper_settings(spec))

    # ------------------------------------------------------------------ lookup

    def get(self, spec: SourceSpec | None) -> Any:
        """The loaded source for ``spec`` (the global one for ``None``)."""
        if spec is None:
            return self._default
        with self._lock:
            source = self._cache.get(spec)
            if source is not None:
                self._cache.move_to_end(spec)
                return source
        with self._build_lock:
            with self._lock:  # another request may have loaded it while we waited
                source = self._cache.get(spec)
                if source is not None:
                    self._cache.move_to_end(spec)
                    return source
            source = self._build(spec)
            with self._lock:
                self._cache[spec] = source
                while len(self._cache) > self.max_loaded:
                    evicted, _ = self._cache.popitem(last=False)
                    _LOGGER.info("Unloading source %s (SOURCE_MAX_LOADED=%d)", evicted.describe(), self.max_loaded)
            return source

    def unload_all(self) -> list[SourceSpec]:
        """Drop the per-voice Piper voices (idle sleep); they load again on their next use.

        Upstream sources hold no model and stay. Returns the specs dropped.
        """
        with self._build_lock, self._lock:
            dropped = [spec for spec in self._cache if spec.kind == "piper"]
            engines = [self._cache.pop(spec) for spec in dropped]
        for engine in engines:
            unload = getattr(engine, "unload", None)
            if unload is not None:
                unload()
        return dropped

    # ------------------------------------------------------------------ construction

    def _piper_settings(self, spec: SourceSpec) -> Settings:
        """Global settings with this voice's Piper choices applied."""
        s = self._settings
        if spec.piper_model:
            voice_choice: dict[str, Any] = {
                "piper_model": Path(spec.piper_model),
                "piper_config": Path(spec.piper_config) if spec.piper_config else None,
            }
        elif spec.piper_voice:
            # A named voice must not be shadowed by a global PIPER_MODEL.
            voice_choice = {"piper_voice": spec.piper_voice, "piper_model": None, "piper_config": None}
        else:
            voice_choice = {}  # only synthesis options differ: reuse the global voice
        return replace(
            s,
            **voice_choice,
            piper_speaker_id=_first(spec.piper_speaker_id, s.piper_speaker_id),
            piper_length_scale=_first(spec.piper_length_scale, s.piper_length_scale),
            piper_noise_scale=_first(spec.piper_noise_scale, s.piper_noise_scale),
            piper_noise_w_scale=_first(spec.piper_noise_w_scale, s.piper_noise_w_scale),
        )

    def _build(self, spec: SourceSpec) -> Any:
        s = self._settings
        if spec.kind == "wyoming":
            from .wyoming_source import WyomingSource

            host, port = replace(s, wyoming_upstream=spec.upstream or s.wyoming_upstream).upstream_address()
            _LOGGER.info("Connecting to upstream TTS %s:%d for a voice...", host, port)
            source = WyomingSource(
                host,
                port,
                spec.upstream_voice if spec.upstream_voice else (None if spec.upstream else s.upstream_voice),
                spec.upstream_speaker if spec.upstream_speaker else (None if spec.upstream else s.upstream_speaker),
                s.upstream_timeout_s,
            )
            source.load()
            return source

        from .model_loader import resolve_piper
        from .piper_engine import PiperEngine, PiperOptions

        ps = self._piper_settings(spec)
        files = resolve_piper(ps)
        engine = PiperEngine(
            files.model,
            files.config,
            PiperOptions(
                use_cuda=ps.piper_use_cuda,
                length_scale=ps.piper_length_scale,
                noise_scale=ps.piper_noise_scale,
                noise_w_scale=ps.piper_noise_w_scale,
                speaker_id=ps.piper_speaker_id,
            ),
        )
        engine.load()
        _LOGGER.info("Piper voice %s ready for a voice (%d Hz, %s)", files.voice, engine.sample_rate, engine.device)
        return engine
