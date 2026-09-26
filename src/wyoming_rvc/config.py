"""Environment-driven configuration."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

F0_METHODS: tuple[str, ...] = ("rmvpe",)
RVC_MODES: tuple[str, ...] = ("whole", "sentence")
SOURCES: tuple[str, ...] = ("piper", "wyoming")
PIPER_DEVICES: tuple[str, ...] = ("auto", "cuda", "cpu")
DEVICES: tuple[str, ...] = ("cuda", "cpu")


class ConfigError(ValueError):
    """Raised when the environment contains an invalid setting."""


@dataclass(frozen=True)
class Settings:
    """All runtime settings. Build with ``Settings.from_env()``."""

    device: str = "cuda"

    http_enabled: bool = True
    http_host: str = "0.0.0.0"
    http_port: int = 8080

    wyoming_host: str = "0.0.0.0"
    wyoming_port: int = 10200
    wyoming_streaming: bool = True
    wyoming_zeroconf: bool = False
    wyoming_zeroconf_name: str | None = None
    wyoming_samples_per_chunk: int = 1024

    models_dir: Path = Path("/models")
    hf_home: Path = Path("/models/huggingface")

    piper_voice: str = "en_US-ljspeech-high"
    piper_model: Path | None = None
    piper_config: Path | None = None
    piper_data_dir: Path = Path("/models/piper")
    piper_voices_repo_id: str = "rhasspy/piper-voices"
    piper_device: str = "auto"
    piper_length_scale: float | None = None
    piper_noise_scale: float | None = None
    piper_noise_w_scale: float | None = None
    piper_speaker_id: int | None = None
    sentence_silence_ms: int = 0

    rvc_enabled: bool = True
    rvc_repo_id: str = "Slichi/KasaneTeto"
    rvc_revision: str | None = None
    rvc_model_file: str | None = None
    rvc_index_file: str | None = None
    rvc_data_dir: Path = Path("/models/rvc")
    rvc_assets_repo_id: str = "IAHispano/Applio"
    rvc_assets_revision: str | None = None
    rvc_f0_method: str = "rmvpe"
    rvc_pitch: int = 0
    rvc_index_rate: float = 0.6
    rvc_protect: float = 0.33
    rvc_mode: str = "sentence"
    rvc_concurrency: int = 1

    warmup_text: str = "System ready."
    max_text_chars: int = 5000

    source: str = "piper"
    wyoming_upstream: str | None = None
    upstream_voice: str | None = None
    upstream_speaker: str | None = None
    upstream_timeout_s: float = 30.0

    program_name: str = "Wyoming RVC"
    voice_name: str = "teto"
    voice_language: str = "en"

    log_level: str = "INFO"
    log_format: str = "text"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        """Parse and validate settings from environment variables."""
        env = os.environ if env is None else env
        r = _Reader(env)

        models_dir = r.path("MODELS_DIR", Path("/models"))
        settings = cls(
            device=r.choice("DEVICE", "cuda", DEVICES),
            http_enabled=r.bool("HTTP_ENABLED", True),
            http_host=r.str("HTTP_HOST", "0.0.0.0"),
            http_port=r.port("HTTP_PORT", 8080),
            wyoming_host=r.str("WYOMING_HOST", "0.0.0.0"),
            wyoming_port=r.port("WYOMING_PORT", 10200),
            wyoming_streaming=r.bool("WYOMING_STREAMING", True),
            wyoming_zeroconf=r.bool("WYOMING_ZEROCONF", False),
            wyoming_zeroconf_name=r.opt_str("WYOMING_ZEROCONF_NAME"),
            wyoming_samples_per_chunk=r.int("WYOMING_SAMPLES_PER_CHUNK", 1024, lo=64, hi=65536),
            models_dir=models_dir,
            hf_home=r.path("HF_HOME", models_dir / "huggingface"),
            piper_voice=r.str("PIPER_VOICE", "en_US-ljspeech-high"),
            piper_model=r.opt_path("PIPER_MODEL"),
            piper_config=r.opt_path("PIPER_CONFIG"),
            piper_data_dir=r.path("PIPER_DATA_DIR", models_dir / "piper"),
            piper_voices_repo_id=r.str("PIPER_VOICES_REPO_ID", "rhasspy/piper-voices"),
            piper_device=r.choice("PIPER_DEVICE", "auto", PIPER_DEVICES),
            piper_length_scale=r.opt_float("PIPER_LENGTH_SCALE", lo=0.1, hi=5.0),
            piper_noise_scale=r.opt_float("PIPER_NOISE_SCALE", lo=0.0, hi=2.0),
            piper_noise_w_scale=r.opt_float("PIPER_NOISE_W_SCALE", lo=0.0, hi=2.0),
            piper_speaker_id=r.opt_int("PIPER_SPEAKER_ID", lo=0),
            sentence_silence_ms=r.int("SENTENCE_SILENCE_MS", 0, lo=0, hi=5000),
            rvc_enabled=r.bool("RVC_ENABLED", True),
            rvc_repo_id=r.str("RVC_REPO_ID", "Slichi/KasaneTeto"),
            rvc_revision=r.opt_str("RVC_REVISION"),
            rvc_model_file=r.opt_str("RVC_MODEL_FILE"),
            rvc_index_file=r.opt_str("RVC_INDEX_FILE"),
            rvc_data_dir=r.path("RVC_DATA_DIR", models_dir / "rvc"),
            rvc_assets_repo_id=r.str("RVC_ASSETS_REPO_ID", "IAHispano/Applio"),
            rvc_assets_revision=r.opt_str("RVC_ASSETS_REVISION"),
            rvc_f0_method=r.choice("RVC_F0_METHOD", "rmvpe", F0_METHODS),
            rvc_pitch=r.int("RVC_PITCH", 0, lo=-24, hi=24),
            rvc_index_rate=r.float("RVC_INDEX_RATE", 0.6, lo=0.0, hi=1.0),
            rvc_protect=r.float("RVC_PROTECT", 0.33, lo=0.0, hi=0.5),
            rvc_mode=r.choice("RVC_MODE", "sentence", RVC_MODES),
            rvc_concurrency=r.int("RVC_CONCURRENCY", 1, lo=1, hi=8),
            warmup_text=r.str("WARMUP_TEXT", "System ready."),
            max_text_chars=r.int("MAX_TEXT_CHARS", 5000, lo=1, hi=100000),
            source=r.choice("SOURCE", "piper", SOURCES),
            wyoming_upstream=r.opt_str("WYOMING_UPSTREAM"),
            upstream_voice=r.opt_str("UPSTREAM_VOICE"),
            upstream_speaker=r.opt_str("UPSTREAM_SPEAKER"),
            upstream_timeout_s=r.float("UPSTREAM_TIMEOUT_S", 30.0, lo=1.0, hi=600.0),
            program_name=r.str("WYOMING_PROGRAM_NAME", "Wyoming RVC"),
            voice_name=r.str("VOICE_NAME", "teto"),
            voice_language=r.str("VOICE_LANGUAGE", "en"),
            log_level=r.choice("LOG_LEVEL", "INFO", ("DEBUG", "INFO", "WARNING", "ERROR"), upper=True),
            log_format=r.choice("LOG_FORMAT", "text", ("text", "json")),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        """Cross-field validation."""
        if self.piper_config is not None and self.piper_model is None:
            raise ConfigError("PIPER_CONFIG requires PIPER_MODEL")
        if self.source == "wyoming":
            self.upstream_address()  # raises ConfigError when missing or malformed
        if self.http_enabled and self.http_port == self.wyoming_port:
            raise ConfigError("HTTP_PORT and WYOMING_PORT must differ")

    @property
    def piper_use_cuda(self) -> bool:
        """Resolve PIPER_DEVICE=auto: Piper follows DEVICE."""
        return self.piper_device == "cuda" or (self.piper_device == "auto" and self.device == "cuda")

    def upstream_address(self) -> tuple[str, int]:
        """Parse WYOMING_UPSTREAM (``tcp://host:port`` or ``host:port``)."""
        from urllib.parse import urlparse

        raw = self.wyoming_upstream
        if not raw:
            raise ConfigError("SOURCE=wyoming requires WYOMING_UPSTREAM, e.g. tcp://wyoming-piper:10200")
        parsed = urlparse(raw if "://" in raw else f"tcp://{raw}")
        try:
            port = parsed.port
        except ValueError as err:
            raise ConfigError(f"WYOMING_UPSTREAM has an invalid port: {raw!r}") from err
        if parsed.scheme != "tcp" or not parsed.hostname or port is None:
            raise ConfigError(f"WYOMING_UPSTREAM must look like tcp://host:port, got {raw!r}")
        return parsed.hostname, port

    def summary(self) -> dict[str, Any]:
        """JSON-friendly view of the settings."""
        data = asdict(self)
        data["piper_use_cuda"] = self.piper_use_cuda
        return {k: (str(v) if isinstance(v, Path) else v) for k, v in data.items()}


class _Reader:
    """Typed accessors over an environment mapping. Empty strings count as unset."""

    def __init__(self, env: Mapping[str, str]) -> None:
        self._env = env

    def _raw(self, key: str) -> str | None:
        value = self._env.get(key)
        if value is None:
            return None
        value = value.strip()
        return value or None

    def str(self, key: str, default: str) -> str:
        return self._raw(key) or default

    def opt_str(self, key: str) -> str | None:
        return self._raw(key)

    def path(self, key: str, default: Path) -> Path:
        raw = self._raw(key)
        return Path(raw) if raw else default

    def opt_path(self, key: str) -> Path | None:
        raw = self._raw(key)
        return Path(raw) if raw else None

    def bool(self, key: str, default: bool) -> bool:
        raw = self._raw(key)
        if raw is None:
            return default
        lowered = raw.lower()
        if lowered in ("1", "true", "yes", "on"):
            return True
        if lowered in ("0", "false", "no", "off"):
            return False
        raise ConfigError(f"{key} must be a boolean, got {raw!r}")

    def int(self, key: str, default: int, lo: int | None = None, hi: int | None = None) -> int:
        raw = self._raw(key)
        value = default if raw is None else self._parse(key, raw, int)
        return self._bounds(key, value, lo, hi)

    def opt_int(self, key: str, lo: int | None = None, hi: int | None = None) -> int | None:
        raw = self._raw(key)
        return None if raw is None else self._bounds(key, self._parse(key, raw, int), lo, hi)

    def float(self, key: str, default: float, lo: float | None = None, hi: float | None = None) -> float:
        raw = self._raw(key)
        value = default if raw is None else self._parse(key, raw, float)
        return self._bounds(key, value, lo, hi)

    def opt_float(self, key: str, lo: float | None = None, hi: float | None = None) -> float | None:
        raw = self._raw(key)
        return None if raw is None else self._bounds(key, self._parse(key, raw, float), lo, hi)

    def port(self, key: str, default: int) -> int:
        return self.int(key, default, lo=1, hi=65535)

    def choice(self, key: str, default: str, choices: tuple[str, ...], upper: bool = False) -> str:
        raw = self._raw(key)
        value = default if raw is None else (raw.upper() if upper else raw.lower())
        if value not in choices:
            raise ConfigError(f"{key} must be one of {', '.join(choices)}; got {raw!r}")
        return value

    @staticmethod
    def _parse(key: str, raw: str, kind: type) -> Any:
        try:
            return kind(raw)
        except ValueError as err:
            raise ConfigError(f"{key} must be a {kind.__name__}, got {raw!r}") from err

    @staticmethod
    def _bounds(key: str, value: Any, lo: Any, hi: Any) -> Any:
        if lo is not None and value < lo:
            raise ConfigError(f"{key} must be >= {lo}, got {value}")
        if hi is not None and value > hi:
            raise ConfigError(f"{key} must be <= {hi}, got {value}")
        return value
