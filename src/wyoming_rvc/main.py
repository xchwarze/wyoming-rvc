"""Entry point: ``python -m wyoming_rvc.main``."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import logging
import os
import signal
import sys
import threading
import time
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

from . import __version__
from .config import ConfigError, Settings
from .http_server import ServiceState, UvicornServer, create_app
from .metrics import now_ms

_LOGGER = logging.getLogger("wyoming_rvc")


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        data.update(getattr(record, "fields", {}) or {})
        if record.exc_info:
            data["exc"] = self.formatException(record.exc_info)
        return json.dumps(data, ensure_ascii=False, default=str)


def setup_logging(level: str, fmt: str) -> None:
    handler = logging.StreamHandler(sys.stdout)
    if fmt == "json":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    for noisy in ("httpx", "httpcore", "urllib3", "filelock", "faiss", "faiss.loader", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def report_torch(settings: Settings) -> dict[str, Any]:
    """Log torch/CUDA facts; fail if CUDA was required but is missing."""
    import torch

    cuda = torch.cuda.is_available()
    facts: dict[str, Any] = {
        "torch": torch.__version__,
        "cuda_available": cuda,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if cuda else None,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "gpu_capability": ".".join(map(str, torch.cuda.get_device_capability(0))) if cuda else None,
        "gpu_total_mb": round(torch.cuda.get_device_properties(0).total_memory / 2**20) if cuda else None,
        "rvc_device": settings.device,
        "source": settings.source,
        "piper_device": ("cuda" if settings.piper_use_cuda else "cpu") if settings.source == "piper" else None,
    }
    _LOGGER.info(
        "torch=%s cuda_available=%s cuda_runtime=%s gpu=%s capability=%s rvc_device=%s source=%s piper_device=%s",
        facts["torch"],
        cuda,
        facts["cuda_runtime"],
        facts["gpu"],
        facts["gpu_capability"],
        facts["rvc_device"],
        facts["source"],
        facts["piper_device"],
        extra={"fields": facts},
    )
    if settings.device == "cuda":
        if not cuda:
            raise RuntimeError(
                "DEVICE=cuda but torch.cuda.is_available() is False. Check the NVIDIA driver, the NVIDIA "
                "Container Toolkit and that the container runs with GPU access (--gpus all). "
                "Use DEVICE=cpu for debugging."
            )
        arch = f"sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}"
        if arch not in torch.cuda.get_arch_list():
            # Not fatal: torch may still run it through PTX JIT; the kernel launch below decides.
            _LOGGER.warning(
                "No native kernels for %s (%s) in this torch build %s; relying on PTX JIT",
                facts["gpu"],
                arch,
                torch.cuda.get_arch_list(),
            )
        torch.zeros(1, device="cuda").sum().item()  # fail now, not on the first request
    return facts


async def run_in_daemon_thread(fn: Callable[[], Any]) -> Any:
    """Like ``asyncio.to_thread`` but on a daemon thread.

    Model loading (downloads included) cannot be interrupted; a daemon thread lets
    SIGTERM end the process at once instead of joining the executor at shutdown.
    """
    loop = asyncio.get_running_loop()
    future: asyncio.Future = loop.create_future()

    def deliver(setter: Callable[[Any], None], value: Any) -> None:
        if not future.done():
            setter(value)

    def target() -> None:
        try:
            result = fn()
        except BaseException as err:  # noqa: BLE001 - re-raised in the awaiting task
            outcome = (future.set_exception, err)
        else:
            outcome = (future.set_result, result)
        with contextlib.suppress(RuntimeError):  # loop already closed after shutdown
            loop.call_soon_threadsafe(deliver, *outcome)

    threading.Thread(target=target, name="load-models", daemon=True).start()
    return await future


def resolve_device(settings: Settings, cuda_available: Callable[[], bool]) -> Settings:
    """Turn DEVICE=auto into cuda or cpu (PIPER_DEVICE=auto then follows it)."""
    if settings.device != "auto":
        return settings
    device = "cuda" if cuda_available() else "cpu"
    _LOGGER.info("DEVICE=auto resolved to %s", device)
    return dataclasses.replace(settings, device=device)


class Service:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.state = ServiceState(max_text_chars=settings.max_text_chars)
        self.torch_facts: dict[str, Any] = {}
        self.source = None
        self.source_name = ""
        self.voices = None  # VoiceManager when RVC is enabled
        self.warmup_ms: float | None = None
        self.wyoming = None
        self.http: UvicornServer | None = None
        self.state.info = self.info
        self.state.reset_peak_memory = self._reset_peak

    def info(self) -> dict[str, Any]:
        s = self.settings
        loaded = self.source is not None and self.state.stage not in ("starting", "loading-source")
        rvc = None
        if s.rvc_enabled and self.voices is not None:
            rvc = {
                "default_voice": self.voices.default_voice,
                "max_loaded_models": self.voices.max_loaded,
                "voices": self.voices.list_voices(),
                "loaded": {vid: e.info.__dict__ for vid, e in self.voices.loaded_engines().items() if e.info},
            }
        return {
            "version": __version__,
            "platform": self.torch_facts,
            "source": {
                "type": s.source,
                "name": self.source_name,
                "sample_rate": self.source.sample_rate if loaded else None,
                "device": self.source.device if self.source is not None else None,
            },
            "rvc": rvc,
            "defaults": self.state.pipeline.defaults.__dict__ if self.state.pipeline else None,
            "wyoming": {"port": s.wyoming_port, "program": s.program_name, "streaming": s.wyoming_streaming},
            "warmup_ms": self.warmup_ms,
            "gpu_memory": self._gpu_memory(),
        }

    def _gpu_memory(self) -> dict[str, float]:
        if not self.settings.rvc_enabled or self.voices is None:
            return {}
        from .rvc_engine import gpu_memory_stats

        return gpu_memory_stats()

    def _reset_peak(self) -> None:
        if self.voices is not None:
            from .rvc_engine import reset_gpu_peak

            reset_gpu_peak()

    def load_source(self) -> None:
        s = self.settings
        self.state.stage = "loading-source"
        if s.source == "wyoming":
            from .wyoming_source import WyomingSource

            host, port = s.upstream_address()
            _LOGGER.info("Connecting to upstream TTS %s:%d...", host, port)
            source = WyomingSource(host, port, s.upstream_voice, s.upstream_speaker, s.upstream_timeout_s)
            source.load()
            self.source, self.source_name = source, f"{source.program} @ {host}:{port}"
            return

        from .model_loader import resolve_piper
        from .piper_engine import PiperEngine, PiperOptions

        _LOGGER.info("Loading Piper...")
        piper_files = resolve_piper(s)
        piper = PiperEngine(
            piper_files.model,
            piper_files.config,
            PiperOptions(
                use_cuda=s.piper_use_cuda,
                length_scale=s.piper_length_scale,
                noise_scale=s.piper_noise_scale,
                noise_w_scale=s.piper_noise_w_scale,
                speaker_id=s.piper_speaker_id,
            ),
        )
        piper.load()
        self.source, self.source_name = piper, piper_files.voice
        _LOGGER.info("Piper ready: %s (%d Hz, %s)", piper_files.voice, piper.sample_rate, piper.device)

    def load_models(self) -> None:
        """Blocking: load the source and shared RVC models, provision every voice."""
        from .model_loader import resolve_rvc_assets, resolve_rvc_files
        from .voices import VoiceManager, load_voice_configs

        s = self.settings
        self.load_source()
        if not s.rvc_enabled:
            _LOGGER.warning("RVC_ENABLED=false: serving the TTS source unchanged")
            return

        from .rvc_engine import RvcEngine, SharedModels, checkpoint_sample_rate

        self.state.stage = "loading-rvc"
        configs, default_voice = load_voice_configs(s)
        _LOGGER.info("Voices configured: %s (default %s)", [c.id for c in configs if c.enabled], default_voice)
        _LOGGER.info("Loading RMVPE / ContentVec (shared by all voices)...")
        shared = SharedModels(s.device, resolve_rvc_assets(s))

        def provision(voice):
            files = resolve_rvc_files(
                voice.repo_id, voice.revision, voice.model_file, voice.index_file, s.rvc_data_dir, f"voice {voice.id}"
            )
            _LOGGER.info("Voice %s: model=%s index=%s", voice.id, files.model, files.index or "none")
            return files, checkpoint_sample_rate(files.model)

        def load(voice, files):
            engine = RvcEngine(shared)
            info = engine.load(files)
            _LOGGER.info(
                "RVC voice %s ready: model=%s version=%s sr=%d f0=%s vocoder=%s epoch=%s device=%s",
                voice.id,
                info.model_name,
                info.version,
                info.sample_rate,
                info.f0,
                info.vocoder,
                info.epoch,
                info.device,
                extra={"fields": {"voice": voice.id, **info.__dict__}},
            )
            return engine

        manager = VoiceManager(configs, default_voice, provision, load, s.rvc_max_loaded_models)
        manager.provision_all()
        manager.resolve(default_voice)  # the default voice must be usable: fail fast otherwise
        self.voices = manager

    async def run(self) -> None:
        s = self.settings
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            with contextlib.suppress(NotImplementedError, RuntimeError):
                loop.add_signal_handler(sig, stop.set)

        tasks: list[asyncio.Task] = []
        if s.http_enabled:
            self.http = UvicornServer(create_app(self.state), s.http_host, s.http_port)
            tasks.append(asyncio.create_task(self.http.serve(), name="http"))
            _LOGGER.info("HTTP listening on %s:%s (ready after model load)", s.http_host, s.http_port)

        try:
            self.torch_facts = await asyncio.to_thread(report_torch, s)
            load = asyncio.create_task(run_in_daemon_thread(self.load_models), name="load")
            done, _ = await asyncio.wait({load, asyncio.create_task(stop.wait())}, return_when=asyncio.FIRST_COMPLETED)
            if load not in done:
                _LOGGER.warning("Shutdown requested during model loading")
                return
            load.result()

            from .pipeline import SynthesisOptions, TtsPipeline
            from .wyoming_server import WyomingService, build_info

            pipeline = TtsPipeline(
                source=self.source,
                rvc=self.voices,
                defaults=SynthesisOptions.from_settings(s),
                rvc_concurrency=s.rvc_concurrency,
                sentence_silence_ms=s.sentence_silence_ms,
            )
            self.state.pipeline = pipeline
            self.state.stage = "warmup"
            _LOGGER.info("Running warmup...")
            started = now_ms()
            preloaded = await self.voices.preload() if self.voices is not None else []
            for voice in preloaded:  # warm every resident voice
                await pipeline.warmup(s.warmup_text, voice)
            if not preloaded:  # no resident RVC voice: at least warm the TTS source
                await pipeline.synthesize(s.warmup_text, pipeline.options(rvc=False))
            self.warmup_ms = now_ms() - started
            _LOGGER.info("Warmup completed in %.0f ms (preloaded voices: %s)", self.warmup_ms, preloaded or "none")
            self._reset_peak()

            if self.voices is not None:
                advertised = self.voices.installed_voices()
            else:
                advertised = [SimpleNamespace(id=s.voice_name, name=self.source_name, language=s.voice_language)]
            info = build_info(s, advertised)
            self.wyoming = WyomingService(s, self.state, lambda: info)
            await self.wyoming.start()

            self.state.ready = True
            self.state.stage = "ready"
            _LOGGER.info("Service ready")
            from .addon import announce_wyoming

            await asyncio.to_thread(announce_wyoming, s.wyoming_port, os.environ)
            await stop.wait()
        finally:
            _LOGGER.info("Shutting down...")
            self.state.ready = False
            self.state.stage = "stopping"
            if self.wyoming is not None:
                await self.wyoming.stop()
            if self.http is not None:
                self.http.stop()
            if tasks:
                await asyncio.wait(tasks, timeout=10)
            _LOGGER.info("Stopped")


def run() -> None:
    started = now_ms()
    from .addon import apply_options

    addon = apply_options(os.environ)  # Home Assistant add-on options, if running as one
    try:
        settings = Settings.from_env()
    except ConfigError as err:
        setup_logging("INFO", "text")
        _LOGGER.error("Invalid configuration: %s", err)
        sys.exit(2)
    setup_logging(settings.log_level, settings.log_format)
    _LOGGER.info("Loading configuration... wyoming-rvc %s%s", __version__, " (Home Assistant add-on)" if addon else "")
    _LOGGER.info("Settings: %s", settings.summary(), extra={"fields": {"settings": settings.summary()}})
    # Caches follow MODELS_DIR; set before torch/onnxruntime/huggingface_hub are imported.
    os.environ.setdefault("HF_HOME", str(settings.hf_home))
    os.environ.setdefault("CUDA_CACHE_PATH", str(settings.models_dir / "cuda-cache"))
    os.environ.setdefault("CUDA_CACHE_MAXSIZE", str(4 << 30))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        if settings.device == "auto":
            import torch

            settings = resolve_device(settings, torch.cuda.is_available)
        asyncio.run(Service(settings).run())
    except Exception:
        _LOGGER.exception("Fatal error after %.0f ms", now_ms() - started)
        sys.exit(1)


if __name__ == "__main__":
    run()
