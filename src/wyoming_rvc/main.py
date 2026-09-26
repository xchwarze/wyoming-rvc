"""Entry point: ``python -m wyoming_rvc.main``."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import signal
import sys
import time
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


class Service:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.state = ServiceState(max_text_chars=settings.max_text_chars)
        self.torch_facts: dict[str, Any] = {}
        self.source = None
        self.source_name = ""
        self.rvc = None
        self.rvc_source = ""
        self.warmup_ms: float | None = None
        self.wyoming = None
        self.http: UvicornServer | None = None
        self.state.info = self.info
        self.state.reset_peak_memory = self._reset_peak

    def info(self) -> dict[str, Any]:
        s = self.settings
        rvc_info = self.rvc.info.__dict__ if self.rvc is not None and self.rvc.info else None
        loaded = self.source is not None and self.state.stage not in ("starting", "loading-source")
        return {
            "version": __version__,
            "platform": self.torch_facts,
            "source": {
                "type": s.source,
                "name": self.source_name,
                "sample_rate": self.source.sample_rate if loaded else None,
                "device": self.source.device if self.source is not None else None,
            },
            "rvc": {"source": self.rvc_source, **(rvc_info or {})} if s.rvc_enabled else None,
            "defaults": self.state.pipeline.defaults.__dict__ if self.state.pipeline else None,
            "wyoming": {
                "port": s.wyoming_port,
                "program": s.program_name,
                "voice": s.voice_name,
                "language": s.voice_language,
                "streaming": s.wyoming_streaming,
            },
            "warmup_ms": self.warmup_ms,
            "gpu_memory": self.rvc.memory_stats() if self.rvc is not None else {},
        }

    def _reset_peak(self) -> None:
        if self.rvc is not None:
            self.rvc.reset_peak_memory()

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
        """Blocking: resolve/download and load every model once."""
        from .model_loader import resolve_rvc, resolve_rvc_assets

        s = self.settings
        self.load_source()
        if not s.rvc_enabled:
            _LOGGER.warning("RVC_ENABLED=false: serving the TTS source unchanged")
            return

        from .rvc_engine import RvcEngine

        self.state.stage = "loading-rvc"
        _LOGGER.info("Loading RVC...")
        rvc_files = resolve_rvc(s)
        _LOGGER.info("RVC model: %s", rvc_files.model)
        _LOGGER.info("RVC index: %s", rvc_files.index or "none")
        assets = resolve_rvc_assets(s)
        _LOGGER.info("Loading RMVPE / ContentVec...")
        self.rvc = RvcEngine(device=s.device)
        info = self.rvc.load(rvc_files, assets)
        self.rvc_source = rvc_files.source
        _LOGGER.info(
            "RVC ready: name=%s version=%s sr=%d f0=%s vocoder=%s epoch=%s device=%s",
            info.model_name,
            info.version,
            info.sample_rate,
            info.f0,
            info.vocoder,
            info.epoch,
            info.device,
            extra={"fields": info.__dict__},
        )

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
            load = asyncio.create_task(asyncio.to_thread(self.load_models), name="load")
            done, _ = await asyncio.wait({load, asyncio.create_task(stop.wait())}, return_when=asyncio.FIRST_COMPLETED)
            if load not in done:
                _LOGGER.warning("Shutdown requested during model loading")
                return
            load.result()

            from .pipeline import SynthesisOptions, TtsPipeline
            from .wyoming_server import WyomingService, build_info

            pipeline = TtsPipeline(
                source=self.source,
                rvc=self.rvc,
                defaults=SynthesisOptions.from_settings(s),
                rvc_concurrency=s.rvc_concurrency,
                sentence_silence_ms=s.sentence_silence_ms,
            )
            self.state.pipeline = pipeline
            self.state.stage = "warmup"
            _LOGGER.info("Running warmup...")
            self.warmup_ms = await pipeline.warmup(s.warmup_text)
            _LOGGER.info("Warmup completed in %.0f ms", self.warmup_ms)
            if self.rvc is not None:
                self.rvc.reset_peak_memory()

            description = (
                f"{s.voice_name} (RVC {self.rvc.info.model_name})" if self.rvc and self.rvc.info else self.source_name
            )
            info = build_info(s, description)
            self.wyoming = WyomingService(s, self.state, lambda: info)
            await self.wyoming.start()

            self.state.ready = True
            self.state.stage = "ready"
            _LOGGER.info("Service ready")
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
    try:
        settings = Settings.from_env()
    except ConfigError as err:
        setup_logging("INFO", "text")
        _LOGGER.error("Invalid configuration: %s", err)
        sys.exit(2)
    setup_logging(settings.log_level, settings.log_format)
    _LOGGER.info("Loading configuration... wyoming-rvc %s", __version__)
    _LOGGER.info("Settings: %s", settings.summary(), extra={"fields": {"settings": settings.summary()}})
    # Caches follow MODELS_DIR; set before torch/onnxruntime/huggingface_hub are imported.
    os.environ.setdefault("HF_HOME", str(settings.hf_home))
    os.environ.setdefault("CUDA_CACHE_PATH", str(settings.models_dir / "cuda-cache"))
    os.environ.setdefault("CUDA_CACHE_MAXSIZE", str(4 << 30))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        asyncio.run(Service(settings).run())
    except Exception:
        _LOGGER.exception("Fatal error after %.0f ms", now_ms() - started)
        sys.exit(1)


if __name__ == "__main__":
    run()
