"""HTTP API for diagnostics and benchmarks. Home Assistant uses Wyoming instead."""

from __future__ import annotations

import contextlib
import json
import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .audio import streaming_wav_header, wav_bytes
from .metrics import Stopwatch, SynthesisMetrics
from .pipeline import SynthesisOptions, TtsPipeline

_LOGGER = logging.getLogger(__name__)

MAX_BODY_BYTES = 1 << 20  # JSON requests only; rejects oversized bodies before buffering them


@dataclass
class ServiceState:
    """Mutable service status shared by both front-ends."""

    pipeline: TtsPipeline | None = None
    ready: bool = False
    stage: str = "starting"
    max_text_chars: int = 5000
    info: Callable[[], dict[str, Any]] = field(default=lambda: {})
    reset_peak_memory: Callable[[], None] = field(default=lambda: None)


class TtsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    pitch: int | None = Field(default=None, ge=-24, le=24)
    index_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    protect: float | None = Field(default=None, ge=0.0, le=0.5)
    f0_method: Literal["rmvpe"] | None = None
    mode: Literal["whole", "sentence"] | None = None
    disable_rvc: bool = False


def create_app(state: ServiceState) -> FastAPI:
    app = FastAPI(title="wyoming-rvc", docs_url="/docs", redoc_url=None)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz() -> JSONResponse:
        status = 200 if state.ready else 503
        return JSONResponse({"ready": state.ready, "stage": state.stage}, status_code=status)

    @app.get("/info")
    async def info() -> dict[str, Any]:
        return {"ready": state.ready, "stage": state.stage, **state.info()}

    @app.post("/v1/metrics/reset")
    async def reset_metrics() -> dict[str, str]:
        state.reset_peak_memory()
        return {"status": "ok"}

    @app.post("/v1/tts")
    async def tts(request: Request) -> Response:
        pipeline, text, options = _prepare(state, await _parse_body(request))
        try:
            result = await pipeline.synthesize(text, options)
            with Stopwatch() as sw:
                payload = wav_bytes(result.pcm, result.sample_rate)
        except Exception as err:  # input was validated in _prepare: anything here is a server error
            _LOGGER.exception("Synthesis failed")
            raise HTTPException(status_code=500, detail=f"Synthesis failed: {err}") from err
        metrics = result.metrics
        metrics.encode_ms += sw.elapsed_ms
        metrics.total_ms += sw.elapsed_ms
        metrics.log(_LOGGER, "http")
        return Response(content=payload, media_type="audio/wav", headers=_metric_headers(metrics))

    @app.post("/v1/tts/stream")
    async def tts_stream(request: Request) -> StreamingResponse:
        """Chunked WAV (header with 0 frames, like Home Assistant's streaming TTS)."""
        pipeline, text, options = _prepare(state, await _parse_body(request))
        rate = pipeline.sample_rate(options)
        metrics = SynthesisMetrics()

        async def body_iter() -> AsyncIterator[bytes]:
            yield streaming_wav_header(rate)
            try:
                async with contextlib.aclosing(pipeline.stream(text, options, metrics)) as pcm_stream:
                    async for chunk in pcm_stream:
                        yield chunk
            except Exception:
                _LOGGER.exception("Streaming synthesis failed")
                raise
            metrics.log(_LOGGER, "http-stream")

        return StreamingResponse(body_iter(), media_type="audio/wav", headers={"X-TTS-Sample-Rate": str(rate)})

    @app.exception_handler(HTTPException)
    async def http_error(_: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse({"error": exc.detail}, status_code=exc.status_code)

    return app


async def _parse_body(request: Request) -> TtsRequest:
    """JSON body; tolerates cp1252 because Windows shells often re-encode curl arguments."""
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > MAX_BODY_BYTES:
        raise HTTPException(status_code=413, detail="request body too large")
    raw = bytearray()
    async for part in request.stream():
        raw += part
        if len(raw) > MAX_BODY_BYTES:
            raise HTTPException(status_code=413, detail="request body too large")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("cp1252", errors="replace")
    try:
        return TtsRequest.model_validate(json.loads(text))
    except json.JSONDecodeError as err:
        raise HTTPException(status_code=422, detail=f"invalid JSON: {err}") from err
    except ValidationError as err:
        detail = "; ".join(f"{'.'.join(map(str, e['loc'])) or 'body'}: {e['msg']}" for e in err.errors())
        raise HTTPException(status_code=422, detail=detail) from err


def _prepare(state: ServiceState, body: TtsRequest) -> tuple[TtsPipeline, str, SynthesisOptions]:
    if not state.ready or state.pipeline is None:
        raise HTTPException(status_code=503, detail=f"Service not ready (stage={state.stage})")
    text = " ".join(body.text.split())
    if not text:
        raise HTTPException(status_code=422, detail="text is empty")
    if len(text) > state.max_text_chars:
        raise HTTPException(status_code=413, detail=f"text longer than {state.max_text_chars} characters")
    try:
        options = state.pipeline.options(
            pitch=body.pitch,
            index_rate=body.index_rate,
            protect=body.protect,
            f0_method=body.f0_method,
            mode=body.mode,
            rvc=False if body.disable_rvc else None,
        )
    except ValueError as err:
        raise HTTPException(status_code=422, detail=str(err)) from err
    return state.pipeline, text, options


def _metric_headers(metrics: SynthesisMetrics) -> dict[str, str]:
    return {f"X-TTS-{key.replace('_', '-').title()}": str(value) for key, value in metrics.as_dict().items()}


class UvicornServer:
    """uvicorn inside our event loop, without uvicorn's own signal handling."""

    def __init__(self, app: FastAPI, host: str, port: int) -> None:
        import uvicorn

        config = uvicorn.Config(app, host=host, port=port, log_config=None, access_log=False, lifespan="off")

        class _Server(uvicorn.Server):
            @contextlib.contextmanager
            def capture_signals(self):  # type: ignore[override]
                yield

        self._server = _Server(config)

    async def serve(self) -> None:
        await self._server.serve()

    @property
    def started(self) -> bool:
        return bool(self._server.started)

    def stop(self) -> None:
        self._server.should_exit = True
