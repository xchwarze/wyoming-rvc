"""End-to-end with real models. Opt-in: WYOMING_RVC_INTEGRATION=1 (downloads ~1 GB on first run).

Uses the regular environment variables (MODELS_DIR, DEVICE, RVC_REPO_ID, ...).
"""

import os

import numpy as np
import pytest

from wyoming_rvc.audio import pcm16_to_float
from wyoming_rvc.config import Settings

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.environ.get("WYOMING_RVC_INTEGRATION") != "1", reason="set WYOMING_RVC_INTEGRATION=1"),
]


@pytest.fixture(scope="module")
def service():
    from wyoming_rvc.main import Service, report_torch

    settings = Settings.from_env()
    svc = Service(settings)
    svc.torch_facts = report_torch(settings)
    svc.load_models()
    return svc


@pytest.fixture
def pipeline(service):
    from wyoming_rvc.pipeline import SynthesisOptions, TtsPipeline
    from wyoming_rvc.rvc_engine import StreamParams

    s = service.settings
    return TtsPipeline(
        service.source,
        service.rvc,
        SynthesisOptions.from_settings(s),
        StreamParams(s.stream_chunk_ms, s.stream_context_ms, s.stream_overlap_ms),
    )


@pytest.mark.parametrize("mode", ["whole", "sentence", "stream"])
async def test_real_synthesis(pipeline, mode):
    from dataclasses import replace

    text = "Hola. Soy HAL. Todos los sistemas están funcionando correctamente."
    result = await pipeline.synthesize(text, replace(pipeline.defaults, mode=mode))
    audio = pcm16_to_float(result.pcm)
    assert result.sample_rate == pipeline.rvc.output_sample_rate
    assert audio.size / result.sample_rate > 2.0
    assert np.sqrt(np.mean(audio**2)) > 0.01, "output is (nearly) silent"
    assert np.max(np.abs(audio)) <= 1.0
    assert result.metrics.rtf < 1.0 or pipeline.rvc.device == "cpu"
