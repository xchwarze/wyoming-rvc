"""RVC_HALF: the whole conversion runs in fp16 and stays close to fp32 (needs torch; skipped otherwise).

Random v2 weights and stand-ins for ContentVec/RMVPE: this checks the dtype plumbing
(generator, features, retrieval, NSF source), not the voice quality.
"""

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from scipy import signal  # noqa: E402

from wyoming_rvc.rvc_engine import INPUT_SR, RvcEngine  # noqa: E402

# Applio's v2 40 kHz generator (the layout of a checkpoint's "config").
# fmt: off
CONFIG = [1025, 32, 192, 192, 768, 2, 6, 3, 0, "1", [3, 7, 11], [[1, 3, 5]] * 3, [10, 10, 2, 2], 512,
          [16, 16, 4, 4], 109, 256, 40000]
# fmt: on


class _Hubert:
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, x):
        assert x.dtype == self.dtype
        frames = torch.randn(1, x.shape[1] // 320, 768, generator=torch.Generator().manual_seed(1))
        return {"last_hidden_state": frames.to(self.dtype)}


class _Rmvpe:
    def infer_from_audio(self, audio, thred):
        return np.full(audio.size // 160 + 1, 220.0, dtype=np.float32)


def _convert(dtype, ckpt, bank):
    shared = SimpleNamespace(
        device="cpu",
        dtype=dtype,
        hubert=_Hubert(dtype),
        rmvpe=_Rmvpe(),
        highpass=signal.butter(N=5, Wn=48, btype="high", fs=INPUT_SR),
    )
    engine = RvcEngine(shared)
    engine._build_generator(ckpt)
    engine._sid = torch.tensor([0])
    engine._bank, engine._bank_sq = bank, (bank * bank).sum(dim=1)
    assert next(engine._net_g.parameters()).dtype == dtype
    audio = (0.3 * np.sin(2 * np.pi * 220 * np.arange(INPUT_SR) / INPUT_SR)).astype(np.float32)
    torch.manual_seed(2)  # the generator samples noise
    out, rate = engine.convert(audio, INPUT_SR, index_rate=0.6, protect=0.33)
    assert out.dtype == np.float32 and rate == 40000 and np.isfinite(out).all()
    return out


def test_fp16_conversion_matches_fp32():
    from wyoming_rvc.vendor.applio.algorithm.synthesizers import Synthesizer

    torch.manual_seed(0)
    net = Synthesizer(*CONFIG, use_f0=True, text_enc_hidden_dim=768, vocoder="HiFi-GAN")
    ckpt = {"config": CONFIG, "weight": net.state_dict(), "f0": 1, "version": "v2"}
    bank = torch.randn(500, 768)
    full, half = _convert(torch.float32, ckpt, bank), _convert(torch.float16, ckpt, bank)
    assert np.corrcoef(full, half)[0, 1] > 0.99
