"""Resident RVC voice conversion engine.

The conversion math follows Applio's ``rvc/infer/pipeline.py`` (MIT, see
``vendor/applio/LICENSE``) but every model is loaded exactly once:
ContentVec, RMVPE, the generator and the index vectors (kept on the device for
exact k-NN retrieval). Applio re-creates RMVPE and re-reads the index per call.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio import prevent_clipping, resample, to_float32
from .model_loader import RvcAssets, RvcFiles

_LOGGER = logging.getLogger(__name__)

INPUT_SR = 16000
WINDOW = 160  # samples per frame at 16 kHz (100 frames/s)
F0_MIN, F0_MAX = 50.0, 1100.0
F0_MEL_MIN = 1127 * np.log(1 + F0_MIN / 700)
F0_MEL_MAX = 1127 * np.log(1 + F0_MAX / 700)
INPUT_PEAK = 0.95
OUTPUT_CEILING = 0.99


class RvcLoadError(RuntimeError):
    """The RVC checkpoint, embedder, pitch model or index could not be loaded."""


@dataclass(frozen=True)
class RvcModelInfo:
    model_path: str
    index_path: str | None
    model_name: str
    version: str
    sample_rate: int
    f0: bool
    embedder: str
    vocoder: str
    speakers: int
    epoch: int | None
    index_vectors: int
    device: str


class RvcEngine:
    """Load once with ``load()``, then call ``convert()``.

    Not internally locked: callers serialize access (see ``TtsPipeline``).
    """

    def __init__(self, device: str = "cuda", x_pad: int = 1, x_query: int = 6, x_center: int = 38, x_max: int = 41):
        self.device = "cuda:0" if device == "cuda" else device
        # Applio's defaults for >=6 GB GPUs: 1 s reflection padding, split audio
        # longer than 41 s at the quietest point near every 38 s.
        self.t_pad = INPUT_SR * x_pad
        self.t_query = INPUT_SR * x_query
        self.t_center = INPUT_SR * x_center
        self.t_max = INPUT_SR * x_max
        self.info: RvcModelInfo | None = None
        self._net_g = None
        self._hubert = None
        self._rmvpe = None
        self._bank = None  # retrieval feature matrix (torch, on device)
        self._bank_sq = None  # its squared norms
        self._version = "v2"
        self._use_f0 = True
        self._tgt_sr = 0
        self._sid = None
        self._highpass: tuple[np.ndarray, np.ndarray] | None = None

    # ------------------------------------------------------------------ loading

    def load(self, files: RvcFiles, assets: RvcAssets) -> RvcModelInfo:
        import torch
        from scipy import signal

        if self.device.startswith("cuda"):
            # Full fp32 matmuls, as in Applio: TF32 would perturb the retrieval distances,
            # which are weighted by 1/d^2 (nearest neighbours dominate).
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.benchmark = False  # input lengths vary per request

        # Same 48 Hz high-pass Applio applies to the 16 kHz input.
        self._highpass = signal.butter(N=5, Wn=48, btype="high", fs=INPUT_SR)

        ckpt = self._load_checkpoint(files.model)
        self._build_generator(ckpt)
        embedder = str(ckpt.get("embedder_model", "contentvec"))
        if embedder != "contentvec":
            raise RvcLoadError(f"Model was trained with embedder {embedder!r}; only 'contentvec' is supported")

        _LOGGER.info("Loading ContentVec from %s", assets.contentvec_dir)
        self._hubert = self._load_hubert(assets.contentvec_dir)

        if self._use_f0:
            _LOGGER.info("Loading RMVPE from %s", assets.rmvpe)
            from .vendor.applio.predictors.rmvpe import RMVPE0Predictor

            self._rmvpe = RMVPE0Predictor(str(assets.rmvpe), device=self.device)

        vectors = self._load_index(files.index) if files.index else 0

        self._sid = torch.tensor([0], device=self.device, dtype=torch.long)

        self.info = RvcModelInfo(
            model_path=str(files.model),
            index_path=str(files.index) if files.index else None,
            model_name=str(ckpt.get("model_name") or files.model.stem),
            version=self._version,
            sample_rate=self._tgt_sr,
            f0=self._use_f0,
            embedder=embedder,
            vocoder=str(ckpt.get("vocoder", "HiFi-GAN")),
            speakers=int(self._net_g.emb_g.weight.shape[0]),
            epoch=int(ckpt["epoch"]) if isinstance(ckpt.get("epoch"), (int, float)) else None,
            index_vectors=vectors,
            device=self.device,
        )
        return self.info

    @staticmethod
    def _load_checkpoint(path: Path) -> dict:
        import torch

        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as err:
            raise RvcLoadError(f"Could not read RVC checkpoint {path}: {err}") from err
        if not isinstance(ckpt, dict) or "weight" not in ckpt or "config" not in ckpt:
            raise RvcLoadError(f"{path} is not an RVC inference checkpoint (missing 'weight'/'config')")
        return ckpt

    def _build_generator(self, ckpt: dict) -> None:
        from torch.nn.utils import parametrize

        from .vendor.applio.algorithm.synthesizers import Synthesizer

        config = list(ckpt["config"])
        config[-3] = ckpt["weight"]["emb_g.weight"].shape[0]
        self._tgt_sr = int(config[-1])
        self._use_f0 = bool(ckpt.get("f0", 1))
        self._version = str(ckpt.get("version", "v1"))
        if self._version not in ("v1", "v2"):
            raise RvcLoadError(f"Unsupported RVC version {self._version!r}")
        net_g = Synthesizer(
            *config,
            use_f0=self._use_f0,
            text_enc_hidden_dim=768 if self._version == "v2" else 256,
            vocoder=ckpt.get("vocoder", "HiFi-GAN"),
        )
        del net_g.enc_q
        result = net_g.load_state_dict(ckpt["weight"], strict=False)
        missing = [k for k in result.missing_keys if not k.startswith("enc_q.")]
        if missing:
            raise RvcLoadError(f"Checkpoint is missing generator weights: {missing[:5]}...")
        # Bake weight-norm parametrizations into plain weights: identical output, faster.
        for module in net_g.modules():
            if parametrize.is_parametrized(module, "weight"):
                parametrize.remove_parametrizations(module, "weight", leave_parametrized=True)
        self._net_g = net_g.to(self.device).float().eval()

    def _load_hubert(self, contentvec_dir: Path):
        from torch import nn
        from transformers import HubertModel

        class HubertModelWithFinalProj(HubertModel):
            def __init__(self, config):
                super().__init__(config)
                self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)

        try:
            model = HubertModelWithFinalProj.from_pretrained(str(contentvec_dir))
        except Exception as err:
            raise RvcLoadError(f"Could not load ContentVec from {contentvec_dir}: {err}") from err
        return model.to(self.device).float().eval()

    def _load_index(self, path: Path) -> int:
        import faiss

        logging.getLogger("faiss").setLevel(logging.WARNING)
        try:
            index = faiss.read_index(str(path))
            big_npy = index.reconstruct_n(0, index.ntotal)
        except Exception as err:
            raise RvcLoadError(f"Could not read FAISS index {path}: {err}") from err
        expected = 768 if self._version == "v2" else 256
        if index.d != expected:
            raise RvcLoadError(
                f"Model/index mismatch: index {path.name} has dimension {index.d}, "
                f"an RVC {self._version} model needs {expected}"
            )
        import torch

        # Retrieval runs on the device as exact k-NN over the reconstructed vectors,
        # avoiding CPU FAISS search and the device->host->device copies around it.
        bank = torch.from_numpy(np.ascontiguousarray(big_npy, dtype=np.float32)).to(self.device)
        self._bank, self._bank_sq = bank, (bank * bank).sum(dim=1)
        _LOGGER.info("RVC index loaded: %s vectors=%d dim=%d", path.name, index.ntotal, index.d)
        return int(index.ntotal)

    # ------------------------------------------------------------------ public API

    @property
    def output_sample_rate(self) -> int:
        if not self._tgt_sr:
            raise RuntimeError("RVC is not loaded")
        return self._tgt_sr

    def convert(
        self,
        audio: np.ndarray,
        sample_rate: int,
        pitch: int = 0,
        f0_method: str = "rmvpe",
        index_rate: float = 0.6,
        protect: float = 0.33,
    ) -> tuple[np.ndarray, int]:
        """Convert a whole utterance. Returns (float32 audio, output sample rate)."""
        x = self._prepare(audio, sample_rate)
        if x.size == 0:
            return np.zeros(0, dtype=np.float32), self._tgt_sr
        audio_pad = np.pad(x, (self.t_pad, self.t_pad), mode="reflect")
        pitch_t, pitchf_t = self._f0(audio_pad, pitch, f0_method)
        rate = index_rate if self._bank is not None else 0.0
        t_pad_tgt = self._tgt_sr * self.t_pad // INPUT_SR

        # Segment bounds (a single segment unless the input exceeds ~41 s).
        spans: list[tuple[int, int | None, int, int | None]] = []
        s = 0
        for t in self._split_points(x) + [None]:
            end = None if t is None else t + 2 * self.t_pad + WINDOW
            f_e = None if t is None else (t + 2 * self.t_pad) // WINDOW
            spans.append((s, end, s // WINDOW, f_e))
            if t is not None:
                s = t
        out: list[np.ndarray] = []
        for a, b, f_s, f_e in spans:
            seg = audio_pad[a:b]
            p, pf = (pitch_t[:, f_s:f_e], pitchf_t[:, f_s:f_e]) if self._use_f0 else (None, None)
            y = self._synthesize(self._features(seg, rate), seg.shape[0], p, pf, protect)
            out.append(y[t_pad_tgt:-t_pad_tgt])
        result = prevent_clipping(np.concatenate(out).astype(np.float32), OUTPUT_CEILING)
        return result, self._tgt_sr

    def memory_stats(self) -> dict[str, float]:
        import torch

        if not self.device.startswith("cuda") or not torch.cuda.is_available():
            return {}
        return {
            "allocated_mb": torch.cuda.memory_allocated() / 2**20,
            "reserved_mb": torch.cuda.memory_reserved() / 2**20,
            "peak_allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
            "peak_reserved_mb": torch.cuda.max_memory_reserved() / 2**20,
        }

    def reset_peak_memory(self) -> None:
        import torch

        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    # ------------------------------------------------------------------ internals

    def _prepare(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        from scipy import signal

        if self._net_g is None or self._highpass is None:
            raise RuntimeError("RVC is not loaded")
        x = resample(to_float32(audio), sample_rate, INPUT_SR)
        if x.size == 0:
            return x
        peak = float(np.max(np.abs(x)))
        if peak > INPUT_PEAK:
            x = x * (INPUT_PEAK / peak)
        padlen = 3 * max(len(self._highpass[0]), len(self._highpass[1]))
        if x.size > padlen:
            x = signal.filtfilt(*self._highpass, x)
        return x.astype(np.float32)

    def _split_points(self, x: np.ndarray) -> list[int]:
        """Applio's split for very long inputs: quietest point near every ``t_center``."""
        if x.size + WINDOW <= self.t_max:
            return []
        padded = np.pad(x, (WINDOW // 2, WINDOW // 2), mode="reflect")
        summed = np.zeros_like(x)
        for i in range(WINDOW):
            summed += padded[i : i - WINDOW]
        points = []
        for t in range(self.t_center, x.size, self.t_center):
            region = np.abs(summed[t - self.t_query : t + self.t_query])
            points.append((t - self.t_query + int(np.argmin(region))) // WINDOW * WINDOW)
        return points

    def _f0(self, audio_pad: np.ndarray, pitch: int, f0_method: str):
        """(coarse pitch, f0 Hz) tensors on the device, or (None, None) for non-f0 models."""
        import torch

        if not self._use_f0:
            return None, None
        if f0_method != "rmvpe":
            raise ValueError(f"Unsupported f0_method {f0_method!r}; available: rmvpe")
        coarse, f0 = self._f0_numpy(audio_pad, pitch)
        pitch_t = torch.from_numpy(coarse).to(self.device).unsqueeze(0)
        pitchf_t = torch.from_numpy(f0.astype(np.float32)).to(self.device).unsqueeze(0)
        return pitch_t, pitchf_t

    def _f0_numpy(self, audio_pad: np.ndarray, pitch: int) -> tuple[np.ndarray, np.ndarray]:
        import torch

        p_len = audio_pad.size // WINDOW
        with torch.inference_mode():
            f0 = self._rmvpe.infer_from_audio(audio_pad, thred=0.03)
        f0 = f0 * pow(2, pitch / 12)
        f0_mel = 1127 * np.log(1 + f0 / 700)
        voiced = f0_mel > 0
        f0_mel[voiced] = (f0_mel[voiced] - F0_MEL_MIN) * 254 / (F0_MEL_MAX - F0_MEL_MIN) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        return np.rint(f0_mel[:p_len]).astype(np.int64), f0[:p_len]

    def _features(self, segment: np.ndarray, index_rate: float):
        """ContentVec features (+ retrieval blend). Returns (feats, feats0) at 50 frames/s."""
        import torch

        with torch.inference_mode():
            x = torch.from_numpy(np.ascontiguousarray(segment)).float().view(1, -1).to(self.device)
            feats = self._hubert(x)["last_hidden_state"]
            if self._version == "v1":
                feats = self._hubert.final_proj(feats[0]).unsqueeze(0)
            feats0 = feats.clone() if self._use_f0 else None
            if index_rate > 0:
                feats = self._retrieve(feats, index_rate)
            return feats, feats0

    def _synthesize(self, features, segment_len: int, pitch, pitchf, protect: float) -> np.ndarray:
        """Generator pass for one segment (Applio's ``voice_conversion`` after feature extraction)."""
        import torch
        import torch.nn.functional as F

        feats, feats0 = features
        with torch.inference_mode():
            feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
            p_len = min(segment_len // WINDOW, feats.shape[1])
            pitch_guidance = pitch is not None and pitchf is not None
            if pitch_guidance:
                feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
                p_len = min(p_len, pitch.shape[1])
                pitch, pitchf = pitch[:, :p_len], pitchf[:, :p_len]
                if protect < 0.5:
                    mask = pitchf.clone()
                    mask[pitchf > 0] = 1
                    mask[pitchf < 1] = protect
                    mask = mask.unsqueeze(-1)
                    feats = feats[:, :p_len] * mask + feats0[:, :p_len] * (1 - mask)
            else:
                pitch = pitchf = None
            feats = feats[:, :p_len]
            lengths = torch.tensor([p_len], device=self.device).long()
            y = self._net_g.infer(feats.float(), lengths, pitch, pitchf, self._sid)[0][0, 0]
            return y.float().cpu().numpy()

    def _retrieve(self, feats, index_rate: float, block: int = 256):
        """Blend features with their 8 nearest training vectors (exact L2, weights 1/d²).

        Queries go in blocks so the (block x index size) distance matrix stays small
        (256 x 25k vectors = 26 MB).
        """
        import torch

        q = feats[0]
        mixed = []
        for i in range(0, q.shape[0], block):
            qb = q[i : i + block]
            dist = (qb * qb).sum(dim=1, keepdim=True) - 2.0 * (qb @ self._bank.T) + self._bank_sq
            score, ix = torch.topk(dist, k=8, dim=1, largest=False)
            weight = 1.0 / score.clamp_min(1e-12).square()
            weight = weight / weight.sum(dim=1, keepdim=True)
            mixed.append((self._bank[ix] * weight.unsqueeze(-1)).sum(dim=1))
        retrieved = torch.cat(mixed).unsqueeze(0)
        return retrieved * index_rate + (1 - index_rate) * feats
