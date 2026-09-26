"""In-process Piper TTS (loaded once)."""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PiperOptions:
    use_cuda: bool = False
    length_scale: float | None = None
    noise_scale: float | None = None
    noise_w_scale: float | None = None
    speaker_id: int | None = None


class PiperEngine:
    """Wraps ``piper.PiperVoice``; yields one float32 array per sentence."""

    def __init__(self, model_path: Path, config_path: Path, options: PiperOptions) -> None:
        self.model_path = model_path
        self.config_path = config_path
        self.options = options
        self._voice = None
        self._syn_config = None
        # espeak-ng phonemization is process-global; serialize synthesis.
        self._lock = threading.Lock()

    def load(self) -> None:
        import onnxruntime
        from piper import PiperVoice, SynthesisConfig

        if self.options.use_cuda:
            # Make the CUDA 12 / cuDNN libraries installed with torch visible to onnxruntime.
            import torch  # noqa: F401

            if hasattr(onnxruntime, "preload_dlls"):
                onnxruntime.preload_dlls()
        if self.options.use_cuda and "CUDAExecutionProvider" not in onnxruntime.get_available_providers():
            raise RuntimeError(
                "Piper on CUDA (PIPER_DEVICE) but onnxruntime has no CUDAExecutionProvider "
                f"(available: {onnxruntime.get_available_providers()}). Install onnxruntime-gpu or disable it."
            )
        self._voice = PiperVoice.load(self.model_path, config_path=self.config_path, use_cuda=self.options.use_cuda)
        providers = self._voice.session.get_providers()
        if self.options.use_cuda and providers[0] != "CUDAExecutionProvider":
            raise RuntimeError(f"Piper on CUDA (PIPER_DEVICE) but the ONNX session runs on {providers}")
        self._syn_config = SynthesisConfig(
            speaker_id=self.options.speaker_id,
            length_scale=self.options.length_scale,
            noise_scale=self.options.noise_scale,
            noise_w_scale=self.options.noise_w_scale,
            # Keep the model's natural level; clipping protection happens once at the end.
            normalize_audio=False,
        )
        _LOGGER.info(
            "Piper loaded: model=%s sample_rate=%s providers=%s",
            self.model_path.name,
            self.sample_rate,
            providers,
        )

    @property
    def device(self) -> str:
        return "cuda" if self.options.use_cuda else "cpu"

    @property
    def sample_rate(self) -> int:
        if self._voice is None:
            raise RuntimeError("Piper is not loaded")
        return int(self._voice.config.sample_rate)

    def sentences(self, text: str) -> Iterator[np.ndarray]:
        """Synthesize ``text`` lazily, one sentence at a time."""
        if self._voice is None:
            raise RuntimeError("Piper is not loaded")
        with self._lock:
            chunks = list(self._voice.phonemize(text))
        for phonemes in chunks:
            if not phonemes:
                continue
            with self._lock:
                ids = self._voice.phonemes_to_ids(phonemes)
                audio = self._voice.phoneme_ids_to_audio(ids, self._syn_config)
            yield np.clip(np.asarray(audio, dtype=np.float32).reshape(-1), -1.0, 1.0)
