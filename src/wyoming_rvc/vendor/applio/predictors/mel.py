# Minimal mel filterbank, replacing ``librosa.filters.mel`` for the RMVPE
# predictor so librosa (and numba) are not runtime dependencies.
#
# Adapted from librosa (https://github.com/librosa/librosa), ISC License,
# Copyright (c) 2013--2023, librosa development team.
# Only the ``htk`` mel scale with Slaney area normalization is implemented,
# which is the configuration RMVPE uses.

import numpy as np


def _hz_to_mel_htk(frequencies: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + frequencies / 700.0)


def _mel_to_hz_htk(mels: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)


def mel(
    *,
    sr: int,
    n_fft: int,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: float | None = None,
    htk: bool = True,
) -> np.ndarray:
    """Return a (n_mels, 1 + n_fft // 2) mel filterbank matrix."""
    if not htk:
        raise NotImplementedError("Only the HTK mel scale is supported")
    if fmax is None:
        fmax = float(sr) / 2

    weights = np.zeros((n_mels, 1 + n_fft // 2), dtype=np.float32)
    fftfreqs = np.fft.rfftfreq(n=n_fft, d=1.0 / sr)

    min_mel = _hz_to_mel_htk(np.asarray(fmin, dtype=np.float64))
    max_mel = _hz_to_mel_htk(np.asarray(fmax, dtype=np.float64))
    mel_f = _mel_to_hz_htk(np.linspace(min_mel, max_mel, n_mels + 2))

    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0, np.minimum(lower, upper))

    # Slaney-style area normalization
    enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
    weights *= enorm[:, np.newaxis]
    return weights
