# Third-party licenses

This project is licensed under **GPL-3.0-or-later** (see `LICENSE`). That choice
is forced by Piper: `piper-tts` (and the espeak-ng it bundles) is GPL-3.0 and is
imported in-process, so the combined program must be GPL-compatible.

## Code included in this repository

| Path | Origin | License | Notes |
|---|---|---|---|
| `src/wyoming_rvc/vendor/applio/algorithm/*` | [IAHispano/Applio](https://github.com/IAHispano/Applio) 3.6.5, `rvc/lib/algorithm/` | MIT, © 2026 AI Hispano (`vendor/applio/LICENSE`) | RVC synthesizer, encoders, HiFi-GAN(-NSF/-MRF) generators. Modified: import paths; RefineGAN removed (needs torchaudio); `print` calls replaced by exceptions. |
| `src/wyoming_rvc/vendor/applio/predictors/rmvpe.py` | Applio `rvc/lib/predictors/RMVPE.py` (itself based on [Dream-High/RMVPE](https://github.com/Dream-High/RMVPE), Apache-2.0) | MIT (Applio) / Apache-2.0 (RMVPE) | Modified: uses the local mel filterbank; per-call `torch.cuda.empty_cache()` removed. |
| `src/wyoming_rvc/vendor/applio/predictors/mel.py` | Adapted from [librosa](https://github.com/librosa/librosa) `filters.mel` | ISC, © librosa development team | HTK mel + Slaney normalization only; verified bit-identical to librosa 0.11 for RMVPE's parameters. |
| `src/wyoming_rvc/rvc_engine.py` (conversion math) | Adapted from Applio `rvc/infer/pipeline.py` | MIT | Rewritten so all models stay resident; retrieval runs as exact k-NN on the GPU (FAISS only reads the index file). |

Applio's [Terms of Use](https://github.com/IAHispano/Applio/blob/main/TERMS_OF_USE.md) also ask users
to respect intellectual property and privacy and not to create harmful content with voice conversion.

## Python dependencies (installed, not vendored)

Exact versions are pinned in `requirements.lock`.

| Package | License |
|---|---|
| piper-tts 1.8.0 ([OHF-Voice/piper1-gpl](https://github.com/OHF-Voice/piper1-gpl)) | GPL-3.0-or-later (bundles espeak-ng, GPL-3.0) |
| onnxruntime | MIT |
| wyoming ([OHF-Voice/wyoming](https://github.com/OHF-Voice/wyoming)) | MIT |
| sentence-stream | Apache-2.0 |
| zeroconf | LGPL-2.1-or-later |
| torch (PyTorch) | BSD-3-Clause and others (see the wheel) |
| NVIDIA CUDA runtime, cuDNN, cuBLAS, … (`nvidia-*-cu12`, pulled by torch) | NVIDIA proprietary license / CUDA EULA (redistributable runtime components) |
| transformers | Apache-2.0 |
| huggingface-hub | Apache-2.0 |
| faiss-cpu | MIT |
| numpy | BSD-3-Clause (and bundled permissive licenses) |
| scipy | BSD-3-Clause |
| soxr (python-soxr / libsoxr) | LGPL-2.1-or-later |
| fastapi, pydantic | MIT |
| uvicorn, starlette | BSD-3-Clause |

The Docker image contains these packages (including NVIDIA's CUDA runtime
libraries as shipped in the PyTorch wheels) and no model weights.

## Model weights (downloaded at run time, never redistributed)

None of these files are in this repository or in the Docker image. They are
fetched from their original location into the `/models` volume on first start.
Check each license yourself before use.

| Weights | Source | License as published by the source |
|---|---|---|
| TetoTalk RVC (`KasaneTetotalkV1_240e_7680s`) | [Slichi/KasaneTeto](https://huggingface.co/Slichi/KasaneTeto) | Model card declares `license: openrail`. Kasane Teto is a character/voicebank by TWINDRILL; its own terms of use apply to the voice. |
| Piper voice `es_AR-daniela-high` | [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices) | Dataset: [OpenSLR 61](https://www.openslr.org/61/), CC BY-SA 4.0 (see the voice's `MODEL_CARD`). |
| ContentVec embedder, RMVPE pitch model | [IAHispano/Applio](https://huggingface.co/IAHispano/Applio) (`Resources/`) | Repo declares MIT. Upstream: [ContentVec](https://github.com/auspicious3000/contentvec) (MIT), [RMVPE](https://github.com/Dream-High/RMVPE) (Apache-2.0). |
