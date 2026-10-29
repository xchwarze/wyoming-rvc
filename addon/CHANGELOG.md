# Changelog

## Unreleased

- Docker only: a second image, `ghcr.io/xchwarze/wyoming-rvc-rocm`, runs RVC on AMD GPUs
  (ROCm, Linux). The add-on is unchanged: Home Assistant OS has no ROCm driver.
- `PIPER_DEVICE=auto` runs Piper on the CPU when onnxruntime has no CUDA support (as in the
  ROCm image) instead of failing at startup.

## 0.5.1

- `SENTENCE_SILENCE_MS` also applies between the sentences of a streamed reply (what Home
  Assistant sends with streaming conversation agents); before, it was only applied inside
  one request.
- With an upstream Wyoming TTS, a sentence ending in "no." ("I said no.") is no longer
  merged with the next one.
- Wyoming requests longer than `MAX_TEXT_CHARS` (5000) get an error, like HTTP ones.
- Docker: experimental `RVC_HALF=true` runs ContentVec and the RVC generators in fp16 on the
  GPU (less VRAM, usually faster). Off by default.

## 0.5.0

- New option `idle_sleep_s` (seconds; default `0` = off, `300` is a good start): after that
  long without a request the models are unloaded to free VRAM, and reloaded by the next
  request, which is slower by the reload time.
- A voice in `voices.yaml` can use its own TTS source: its own Piper voice (`piper_voice`, or
  `piper_model`) or any Wyoming TTS (`upstream`). Voices without these keys work as before.

## 0.4.1

- The voices are listed under the language of `piper_voice` (for example `es` for
  `es_MX-claude-high`), so Home Assistant offers them to assistants in that language without
  setting `voice_language`. Previously they stayed `en` and were hidden from a Spanish
  assistant. Reload the Wyoming integration after updating.

## 0.4.0

- Comes with two voices, Kasane Teto (default) and Hatsune Miku, so the voice selector in
  Home Assistant works out of the box. Setting your own `rvc_repo_id` or `voice_name`, or a
  `voices.yaml`, replaces them as before. The first start downloads Miku (about 280 MB).
- After updating, reload the Wyoming integration so Home Assistant sees the new voice.

## 0.3.1

- Models are now stored in the app's `/data` folder, so they survive updates (they were
  re-downloaded after each update). The first start after updating downloads them once more.
- A voice with a broken index is no longer offered to Home Assistant.
- The default voice is loaded at startup when no voice sets `preload`, and another voice
  that fails to load no longer stops the app.
- A cancelled request during a voice switch no longer leaves a second copy of the model in VRAM.

## 0.3.0

- Several RVC voices in one instance: list them in `voices.yaml` in the app's config
  folder and pick one per assistant in Home Assistant. ContentVec, RMVPE and Piper are
  shared; voices load into VRAM on first use (`rvc_max_loaded_models`, default 1).
- New options `default_voice` and `rvc_max_loaded_models`.
- Requests for an unknown voice fail instead of silently using the default voice.
- Without `voices.yaml` nothing changes.

## 0.2.1

- Rely on the image's Docker HEALTHCHECK instead of the obsolete `watchdog` option
  (0.2.0 was never published).

## 0.2.0

- First release as a Home Assistant add-on: options from the add-on UI, automatic
  Wyoming discovery, `device: auto` (GPU when available, otherwise CPU).
