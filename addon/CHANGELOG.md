# Changelog

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
