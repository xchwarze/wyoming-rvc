# Changelog

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
