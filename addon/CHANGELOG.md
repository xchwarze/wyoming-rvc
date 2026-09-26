# Changelog

## 0.2.1

- Rely on the image's Docker HEALTHCHECK instead of the obsolete `watchdog` option
  (0.2.0 was never published).

## 0.2.0

- First release as a Home Assistant add-on: options from the add-on UI, automatic
  Wyoming discovery, `device: auto` (GPU when available, otherwise CPU).
