# Wyoming RVC (CPU)

The CPU profile uses `ghcr.io/xchwarze/wyoming-rvc-cpu`, with the CPU-only build of
PyTorch 2.11.0. It needs an amd64 host, disk space for the image and about 1.3 GB
for models downloaded on first start. It does not require GPU devices or drivers.
RVC conversion can run at roughly real time or slower on the CPU.

## Setup

1. Leave `device: cpu` and select a `piper_voice` in your assistant's language.
2. Start the add-on and wait for the model downloads and the `Service ready` log.
3. Accept the new **Wyoming Protocol** entry in **Settings → Devices & Services**.
4. In **Settings → Voice assistants**, select the discovered **Wyoming RVC** TTS
   engine, then choose Kasane Teto or Hatsune Miku as the voice.

`device` accepts `cpu` or `auto`; this image always runs on the CPU. Changing the
option cannot add GPU support or change the image. All other options, including
`probe_text`, match the [shared option reference](https://github.com/xchwarze/wyoming-rvc/blob/main/addon/DOCS.md#options).

## Configuration and migration

Put custom `voices.yaml` and local models in
`/addon_configs/<id>_wyoming_rvc_cpu/` (File editor or Samba). Supervisor assigns
the repository `<id>`; the container sees this folder as `/config`.

This profile has its own private `/data` model cache. It does not share settings,
files or downloads with the NVIDIA or AMD profiles. Back up and keep the old
installation, copy your changed options and public configuration files, and let
this profile download its models. Adjust any absolute model paths. Select the new
Wyoming integration and test it before uninstalling the old profile. See the
[migration guide and voice examples](https://github.com/xchwarze/wyoming-rvc/blob/main/addon/DOCS.md#profiles-data-and-migration).

Host port 10200 is disabled by default; automatic discovery needs no published
port. For external clients, set a unique host port if several profiles run at
once. Wyoming has no authentication; use a trusted network only. The HTTP debug
API is not published.

## License notes

Models download from their original locations at runtime. Check each voice's
license; see [THIRD_PARTY_LICENSES.md](https://github.com/xchwarze/wyoming-rvc/blob/main/THIRD_PARTY_LICENSES.md).
