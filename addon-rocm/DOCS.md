# Wyoming RVC (AMD ROCm)

Experimental profile using `ghcr.io/xchwarze/wyoming-rvc-rocm` and PyTorch 2.11.0
with ROCm. RVC runs on the AMD GPU; Piper runs on the CPU because the image's
ONNX Runtime has no ROCm provider.

## Host requirements

- amd64 Linux and an AMD GPU supported by the image's PyTorch ROCm 7.2 wheels.
  See [AMD GPUs (ROCm)](https://github.com/xchwarze/wyoming-rvc#amd-gpus-rocm) for
  GPU architectures and Docker usage.
- A compatible host kernel with the `amdgpu` compute driver and accessible
  `/dev/kfd` and `/dev/dri` devices. Container permissions and the host security
  policy must also permit compute access.
- Disk space for the large ROCm image (about 15 GB), plus about 1.3 GB of models.

**Official Home Assistant OS does not provide ROCm compute access to add-ons.**
This manifest cannot install missing drivers or make an unsupported GPU work.
Proposed [HAOS GPU access](https://github.com/home-assistant/architecture/discussions/1568)
is not a released feature. On official HAOS, use **Wyoming RVC (CPU)**; for GPU
acceleration, run the ROCm Docker image on a compatible external Linux host and
connect Home Assistant through Wyoming on port 10200.

The manifest requests `/dev/kfd` and Supervisor's `video: true` access for DRM
devices. It does not request broad host privileges or disable AppArmor. These
requests alone do not guarantee GPU access on a particular host.

## Setup on a compatible host

1. Confirm the host provides the required GPU devices and permissions before
   installing. A missing `/dev/kfd` can prevent Supervisor from starting the app.
2. Keep `device: cuda`. PyTorch uses the name `cuda` for AMD HIP devices too; it
   does not mean this image needs an NVIDIA GPU. This setting fails startup if
   PyTorch cannot use the GPU. `auto` allows a CPU fallback; `cpu` forces CPU
   execution, but neither setting removes the manifest's host-device requirements.
3. Set `piper_voice` and `warmup_text` for your assistant's language. Start the
   add-on and wait for model downloads and the `Service ready` log.
4. Accept the **Wyoming Protocol** discovery in **Settings → Devices & Services**.
   In **Settings → Voice assistants**, select its **Wyoming RVC** TTS engine and voice.

All options except the `device` default match the
[shared option reference](https://github.com/xchwarze/wyoming-rvc/blob/main/addon/DOCS.md#options),
including `probe_text` for upstream TTS validation.

## Configuration and migration

Put custom `voices.yaml` and local models in
`/addon_configs/<id>_wyoming_rvc_rocm/` (File editor or Samba). Supervisor assigns
the repository `<id>`; the container sees this folder as `/config`.

Settings and the private `/data` model cache are separate from the other profiles.
Back up and keep the old installation, copy changed options and public files,
adjust absolute paths, and allow the new profile to download its models. Select
the new Wyoming integration and test speech before removing the old installation.
See the [migration guide and voice examples](https://github.com/xchwarze/wyoming-rvc/blob/main/addon/DOCS.md#profiles-data-and-migration).

Host port 10200 is disabled by default; automatic discovery needs no published
port. If external clients need it, assign a unique host port for each running
profile. Wyoming has no authentication; use a trusted network only. The HTTP
debug API is not published.

## License notes

Models download from their original locations at runtime. Check each voice's
license; see [THIRD_PARTY_LICENSES.md](https://github.com/xchwarze/wyoming-rvc/blob/main/THIRD_PARTY_LICENSES.md).
