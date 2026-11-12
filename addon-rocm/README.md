# Wyoming RVC (AMD ROCm)

Experimental AMD GPU profile for RVC voice conversion in Home Assistant Assist.
Uses the ROCm image, not the NVIDIA image. amd64 Linux only.

Requires a supported AMD GPU, a compatible host `amdgpu` compute driver, and
accessible `/dev/kfd` and `/dev/dri` devices. **Official Home Assistant OS does not
provide this compute access.** Installing this profile does not add host drivers.
Choose **Wyoming RVC (CPU)** on official HAOS, or run the ROCm Docker image on a
separate compatible Linux machine.

This is a separate installation with its own configuration and model cache.
See the Documentation tab for prerequisites and setup.
