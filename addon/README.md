# Wyoming RVC (NVIDIA)

Give any text-to-speech voice an RVC timbre for Home Assistant Assist (Piper + RVC
over Wyoming), using the NVIDIA CUDA image. Existing installations keep their
configuration and model cache; the add-on slug and image have not changed.

GPU acceleration requires a compatible host that exposes the NVIDIA runtime and
driver to the container. Official Home Assistant OS does not provide this access;
choose **Wyoming RVC (CPU)** there to avoid downloading GPU libraries. The NVIDIA
profile still supports its existing CPU fallback with `device: auto`.

amd64 only. See the Documentation tab for setup, options and hardware notes.
