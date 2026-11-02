# syntax=docker/dockerfile:1

# GPU=cuda (NVIDIA, the default), rocm (AMD), or cpu:
# docker build --build-arg GPU=cpu -t wyoming-rvc-cpu .
# No CUDA/ROCm base image is needed: the PyTorch wheels ship their own GPU runtime.
# The CPU image removes those pinned GPU packages, then installs a CPU-only PyTorch wheel.
# NVIDIA: the Container Toolkit injects the driver at run time (host driver >= 570).
# AMD: the host's amdgpu driver, reached through /dev/kfd and /dev/dri.
ARG PYTHON_IMAGE=python:3.12.14-slim-trixie

# ---------------------------------------------------------------- builder
FROM ${PYTHON_IMAGE} AS builder
ARG GPU=cuda

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
COPY requirements.lock requirements-rocm.lock /tmp/
# The CPU index no longer serves a 2.11 CPU wheel; pin its current Linux amd64 CPU wheel.
RUN case "$GPU" in \
      cuda) lock=/tmp/requirements.lock ;; \
      rocm) lock=/tmp/requirements-rocm.lock ;; \
      cpu) \
        awk '/^[^[:space:]]/ { skip = $1 ~ /^(cuda-|nvidia-|torch|triton)/ } !skip { print }' \
          /tmp/requirements.lock > /tmp/requirements-cpu.lock \
        && lock=/tmp/requirements-cpu.lock ;; \
      *) echo "GPU must be cuda, rocm, or cpu, got '$GPU'" >&2; exit 1 ;; \
    esac \
    && /opt/venv/bin/pip install --require-hashes -r "$lock" \
    && if [ "$GPU" = cpu ]; then \
      printf '%s\n' \
        'torch==2.15.0+cpu --hash=sha256:0955c7e3446326293f48af18fad67872fa31dee5a36a87690eb9e5be8b465be3' \
        > /tmp/requirements-torch-cpu.lock \
      && /opt/venv/bin/pip install --no-deps --require-hashes \
        --index-url https://download.pytorch.org/whl/cpu \
        -r /tmp/requirements-torch-cpu.lock; \
    fi
COPY requirements-onnxruntime-gpu.lock /tmp/requirements-onnxruntime-gpu.lock
# CUDA: swap the CPU-only onnxruntime for the CUDA 12 build (CPU stays the default).
# ROCm: Piper keeps the CPU build (no ROCm onnxruntime wheel), so PIPER_DEVICE=auto means CPU.
RUN if [ "$GPU" = cuda ]; then \
      /opt/venv/bin/pip uninstall -y onnxruntime \
      && /opt/venv/bin/pip install --no-deps --require-hashes -r /tmp/requirements-onnxruntime-gpu.lock; \
    fi

COPY pyproject.toml README.md LICENSE THIRD_PARTY_LICENSES.md /build/
COPY src /build/src
RUN /opt/venv/bin/pip install --no-deps --no-build-isolation /build

# ---------------------------------------------------------------- runtime
FROM ${PYTHON_IMAGE} AS runtime

# Labels the Home Assistant Supervisor reads when the image runs as an add-on.
ARG BUILD_VERSION=dev
LABEL io.hass.type="addon" \
      io.hass.arch="amd64" \
      io.hass.version="${BUILD_VERSION}"

# libgomp: OpenMP runtime used by faiss-cpu and onnxruntime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY scripts /app/scripts

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODELS_DIR=/models \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

VOLUME ["/models"]
EXPOSE 8080 10200

# Ready only after every model is loaded and warmed up (first start downloads ~1 GB).
HEALTHCHECK --interval=30s --timeout=5s --start-period=900s --retries=3 \
    CMD ["python", "-m", "wyoming_rvc.healthcheck"]

STOPSIGNAL SIGTERM
ENTRYPOINT ["python", "-m", "wyoming_rvc.main"]
