# syntax=docker/dockerfile:1

# No CUDA base image is needed: the PyTorch cu128 wheels ship their own CUDA
# 12.8 runtime and cuDNN as pip packages; the NVIDIA Container Toolkit injects
# the driver at run time (host driver >= 570 required).
ARG PYTHON_IMAGE=python:3.12.14-slim-trixie

# ---------------------------------------------------------------- builder
FROM ${PYTHON_IMAGE} AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
COPY requirements.lock /tmp/requirements.lock
RUN /opt/venv/bin/pip install --require-hashes -r /tmp/requirements.lock
COPY requirements-onnxruntime-gpu.lock /tmp/requirements-onnxruntime-gpu.lock
# Swap the CPU-only onnxruntime for the CUDA 12 build (CPU stays the default).
RUN /opt/venv/bin/pip uninstall -y onnxruntime \
    && /opt/venv/bin/pip install --no-deps --require-hashes -r /tmp/requirements-onnxruntime-gpu.lock

COPY pyproject.toml README.md LICENSE THIRD_PARTY_LICENSES.md /build/
COPY src /build/src
RUN /opt/venv/bin/pip install --no-deps --no-build-isolation /build

# ---------------------------------------------------------------- runtime
FROM ${PYTHON_IMAGE} AS runtime

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
    HF_HOME=/models/huggingface \
    CUDA_CACHE_PATH=/models/cuda-cache \
    CUDA_CACHE_MAXSIZE=4294967296 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

VOLUME ["/models"]
EXPOSE 8080 10200

# Ready only after every model is loaded and warmed up (first start downloads ~1 GB).
HEALTHCHECK --interval=30s --timeout=5s --start-period=900s --retries=3 \
    CMD ["python", "-c", "import os,sys,urllib.request; p=os.environ.get('HTTP_PORT','8080'); sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{p}/readyz', timeout=4).status == 200 else 1)"]

STOPSIGNAL SIGTERM
ENTRYPOINT ["python", "-m", "wyoming_rvc.main"]
