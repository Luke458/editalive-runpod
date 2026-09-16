# EditaLive — prebuilt RunPod image.
#
# Bakes CUDA/PyTorch + EditaLive requirements + flash-attn + fastvideo-kernel
# so a fresh Pod is ready seconds after boot instead of waiting ~40 min for
# source builds.
#
# Base: torch 2.6.0, Python 3.10, CUDA 12.8.1 (matches requirements.txt).
#
# Build (no GPU needed on the build host — nvcc comes from the base image):
#   ./build-and-push.sh
#
# CUDA arch list: must cover the GPUs you will rent. Default 8.0 / 8.9 / 9.0
# covers A100, RTX 4090 & L40S, and H100. Add 8.6 for A10/A40.
# Each extra arch lengthens the build.
#
# Models are NOT baked in by default (keep the image small; weights live on a
# network volume). Set --build-arg BAKE_WEIGHTS=1 to include them.
FROM runpod/pytorch:1.3.1-cu1281-torch260-ubuntu2204

ARG EDITALIVE_REPO=https://github.com/GVCLab/EditaLive.git
ARG EDITALIVE_REF=main
ARG TORCH_CUDA_ARCH_LIST="8.0;8.9;9.0"
ARG MAX_JOBS=4
ARG CMAKE_BUILD_PARALLEL_LEVEL=4
ARG BAKE_WEIGHTS=0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONNOUSERSITE=1 \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    HF_HUB_ENABLE_HF_TRANSFER=1

# System deps: ffmpeg for preprocessing/audio, build toolchain for the kernels,
# and the shared libraries OpenCV needs at import time.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg git build-essential cmake ninja-build pkg-config \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
        ca-certificates curl wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN git clone "${EDITALIVE_REPO}" /opt/EditaLive \
    && git -C /opt/EditaLive checkout "${EDITALIVE_REF}"

WORKDIR /opt/EditaLive

# Python deps (torch already satisfied by the base image).
RUN python -m pip install --no-cache-dir --upgrade pip wheel setuptools \
    && python -m pip install --no-cache-dir packaging psutil ninja hf_transfer \
    && python -m pip install --no-cache-dir -r requirements.txt

# flash-attn (compiled against the installed torch/CUDA).
RUN MAX_JOBS="${MAX_JOBS}" \
    python -m pip install --no-cache-dir flash-attn==2.7.2.post1 --no-build-isolation

# FastVideo block-sparse kernel (CUTLASS, CUDA-only).
RUN CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL}" \
    bash tools/install_fastvideo_kernel.sh

# Optional: bake the ~50 GB of weights into the image (not recommended).
RUN if [ "${BAKE_WEIGHTS}" = "1" ]; then python tools/download_weights.py; fi

# Fail the build now if either compiled kernel is unusable.
RUN python -c "import torch, flash_attn, fastvideo_kernel; \
print('torch', torch.__version__); \
print('flash-attn', flash_attn.__version__); \
print('fastvideo_kernel OK')"

# First-boot helper: clone the repo onto the network volume (if absent) and run
# the idempotent setup (downloads weights once, writes run-streaming.sh).
COPY runpod-setup.sh /opt/editalive-setup.sh
RUN printf '%s\n' \
    '#!/usr/bin/env bash' \
    'set -euo pipefail' \
    'REPO_DIR=/workspace/EditaLive' \
    'if [[ ! -d "$REPO_DIR/.git" ]]; then' \
    '  git clone https://github.com/GVCLab/EditaLive.git "$REPO_DIR"' \
    'fi' \
    'cd "$REPO_DIR"' \
    'exec bash /opt/editalive-setup.sh' \
    > /usr/local/bin/editalive-init \
    && chmod +x /usr/local/bin/editalive-init

WORKDIR /workspace
