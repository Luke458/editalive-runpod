#!/usr/bin/env bash
#
# Build the EditaLive RunPod image and push it to a registry.
#
# No GPU is required on this machine: nvcc comes from the base image inside
# the build container. Podman or Docker both work.
#
# Set your registry first, e.g.:
#   export REGISTRY=ghcr.io/<your-github-user>
#   export REGISTRY=docker.io/<your-dockerhub-user>
#
# Then:
#   ./build-and-push.sh              # build + push
#   ./build-and-push.sh --no-push    # build only
#
# Env overrides:
#   IMAGE_NAME=editalive   TAG=cu128-torch260
#   TORCH_CUDA_ARCH_LIST="8.0;8.9;9.0"
#   MAX_JOBS=8   CMAKE_BUILD_PARALLEL_LEVEL=8
#   BAKE_WEIGHTS=1         # bake ~50 GB of weights into the image
set -euo pipefail

cd "$(dirname "$0")"

IMAGE_NAME="${IMAGE_NAME:-editalive}"
TAG="${TAG:-cu128-torch260}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.9;9.0}"
MAX_JOBS="${MAX_JOBS:-$(nproc)}"
CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc)}"
BAKE_WEIGHTS="${BAKE_WEIGHTS:-0}"

DO_PUSH=1
[[ "${1:-}" == "--no-push" ]] && DO_PUSH=0

: "${REGISTRY:?Set REGISTRY first, e.g. export REGISTRY=ghcr.io/<user>}"
FULL_IMAGE="${REGISTRY%/}/${IMAGE_NAME}:${TAG}"

# Pick a container engine (docker may be podman under the hood).
if command -v docker >/dev/null 2>&1; then ENGINE=docker
elif command -v podman >/dev/null 2>&1; then ENGINE=podman
else echo "No docker or podman found." >&2; exit 1; fi
echo "Engine: ${ENGINE}"

echo "Building ${FULL_IMAGE}"
echo "  arch list : ${TORCH_CUDA_ARCH_LIST}"
echo "  max jobs  : ${MAX_JOBS} / ${CMAKE_BUILD_PARALLEL_LEVEL}"
echo "  weights   : BAKE_WEIGHTS=${BAKE_WEIGHTS}"
echo "This takes roughly 40-90 min the first time."

"${ENGINE}" build \
  --platform linux/amd64 \
  --build-arg TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
  --build-arg MAX_JOBS="${MAX_JOBS}" \
  --build-arg CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL}" \
  --build-arg BAKE_WEIGHTS="${BAKE_WEIGHTS}" \
  -t "${FULL_IMAGE}" \
  .

if (( DO_PUSH )); then
  echo "Pushing ${FULL_IMAGE}"
  "${ENGINE}" push "${FULL_IMAGE}"
  echo
  echo "Push complete. Register it as a RunPod template:"
  echo "  IMAGE=${FULL_IMAGE} TEMPLATE_NAME=editalive ./runpod-pod.sh template"
  echo "Then boot a Pod from it:"
  echo "  TEMPLATE_ID=<id> ./runpod-pod.sh create 4090"
else
  echo "Built ${FULL_IMAGE} (not pushed)."
fi
