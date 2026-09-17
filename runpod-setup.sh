#!/usr/bin/env bash
#
# EditaLive — one-shot setup for a RunPod GPU pod.
#
# Run this ON THE POD (web terminal or SSH), not on your local machine:
#     cd /workspace && bash runpod-setup.sh
#
# Designed for `runpod/pytorch:1.3.1-cu1281-torch260-ubuntu2204`
# (PyTorch 2.6.0, Python 3.12, CUDA 12.x, nvcc included).
#
# It is idempotent: re-running skips completed steps.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/EditaLive}"
REPO_URL="${REPO_URL:-https://github.com/GVCLab/EditaLive.git}"
export HF_HOME="${HF_HOME:-/workspace/hf}"
export HF_HUB_ENABLE_HF_TRANSFER=1

log()  { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[warn] %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[1;31m[error] %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- sanity ----
log "Checking environment"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found — this must run on a GPU pod."
nvidia-smi -L || die "No GPU detected by driver."

if ! command -v nvcc >/dev/null; then
  die "nvcc not found. Use a CUDA *devel* image, e.g. runpod/pytorch:1.3.1-cu1281-torch260-ubuntu2204"
fi
nvcc --version | tail -n 1

PY_VER="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
echo "Python ${PY_VER}"
case "${PY_VER}" in
  3.10|3.11|3.12) : ;;
  *) warn "Python ${PY_VER} is untested; 3.10-3.12 expected." ;;
esac

python - <<'PY'
import torch
print(f"torch {torch.__version__} | cuda {torch.version.cuda} | available {torch.cuda.is_available()}")
assert torch.cuda.is_available(), "torch cannot see the GPU"
PY

# ------------------------------------------------------------------ repo ----
if [[ -d "${REPO_DIR}/.git" ]]; then
  log "Repo already present at ${REPO_DIR}"
else
  log "Cloning EditaLive into ${REPO_DIR}"
  git clone "${REPO_URL}" "${REPO_DIR}"
fi
cd "${REPO_DIR}"

# ----------------------------------------------------------- python deps ----
log "Installing Python requirements"
pip install --upgrade pip wheel
pip install -r requirements.txt

# ---------------------------------------------------------- cuda kernels ----
log "Installing ninja"
pip install ninja

if python -c 'import flash_attn' 2>/dev/null; then
  log "flash-attn already installed"
else
  log "Building flash-attn (this takes a while)"
  MAX_JOBS="${MAX_JOBS:-$(nproc)}" \
    pip install flash-attn==2.7.2.post1 --no-build-isolation
fi

if python -c 'import fastvideo_kernel' 2>/dev/null; then
  log "fastvideo-kernel already installed"
else
  log "Building fastvideo-kernel (this takes a while)"
  CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc)}" \
    bash tools/install_fastvideo_kernel.sh
fi

# --------------------------------------------------------------- weights ----
weights_present() {
  [[ -f weights/EditaLive/editalive_edit.safetensors ]] \
    && [[ -f weights/EditaLive/lightx2v.safetensors ]] \
    && [[ -f weights/EditaLive/editalive_streaming.safetensors ]] \
    && [[ -f weights/Flash-VAED/Flash_VAED_Wan.pth ]] \
    && [[ -d weights/Wan-Animate ]] \
    && [[ -n "$(ls -A weights/Wan-Animate 2>/dev/null)" ]]
}

if weights_present; then
  log "Weights already present in ${REPO_DIR}/weights — skipping download"
else
  log "Downloading model weights into ${REPO_DIR}/weights (~50 GB)"
  pip install hf_transfer
  python tools/download_weights.py
fi

weights_present || die "Weight download incomplete"

# --------------------------------------------- pick flags from the GPU ----
log "Choosing inference flags for this GPU"
read -r VRAM_MIB COMPUTE_CAP <<<"$(
  python - <<'PY'
import torch
p = torch.cuda.get_device_properties(0)
print(p.total_memory // (1024 * 1024), f"{p.major}.{p.minor}")
PY
)"

VRAM_GB=$(( VRAM_MIB / 1024 ))
echo "Detected: ${VRAM_GB} GB VRAM, compute capability ${COMPUTE_CAP}"

FP8=""
if awk "BEGIN{exit !(${COMPUTE_CAP} >= 8.9)}"; then FP8="--fp8"; else
  warn "Compute capability ${COMPUTE_CAP} < 8.9: FP8 disabled."
fi

if (( VRAM_GB >= 45 )); then
  LOW_VRAM="off";             SHORT=480; LONG=832
elif (( VRAM_GB >= 30 )); then
  LOW_VRAM="off";             SHORT=480; LONG=832
elif (( VRAM_GB >= 20 )); then
  LOW_VRAM="weights+kv";      SHORT=384; LONG=672
else
  LOW_VRAM="weights+kv";      SHORT=384; LONG=672
  warn "${VRAM_GB} GB is below the repo's tested floor; expect OOM."
fi
echo "Flags: ${FP8:-<no fp8>} --low_vram ${LOW_VRAM} --short_edge ${SHORT} --long_edge ${LONG}"

COMMON=(
  --ckpt_dir ./weights/Wan-Animate
  --lora_paths
    ./weights/EditaLive/editalive_edit.safetensors
    ./weights/EditaLive/lightx2v.safetensors
    ./weights/EditaLive/editalive_streaming.safetensors
  --short_edge "${SHORT}" --long_edge "${LONG}"
  --fast_decode True
  --enable_compile False
)
[[ -n "${FP8}" ]] && COMMON+=("${FP8}")
COMMON+=( --low_vram "${LOW_VRAM}" )

# ------------------------------------------------- write convenience runs ----
log "Writing run-streaming.sh and run-webcam.sh"
{
  echo '#!/usr/bin/env bash'
  echo 'set -euo pipefail'
  echo 'cd "$(dirname "$0")"'
  echo '# Usage: ./run-streaming.sh [video] ["prompt"] [extra edit_streaming.py flags...]'
  echo 'VIDEO="${1:-./demo/demo_1.mp4}"'
  echo 'PROMPT="${2:-Transform it into a soft plush-like aesthetic with smooth textures and gentle shading.}"'
  echo 'if (( $# >= 2 )); then shift 2; elif (( $# == 1 )); then shift 1; fi'
  printf 'python edit_streaming.py \\\n'
  for a in "${COMMON[@]}"; do printf '  %q \\\n' "$a"; done
  echo '  --video "${VIDEO}" \'
  echo '  --prompt "${PROMPT}" \'
  echo '  --frame_num -1 \'
  echo '  --save_dir ./outputs_streaming \'
  echo '  "$@"'
} > run-streaming.sh
chmod +x run-streaming.sh

{
  echo '#!/usr/bin/env bash'
  echo 'set -euo pipefail'
  echo 'cd "$(dirname "$0")"'
  echo 'mkdir -p .cache/compile/webcam'
  echo 'python edit_webcam.py --device 0 --port 7860 "$@"'
} > run-webcam.sh
chmod +x run-webcam.sh

log "Done."
cat <<EOF

Weights : ${REPO_DIR}/weights
Test run: ./run-streaming.sh
Webcam  : ./run-webcam.sh    (then open the pod's 7860 proxy URL)

Result videos land in ./outputs_streaming
EOF
