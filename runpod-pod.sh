#!/usr/bin/env bash
#
# Create and manage an EditaLive GPU pod on RunPod from your local machine.
#
# Auth: export RUNPOD_API_KEY=...   (never hardcode it here)
#       Get one at https://www.runpod.io/console/user/settings
#
# Usage:
#   ./runpod-pod.sh up [4090|l40s|h100|a100|5090]       # create + wait + init
#   ./runpod-pod.sh create [4090|l40s|h100|a100|5090]   # from an image, or
#   TEMPLATE_ID=<id> ./runpod-pod.sh create 4090         # from a saved template
#   IMAGE=<registry/image:tag> ./runpod-pod.sh template  # register baked image
#   ./runpod-pod.sh list
#   ./runpod-pod.sh ssh      <podId>
#   ./runpod-pod.sh status   <podId>
#   ./runpod-pod.sh stop     <podId>
#   ./runpod-pod.sh terminate <podId>
#
# Env overrides:
#   CLOUD_TYPE=SECURE|COMMUNITY   (default SECURE)
#   IMAGE=runpod/pytorch:1.3.1-cu1281-torch260-ubuntu2204
#   DISK_GB=150  VOLUME_GB=100  POD_NAME=editalive
#   TEMPLATE_ID=<id>  TEMPLATE_NAME=editalive
#   EDITALIVE_INIT=0   SSH_READY_TIMEOUT=900
set -euo pipefail

API="https://rest.runpod.io/v1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load the API key from .env (gitignored) if it isn't already exported.
# Accepts either RUNPOD_API_KEY or runpod_api as the variable name.
if [[ -z "${RUNPOD_API_KEY:-}" && -f "${SCRIPT_DIR}/.env" ]]; then
  RUNPOD_API_KEY="$(sed -n 's/^[[:space:]]*\(RUNPOD_API_KEY\|runpod_api\)[[:space:]]*=[[:space:]]*//p' "${SCRIPT_DIR}/.env" \
    | head -1 | tr -d '\r' | sed -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//")"
fi

IMAGE="${IMAGE:-runpod/pytorch:1.3.1-cu1281-torch260-ubuntu2204}"
CLOUD_TYPE="${CLOUD_TYPE:-SECURE}"
DISK_GB="${DISK_GB:-150}"
VOLUME_GB="${VOLUME_GB:-100}"
POD_NAME="${POD_NAME:-editalive}"
TEMPLATE_ID="${TEMPLATE_ID:-}"
TEMPLATE_NAME="${TEMPLATE_NAME:-editalive}"

command -v jq >/dev/null || { echo "jq is required" >&2; exit 1; }
command -v curl >/dev/null || { echo "curl is required" >&2; exit 1; }

log()  { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[warn] %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[1;31m[error] %s\033[0m\n' "$*" >&2; exit 1; }

api() {
  : "${RUNPOD_API_KEY:?Set RUNPOD_API_KEY in your environment first}"
  local method="$1" path="$2" body="${3:-}"
  if [[ -n "${body}" ]]; then
    curl -sS -X "${method}" "${API}${path}" \
      -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
      -H "Content-Type: application/json" \
      -d "${body}"
  else
    curl -sS -X "${method}" "${API}${path}" \
      -H "Authorization: Bearer ${RUNPOD_API_KEY}"
  fi
}

gpu_id() {
  case "${1:-4090}" in
    4090)      echo "NVIDIA GeForce RTX 4090" ;;
    5090)      echo "NVIDIA GeForce RTX 5090" ;;
    l40s)      echo "NVIDIA L40S" ;;
    a6000)     echo "NVIDIA RTX A6000" ;;
    h100)      echo "NVIDIA H100 PCIe" ;;
    h100-sxm)  echo "NVIDIA H100 80GB HBM3" ;;
    a100)      echo "NVIDIA A100 80GB PCIe" ;;
    *)         echo "$1" ;;   # pass a raw RunPod GPU type id through
  esac
}

cmd_template() {
  [[ "${IMAGE}" == runpod/* ]] && {
    echo "Set IMAGE to your pushed registry image, e.g." >&2
    echo "  IMAGE=ghcr.io/<user>/editalive:cu128-torch260 ./runpod-pod.sh template" >&2
    exit 1
  }
  echo "Registering template '${TEMPLATE_NAME}' -> ${IMAGE}"

  local body
  body="$(jq -n \
    --arg name  "${TEMPLATE_NAME}" \
    --arg image "${IMAGE}" \
    --argjson disk "${DISK_GB}" \
    --argjson vol  "${VOLUME_GB}" \
    '{
      name: $name,
      imageName: $image,
      category: "NVIDIA",
      isServerless: false,
      isPublic: false,
      containerDiskInGb: $disk,
      volumeInGb: $vol,
      volumeMountPath: "/workspace",
      ports: ["22/tcp", "7860/http"],
      env: { HF_HOME: "/workspace/hf", HF_HUB_ENABLE_HF_TRANSFER: "1" },
      readme: "EditaLive prebuilt (CUDA 12.x, torch 2.6.0, flash-attn, fastvideo-kernel)."
    }')"

  local resp; resp="$(api POST /templates "${body}")"
  if ! jq -e '.id' >/dev/null 2>&1 <<<"${resp}"; then
    echo "Template create failed:" >&2
    jq . <<<"${resp}" >&2 || echo "${resp}" >&2
    exit 1
  fi

  local id; id="$(jq -r '.id' <<<"${resp}")"
  echo "Template ${id} created."
  echo "$id" > .last_template_id
  echo
  echo "Boot a Pod from it:"
  echo "  TEMPLATE_ID=${id} ./runpod-pod.sh create 4090"
}

cmd_create() {
  local gpu; gpu="$(gpu_id "${1:-4090}")"
  local use_template="${TEMPLATE_ID:-}"
  if [[ -n "${use_template}" ]]; then
    echo "Creating '${POD_NAME}' on ${gpu} (${CLOUD_TYPE}) from template ${use_template}..."
  else
    echo "Creating '${POD_NAME}' on ${gpu} (${CLOUD_TYPE}) from image ${IMAGE}..."
  fi

  local body
  if [[ -n "${use_template}" ]]; then
    body="$(jq -n \
      --arg name   "${POD_NAME}" \
      --arg cloud  "${CLOUD_TYPE}" \
      --arg gpu    "${gpu}" \
      --arg tid    "${use_template}" \
      '{
        name: $name,
        templateId: $tid,
        cloudType: $cloud,
        computeType: "GPU",
        gpuTypeIds: [$gpu],
        gpuTypePriority: "availability",
        gpuCount: 1
      }')"
  else
    body="$(jq -n \
      --arg name   "${POD_NAME}" \
      --arg image  "${IMAGE}" \
      --arg cloud  "${CLOUD_TYPE}" \
      --arg gpu    "${gpu}" \
      --argjson disk   "${DISK_GB}" \
      --argjson vol    "${VOLUME_GB}" \
      '{
        name: $name,
        imageName: $image,
        cloudType: $cloud,
        computeType: "GPU",
        gpuTypeIds: [$gpu],
        gpuTypePriority: "availability",
        gpuCount: 1,
        containerDiskInGb: $disk,
        volumeInGb: $vol,
        volumeMountPath: "/workspace",
        ports: ["22/tcp", "7860/http"],
        env: { HF_HOME: "/workspace/hf", HF_HUB_ENABLE_HF_TRANSFER: "1" },
        minRAMPerGPU: 32,
        minVCPUPerGPU: 4
      }')"
  fi

  local resp; resp="$(api POST /pods "${body}")"
  if ! jq -e '.id' >/dev/null 2>&1 <<<"${resp}"; then
    echo "Create failed:" >&2
    jq . <<<"${resp}" >&2 || echo "${resp}" >&2
    exit 1
  fi

  local id; id="$(jq -r '.id' <<<"${resp}")"
  echo "Pod ${id} created."
  echo "$id" > .last_pod_id
  echo
  echo "Wait ~1-2 min for the container to boot, then over SSH:"
  echo "  ./runpod-pod.sh ssh ${id}"
  if [[ -n "${use_template}" ]]; then
    echo "  editalive-init            # clones repo + fetches weights (skipped if cached)"
    echo "  cd /workspace/EditaLive"
    echo "  ./run-streaming.sh"
  else
    echo "  scp -P <port> runpod-setup.sh root@<ip>:/workspace/"
    echo "  cd /workspace && bash runpod-setup.sh"
  fi
}

# Create a pod, wait until SSH is reachable, then start editalive-init remotely.
cmd_up() {
  local gpu="${1:-4090}"
  cmd_create "${gpu}"

  local id
  id="$(cat "${SCRIPT_DIR}/.last_pod_id" 2>/dev/null || true)"
  [[ -n "${id}" ]] || die "Could not determine the new pod id."

  local timeout="${SSH_READY_TIMEOUT:-900}" waited=0 ip="" port=""
  log "Waiting for pod ${id} to expose SSH (up to ${timeout}s)"
  while (( waited < timeout )); do
    local json; json="$(api GET "/pods/${id}")"
    ip="$(jq -r '.publicIp // empty' <<<"${json}")"
    port="$(jq -r '.portMappings["22"] // empty' <<<"${json}")"
    [[ -n "${ip}" && -n "${port}" ]] && break
    sleep 15; waited=$((waited + 15))
    printf '  ... %ss\n' "${waited}"
  done

  if [[ -z "${ip}" || -z "${port}" ]]; then
    warn "SSH not ready after ${waited}s. Check the RunPod console, then:"
    echo "  ./runpod-pod.sh ssh ${id}"
    return 1
  fi
  log "SSH ready: ssh root@${ip} -p ${port}"

  if [[ "${EDITALIVE_INIT:-1}" != "1" ]]; then
    echo "EDITALIVE_INIT=0 — skipping remote init."
    echo "  ./runpod-pod.sh ssh ${id}"
    return 0
  fi

  local ssh_opts=(-n -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -p "${port}")
  log "Starting remote init (weights download runs in the background)"
  if [[ -n "${TEMPLATE_ID:-}" ]]; then
    ssh "${ssh_opts[@]}" "root@${ip}" \
      'nohup bash /usr/local/bin/editalive-init > /workspace/init.log 2>&1 & echo "init started (pid $!)"'
  else
    scp -o StrictHostKeyChecking=accept-new -P "${port}" \
      "${SCRIPT_DIR}/runpod-setup.sh" "root@${ip}:/workspace/runpod-setup.sh"
    ssh "${ssh_opts[@]}" "root@${ip}" \
      'nohup bash /workspace/runpod-setup.sh > /workspace/init.log 2>&1 & echo "init started (pid $!)"'
  fi

  cat <<EOF

Pod ${id} is initializing. Next:
  ./runpod-pod.sh ssh ${id}                      # open the pod
  tail -f /workspace/init.log                    # watch progress (~9 min)
  cd /workspace/EditaLive && ./run-streaming.sh  # once init prints "Done."
EOF
}

cmd_list() {
  api GET /pods | jq -r '
    (["ID","NAME","STATUS","GPU","$/HR"] | @tsv),
    (.[] | [.id, .name, (.desiredStatus // "?"),
            (.machine.gpuTypeId // "?"),
            ((.costPerHr // "?") | tostring)] | @tsv)' | column -t -s $'\t'
}

cmd_get() { api GET "/pods/$1" | jq .; }

cmd_status() {
  api GET "/pods/$1" | jq -r '"\(.id)  \(.name)  \(.desiredStatus)  $(\(.costPerHr // "?" )/hr)"'
}

cmd_ssh() {
  local id="$1" json
  json="$(api GET "/pods/${id}")"
  local ip port
  ip="$(jq -r '.publicIp // empty' <<<"${json}")"
  port="$(jq -r '.portMappings["22"] // empty' <<<"${json}")"
  if [[ -n "${ip}" && -n "${port}" ]]; then
    echo "ssh root@${ip} -p ${port}"
  else
    cat <<EOF
No direct IP/port yet (still starting, or Community Cloud without a public IP).
Wait a moment and retry, or use the RunPod web terminal, or:
  pip install runpodctl && runpodctl ssh ${id}
EOF
  fi
}

cmd_stop()      { api POST "/pods/$1/stop"      >/dev/null && echo "Stop requested for $1"; }
cmd_terminate() { api DELETE "/pods/$1"         >/dev/null && echo "Terminate requested for $1"; }

case "${1:-}" in
  up)        shift; cmd_up "$@" ;;
  create)    shift; cmd_create "$@" ;;
  template)  cmd_template ;;
  list)      cmd_list ;;
  get)       shift; cmd_get "$@" ;;
  status)    shift; cmd_status "$@" ;;
  ssh)       shift; cmd_ssh "$@" ;;
  stop)      shift; cmd_stop "$@" ;;
  terminate) shift; cmd_terminate "$@" ;;
  *) sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//' ;;
esac
