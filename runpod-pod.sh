#!/usr/bin/env bash
#
# Create and manage an EditaLive GPU pod on RunPod from your local machine.
#
# Auth: export RUNPOD_API_KEY=...   (never hardcode it here)
#       Get one at https://www.runpod.io/console/user/settings
#
# Usage:
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
set -euo pipefail

API="https://rest.runpod.io/v1"

IMAGE="${IMAGE:-runpod/pytorch:1.3.1-cu1281-torch260-ubuntu2204}"
CLOUD_TYPE="${CLOUD_TYPE:-SECURE}"
DISK_GB="${DISK_GB:-150}"
VOLUME_GB="${VOLUME_GB:-100}"
POD_NAME="${POD_NAME:-editalive}"
TEMPLATE_ID="${TEMPLATE_ID:-}"
TEMPLATE_NAME="${TEMPLATE_NAME:-editalive}"

command -v jq >/dev/null || { echo "jq is required" >&2; exit 1; }
command -v curl >/dev/null || { echo "curl is required" >&2; exit 1; }

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
      readme: "EditaLive prebuilt (CUDA 12.8, torch 2.6.0, flash-attn, fastvideo-kernel)."
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
  create)    shift; cmd_create "$@" ;;
  template)  cmd_template ;;
  list)      cmd_list ;;
  get)       shift; cmd_get "$@" ;;
  status)    shift; cmd_status "$@" ;;
  ssh)       shift; cmd_ssh "$@" ;;
  stop)      shift; cmd_stop "$@" ;;
  terminate) shift; cmd_terminate "$@" ;;
  *) sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//' ;;
esac
