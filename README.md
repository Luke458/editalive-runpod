# EditaLive on RunPod

Tooling to run [EditaLive](https://github.com/GVCLab/EditaLive) — real-time, streaming
character video editing — on rented RunPod GPUs.

EditaLive's inference stack is **CUDA-only** (FlashAttention + custom CUTLASS kernels),
so it cannot run on the AMD/ROCm machine this tooling was written on. Everything here
targets NVIDIA GPUs on RunPod.

The main problem this repo solves: a fresh pod spends several minutes compiling
`flash-attn` and `fastvideo-kernel` and then **downloads ~53 GB** of weights before it
can run anything. This repo bakes the compiles into an image, so the only first-boot
cost is the weight download (~9 min).

## Contents

| File | Purpose |
|---|---|
| `Dockerfile` | Prebuilt image: CUDA 12.x + torch 2.6.0 + Python 3.12 + ffmpeg + repo deps + `flash-attn` + `fastvideo-kernel`, verified at build. Adds `editalive-init`. |
| `build-and-push.sh` | Build and push the image locally (docker/podman; no GPU or local CUDA required). |
| `runpod-pod.sh` | Create/manage pods and register templates via the RunPod REST API. |
| `runpod-setup.sh` | On-pod setup (idempotent): clones the repo, installs anything missing, downloads weights, writes ready-to-run scripts. |
| `.github/workflows/build-image.yml` | CI that builds and pushes the image to GHCR. |

## Requirements

- A RunPod account and an [API key](https://www.runpod.io/console/user/settings).
- `jq` and `curl` locally (for `runpod-pod.sh`).
- Optional: `docker` or `podman` if you build the image yourself.
- Pulling the image needs no credentials if the GHCR package is public; otherwise see
  [Package visibility](#package-visibility).

> **Security:** never commit your RunPod API key. `runpod-pod.sh` reads it from
> `RUNPOD_API_KEY`, or automatically from a gitignored `.env` in this directory
> (accepting either `RUNPOD_API_KEY` or `runpod_api` as the variable name). Keep that
> file `chmod 600`. The repo's `.gitignore` already excludes `.env` and `outputs/`.

## Quick start

### 1. Register the prebuilt image as a template

The CI publishes the image to `ghcr.io/luke458/editalive-runpod:latest`.

```bash
export RUNPOD_API_KEY=...
cd editalive-runpod
IMAGE=ghcr.io/luke458/editalive-runpod:latest ./runpod-pod.sh template
# -> Template <id> created.  (also saved to .last_template_id)
```

Or build your own image first (see [Building the image](#building-the-image)).

### 2. Boot a pod and initialize it

One command creates the pod, waits for SSH, then starts the weight download in the
background:

```bash
TEMPLATE_ID=<id> ./runpod-pod.sh up 4090     # 4090 | l40s | h100 | h100-sxm | a100 | 5090
```

Then:

```bash
./runpod-pod.sh ssh <podId>          # open the pod
tail -f /workspace/init.log          # watch the ~53 GB download (~9 min)
cd /workspace/EditaLive
./run-streaming.sh                   # render the included demo
./run-webcam.sh                      # real-time webcam mode
```

To do it manually instead: `./runpod-pod.sh create 4090`, wait for the SSH mapping,
then run `editalive-init`. Set `EDITALIVE_INIT=0` on `up` to skip the automatic init.

`editalive-init` is idempotent — it skips the clone and any weights already present.

### 3. Webcam / streaming mode

`./run-webcam.sh` binds port 7860. Open the pod's proxy URL:

```
https://<pod-id>-7860.proxy.runpod.net
```

It is HTTPS, so browser camera access works without port forwarding. In the UI:
**Prepare model** → enter prompt → **Prepare prompt** → **Enable camera** →
**Start editing**. Defaults (FP8, Flash-VAED decoder, `torch.compile`) are tuned for a
24 GB card.

## Which GPU

`runpod-setup.sh` auto-detects VRAM and compute capability and writes `run-streaming.sh`
with sensible flags. Rough guidance for the 14B model:

| GPU | VRAM | FP8 | Flags | Resolution |
|---|---|---|---|---|
| RTX 4090 | 24 GB | yes | `--fp8 --low_vram weights+kv` | 384×672 |
| L40S | 48 GB | yes | `--fp8 --low_vram off` | 480×832 |
| H100 | 80 GB | yes | `--fp8 --low_vram off` | 480×832 |
| A100 | 80 GB | **no** (SM 8.0) | `--low_vram off` | 480×832 |
| RTX 5090 | 32 GB | yes | `--fp8 --low_vram weights+kv` | 384×672 |

A 16 GB card is below the project's tested floor and will likely OOM.

Indicative RunPod Secure Cloud rates (check current pricing): 4090 ~$0.74/hr,
L40S ~$0.99/hr, A100 ~$1.39/hr, H100 PCIe ~$2.89/hr, H100 SXM ~$3.49/hr. The image is
built for arch `8.0;8.9;9.0`, which covers all of the above.

## Building the image

Local build needs no GPU — `nvcc` comes from the base image inside the build container.
A full build takes ~10-20 min (flash-attn installs from a prebuilt wheel when one
matches; the CUTLASS kernel compiles from source); CI rebuilds reuse the registry cache
(~8 min).

Note: at runtime `nvcc` is not on `PATH` (it lives at `/usr/local/cuda/bin`). That is
fine because the image already ships the compiled kernels; `runpod-setup.sh` only needs
`nvcc` when it has to build one from scratch.

```bash
export REGISTRY=ghcr.io/<your-user>     # or docker.io/<your-user>
docker login ghcr.io -u <your-user>     # or: docker login

REGISTRY=ghcr.io/<your-user> ./build-and-push.sh
#   --no-push   build only
```

Build variables (env):

| Variable | Default | Notes |
|---|---|---|
| `IMAGE_NAME` | `editalive` | |
| `TAG` | `cu128-torch260` | |
| `TORCH_CUDA_ARCH_LIST` | `8.0;8.9;9.0` | **Must cover the GPU you'll rent.** Add `8.6` for A10/A40. Each extra arch lengthens the build. |
| `MAX_JOBS` | `nproc` | flash-attn compile parallelism; lower it if the build OOMs. |
| `CMAKE_BUILD_PARALLEL_LEVEL` | `nproc` | fastvideo-kernel compile parallelism. |
| `BAKE_WEIGHTS` | `0` | Set `1` to bake ~53 GB of weights into the image (not recommended). |

Then register the template:

```bash
IMAGE=ghcr.io/<your-user>/editalive:cu128-torch260 ./runpod-pod.sh template
```

### CI (GitHub Actions)

`.github/workflows/build-image.yml` builds and pushes to
`ghcr.io/<owner>/<repo>` on pushes touching `Dockerfile` / `runpod-setup.sh`, or
manually via **Actions → Build EditaLive image → Run workflow**. Manual runs accept
`torch_cuda_arch_list`, `bake_weights`, and `max_jobs` inputs.

The runner has 4 vCPU / 16 GB; if flash-attn fails with an out-of-memory error, re-run
with `-f max_jobs=2`:

```bash
gh workflow run "Build EditaLive image" -R <owner>/<repo> -f max_jobs=2
gh run watch -R <owner>/<repo>
```

#### Package visibility

RunPod must be able to pull the image. If the GHCR package is public, no credentials
are needed. If it stays private, either make it public (profile → Packages →
`editalive-runpod` → Package settings → Change visibility → Public) or add a RunPod
registry credential and pass `containerRegistryAuthId`.

## Building on the pod instead (no image)

If you'd rather not deal with a registry, use the stock RunPod PyTorch image
(`runpod/pytorch:1.3.1-cu1281-torch260-ubuntu2204`) and let the pod compile:

```bash
./runpod-pod.sh create 4090            # no TEMPLATE_ID -> uses the stock image
./runpod-pod.sh ssh <podId>
# from your local machine:
scp -P <port> runpod-setup.sh root@<ip>:/workspace/
# on the pod:
cd /workspace && bash runpod-setup.sh
```

`runpod-setup.sh` skips any step already completed, so it is safe to re-run.

## `runpod-pod.sh` reference

| Command | Description |
|---|---|
| `up [gpu]` | `create` + wait for SSH + start `editalive-init`. `EDITALIVE_INIT=0` skips init; `SSH_READY_TIMEOUT=900` changes the wait. |
| `create [4090\|5090\|l40s\|a6000\|h100\|h100-sxm\|a100]` | Create a pod. Raw RunPod GPU type IDs also work. |
| `template` | Register `IMAGE` as a RunPod template. |
| `list` | List pods with status and cost. |
| `get <podId>` | Full pod JSON. |
| `status <podId>` | One-line status. |
| `ssh <podId>` | Print the `ssh root@IP -p PORT` command. |
| `stop <podId>` | Stop compute billing; the volume keeps billing at $0.20/GB/mo. |
| `terminate <podId>` | Delete the pod and its volume (no further charges). |

Environment overrides:

| Variable | Default | Notes |
|---|---|---|
| `RUNPOD_API_KEY` | — | **Required.** |
| `TEMPLATE_ID` | — | When set, `create` boots from this template. |
| `TEMPLATE_NAME` | `editalive` | Name used by `template`. |
| `IMAGE` | stock RunPod PyTorch | Image used by `template` and by `create` when no `TEMPLATE_ID`. |
| `CLOUD_TYPE` | `SECURE` | Or `COMMUNITY` (cheaper, no public IP sometimes). |
| `DISK_GB` | `150` | Container disk (ephemeral; holds the image + build artifacts). |
| `VOLUME_GB` | `100` | Persistent pod volume mounted at `/workspace` (weights live here). |
| `POD_NAME` | `editalive` | |
| `EDITALIVE_INIT` | `1` | `up` runs `editalive-init` unless set to `0`. |
| `SSH_READY_TIMEOUT` | `900` | Seconds `up` waits for the SSH port mapping. |

## Storage and cost

Weights live on the pod volume at `/workspace` (`HF_HOME=/workspace/hf`, repo at
`/workspace/EditaLive`). Compute is billed per second while running.

| Storage | Running | Stopped |
|---|---|---|
| Container disk (150 GB) | $0.10/GB/mo | not charged |
| Volume disk (100 GB) | $0.10/GB/mo | **$0.20/GB/mo** |
| Network volume | $0.07/GB/mo | $0.07/GB/mo |

- **"Stop" is not free**: a stopped 100 GB pod volume costs ~$20/mo. For occasional use,
  `terminate` between sessions and let the weights re-download (~9 min) — that is what
  this tooling assumes.
- If your balance reaches $0, a pod **without** a network volume is terminated and its
  data is unrecoverable.
- Using it often? Put the weights on a **network volume** (~$7/mo for 100 GB, portable
  across pods) instead of paying the stopped-volume rate.

## Generated run scripts

`runpod-setup.sh` writes two wrappers next to the repo's `edit_streaming.py`:

- `run-streaming.sh [video] ["prompt"] [extra flags...]` — single render, output in
  `./outputs_streaming`. Add `--enable_compile True` for long/repeated runs (~1.3× faster
  after a one-time warm-up).
- `run-webcam.sh [extra flags...]` — the interactive server on port 7860.

Both are generated from the detected GPU, so review them if you change GPUs.

## Upstream

- Repo: https://github.com/GVCLab/EditaLive
- Weights: https://huggingface.co/huaichang/EditaLive
- EditaLive is released for **academic research only**. Follow its license and
  disclaimer; do not generate harmful, defamatory, or illegal content.
