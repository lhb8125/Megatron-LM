#!/bin/bash
# Build a Megatron-LM image that bakes the CURRENT working tree on top of a
# prebuilt sync-free base image (CUDA/cuDNN/cuBLAS + PyTorch + TransformerEngine
# + DeepEP/HybridEP + resiliency-ext already installed there).
#
# Two prebuilt bases are supported, pick one with --deepep / --hybridep:
#   --deepep    : DeepEP cross-node base (NVSHMEM built with IBGDA/GDRCopy),
#                 for moe_flex_dispatcher_backend=deepep multinode runs.
#                 -> megatron-sync-free:e60981c20-20260814-1306
#   --hybridep  : HybridEP base (validated for grouped_tensor training),
#                 for moe_flex_dispatcher_backend=hybridep runs.
#                 -> megatron-sync-free:6401e2645-20260812-2227  (default)
#
# Usage:
#   docker/mm/build_docker.sh [--deepep|--hybridep] [--no-sudo] [--bake-helpers] [--no-push]
#
# By default the image is pushed to harbor (harbor.xaminim.com/minimax-dialogue).
# Pass --no-push to build locally only.
#
# Env overrides:
#   BASE_IMAGE   base image to overlay onto (overrides --deepep/--hybridep)
#   IMAGE_NAME   output repo name  (default: megatron-sync-free)
#   IMAGE_TAG    output tag        (default: <commit>-<timestamp>[-dirty])
#   REGISTRY     push destination  (default: harbor.xaminim.com/minimax-dialogue;
#                set empty with REGISTRY= to build a local-only tag)
set -xve

DEEPEP_BASE="harbor.xaminim.com/minimax-dialogue/megatron-sync-free:e60981c20-20260814-1306"
HYBRIDEP_BASE="harbor.xaminim.com/minimax-dialogue/megatron-sync-free:6401e2645-20260812-2227"

DOCKER_CMD="sudo docker"
BAKE_HELPERS=0
DO_PUSH=1
BACKEND="hybridep"
for arg in "$@"; do
    case "$arg" in
        --no-sudo)      DOCKER_CMD="docker" ;;
        --bake-helpers) BAKE_HELPERS=1 ;;
        --push)         DO_PUSH=1 ;;
        --no-push)      DO_PUSH=0 ;;
        --deepep)       BACKEND="deepep" ;;
        --hybridep)     BACKEND="hybridep" ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_DIR}"

# Base image: pick by backend; BASE_IMAGE env overrides.
if [ "$BACKEND" = "deepep" ]; then
    BASE_IMAGE=${BASE_IMAGE:-${DEEPEP_BASE}}
else
    BASE_IMAGE=${BASE_IMAGE:-${HYBRIDEP_BASE}}
fi
echo "Selected backend=${BACKEND}, BASE_IMAGE=${BASE_IMAGE}"

COMMIT_HASH=$(git rev-parse --short=9 HEAD 2>/dev/null || echo nogit)
DIRTY=""
if ! git diff --quiet 2>/dev/null || ! git diff --cached --quiet 2>/dev/null; then
    DIRTY="-dirty"
fi
TIMESTAMP=$(date +%Y%m%d-%H%M)
IMAGE_NAME=${IMAGE_NAME:-megatron-sync-free}
IMAGE_TAG=${IMAGE_TAG:-${COMMIT_HASH}-${TIMESTAMP}${DIRTY}}

# Push to harbor by default (matches the other Megatron repo's build_docker.sh).
# Override REGISTRY to change the destination, or pass --no-push to keep it local.
REGISTRY=${REGISTRY:-harbor.xaminim.com/minimax-dialogue}

if [ -n "${REGISTRY:-}" ]; then
    VERSION="${REGISTRY%/}/${IMAGE_NAME}:${IMAGE_TAG}"
else
    VERSION="${IMAGE_NAME}:${IMAGE_TAG}"
fi

${DOCKER_CMD} build \
    --build-arg BASE_IMAGE="${BASE_IMAGE}" \
    --build-arg BAKE_HELPERS="${BAKE_HELPERS}" \
    -t "${VERSION}" \
    -f docker/mm/Dockerfile \
    "${REPO_DIR}"

if [ "$DO_PUSH" -eq 1 ]; then
    if [ -z "${REGISTRY:-}" ]; then
        echo "WARNING: --push given but REGISTRY is empty; skipping push." >&2
    else
        ${DOCKER_CMD} push "${VERSION}"
    fi
fi

echo "BUILT_IMAGE=${VERSION}"
