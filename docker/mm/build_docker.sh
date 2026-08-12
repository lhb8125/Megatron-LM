#!/bin/bash
# Build a Megatron-LM image that bakes the CURRENT working tree on top of the
# prebuilt sync-free base image (CUDA/cuDNN/cuBLAS + PyTorch + TransformerEngine
# + DeepEP hybrid-ep + resiliency-ext already installed there).
#
# Usage:
#   docker/mm/build_docker.sh [--no-sudo] [--bake-helpers] [--path-a] [--push]
#
# Env overrides:
#   BASE_IMAGE   base image to overlay onto
#                (default: megatron-sync-free-grouped-tensor:b300  == Path B;
#                 use --path-a for megatron-sync-free-hybridep:b300)
#   IMAGE_NAME   output repo name  (default: megatron-sync-free)
#   IMAGE_TAG    output tag        (default: <commit>-<timestamp>[-dirty])
#   REGISTRY     if set (or --push given with a registry), the image is pushed
set -xve

DOCKER_CMD="sudo docker"
BAKE_HELPERS=0
DO_PUSH=0
PATH_A=0
for arg in "$@"; do
    case "$arg" in
        --no-sudo)      DOCKER_CMD="docker" ;;
        --bake-helpers) BAKE_HELPERS=1 ;;
        --push)         DO_PUSH=1 ;;
        --path-a)       PATH_A=1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_DIR}"

# Base image: Path B (device-init GroupedTensor) by default; Path A on request.
if [ "$PATH_A" -eq 1 ]; then
    BASE_IMAGE=${BASE_IMAGE:-megatron-sync-free-hybridep:b300}
else
    BASE_IMAGE=${BASE_IMAGE:-megatron-sync-free-grouped-tensor:b300}
fi

COMMIT_HASH=$(git rev-parse --short=9 HEAD 2>/dev/null || echo nogit)
DIRTY=""
if ! git diff --quiet 2>/dev/null || ! git diff --cached --quiet 2>/dev/null; then
    DIRTY="-dirty"
fi
TIMESTAMP=$(date +%Y%m%d-%H%M)
IMAGE_NAME=${IMAGE_NAME:-megatron-sync-free}
IMAGE_TAG=${IMAGE_TAG:-${COMMIT_HASH}-${TIMESTAMP}${DIRTY}}

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
