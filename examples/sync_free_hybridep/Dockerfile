# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# IMAGE_NAME: b300-sync-free-hybridep-te215
#
# Single-node B300 (SM103a) image for the HybridEP sync-free MoE + 1F1B combined
# overlap experiment. Path A of docs/hybridep_paged_stash_sync_free:
#   TE op-fuser + CuTe DSL fused GroupedMLP + full-iteration CUDA graph, on top
#   of HybridEP static-budget dispatch + paged stash.
#
# Adapted from
#   examples/moe_recipes/qwen3_235b/gb200/mxfp8_128GPU_TP1PP1EP64_paged_stash_fullcg_overlap.yaml
# for a single B300 node (compute cap 10.3):
#   - NVTE_CUDA_ARCHS / TORCH_CUDA_ARCH_LIST target SM103a only.
#   - DeepEP built from the hybrid-ep branch (provides deep_ep.HybridEPBuffer).

ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:26.04-py3
FROM ${FROM_IMAGE_NAME} AS base

ENV SHELL=/bin/bash
ENV DEBIAN_FRONTEND=noninteractive

ARG YQ_VERSION=4.27.5
# TE v2.15 main (matches the full-CG CuTe DSL fused grouped MLP recipe).
ARG TE_COMMIT="6cdd7115e1c1e6605feaff6ebe111ea0466cd71a"
# 103a = SM103a (B300 / GB300).
ARG NVTE_CUDA_ARCHS="103a"

# Install system dependencies
RUN bash -ex <<"EOF"
    rm -rf /opt/megatron-lm
    apt-get update
    apt-get install -y --no-install-recommends git curl gettext sudo wget
    ARCH=$(uname -m)
    case "${ARCH}" in
      "x86_64") YQ_ARCH=amd64; YQ_SHA256=9a54846e81720ae22814941905cd3b056ebdffb76bf09acffa30f5e90b22d615 ;;
      "aarch64") YQ_ARCH=arm64; YQ_SHA256=ea360a0ecdff30c8625ccd0b97f8714b8308a429fd839cf8ccc481f311e217c6 ;;
      *) echo "Unsupported architecture: ${ARCH}" && exit 1 ;;
    esac
    wget https://github.com/mikefarah/yq/releases/download/v${YQ_VERSION}/yq_linux_${YQ_ARCH} -O /usr/bin/yq
    echo "${YQ_SHA256}  /usr/bin/yq" | sha256sum --check -
    chmod +x /usr/bin/yq
    apt-get clean
    rm -rf /var/lib/apt/lists/*
EOF

# Install Megatron-Core python dependencies (mlm + dev + test groups)
RUN bash -ex <<"EOF"
    unset PIP_CONSTRAINT
    pip install \
        sentencepiece tiktoken transformers accelerate \
        wandb einops tqdm datasets nvtx \
        flask-restful flask[async] fastapi hypercorn \
        nltk wrapt pydantic pyyaml omegaconf \
        nvidia-resiliency-ext==0.6.0 \
        pytest==8.3.5 pytest-mock pytest-cov pytest-random-order pytest-asyncio \
        coverage tensorboard
EOF

# Install latest cuDNN (libcudnn9-cuda-13) from NVIDIA's CUDA apt repo.
RUN bash -ex <<"EOF"
    ARCH=$(uname -m)
    case "${ARCH}" in
      "x86_64") REPO_ARCH=x86_64 ;;
      "aarch64") REPO_ARCH=sbsa ;;
      *) echo "Unsupported architecture: ${ARCH}" && exit 1 ;;
    esac
    apt-get update
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/${REPO_ARCH}/cuda-keyring_1.1-1_all.deb
    dpkg -i cuda-keyring_1.1-1_all.deb
    rm cuda-keyring_1.1-1_all.deb
    apt-get update
    apt-get install -y --no-install-recommends libcudnn9-cuda-13
    apt-get clean
    rm -rf /var/lib/apt/lists/*
EOF

# Install TransformerEngine (v2.15 main)
RUN unset PIP_CONSTRAINT && \
    NVTE_CUDA_ARCHS="${NVTE_CUDA_ARCHS}" NVTE_BUILD_THREADS_PER_JOB=8 NVTE_FRAMEWORK=pytorch \
    pip install --no-build-isolation --no-cache-dir \
    "git+https://github.com/NVIDIA/TransformerEngine.git@${TE_COMMIT}"

# Flash Attention 4, CUTLASS DSL, and cuDNN frontend with CuTe DSL support.
# nvidia-cudnn-frontend[cutedsl] is required for the TE v2.15 CuTe DSL fused
# grouped MLP (grouped_gemm_glu_wrapper_sm100).
# Pin nvidia-cutlass-dsl==4.5.2: it is the version quack-kernels 0.5.0 (pulled in
# by cudnn-frontend[cutedsl]) is built against. 4.7.0 renamed/removed
# cutlass.cute.core.ThrMma and breaks quack's fused grouped MLP import.
RUN pip install --no-cache-dir flash-attn-4==4.0.0b10 "nvidia-cutlass-dsl[cu13]==4.5.2" && \
    pip install --no-cache-dir "nvidia-cudnn-frontend[cutedsl]>=1.23.0" && \
    pip install --no-cache-dir "nvidia-cutlass-dsl[cu13]==4.5.2"

# =========================
# HybridEP (DeepEP hybrid-ep branch -> deep_ep.HybridEPBuffer)
# =========================
FROM base AS hybridep

RUN bash -ex <<"EOF"
    cd /workspace
    git clone https://github.com/linux-rdma/rdma-core.git
    cd rdma-core && git checkout tags/v60.0 && sh build.sh
    apt-get update
    apt-get install -y --no-install-recommends libnvidia-ml-dev
    git clone --branch hybrid-ep https://github.com/deepseek-ai/DeepEP.git
    # HybridEP hybrid-ep tip (DOCA path; drop the NIXL marker to build without NIXL).
    cd DeepEP && git checkout d28bd676c2120573c9f1425f0c16c39faa4117e6
    rm -f .use_nixl
    RDMA_CORE_HOME=/workspace/rdma-core/build HYBRID_EP_MULTINODE=1 USE_NIXL=0 \
        TORCH_CUDA_ARCH_LIST="10.3" MAX_JOBS=8 \
        pip install --no-build-isolation .
    apt-get purge -y libnvidia-ml-dev
    apt-get autoremove -y
    rm -rf /root/.cache /tmp/* /var/lib/apt/lists/*
EOF

WORKDIR /workspace/
