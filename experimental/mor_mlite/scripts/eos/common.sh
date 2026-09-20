#!/usr/bin/env bash

# Shared constants and helpers for the fixed EOS v1 environment.  This file is
# sourced by the other launchers; it performs no work on its own.

readonly EOS_LOGIN="login-eos"
readonly EOS_WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
readonly CONTAINER_IMAGE_SOURCE="nvcr.io/nvidia/pytorch:26.01-py3"
readonly MEGATRON_SHA="5c8315f12a64a7279eec58896af9e74ee3351b74"
readonly MAGI_TAG="v1.1.1"
readonly MAX_WORLD_SIZE=8
readonly TORCH_WHEEL_VERSION="2.10.0+cu129"
readonly TORCH_WHEEL_INDEX="https://download.pytorch.org/whl/cu129"
# Megatron's pinned MLite sources require
# transformer_engine.pytorch.permutation.moe_permute_and_pad_with_probs.  It is
# absent from TE 2.11.  TE 2.13 is the newest release that both exports that
# API and whose PyTorch sdist resolves the CUDA-12 core package; TE 2.14's
# published PyTorch sdist metadata unconditionally selects the CUDA-13 core.
readonly TRANSFORMER_ENGINE_VERSION="2.13.0"
readonly NVRX_VERSION="0.6.0"

readonly DEPS_ROOT="${MOR_DEPS_ROOT:-${EOS_WORKDIR}/.deps}"
readonly MEGATRON_LM_ROOT="${DEPS_ROOT}/Megatron-LM"
readonly MAGI_ATTENTION_SOURCE="${DEPS_ROOT}/MagiAttention"
# Bump the environment key whenever the dependency lock changes.  This keeps a
# failed/partially upgraded install isolated without deleting recoverable EOS
# state shared by earlier jobs.
readonly MOR_VENV="${DEPS_ROOT}/venv-torch210-cu129-v3"
readonly MOR_PIP_CONSTRAINT="${EOS_WORKDIR}/scripts/eos/constraints-cu129.txt"

die() {
    echo "error: $*" >&2
    exit 2
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

require_eos_project() {
    local resolved
    resolved="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
    [[ "${resolved}" == "${EOS_WORKDIR}" ]] || \
        die "expected project at ${EOS_WORKDIR}, resolved ${resolved}"
}

activate_mor_environment() {
    local pinned_pythonpath
    [[ -x "${MOR_VENV}/bin/python" ]] || \
        die "venv is missing; run scripts/eos/setup_env.sh first"
    export PATH="${MOR_VENV}/bin:${PATH}"
    export MOR_VENV
    export MOR_PROJECT_ROOT="${EOS_WORKDIR}"
    export MEGATRON_LM_ROOT
    export MAGI_ATTENTION_SOURCE
    export MOR_CONTAINER_IMAGE_SOURCE="${CONTAINER_IMAGE_SOURCE}"
    pinned_pythonpath="${EOS_WORKDIR}/src:${MEGATRON_LM_ROOT}/experimental/lite"
    pinned_pythonpath="${pinned_pythonpath}:${MEGATRON_LM_ROOT}"
    export PYTHONPATH="${pinned_pythonpath}${PYTHONPATH:+:${PYTHONPATH}}"
    export HF_HOME="${MOR_HF_HOME:-${EOS_WORKDIR}/.cache/huggingface}"
    export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
    export TOKENIZERS_PARALLELISM="false"
    export PYTHONNOUSERSITE=1
    # Freeze the same sm90 execution switches as pinned MLite's sanctioned Magi
    # wrapper. Slurm inherits the submitting shell, so every switch must be set
    # explicitly rather than accepting a caller's unrelated experiment flags.
    unset MAGI_ATTENTION_SDPA_BACKEND MAGI_ATTENTION_FA4_BACKEND
    export MAGI_ATTENTION_KERNEL_BACKEND=ffa
    export MAGI_ATTENTION_NATIVE_GRPCOLL=0
    export MAGI_ATTENTION_HIERARCHICAL_COMM=0
    export MAGI_ATTENTION_QO_COMM=0
    export MAGI_ATTENTION_DETERMINISTIC_MODE=0
    export MAGI_ATTENTION_FORWARD_HIGH_PRECISION_REDUCE=0
    export MAGI_ATTENTION_BACKWARD_HIGH_PRECISION_REDUCE=0
    export MAGI_ATTENTION_BWD_HIDE_TAIL_REDUCE=0
}
