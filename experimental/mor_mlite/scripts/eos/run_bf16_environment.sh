#!/usr/bin/env bash
# Reuse the independently prepared runtime without installing during tests.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/common.sh"
require_eos_project
activate_mor_environment
[[ "$#" -gt 0 ]] || die "expected an acceptance script"
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST=9.0a
export MAGI_ATTENTION_WORKSPACE_BASE="${MOR_FFA_WORKSPACE:-${EOS_WORKDIR}}"
NVIDIA_LIB_ROOT="${MOR_VENV}/lib/python3.12/site-packages/nvidia"
export CUDNN_HOME="${NVIDIA_LIB_ROOT}/cudnn"
export NVRTC_HOME="${NVIDIA_LIB_ROOT}/cuda_nvrtc"
export CURAND_HOME="${NVIDIA_LIB_ROOT}/curand"
export NVTE_CUDA_INCLUDE_DIR="${NVIDIA_LIB_ROOT}/cuda_runtime/include"
export LD_LIBRARY_PATH="${NVIDIA_LIB_ROOT}/cuda_runtime/lib:${NVIDIA_LIB_ROOT}/cudnn/lib:${NVIDIA_LIB_ROOT}/cublas/lib:${NVIDIA_LIB_ROOT}/cuda_nvrtc/lib:${NVIDIA_LIB_ROOT}/curand/lib:${LD_LIBRARY_PATH:-}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
exec bash "$@"
