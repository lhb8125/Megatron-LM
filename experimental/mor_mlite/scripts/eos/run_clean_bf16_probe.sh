#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROBE_PROJECT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
SOURCE_DEPS="${MOR_DEPS_ROOT:?set the existing pinned dependency root}"
PROBE_DEPS="${PROBE_PROJECT}/.deps-cu129-clean"
PROBE_VENV="${PROBE_DEPS}/venv-torch210-cu129-v3"
# Copy the installed cu129 distributions, not the base image's site-packages.
# The source environment is read-only throughout this workflow.
if [[ ! -f "${PROBE_VENV}/pyvenv.cfg" ]]; then
    /usr/bin/python3 -m venv --without-pip "${PROBE_VENV}"
    cp -a "${SOURCE_DEPS}/venv-torch210-cu129-v3/lib/python3.12/site-packages/." \
        "${PROBE_VENV}/lib/python3.12/site-packages/"
fi
for dependency in Megatron-LM MagiAttention; do
    if [[ ! -e "${PROBE_DEPS}/${dependency}" ]]; then
        ln -s "${SOURCE_DEPS}/${dependency}" "${PROBE_DEPS}/${dependency}"
    fi
done
export MOR_DEPS_ROOT="${PROBE_DEPS}"
source "${SCRIPT_DIR}/common.sh"
activate_mor_environment
# Keep support dependencies at the versions used by the successful original
# runtime, but install them physically in the clean environment.
mapfile -t SUPPORT_PACKAGES < <("${SOURCE_DEPS}/venv-torch210-cu129-v3/bin/python" -c '
from importlib.metadata import version
names = ["huggingface-hub", "safetensors", "defusedxml", "httpx", "nvidia-ml-py",
         "psutil", "iniconfig", "pluggy", "debugpy"]
for name in names:
    print(f"{name}=={version(name)}")
')
[[ "${#SUPPORT_PACKAGES[@]}" -eq 9 ]] || die "failed to resolve original support package versions"
export PIP_CACHE_DIR="${PROBE_DEPS}/pip-cache"
python -m pip install --constraint "${MOR_PIP_CONSTRAINT}" "${SUPPORT_PACKAGES[@]}"
python "${SCRIPT_DIR}/repair_te_wheel_tag.py" --venv "${MOR_VENV}"
python -m pip check
NVIDIA_LIB_ROOT="${MOR_VENV}/lib/python3.12/site-packages/nvidia"
export CUDNN_HOME="${NVIDIA_LIB_ROOT}/cudnn"
export NVRTC_HOME="${NVIDIA_LIB_ROOT}/cuda_nvrtc"
export CURAND_HOME="${NVIDIA_LIB_ROOT}/curand"
export CUDA_HOME="${NVIDIA_LIB_ROOT}/cuda_runtime"
if [[ "${MOR_CORE_PROBE_BACKEND:-te_fused}" == magi_local ]]; then
    # FFA may JIT an exact deterministic/GQA variant. Use the real compiler
    # toolkit pinned by env_check, not the header-only Torch runtime wheel.
    export CUDA_HOME=/usr/local/cuda
    export XDG_CACHE_HOME="${PROBE_PROJECT}/.cache"
    export TORCH_CUDA_ARCH_LIST=9.0a
fi
export NVTE_CUDA_INCLUDE_DIR="${NVIDIA_LIB_ROOT}/cuda_runtime/include"
export LD_LIBRARY_PATH="${NVIDIA_LIB_ROOT}/cuda_runtime/lib:${NVIDIA_LIB_ROOT}/cudnn/lib:${NVIDIA_LIB_ROOT}/cublas/lib:${NVIDIA_LIB_ROOT}/cuda_nvrtc/lib:${NVIDIA_LIB_ROOT}/curand/lib:${LD_LIBRARY_PATH:-}"
"${SCRIPT_DIR}/version_probe.sh" 1
export NVTE_FLASH_ATTN=0 NVTE_FUSED_ATTN=1 NVTE_UNFUSED_ATTN=0
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
python "${SCRIPT_DIR}/probe_bf16_fused.py" \
    --backend "${MOR_CORE_PROBE_BACKEND:-te_fused}" \
    --traces "${PROBE_PROJECT}/artifacts/attention-6006269/baseline.pt" \
    --report "${PROBE_PROJECT}/artifacts/fused-${SLURM_JOB_ID}/kernel.json"
