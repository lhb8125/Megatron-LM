#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/common.sh"
require_eos_project
activate_mor_environment
INIT_ROOT="${MOR_FOLDED_INIT_ROOT:?set the existing EP1/EP4 folded checkpoint root}"
CORE_TRACES="${MOR_CORE_TRACES:?set the captured BF16 QKV trace file}"
[[ -d "${INIT_ROOT}/folded_init_ep1" && -d "${INIT_ROOT}/folded_init_ep4" ]] || die "missing folded checkpoints"
[[ -f "${CORE_TRACES}" ]] || die "missing core traces"
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
"${SCRIPT_DIR}/version_probe.sh" 8
ROOT="${EOS_WORKDIR}/artifacts/bf16-${SLURM_JOB_ID}/qwen30b"
mkdir -p "${ROOT}/reports"
python -m pytest -q tests --junitxml="${ROOT}/reports/pytest.xml"
python "${SCRIPT_DIR}/probe_bf16_fused.py" --backend magi_adapter \
    --traces "${CORE_TRACES}" --report "${ROOT}/reports/core.json"
COMMON=(--backend mlite --preset qwen3-30b --precision bf16 --seq-lens 128,128
    --num-microbatches 1 --steps 1 --adam-eps 1e-6 --forward-only --no-checkpoint-roundtrip)
python -m torch.distributed.run --standalone --nproc-per-node=1 \
    -m mor_mlite.parity run "${COMMON[@]}" --topology baseline --route-mode learned \
    --reference-dp-shards 2 --init-checkpoint "${INIT_ROOT}/folded_init_ep1" \
    --output "${ROOT}/baseline"
python -m torch.distributed.run --standalone --nproc-per-node=8 \
    -m mor_mlite.parity run "${COMMON[@]}" --topology all --route-mode replay \
    --replay-from "${ROOT}/baseline" --init-checkpoint "${INIT_ROOT}/folded_init_ep4" \
    --cp-transition magi_direct --output "${ROOT}/all"
python -m mor_mlite.parity compare "${ROOT}/baseline" "${ROOT}/all" \
    --scope forward --report "${ROOT}/reports/all_vs_baseline.json"
echo "Native BF16 full forward passed: ${ROOT} (training validation remains separate)"
