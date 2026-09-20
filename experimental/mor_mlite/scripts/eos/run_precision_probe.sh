#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/common.sh"
require_eos_project
activate_mor_environment
"${SCRIPT_DIR}/version_probe.sh" 8
ARTIFACT_ROOT="${EOS_WORKDIR}/artifacts/precision-${SLURM_JOB_ID}"
INIT_ROOT="/lustre/fsw/coreai_devtech_all/hongbinl/mor_mlite/artifacts/eos/5998357/qwen30b"
mkdir -p "${ARTIFACT_ROOT}/reports"
IFS=',' read -r -a MODES <<<"${MOR_PRECISION_MODES:-residual_fp32,both}"
for mode in "${MODES[@]}"; do
    python -m torch.distributed.run --standalone --nproc-per-node=1 \
        "${SCRIPT_DIR}/probe_precision.py" --mode "${mode}" \
        --backend mlite --preset qwen3-30b --topology baseline --precision bf16 \
        --init-checkpoint "${INIT_ROOT}/folded_init_ep1" --seq-lens 128,128 \
        --num-microbatches 1 --steps 1 --adam-eps 1e-6 --route-mode learned \
        --forward-only --reference-dp-shards 2 --no-checkpoint-roundtrip \
        --output "${ARTIFACT_ROOT}/${mode}_baseline"
    python -m torch.distributed.run --standalone --nproc-per-node=8 \
        "${SCRIPT_DIR}/probe_precision.py" --mode "${mode}" \
        --backend mlite --preset qwen3-30b --topology all --precision bf16 \
        --init-checkpoint "${INIT_ROOT}/folded_init_ep4" --seq-lens 128,128 \
        --num-microbatches 1 --steps 1 --adam-eps 1e-6 --route-mode replay \
        --replay-from "${ARTIFACT_ROOT}/${mode}_baseline" --cp-transition magi_direct \
        --forward-only --no-checkpoint-roundtrip --output "${ARTIFACT_ROOT}/${mode}_all"
    if python -m mor_mlite.parity compare "${ARTIFACT_ROOT}/${mode}_baseline" \
        "${ARTIFACT_ROOT}/${mode}_all" --scope forward --report "${ARTIFACT_ROOT}/reports/${mode}.json"; then
        echo "Precision diagnostic ${mode} passed"
    else
        echo "Precision diagnostic ${mode} failed; collecting the next independent experiment" >&2
    fi
done
