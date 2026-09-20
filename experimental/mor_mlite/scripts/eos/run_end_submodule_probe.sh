#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/common.sh"
require_eos_project
activate_mor_environment
"${SCRIPT_DIR}/version_probe.sh" 8
: "${MOR_OPERATOR_ORACLE:?set the canonical native baseline artifact}"
ARTIFACT_ROOT="${EOS_WORKDIR}/artifacts/submodules-${SLURM_JOB_ID}"
INIT_ROOT="/lustre/fsw/coreai_devtech_all/hongbinl/mor_mlite/artifacts/eos/5998357/qwen30b"
for mode in native projection_fp32; do
    OPTIONS=(--submodules)
    if [[ "${mode}" == projection_fp32 ]]; then OPTIONS+=(--projection-fp32); fi
    python -m torch.distributed.run --standalone --nproc-per-node=8 \
        "${SCRIPT_DIR}/probe_end_operators.py" --oracle "${MOR_OPERATOR_ORACLE}" \
        --report "${ARTIFACT_ROOT}/reports/${mode}.json" "${OPTIONS[@]}" \
        --backend mlite --preset qwen3-30b --topology all --precision bf16 \
        --init-checkpoint "${INIT_ROOT}/folded_init_ep4" --seq-lens 128,128 \
        --num-microbatches 1 --steps 1 --adam-eps 1e-6 --route-mode replay \
        --replay-from "${MOR_OPERATOR_ORACLE}" --cp-transition magi_direct \
        --forward-only --no-checkpoint-roundtrip --output "${ARTIFACT_ROOT}/${mode}"
done
