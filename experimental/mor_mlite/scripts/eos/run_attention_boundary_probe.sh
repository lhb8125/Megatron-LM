#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/common.sh"
require_eos_project
activate_mor_environment
"${SCRIPT_DIR}/version_probe.sh" 8
python -m pytest -q tests/test_attention_boundaries.py
ARTIFACT_ROOT="${EOS_WORKDIR}/artifacts/attention-${SLURM_JOB_ID}"
INIT_ROOT="/lustre/fsw/coreai_devtech_all/hongbinl/mor_mlite/artifacts/eos/5998357/qwen30b"
ORACLE="/lustre/fsw/coreai_devtech_all/hongbinl/agentic-mcore-dev/.claude/worktrees/mor-mlite-validation/vendor/mor_mlite_precision/artifacts/operators-6006133/baseline"
COMMON=(--oracle "${ORACLE}" --backend mlite --preset qwen3-30b --precision bf16
    --seq-lens 128,128 --num-microbatches 1 --steps 1 --adam-eps 1e-6
    --forward-only --no-checkpoint-roundtrip)
python -m torch.distributed.run --standalone --nproc-per-node=1 \
    "${SCRIPT_DIR}/probe_attention_boundaries.py" "${COMMON[@]}" \
    --topology baseline --route-mode learned --reference-dp-shards 2 --init-checkpoint "${INIT_ROOT}/folded_init_ep1" \
    --traces "${ARTIFACT_ROOT}/baseline.pt" --output "${ARTIFACT_ROOT}/baseline"
for mode in native identical_core; do
    OPTIONS=()
    if [[ "${mode}" == identical_core ]]; then OPTIONS+=(--inject-core); fi
    python -m torch.distributed.run --standalone --nproc-per-node=8 \
        "${SCRIPT_DIR}/probe_attention_boundaries.py" "${COMMON[@]}" "${OPTIONS[@]}" \
        --topology all --init-checkpoint "${INIT_ROOT}/folded_init_ep4" \
        --route-mode replay --replay-from "${ORACLE}" \
        --compare-traces "${ARTIFACT_ROOT}/baseline.pt" --traces "${ARTIFACT_ROOT}/${mode}.pt" \
        --cp-transition magi_direct --output "${ARTIFACT_ROOT}/${mode}"
done
