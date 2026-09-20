#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

require_eos_project
activate_mor_environment
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

readonly SOURCE_JOB=5997893
readonly SOURCE_ROOT="${EOS_WORKDIR}/artifacts/eos/${SOURCE_JOB}/qwen30b"
readonly ARTIFACT_ROOT="${EOS_WORKDIR}/artifacts/eos/${SLURM_JOB_ID}/qwen_optimizer_diagnostic"
mkdir -p "${ARTIFACT_ROOT}"

run_qwen() {
    python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=8 \
        -m mor_mlite.parity run \
        --backend mlite \
        --preset qwen3-30b \
        --topology all \
        --precision bf16 \
        --seq-lens 128,128 \
        --num-microbatches 1 \
        --steps 1 \
        --adam-eps 1e-6 \
        --route-mode replay \
        --replay-from "${SOURCE_ROOT}/baseline_forward" \
        --cp-transition magi_direct \
        "$@"
}

run_qwen \
    --init-checkpoint "${SOURCE_ROOT}/folded_init_ep4" \
    --checkpoint-save-only \
    --output "${ARTIFACT_ROOT}/save"

run_qwen \
    --resume-checkpoint "${ARTIFACT_ROOT}/save/runtime-checkpoint" \
    --no-checkpoint-roundtrip \
    --output "${ARTIFACT_ROOT}/resume"
