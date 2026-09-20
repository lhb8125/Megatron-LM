#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

require_eos_project
activate_mor_environment
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export MOR_DIAGNOSTIC_CONTINUE_AFTER_CHECKPOINT_MISMATCH=1

readonly SOURCE_JOB="${MOR_DIAGNOSTIC_SOURCE_JOB:-5997775}"
readonly SOURCE_ROOT="${EOS_WORKDIR}/artifacts/eos/${SOURCE_JOB}/qwen30b"
readonly OUTPUT_ROOT="${EOS_WORKDIR}/artifacts/eos/${SLURM_JOB_ID:-manual}/qwen_resume_diagnostic"

python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=8 \
    -m mor_mlite.parity run \
    --backend mlite \
    --preset qwen3-30b \
    --topology all \
    --precision bf16 \
    --resume-checkpoint "${SOURCE_ROOT}/all_train_save/runtime-checkpoint" \
    --seq-lens 128,128 \
    --num-microbatches 1 \
    --steps 1 \
    --adam-eps 1e-6 \
    --route-mode replay \
    --replay-from "${SOURCE_ROOT}/baseline_forward" \
    --cp-transition magi_direct \
    --no-checkpoint-roundtrip \
    --output "${OUTPUT_ROOT}"

echo "Diagnostic resume artifact: ${OUTPUT_ROOT}"
