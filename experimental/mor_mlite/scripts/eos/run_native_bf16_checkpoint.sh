#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/common.sh"
require_eos_project
activate_mor_environment
"${SCRIPT_DIR}/version_probe.sh" 8
INIT_ROOT="${MOR_FOLDED_INIT_ROOT:?set the existing folded checkpoint root}"
ORACLE="${MOR_BF16_ORACLE:?set the freshly validated native BF16 forward baseline}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
ROOT="${EOS_WORKDIR}/artifacts/bf16-${SLURM_JOB_ID}/qwen30b_train"
mkdir -p "${ROOT}/reports"
COMMON=(--backend mlite --preset qwen3-30b --topology all --precision bf16
    --seq-lens 128,128 --num-microbatches 1 --steps 1 --adam-eps 1e-6
    --route-mode replay --replay-from "${ORACLE}" --cp-transition magi_direct)
python -m torch.distributed.run --standalone --nproc-per-node=8 \
    -m mor_mlite.parity run "${COMMON[@]}" --init-checkpoint "${INIT_ROOT}/folded_init_ep4" \
    --checkpoint-save-only --output "${ROOT}/save"
python -m torch.distributed.run --standalone --nproc-per-node=8 \
    -m mor_mlite.parity run "${COMMON[@]}" --resume-checkpoint "${ROOT}/save/runtime-checkpoint" \
    --no-checkpoint-roundtrip --output "${ROOT}/resume"
python -m mor_mlite.parity certify-checkpoint "${ROOT}/save/runtime-checkpoint" "${ROOT}/resume" \
    --report "${ROOT}/reports/checkpoint_external_resume.json"
