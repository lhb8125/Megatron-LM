#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/common.sh"
require_eos_project
activate_mor_environment
"${SCRIPT_DIR}/version_probe.sh" 8
ARTIFACT_ROOT="${EOS_WORKDIR}/artifacts/operators-${SLURM_JOB_ID}"
INIT_ROOT="/lustre/fsw/coreai_devtech_all/hongbinl/mor_mlite/artifacts/eos/5998357/qwen30b"
python -m torch.distributed.run --standalone --nproc-per-node=1 \
    -m mor_mlite.parity run --backend mlite --preset qwen3-30b --topology baseline --precision bf16 \
    --init-checkpoint "${INIT_ROOT}/folded_init_ep1" --seq-lens 128,128 \
    --num-microbatches 1 --steps 1 --adam-eps 1e-6 --route-mode learned \
    --forward-only --reference-dp-shards 2 --no-checkpoint-roundtrip --output "${ARTIFACT_ROOT}/baseline"
python -m torch.distributed.run --standalone --nproc-per-node=8 \
    "${SCRIPT_DIR}/probe_end_operators.py" --oracle "${ARTIFACT_ROOT}/baseline" \
    --report "${ARTIFACT_ROOT}/reports/end_operators.json" \
    --backend mlite --preset qwen3-30b --topology all --precision bf16 \
    --init-checkpoint "${INIT_ROOT}/folded_init_ep4" --seq-lens 128,128 \
    --num-microbatches 1 --steps 1 --adam-eps 1e-6 --route-mode replay \
    --replay-from "${ARTIFACT_ROOT}/baseline" --cp-transition magi_direct \
    --forward-only --no-checkpoint-roundtrip --output "${ARTIFACT_ROOT}/end_operators"
