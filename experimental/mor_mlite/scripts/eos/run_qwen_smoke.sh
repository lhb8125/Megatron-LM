#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

require_eos_project
activate_mor_environment
# The 30B distributed-optimizer restore is close to the H100 memory ceiling.
# Expandable segments prevent allocator fragmentation from rejecting a small
# optimizer shard even when the caching pool still has free bytes.
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
"${SCRIPT_DIR}/version_probe.sh" 8
"${SCRIPT_DIR}/magi_canary.sh"

HF_SOURCE="${MOR_QWEN_HF_PATH:-Qwen/Qwen3-30B-A3B-Base}"
# MLite's config loader accepts Hub IDs, but its pinned SafeTensorReader reads
# local directories only.  Resolve the Hub snapshot once, before torchrun, so
# eight ranks never discover that mismatch after allocating the 30B model.
if [[ -d "${HF_SOURCE}" ]]; then
    HF_PATH="$(cd "${HF_SOURCE}" && pwd -P)"
else
    HF_PATH="$(python - "${HF_SOURCE}" <<'PY'
import sys

from huggingface_hub import snapshot_download

print(
    snapshot_download(
        repo_id=sys.argv[1],
        allow_patterns=(
            "config.json",
            "model.safetensors",
            "model.safetensors.index.json",
            "model-*.safetensors",
        ),
    )
)
PY
)"
fi
if [[ ! -f "${HF_PATH}/config.json" ]]; then
    echo "Resolved HF checkpoint has no config.json: ${HF_PATH}" >&2
    exit 2
fi
if [[ ! -f "${HF_PATH}/model.safetensors" && ! -f "${HF_PATH}/model.safetensors.index.json" ]]; then
    echo "Resolved HF checkpoint has no safetensors weights: ${HF_PATH}" >&2
    exit 2
fi
export MOR_HF_SOURCE="${HF_SOURCE}"
ARTIFACT_ROOT="${MOR_ARTIFACT_ROOT:-${EOS_WORKDIR}/artifacts/eos/${SLURM_JOB_ID:-manual}/qwen30b}"
mkdir -p "${ARTIFACT_ROOT}/reports"

# Exercise the public HF -> folded MoR DCP converter for each EP layout.  MLite
# names grouped-expert parameters by EP-local index, so DCP can reshard dense
# DP/TP/CP axes but cannot reinterpret an EP=1 expert key as EP=4.  Both imports
# stream the same HF tensors through the same deterministic mean fold and router
# seed; the subsequent replay comparison proves their effective initial models
# agree across topology.
python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=1 \
    -m mor_mlite.convert_hf \
    --hf-path "${HF_PATH}" \
    --output "${ARTIFACT_ROOT}/folded_init_ep1" \
    --tp 1 --cp 1 --dp 1 --ep 1 --etp 1 \
    --cp-transition magi_direct

python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=8 \
    -m mor_mlite.convert_hf \
    --hf-path "${HF_PATH}" \
    --output "${ARTIFACT_ROOT}/folded_init_ep4" \
    --tp 2 --cp 2 --dp 2 --ep 4 --etp 1 \
    --cp-transition magi_direct

# One-GPU forward-only baseline.  No optimizer state is allocated on this path.
python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=1 \
    -m mor_mlite.parity run \
    --backend mlite \
    --preset qwen3-30b \
    --topology baseline \
    --precision bf16 \
    --init-checkpoint "${ARTIFACT_ROOT}/folded_init_ep1" \
    --seq-lens 128,128 \
    --num-microbatches 1 \
    --steps 1 \
    --adam-eps 1e-6 \
    --route-mode learned \
    --forward-only \
    --reference-dp-shards 2 \
    --no-checkpoint-roundtrip \
    --output "${ARTIFACT_ROOT}/baseline_forward"

# First compare the 8-GPU initial model with the single-card baseline.  This is
# forward-only so the artifact is not shifted by an optimizer update.
python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=8 \
    -m mor_mlite.parity run \
    --backend mlite \
    --preset qwen3-30b \
    --topology all \
    --precision bf16 \
    --init-checkpoint "${ARTIFACT_ROOT}/folded_init_ep4" \
    --seq-lens 128,128 \
    --num-microbatches 1 \
    --steps 1 \
    --adam-eps 1e-6 \
    --route-mode replay \
    --replay-from "${ARTIFACT_ROOT}/baseline_forward" \
    --cp-transition magi_direct \
    --forward-only \
    --no-checkpoint-roundtrip \
    --output "${ARTIFACT_ROOT}/all_forward"

forward_compare_status=0
if python -m mor_mlite.parity compare \
    "${ARTIFACT_ROOT}/baseline_forward" "${ARTIFACT_ROOT}/all_forward" \
    --scope forward \
    --report "${ARTIFACT_ROOT}/reports/all_vs_baseline.json"; then
    :
else
    forward_compare_status=$?
    echo "Qwen forward parity failed; continuing to collect independent checkpoint evidence." >&2
fi

# A 30B second MLite handle in the same process retains the first handle's Lite
# process groups and is not a true clean-room restore.  Save and resume in two
# separate torchrun processes so process exit releases the complete CUDA/NCCL
# context.  The receipt binds model, optimizer, and RNG hashes to the save;
# resume verifies all three before executing global step 1. The save process
# also executes step 1 as the uninterrupted oracle, and certification requires
# exact post-step model/optimizer/RNG fingerprints from the fresh process.
python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=8 \
    -m mor_mlite.parity run \
    --backend mlite \
    --preset qwen3-30b \
    --topology all \
    --precision bf16 \
    --init-checkpoint "${ARTIFACT_ROOT}/folded_init_ep4" \
    --seq-lens 128,128 \
    --num-microbatches 1 \
    --steps 1 \
    --adam-eps 1e-6 \
    --route-mode replay \
    --replay-from "${ARTIFACT_ROOT}/baseline_forward" \
    --cp-transition magi_direct \
    --checkpoint-save-only \
    --output "${ARTIFACT_ROOT}/all_train_save"

python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=8 \
    -m mor_mlite.parity run \
    --backend mlite \
    --preset qwen3-30b \
    --topology all \
    --precision bf16 \
    --resume-checkpoint "${ARTIFACT_ROOT}/all_train_save/runtime-checkpoint" \
    --seq-lens 128,128 \
    --num-microbatches 1 \
    --steps 1 \
    --adam-eps 1e-6 \
    --route-mode replay \
    --replay-from "${ARTIFACT_ROOT}/baseline_forward" \
    --cp-transition magi_direct \
    --no-checkpoint-roundtrip \
    --output "${ARTIFACT_ROOT}/all_resume"

checkpoint_status=0
if python -m mor_mlite.parity certify-checkpoint \
    "${ARTIFACT_ROOT}/all_train_save/runtime-checkpoint" \
    "${ARTIFACT_ROOT}/all_resume" \
    --report "${ARTIFACT_ROOT}/reports/checkpoint_external_resume.json"; then
    :
else
    checkpoint_status=$?
fi

if [[ "${forward_compare_status}" -ne 0 || "${checkpoint_status}" -ne 0 ]]; then
    echo "Qwen3-30B-A3B smoke test failed one or more acceptance gates. Artifacts: ${ARTIFACT_ROOT}" >&2
    exit 1
fi

echo "Qwen3-30B-A3B smoke test passed. Artifacts: ${ARTIFACT_ROOT}"
