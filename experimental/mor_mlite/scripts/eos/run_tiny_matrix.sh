#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

require_eos_project
activate_mor_environment
"${SCRIPT_DIR}/version_probe.sh" "${MOR_EXPECTED_WORLD_SIZE:-8}"
"${SCRIPT_DIR}/magi_canary.sh"

ARTIFACT_ROOT="${MOR_ARTIFACT_ROOT:-${EOS_WORKDIR}/artifacts/eos/${SLURM_JOB_ID:-manual}/tiny}"
mkdir -p "${ARTIFACT_ROOT}/reports"

# This includes the in-process and spawned 2/4-rank Gloo tests for routing,
# active packing, differentiable variable All-to-All, and DP-group isolation.
python -m pytest -q "${EOS_WORKDIR}/tests" \
    --junitxml="${ARTIFACT_ROOT}/reports/pytest.xml"

run_mlite() {
    local world_size="$1"
    local topology="$2"
    local output="$3"
    shift 3
    [[ "${world_size}" -le "${MAX_WORLD_SIZE}" ]] || die "world size exceeds ${MAX_WORLD_SIZE}"
    # Invoke torch.distributed through the pinned cu129 venv interpreter; a
    # system `torchrun` shebang would select the bootstrap image's cu131 Torch.
    python -m torch.distributed.run \
        --standalone --nnodes=1 --nproc-per-node="${world_size}" \
        -m mor_mlite.parity run \
        --backend mlite \
        --preset tiny \
        --topology "${topology}" \
        --precision bf16 \
        --steps 1 \
        --num-microbatches 2 \
        --adam-eps 1e-6 \
        --seq-lens 9,6,3 \
        --output "${output}" \
        "$@"
}

compare_with_baseline() {
    local candidate="$1"
    local report_name="$2"
    python -m mor_mlite.parity compare \
        "${ARTIFACT_ROOT}/mlite_baseline" "${candidate}" \
        --report "${ARTIFACT_ROOT}/reports/${report_name}.json"
}

# FP32, single-card semantic oracle.  Distributed MLite production execution is
# BF16, so it also receives a separate single-rank BF16 baseline below.
python -m mor_mlite.parity run \
    --backend reference \
    --preset tiny \
    --topology baseline \
    --precision fp32 \
    --device cuda \
    --steps 1 \
    --num-microbatches 2 \
    --adam-eps 1e-6 \
    --seq-lens 9,6,3 \
    --output "${ARTIFACT_ROOT}/reference_fp32"
python -m mor_mlite.parity run \
    --backend reference \
    --preset tiny \
    --topology baseline \
    --precision fp32 \
    --device cuda \
    --steps 1 \
    --num-microbatches 2 \
    --adam-eps 1e-6 \
    --seq-lens 9,6,3 \
    --route-mode replay \
    --replay-from "${ARTIFACT_ROOT}/reference_fp32" \
    --output "${ARTIFACT_ROOT}/reference_fp32_replay"
python -m mor_mlite.parity compare \
    "${ARTIFACT_ROOT}/reference_fp32" "${ARTIFACT_ROOT}/reference_fp32_replay" \
    --report "${ARTIFACT_ROOT}/reports/reference_fp32_replay.json"

run_mlite 1 baseline "${ARTIFACT_ROOT}/mlite_baseline" \
    --route-mode learned

# Exercise the same process-isolated full-state save/resume path used by the
# 30B smoke while the tiny in-process roundtrip below remains the stronger
# uninterrupted numerical oracle.
run_mlite 1 baseline "${ARTIFACT_ROOT}/external_checkpoint_save" \
    --route-mode replay \
    --replay-from "${ARTIFACT_ROOT}/mlite_baseline" \
    --checkpoint-save-only
run_mlite 1 baseline "${ARTIFACT_ROOT}/external_checkpoint_resume" \
    --route-mode replay \
    --replay-from "${ARTIFACT_ROOT}/mlite_baseline" \
    --resume-checkpoint "${ARTIFACT_ROOT}/external_checkpoint_save/runtime-checkpoint" \
    --no-checkpoint-roundtrip
python -m mor_mlite.parity certify-checkpoint \
    "${ARTIFACT_ROOT}/external_checkpoint_save/runtime-checkpoint" \
    "${ARTIFACT_ROOT}/external_checkpoint_resume" \
    --report "${ARTIFACT_ROOT}/reports/checkpoint_external_resume.json"

declare -A WORLD_SIZES=(
    [zero1]=2
    [tp]=2
    [cp]=2
    [ep]=2
    [tp_cp_ep]=4
    [tp_dp_ep]=4
    [cp_dp_ep]=4
    [all]=8
)

FILTERED_DIAGNOSTIC=0
if [[ -n "${MOR_TOPOLOGIES:-}" ]]; then
    FILTERED_DIAGNOSTIC=1
    IFS=',' read -r -a TOPOLOGIES <<<"${MOR_TOPOLOGIES}"
else
    TOPOLOGIES=(zero1 tp cp ep tp_cp_ep tp_dp_ep cp_dp_ep all)
fi

for topology in "${TOPOLOGIES[@]}"; do
    world_size="${WORLD_SIZES[${topology}]:-}"
    [[ -n "${world_size}" ]] || die "unknown tiny topology: ${topology}"
    learned="${ARTIFACT_ROOT}/${topology}_learned"
    run_mlite "${world_size}" "${topology}" "${learned}" \
        --route-mode learned \
        --cp-transition magi_direct
    compare_with_baseline "${learned}" "${topology}_learned"

    candidate="${ARTIFACT_ROOT}/${topology}_replay"
    run_mlite "${world_size}" "${topology}" "${candidate}" \
        --route-mode replay \
        --replay-from "${ARTIFACT_ROOT}/mlite_baseline" \
        --cp-transition magi_direct
    compare_with_baseline "${candidate}" "${topology}"
done

# CP canonical is the correctness oracle for the direct layout transition.
if [[ " ${TOPOLOGIES[*]} " == *" cp "* ]]; then
    canonical="${ARTIFACT_ROOT}/cp_canonical_replay"
    run_mlite 2 cp "${canonical}" \
        --route-mode replay \
        --replay-from "${ARTIFACT_ROOT}/mlite_baseline" \
        --cp-transition magi_canonical
    compare_with_baseline "${canonical}" "cp_canonical"
    python -m mor_mlite.parity compare \
        "${canonical}" "${ARTIFACT_ROOT}/cp_replay" \
        --report "${ARTIFACT_ROOT}/reports/cp_canonical_vs_direct.json"
fi

if [[ "${FILTERED_DIAGNOSTIC}" -eq 1 ]]; then
    echo "Filtered tiny topology diagnostic passed (${TOPOLOGIES[*]}). Artifacts: ${ARTIFACT_ROOT}"
else
    python -m mor_mlite.parity.receipt \
        --artifact-root "${ARTIFACT_ROOT}" \
        --job-id "${SLURM_JOB_ID:-manual}"
    echo "Complete tiny topology matrix passed. Artifacts: ${ARTIFACT_ROOT}"
fi
