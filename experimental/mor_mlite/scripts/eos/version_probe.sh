#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

require_eos_project
activate_mor_environment

EXPECTED_WORLD_SIZE="${1:-${MOR_EXPECTED_WORLD_SIZE:-1}}"
[[ "${EXPECTED_WORLD_SIZE}" =~ ^[1-8]$ ]] || \
    die "expected world size must be an integer from 1 to ${MAX_WORLD_SIZE}"

MANIFEST_PATH="${MOR_VERSION_MANIFEST:-${EOS_WORKDIR}/artifacts/eos/${SLURM_JOB_ID:-manual}/versions.json}"
mkdir -p "$(dirname "${MANIFEST_PATH}")"

python -m mor_mlite.env_check \
    --megatron-root "${MEGATRON_LM_ROOT}" \
    --expected-world-size "${EXPECTED_WORLD_SIZE}" \
    --output "${MANIFEST_PATH}"

TE_CANARY_PATH="${MOR_TE_CANARY_MANIFEST:-$(dirname "${MANIFEST_PATH}")/te_canary.json}"
python -m mor_mlite.runtime_canary te --output "${TE_CANARY_PATH}"

echo "Validated version manifest: ${MANIFEST_PATH}"
echo "Validated Transformer Engine canary: ${TE_CANARY_PATH}"
