#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

require_command sbatch
require_eos_project
mkdir -p "${EOS_WORKDIR}/logs"
export MOR_PROJECT_ROOT="${EOS_WORKDIR}"
sbatch --chdir="${EOS_WORKDIR}" \
    --output="${EOS_WORKDIR}/logs/%x-%j.out" --error="${EOS_WORKDIR}/logs/%x-%j.err" \
    "${EOS_WORKDIR}/slurm/qwen30b_smoke.sbatch"
