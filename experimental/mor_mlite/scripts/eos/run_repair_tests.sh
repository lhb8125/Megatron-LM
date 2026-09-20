#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/common.sh"
require_eos_project
activate_mor_environment
cd "${EOS_WORKDIR}"
python -c 'import torch, mor_mlite; print(torch.__version__, mor_mlite.__file__)'
python -m pytest -q tests --junitxml="${EOS_WORKDIR}/artifacts/repair-${SLURM_JOB_ID}.xml"
