#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

require_command ssh
require_command rsync

LOCAL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"

REMOTE_ROOT="${MOR_EOS_REMOTE_ROOT:-/lustre/fsw/coreai_devtech_all/hongbinl/mor_mlite}"
[[ "${REMOTE_ROOT}" =~ ^/[a-zA-Z0-9_./-]+$ ]] || die "remote root must be an absolute shell-safe path"
echo "Syncing ${LOCAL_ROOT} to ${EOS_LOGIN}:${REMOTE_ROOT}"
ssh "${EOS_LOGIN}" mkdir -p "${REMOTE_ROOT}" "${REMOTE_ROOT}/logs"
rsync -az \
    --exclude '.git/' \
    --exclude '.deps/' \
    --exclude '.venv/' \
    --exclude '.cache/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude 'artifacts/eos/' \
    --exclude 'artifacts/local_validation/' \
    "${LOCAL_ROOT}/" "${EOS_LOGIN}:${REMOTE_ROOT}/"

echo "Sync complete. Remote dependency caches and prior EOS artifacts were preserved."
