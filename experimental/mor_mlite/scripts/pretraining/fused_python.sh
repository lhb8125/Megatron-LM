#!/bin/sh
# Explicit startup entry for the pinned four-arm GB200 experiment.
# Establish TE selectors before Python/NCCL starts any background threads.
# Pinned native MLite selects these same values during model construction;
# doing the first insertion there can race with NCCL heartbeat getenv().
set -eu
: "${PROJECT_ROOT:?PROJECT_ROOT must identify the experiment checkout}"

if [ "${NVTE_FLASH_ATTN:-0}" != 0 ] || \
   [ "${NVTE_FUSED_ATTN:-1}" != 1 ] || \
   [ "${NVTE_UNFUSED_ATTN:-0}" != 0 ]; then
    echo "startup environment conflicts with the approved fused attention backend" >&2
    exit 2
fi
export NVTE_FLASH_ATTN=0
export NVTE_FUSED_ATTN=1
export NVTE_UNFUSED_ATTN=0

exec "${PROJECT_ROOT}/runtime/mor-pretraining/gb200-env-nvrx060/bin/python" "$@"
