#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

require_eos_project
activate_mor_environment

# Reuse the operator-level test shipped by the exact pinned MLite checkout.
# It exercises standard torch.distributed All-to-All dispatch, BF16 Magi
# attention forward/backward, undispatch, and token-wise dQ/dK/dV parity.
CANARY="${MEGATRON_LM_ROOT}/experimental/lite/tests/smoke/primitive/test_magi_attention_operator.py"
[[ -f "${CANARY}" ]] || die "pinned MLite Magi canary is missing: ${CANARY}"
[[ "${MAGI_ATTENTION_NATIVE_GRPCOLL}" == "0" ]] || \
    die "v1 Magi canary requires standard All-to-All"

python - <<'PY'
import torch
import magi_attention  # noqa: F401
import magi_attention.magi_attn_ext  # noqa: F401

if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
    raise SystemExit("Magi canary requires two visible CUDA devices")
for index in range(2):
    if torch.cuda.get_device_capability(index) != (9, 0):
        raise SystemExit(f"Magi canary requires sm90, device {index} is not sm90")
PY

# The pinned MLite conftest intentionally skips every GPU-marked test unless
# it is launched by the sanctioned harness.  Set the same flag as the
# upstream run_magi_attention_e2e.sh wrapper so this canary cannot silently
# succeed with two skipped ranks.
MLITE_TEST_HARNESS=1 python -m torch.distributed.run \
    --standalone --nnodes=1 --nproc-per-node=2 \
    -m pytest -q "${CANARY}" -k static-degree1
