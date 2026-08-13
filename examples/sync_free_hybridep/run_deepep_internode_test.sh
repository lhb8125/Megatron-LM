#!/bin/bash
# =============================================================================
# 2-node DeepEP internode (v1) smoke test for the sync-free image.
#
# Runs DeepEP's own tests/test_internode.py across 2 nodes x 8 GPUs (16 ranks)
# to verify cross-node RDMA dispatch/combine over NVSHMEM actually works. This
# is the exact v1 Buffer API (Buffer.get_dispatch_layout / dispatch / combine)
# that Megatron's `moe_flex_dispatcher_backend: deepep` uses, so a passing run
# validates the same code path the training job exercises.
#
# Platform env contract (same as pretrain.sh, injected by canoe):
#   WORLD_SIZE  = number of NODES        (test_internode's init_dist num_nodes)
#   RANK        = node rank (0-based)    (test_internode's init_dist node_rank)
#   MASTER_ADDR / MASTER_PORT = rendezvous
#   GPUS_PER_NODE = 8 (required; test asserts num_local_ranks == 8)
#
# test_internode.py hard requirements:
#   * exactly 8 GPUs per node (num_local_ranks == 8)
#   * num_ranks > 8  => at least 2 nodes (this is a CROSS-NODE test)
#
# Canoe usage (2 nodes x 8 GPU):
#   image:  harbor.xaminim.com/minimax-dialogue/megatron-sync-free:<tag-with-nvshmem>
#   entry:  bash -cx 'bash -x examples/sync_free_hybridep/run_deepep_internode_test.sh'
#   workers: 2   gpu: 8
#
# Optional env overrides:
#   NUM_TOKENS (default 4096), HIDDEN (7168), NUM_EXPERTS (256), NUM_TOPK (8)
#   TEST_LL_COMPAT=1   also test low-latency kernel compatibility
#   NVSHMEM_DEBUG=INFO to get verbose NVSHMEM bring-up logs
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEEPEP_DIR="${DEEPEP_DIR:-/workspace/DeepEP}"
TEST_PY="${DEEPEP_DIR}/tests/test_internode.py"

# --------------------------------------------------- distributed (platform env)
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export WORLD_SIZE=${WORLD_SIZE:-1}          # = number of nodes (init_dist num_nodes)
export RANK=${RANK:-0}                       # = node rank      (init_dist node_rank)
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-8361}

if [[ "${GPUS_PER_NODE}" != "8" ]]; then
  echo "ERROR: test_internode.py requires exactly 8 GPUs per node (num_local_ranks==8); got GPUS_PER_NODE=${GPUS_PER_NODE}" >&2
  exit 1
fi
if [[ "${WORLD_SIZE}" -lt 2 ]]; then
  echo "ERROR: this is a CROSS-NODE test; need WORLD_SIZE (node count) >= 2, got ${WORLD_SIZE}." >&2
  echo "       Launch a 2-node x 8-GPU canoe job." >&2
  exit 1
fi

if [[ ! -f "${TEST_PY}" ]]; then
  echo "ERROR: ${TEST_PY} not found. Is this the DeepEP-enabled image?" >&2
  exit 1
fi

# --------------------------------------------------- NVSHMEM / NCCL RDMA env
# IBGDA is the fast path; if the cluster driver is not configured for it,
# NVSHMEM falls back to CPU-assisted IBGDA (needs gdrdrv). Leave these as
# opt-in overrides rather than forcing a mode that the cluster may not allow.
export NVSHMEM_DISABLE_CUDA_VMM=${NVSHMEM_DISABLE_CUDA_VMM:-1}
export NVSHMEM_IB_ENABLE_IBGDA=${NVSHMEM_IB_ENABLE_IBGDA:-1}
export NVSHMEM_IBGDA_NIC_HANDLER=${NVSHMEM_IBGDA_NIC_HANDLER:-gpu}
# Adaptive routing MUST be off for DeepEP (see fused_a2a.py note); surface it.
export NVSHMEM_IB_TRAFFIC_CLASS=${NVSHMEM_IB_TRAFFIC_CLASS:-0}
[[ -n "${NVSHMEM_DEBUG:-}" ]] && export NVSHMEM_DEBUG

# Test parameters (mirror the training model where it matters).
NUM_TOKENS=${NUM_TOKENS:-4096}
HIDDEN=${HIDDEN:-7168}
NUM_EXPERTS=${NUM_EXPERTS:-256}
NUM_TOPK=${NUM_TOPK:-8}
LL_FLAG=""
[[ "${TEST_LL_COMPAT:-0}" == "1" ]] && LL_FLAG="--test-ll-compatibility"

echo "============================================"
echo "DeepEP internode (v1) test"
echo "  nodes (WORLD_SIZE) : ${WORLD_SIZE}"
echo "  node_rank (RANK)   : ${RANK}"
echo "  gpus/node          : ${GPUS_PER_NODE}"
echo "  master             : ${MASTER_ADDR}:${MASTER_PORT}"
echo "  total ranks        : $((WORLD_SIZE * GPUS_PER_NODE))"
echo "  num_tokens=${NUM_TOKENS} hidden=${HIDDEN} num_experts=${NUM_EXPERTS} num_topk=${NUM_TOPK}"
echo "============================================"

cd "${DEEPEP_DIR}"
# test_internode.py spawns 8 local ranks itself via torch.multiprocessing.spawn;
# we launch ONE process per node and let it fan out.
exec python3 tests/test_internode.py \
  --num-processes "${GPUS_PER_NODE}" \
  --num-tokens "${NUM_TOKENS}" \
  --hidden "${HIDDEN}" \
  --num-experts "${NUM_EXPERTS}" \
  --num-topk "${NUM_TOPK}" \
  ${LL_FLAG}
