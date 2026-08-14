#!/bin/bash
# =============================================================================
# 2-node HybridEP internode smoke test for the sync-free image.
#
# Runs DeepEP's tests/test_hybrid_ep.py across 2 nodes to verify HybridEP's
# cross-node path (deep_ep.HybridEPBuffer, the DOCA/RDMA multinode coordinator
# built with HYBRID_EP_MULTINODE=1). This is the HybridEP counterpart of
# run_deepep_internode_test.sh (which tests the DeepEP v1 Buffer / NVSHMEM path).
#
# Unlike test_internode.py, test_hybrid_ep.py does NOT hardcode 8 GPUs/node:
#   NUM_OF_RANKS_PER_NODE = --num-processes (per-node GPU count)
#   NUM_OF_NODES          = world_ranks // NUM_OF_RANKS_PER_NODE
#   multinode             = NUM_OF_NODES > 1
# so it just needs >= 2 nodes to exercise the cross-node path.
#
# Platform env contract (same as pretrain.sh, injected by canoe):
#   WORLD_SIZE  = number of NODES        (init_dist num_nodes)
#   RANK        = node rank (0-based)    (init_dist node_rank)
#   MASTER_ADDR / MASTER_PORT = rendezvous
#   GPUS_PER_NODE = per-node GPU count   (-> --num-processes)
#
# Canoe usage (2 nodes x 8 GPU):
#   image:  harbor.xaminim.com/minimax-dialogue/megatron-sync-free:<tag>
#   entry:  bash -cx 'bash -x examples/sync_free_hybridep/run_hybridep_internode_test.sh'
#   workers: 2   gpu: 8
#
# Optional env overrides (consumed directly by test_hybrid_ep.py):
#   HIDDEN_DIM (7168), MAX_NUM_OF_TOKENS_PER_RANK (4096),
#   NUM_TOKENS_PER_RANK (4096), NUM_LOCAL_EXPERTS (8), TOPK (8)
#   USE_MNNVL=1  only for multi-node NVLink fabrics (GB200 NVL72); leave off on
#                RoCE/IB clusters so HybridEP uses the RDMA coordinator.
# =============================================================================
set -euo pipefail

DEEPEP_DIR="${DEEPEP_DIR:-/workspace/DeepEP}"
TEST_PY="${DEEPEP_DIR}/tests/test_hybrid_ep.py"

# --------------------------------------------------- distributed (platform env)
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export WORLD_SIZE=${WORLD_SIZE:-1}          # = number of nodes (init_dist num_nodes)
export RANK=${RANK:-0}                       # = node rank      (init_dist node_rank)
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-8361}

if [[ "${WORLD_SIZE}" -lt 2 ]]; then
  echo "ERROR: this is a CROSS-NODE test; need WORLD_SIZE (node count) >= 2, got ${WORLD_SIZE}." >&2
  echo "       Launch a 2-node canoe job." >&2
  exit 1
fi
if [[ ! -f "${TEST_PY}" ]]; then
  echo "ERROR: ${TEST_PY} not found. Is this the DeepEP/HybridEP-enabled image?" >&2
  exit 1
fi

# HybridEP cross-node uses the DOCA/RDMA coordinator (HYBRID_EP_MULTINODE build),
# not NVSHMEM, so no NVSHMEM env is required here.

echo "============================================"
echo "HybridEP internode test"
echo "  nodes (WORLD_SIZE) : ${WORLD_SIZE}"
echo "  node_rank (RANK)   : ${RANK}"
echo "  gpus/node          : ${GPUS_PER_NODE}  (-> --num-processes)"
echo "  master             : ${MASTER_ADDR}:${MASTER_PORT}"
echo "  total ranks        : $((WORLD_SIZE * GPUS_PER_NODE))"
echo "  HIDDEN_DIM=${HIDDEN_DIM:-7168} NUM_LOCAL_EXPERTS=${NUM_LOCAL_EXPERTS:-8} TOPK=${TOPK:-8}"
echo "  USE_MNNVL=${USE_MNNVL:-0}"
echo "============================================"

cd "${DEEPEP_DIR}"
# test_hybrid_ep.py spawns GPUS_PER_NODE local ranks itself; launch ONE process
# per node and let it fan out.
exec python3 tests/test_hybrid_ep.py --num-processes "${GPUS_PER_NODE}"
