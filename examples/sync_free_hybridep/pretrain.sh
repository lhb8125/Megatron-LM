#!/bin/bash
# =============================================================================
# Platform entrypoint (meepo-style) for THIS repo's sync-free HybridEP GPT-MoE.
#
# Mirrors examples/meepo/pretrain.sh: the experiment platform injects a config
# YAML at CONFIG_FILE (default /canoe/exp/config.yaml) and the distributed
# rendezvous via WORLD_SIZE / RANK / MASTER_ADDR / MASTER_PORT / GPUS_PER_NODE.
#
# Unlike meepo (pretrain_meepo_multimodal.py --config-file), this repo's
# pretrain_gpt.py takes plain CLI flags, so we convert the YAML's ARGS/ENV_VARS
# blocks with yaml_to_shell.py.
#
# Platform usage:
#   image:  harbor.xaminim.com/minimax-dialogue/megatron-sync-free:<tag>
#   entry:  bash -cx 'bash -x examples/sync_free_hybridep/pretrain.sh'
#   (mount / point CONFIG_FILE at examples/sync_free_hybridep/config_m3mini_syncfree.yaml)
#
# Local usage:
#   CONFIG_FILE=examples/sync_free_hybridep/config_m3mini_syncfree.yaml \
#     bash examples/sync_free_hybridep/pretrain.sh
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

CONFIG_FILE=${CONFIG_FILE:-/canoe/exp/config.yaml}
ENTRY=${ENTRY:-pretrain_gpt.py}
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "Error: Config file not found: ${CONFIG_FILE}" && exit 1
fi
if [ ! -f "${ENTRY}" ]; then
    echo "Error: Entry python file not found: ${ENTRY}" && exit 1
fi

# --------------------------------------------------- YAML -> env exports + CLI
# ENV_VARS block -> exported into this shell (so torchrun children inherit them).
while IFS= read -r line; do
    [ -n "${line}" ] && eval "${line}"
done < <(python3 "${SCRIPT_DIR}/yaml_to_shell.py" "${CONFIG_FILE}" env)

TRAIN_ARGS="$(python3 "${SCRIPT_DIR}/yaml_to_shell.py" "${CONFIG_FILE}" args)"

# Optional overrides handy on the platform.
if [ -n "${TRAIN_ITERS:-}" ]; then
    TRAIN_ARGS="$(echo "${TRAIN_ARGS}" | sed -E "s/--train-iters [0-9]+/--train-iters ${TRAIN_ITERS}/")"
fi

# --------------------------------------------------- distributed (platform env)
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export NNODES=${WORLD_SIZE:-1}
export NODE_RANK=${RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-1234}
DISTRIBUTED_ARGS="--nproc_per_node ${GPUS_PER_NODE} --nnodes ${NNODES} \
  --node_rank ${NODE_RANK} --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT}"

# --------------------------------------------------- optional nsys (meepo-style)
NSYS_PROFILE_ARGS=""
if [[ ${NSYS_PROFILE_ENABLED:-0} -eq 1 ]]; then
    TRAIN_ARGS+=" --profile --profile-step-start ${PROFILE_STEP_START:-45} \
      --profile-step-end ${PROFILE_STEP_END:-47} --profile-ranks ${PROFILE_RANKS:-0}"
    NSYS_OUTPUT_DIR=${TENSORBOARD_LOG_PATH:-/tensorboard-logs}/nsys_output
    mkdir -p "${NSYS_OUTPUT_DIR}" && chmod -R 777 "${NSYS_OUTPUT_DIR}" || true
    TRACE="cuda"
    if [[ ${NVTX_PROFILE_ENABLED:-0} -eq 1 ]]; then
        TRACE="cuda,nvtx"
        # This repo gates framework NVTX ranges behind --nvtx-ranges; without it
        # nvtx_range_push/pop are no-ops and no framework tags show up in nsys.
        TRAIN_ARGS+=" --nvtx-ranges"
    fi
    NSYS_PROFILE_ARGS="nsys profile -s none -o ${NSYS_OUTPUT_DIR}/${NODE_RANK} \
      -t ${TRACE} ${NSYS_CUDA_GRAPH_TRACE:+--cuda-graph-trace=${NSYS_CUDA_GRAPH_TRACE}} \
      --force-overwrite true --capture-range=cudaProfilerApi \
      --capture-range-end=stop --cpuctxsw=none"
fi

echo "============================================"
echo "sync-free HybridEP pretrain (--config-file style, converted to CLI flags)"
echo "CONFIG_FILE : ${CONFIG_FILE}"
echo "ENTRY       : ${ENTRY}"
echo "WORLD_SIZE  : $((GPUS_PER_NODE * NNODES)) (${NNODES} nodes x ${GPUS_PER_NODE} GPUs)"
echo "============================================"

${NSYS_PROFILE_ARGS} torchrun ${DISTRIBUTED_ARGS} \
    "${ENTRY}" \
    ${TRAIN_ARGS} \
    ${EXTRA_OPTIONS:-}
