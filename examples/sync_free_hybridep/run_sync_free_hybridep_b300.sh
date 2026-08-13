#!/bin/bash
# =============================================================================
# HybridEP sync-free MoE + 1F1B combined-overlap — single-node B300 launcher
# =============================================================================
# Runs the recipe in examples/sync_free_hybridep/sync_free_hybridep_b300_1node.yaml
# inside the prebuilt image (megatron-sync-free-hybridep:b300) on one 8x B300 node.
#
# Path A: HybridEP static budget + paged stash + CuTe DSL fused GroupedMLP +
#         TE op-fuser + full-iteration CUDA graph + 1F1B combined EP-A2A overlap.
#
# Usage:
#   ./run_sync_free_hybridep_b300.sh                                   # normal training
#   NSYS_PROFILE_ENABLED=1 ./run_sync_free_hybridep_b300.sh            # nsys, cuda only
#   NSYS_PROFILE_ENABLED=1 NVTX_PROFILE_ENABLED=1 ./run_..._b300.sh    # nsys + NVTX
#
# nsys profiling is controlled the same way as the reference Megatron repo
# (examples/meepo/pretrain.sh):
#   NSYS_PROFILE_ENABLED=1   enable nsys capture (cudaProfilerApi range)
#   NVTX_PROFILE_ENABLED=1   also record NVTX ranges (-t cuda,nvtx). Because THIS
#                            repo gates framework NVTX behind --nvtx-ranges, we
#                            additionally pass that flag so nvtx_range_push/pop fire.
#   NSYS_CUDA_GRAPH_TRACE=node   opt-in: expand cudagraph nodes in the trace.
# Reports land in $OUTPUT_PATH/nsys_output/<NODE_RANK>.nsys-rep.
#
# Env overrides: IMAGE, GPUS_PER_NODE, MASTER_PORT, OUTPUT_PATH,
#                PROFILE_STEP_START, PROFILE_STEP_END, PROFILE_RANKS
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RECIPE="${RECIPE:-${SCRIPT_DIR}/sync_free_hybridep_b300_1node.yaml}"

IMAGE="${IMAGE:-megatron-sync-free-hybridep:b300}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29511}"
OUTPUT_PATH="${OUTPUT_PATH:-${SCRIPT_DIR}/output}"
# In-container tensorboard log dir. Like the reference repo, nsys reports live
# under ${TENSORBOARD_LOG_PATH}/nsys_output and --tensorboard-dir == this path.
TENSORBOARD_LOG_PATH="${TENSORBOARD_LOG_PATH:-/workspace/output/tensorboard}"
TOKENIZER_DIR="${TOKENIZER_DIR:-/data/minimax-dialogue/pretrain_model/m2-mini/tokenizer}"
NODE_RANK="${NODE_RANK:-0}"
NSYS_PROFILE_ENABLED="${NSYS_PROFILE_ENABLED:-0}"
NVTX_PROFILE_ENABLED="${NVTX_PROFILE_ENABLED:-0}"
NSYS_CUDA_GRAPH_TRACE="${NSYS_CUDA_GRAPH_TRACE:-}"
PROFILE_STEP_START="${PROFILE_STEP_START:-20}"
PROFILE_STEP_END="${PROFILE_STEP_END:-23}"
PROFILE_RANKS="${PROFILE_RANKS:-0}"

DOCKER="${DOCKER:-sudo docker}"

mkdir -p "${OUTPUT_PATH}"

# ------------------------------------------------------------------ build args/env
ENV_EXPORTS="$(python3 "${SCRIPT_DIR}/yaml_to_shell.py" "${RECIPE}" env)"
TRAIN_ARGS="$(python3 "${SCRIPT_DIR}/yaml_to_shell.py" "${RECIPE}" args)"

# Optional override of train_iters (e.g. run longer so an nsys time-window
# comfortably lands in steady state).
if [[ -n "${TRAIN_ITERS:-}" ]]; then
  TRAIN_ARGS="$(echo "${TRAIN_ARGS}" | sed -E "s/--train-iters [0-9]+/--train-iters ${TRAIN_ITERS}/")"
fi

LAST_RANK=$((GPUS_PER_NODE - 1))
DIST_ARGS="--nproc_per_node ${GPUS_PER_NODE} --nnodes 1 --node_rank 0 \
  --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT}"

# nsys profiling — aligned with the reference repo (examples/meepo/pretrain.sh):
# a single NSYS_PROFILE_ARGS prefix placed before torchrun, cudaProfilerApi
# capture range driven by Megatron's --profile (steps PROFILE_STEP_START..END).
NSYS_PROFILE_ARGS=""
if [[ "${NSYS_PROFILE_ENABLED}" == "1" ]]; then
  # Reference repo: NSYS_OUTPUT_DIR=$TENSORBOARD_LOG_PATH/nsys_output, created
  # (mkdir+chmod 777) in-container next to torchrun. See INNER below.
  NSYS_OUTPUT_DIR="${TENSORBOARD_LOG_PATH}/nsys_output"
  TRAIN_ARGS="${TRAIN_ARGS} --profile --profile-step-start ${PROFILE_STEP_START} \
    --profile-step-end ${PROFILE_STEP_END} --profile-ranks ${PROFILE_RANKS}"
  # Opt-in: export NSYS_CUDA_GRAPH_TRACE=node to expand cudagraph nodes in traces.
  CUDA_GRAPH_TRACE_ARG=""
  if [[ -n "${NSYS_CUDA_GRAPH_TRACE}" ]]; then
    CUDA_GRAPH_TRACE_ARG="--cuda-graph-trace=${NSYS_CUDA_GRAPH_TRACE}"
  fi
  if [[ "${NVTX_PROFILE_ENABLED}" == "1" ]]; then
    # This repo gates framework NVTX behind --nvtx-ranges; add it so the
    # nvtx_range_push/pop tags actually show up in the trace.
    TRAIN_ARGS="${TRAIN_ARGS} --nvtx-ranges"
    NSYS_TRACE="cuda,nvtx"
  else
    NSYS_TRACE="cuda"
  fi
  NSYS_PROFILE_ARGS="nsys profile -s none -o ${NSYS_OUTPUT_DIR}/${NODE_RANK} \
    -t ${NSYS_TRACE} ${CUDA_GRAPH_TRACE_ARG} --force-overwrite true \
    --capture-range=cudaProfilerApi --capture-range-end=stop --cpuctxsw=none"
fi

# ------------------------------------------------------------------ in-container cmd
read -r -d '' INNER <<INNER_EOF || true
set -uo pipefail
cd /workspace/Megatron-LM
export PYTHONPATH=/workspace/Megatron-LM:\${PYTHONPATH:-}
${ENV_EXPORTS}
if [[ "${NSYS_PROFILE_ENABLED}" == "1" ]]; then
  mkdir -p "${NSYS_OUTPUT_DIR}"
  chmod -R 777 "${NSYS_OUTPUT_DIR}"
fi
echo "=================== effective env (subset) ==================="
env | grep -E 'NVTE_|HYBRID|NVLINK|CUDA_DEVICE_MAX' || true
echo "=============================================================="
echo "NSYS_PROFILE_ARGS: ${NSYS_PROFILE_ARGS}"
${NSYS_PROFILE_ARGS} python -u -m torch.distributed.run ${DIST_ARGS} \
  /workspace/Megatron-LM/pretrain_gpt.py \
  ${TRAIN_ARGS} \
  --save-interval 100000 \
  --tensorboard-dir ${TENSORBOARD_LOG_PATH} \
  2>&1 | stdbuf -oL tee /workspace/output/train.log
INNER_EOF

# nsys/CUPTI kernel & CUDA-API tracing needs the SYS_ADMIN capability inside the
# container (otherwise only NVTX is recorded). Only added when profiling.
CAP_ARGS=""
if [[ "${NSYS_PROFILE_ENABLED}" == "1" ]]; then
  CAP_ARGS="--cap-add=SYS_ADMIN"
fi

# ------------------------------------------------------------------ run
${DOCKER} run --rm --gpus all ${CAP_ARGS} \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  --shm-size=32g \
  -v "${REPO_DIR}:/workspace/Megatron-LM" \
  -v "${OUTPUT_PATH}:/workspace/output" \
  -v "${TOKENIZER_DIR}:${TOKENIZER_DIR}:ro" \
  -w /workspace/Megatron-LM \
  --entrypoint bash \
  "${IMAGE}" -c "${INNER}"
