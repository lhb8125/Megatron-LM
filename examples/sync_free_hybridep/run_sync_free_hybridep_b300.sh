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
#   ./run_sync_free_hybridep_b300.sh                 # normal training
#   NSYS=1 ./run_sync_free_hybridep_b300.sh          # profile steps with nsys
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
TOKENIZER_DIR="${TOKENIZER_DIR:-/data/minimax-dialogue/pretrain_model/m2-mini/tokenizer}"
NSYS="${NSYS:-0}"
PROFILE_STEP_START="${PROFILE_STEP_START:-20}"
PROFILE_STEP_END="${PROFILE_STEP_END:-23}"
PROFILE_RANKS="${PROFILE_RANKS:-0 1 2 3 4 5 6 7}"

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

# nsys: two modes.
#   NSYS=1  -> cudaProfilerApi capture range driven by Megatron's --profile
#              (steps PROFILE_STEP_START..END), wrapped per RANK so stop-shutdown
#              cleanly writes each rank's report.
#   NSYS=2  -> time-window capture (--delay/--duration). Robust fallback that
#              does not depend on the app calling cudaProfilerStop; writes the
#              report unconditionally after NSYS_DURATION seconds.
NSYS_PREFIX=""
NO_PYTHON=""
# PY_BIN is the interpreter placed *after* the nsys wrapper. With torchrun
# --no-python (nsys per-rank wrapping) we must re-add `python`; without nsys,
# torchrun runs the .py script directly so PY_BIN stays empty.
PY_BIN=""
if [[ "${NSYS}" == "1" ]]; then
  NSYS_OUT="${OUTPUT_PATH}/nsys"
  mkdir -p "${NSYS_OUT}"
  TRAIN_ARGS="${TRAIN_ARGS} --profile --profile-step-start ${PROFILE_STEP_START} \
    --profile-step-end ${PROFILE_STEP_END} --profile-ranks ${PROFILE_RANKS}"
  NO_PYTHON="--no-python"
  PY_BIN="python"
  NSYS_PREFIX="nsys profile -s none -o ${NSYS_OUT}/report_rank%q{RANK} \
    -t cuda,nvtx --cuda-graph-trace=node \
    --force-overwrite true --capture-range=cudaProfilerApi \
    --capture-range-end=stop-shutdown --kill=sigterm --cpuctxsw=none"
elif [[ "${NSYS}" == "2" ]]; then
  NSYS_OUT="${OUTPUT_PATH}/nsys"
  mkdir -p "${NSYS_OUT}"
  NSYS_DELAY="${NSYS_DELAY:-150}"
  NSYS_DURATION="${NSYS_DURATION:-8}"
  NO_PYTHON="--no-python"
  PY_BIN="python"
  NSYS_PREFIX="nsys profile -s none -o ${NSYS_OUT}/report_rank%q{RANK} \
    -t cuda,nvtx --cuda-graph-trace=node \
    --force-overwrite true --delay=${NSYS_DELAY} --duration=${NSYS_DURATION} \
    --kill=none --cpuctxsw=none"
elif [[ "${NSYS}" == "3" ]]; then
  # Single-rank (default rank0) full CUDA+NVTX trace over a few steady iterations.
  # Only the profiled rank is wrapped by nsys (via nsys_rank_shim.sh); other ranks
  # run plain python. Uses a time window (--delay/--duration, --kill=none) which is
  # the mode that reliably records CUPTI kernel/API activity + writes the report
  # in this docker+torchrun setup. Size NSYS_DELAY so the window lands in steady
  # state and NSYS_DURATION to cover ~3 iterations.
  NSYS_OUT="${OUTPUT_PATH}/nsys"
  mkdir -p "${NSYS_OUT}"
  NO_PYTHON="--no-python"
  PY_BIN="/workspace/Megatron-LM/examples/sync_free_hybridep/nsys_rank_shim.sh"
fi

# ------------------------------------------------------------------ in-container cmd
read -r -d '' INNER <<INNER_EOF || true
set -uo pipefail
cd /workspace/Megatron-LM
export PYTHONPATH=/workspace/Megatron-LM:\${PYTHONPATH:-}
${ENV_EXPORTS}
# Make sure any nsys reports (incl. nsys' /tmp/nsys-root fallback) land in the
# mounted output dir even if the run is torn down at capture-range shutdown.
collect_nsys() {
  if [[ "${NSYS}" != "0" ]]; then
    mkdir -p /workspace/output/nsys
    find /tmp/nsys-root /workspace/Megatron-LM /workspace/output -maxdepth 2 \
      -name '*.nsys-rep' -newermt '-1 hour' 2>/dev/null \
      -exec cp -f {} /workspace/output/nsys/ \; || true
  fi
}
trap collect_nsys EXIT
export NSYS_OUT_DIR=/workspace/output/nsys
export NSYS_PROFILE_RANK=${PROFILE_RANK:-0}
export NSYS_TRACE=${NSYS_TRACE:-cuda,nvtx}
export NSYS_SHIM_MODE=${NSYS_SHIM_MODE:-window}
export NSYS_SHIM_DELAY=${NSYS_DELAY:-125}
export NSYS_SHIM_DURATION=${NSYS_DURATION:-4}
echo "=================== effective env (subset) ==================="
env | grep -E 'NVTE_|HYBRID|NVLINK|CUDA_DEVICE_MAX' || true
echo "=============================================================="
stdbuf -oL -eL python -u -m torch.distributed.run ${DIST_ARGS} ${NO_PYTHON} \
  ${NSYS_PREFIX} \
  ${PY_BIN} /workspace/Megatron-LM/pretrain_gpt.py \
  ${TRAIN_ARGS} \
  --save-interval 100000 \
  --tensorboard-dir /workspace/output/tensorboard \
  2>&1 | stdbuf -oL tee /workspace/output/train.log
collect_nsys
INNER_EOF

# nsys/CUPTI kernel & CUDA-API tracing needs the SYS_ADMIN capability inside the
# container (otherwise only NVTX is recorded). Only added when profiling.
CAP_ARGS=""
if [[ "${NSYS}" != "0" ]]; then
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
