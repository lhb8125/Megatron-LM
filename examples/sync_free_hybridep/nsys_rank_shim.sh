#!/bin/bash
# Per-rank launch shim for torchrun --no-python.
# Wraps ONLY the profiled rank (default rank 0) with nsys; all other ranks run
# plain python so the job completes normally (no NCCL shutdown hang).
#
# nsys uses cudaProfilerApi capture range (Megatron --profile drives
# cudaProfilerStart/Stop), so CUPTI attaches at process launch and records the
# full CUDA API + kernel trace (not just NVTX). Collection is bracketed to the
# --profile-step-start..end window; the report is flushed on clean process exit.
#
# Env (exported by the launcher):
#   NSYS_PROFILE_RANK   rank to profile (default 0)
#   NSYS_OUT_DIR        output dir for the .nsys-rep
#   NSYS_TRACE          nsys -t value (default cuda,nvtx)
set -uo pipefail

PROFILE_RANK="${NSYS_PROFILE_RANK:-0}"
OUT_DIR="${NSYS_OUT_DIR:-/workspace/output/nsys}"
TRACE="${NSYS_TRACE:-cuda,nvtx}"
NSYS_MODE="${NSYS_SHIM_MODE:-caprange}"   # caprange | window
NSYS_DELAY="${NSYS_SHIM_DELAY:-125}"
NSYS_DURATION="${NSYS_SHIM_DURATION:-4}"
mkdir -p "${OUT_DIR}"
echo "[nsys_rank_shim] RANK=${RANK:-unset} LOCAL_RANK=${LOCAL_RANK:-unset} PROFILE_RANK=${PROFILE_RANK} MODE=${NSYS_MODE} OUT_DIR=${OUT_DIR}" >&2

if [[ "${RANK:-0}" == "${PROFILE_RANK}" ]]; then
  echo "[nsys_rank_shim] wrapping rank ${RANK:-0} with nsys (${NSYS_MODE})" >&2
  if [[ "${NSYS_MODE}" == "window" ]]; then
    # Time-window capture: CUPTI is attached for the whole session, so kernel/API
    # activity is recorded (not only NVTX). --kill=none leaves the app running so
    # nsys flushes the report without SIGTERM'ing a rank mid-JIT.
    exec nsys profile \
      -s none \
      -o "${OUT_DIR}/report_rank${PROFILE_RANK}" \
      -t "${TRACE}" \
      --cuda-graph-trace=node \
      --force-overwrite true \
      --delay="${NSYS_DELAY}" --duration="${NSYS_DURATION}" \
      --kill=none \
      --cpuctxsw=none \
      python "$@"
  fi
  exec nsys profile \
    -s none \
    -o "${OUT_DIR}/report_rank${PROFILE_RANK}" \
    -t "${TRACE}" \
    --cuda-graph-trace=node \
    --force-overwrite true \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop-shutdown \
    --cpuctxsw=none \
    python "$@"
else
  exec python "$@"
fi
