# HybridEP Sync-Free MoE + 1F1B Combined Overlap — single-node B300 experiment

This directory delivers **two runnable, high-throughput single-node (8× B300)**
Megatron-LM experiments for the *HybridEP sync-free MoE* design in
`hybridep_paged_stash_sync_free.zh.pdf`, both combined with the **1F1B combined
EP-A2A overlap** schedule, each with a Docker image that contains everything needed:

- **Path A** — CuTe DSL fused GroupedMLP + TE op-fuser + full-iteration CUDA graph.
- **Path B** — device-init GroupedTensor/cublas grouped GEMM (Megatron PR #6000 +
  TE PR #3224), **no op-fuser and no full CUDA graph** (for SFT / post-training /
  dynamic control flow).

Both are verified end-to-end:

- Training runs to completion with a decreasing loss.
- Nsight Systems confirms **no CPU–GPU synchronization** in the steady-state
  layer forward/backward on **all 8 ranks** (`0` synchronize/memcpy CUDA API
  calls, no sync-like NVTX ranges).

## What "sync-free" means here (and how it is achieved)

The MoE steady-state data path
`router → dispatch → FC1 → activation → FC2 → combine` runs with **no dynamic
receive-sizing / expert-split-metadata D2H sync**. This is the union of:

| Building block | Path A flags | Path B flags |
|---|---|---|
| HybridEP static-budget dispatch | `--moe-flex-dispatcher-backend hybridep` + `--moe-expert-rank-capacity-factor 1.5` | same |
| Paged stash (static fwd/bwd activation storage) | `--moe-paged-stash` (+ page-size / buffer-size-factor) | same |
| Sync-free expert compute | CuTe DSL fused GroupedMLP + TE op-fuser (`--use-transformer-engine-op-fuser` + `NVTE_CUTEDSL_FUSED_GROUPED_MLP=1`) | device-init GroupedTensor/cublas (`--moe-use-grouped-tensor`, **no op-fuser**) |
| Host-launch amortization | Full-iteration CUDA graph (`--cuda-graph-impl local` + `--cuda-graph-scope full_iteration`) | **none** (eager; GroupedTensor consumes CUDA int64 splits directly) |
| 1F1B combined EP-A2A overlap | `--overlap-moe-expert-parallel-comm` + `--delay-wgrad-compute` | same |

The static budget is kept balanced under mock data via
`--moe-router-force-load-balancing`, so overflow/rerun (the control-plane
`.item()` at step boundary, which is *outside* the sync-free scope) stays rare.

> **Path B enablement**: `--moe-use-grouped-tensor` was ported from
> Megatron-LM PR #6000 onto this `dev` checkout (branch
> `feat/pr6000-grouped-tensor`), and the Path-B image builds TransformerEngine
> from PR #3224 (`GroupedLinear(use_grouped_tensor=...)`), which is not yet in
> TE main.

## Files

| File | Purpose |
|---|---|
| `Dockerfile` | **Path A** B300 (SM103a) image: TE v2.15 main + CuTe DSL 4.5.2 + cuDNN-fe 1.23 + FA4 + DeepEP `hybrid-ep` + `nvidia-resiliency-ext`. |
| `Dockerfile.pathB` | **Path B** image: TransformerEngine PR #3224 fork branch (adds `use_grouped_tensor`) + DeepEP `hybrid-ep` + resiliency-ext. |
| `sync_free_hybridep_b300_1node.yaml` | **Path A** recipe — Qwen3-30B-style MoE scaled to 8 GPUs (TP1/PP1/**EP8**, 64 experts, top-8, SL 4096), mock data, meepo tokenizer, full attention. |
| `sync_free_grouped_tensor_b300_1node.yaml` | **Path B** recipe — same model, `moe_use_grouped_tensor=true`, no op-fuser, no full CG. |
| `run_sync_free_hybridep_b300.sh` | Launcher (used for both paths via `RECIPE=` / `IMAGE=`): YAML → CLI, runs `pretrain_gpt.py` under `torchrun`, optional nsys profiling (adds `--cap-add=SYS_ADMIN`). |
| `nsys_rank_shim.sh` | Per-rank shim for `NSYS=3`: wraps only `PROFILE_RANK` with nsys so a single-rank report with a real CUDA trace is produced. |
| `yaml_to_shell.py` | Recipe YAML → shell args/env converter. |
| `check_no_cpu_gpu_sync.py` | Parses an nsys sqlite export and reports the CPU-GPU sync verdict. |
| `artifacts/`, `artifacts_pathB/` | Verified training logs + 8-rank nsys reports for Path A / Path B. |

| File | Purpose |
|---|---|
| `Dockerfile` | B300 (SM103a) image: TE v2.15 + CuTe DSL 4.5.2 + cuDNN-fe 1.23 + FA4 + DeepEP `hybrid-ep` (`deep_ep.HybridEPBuffer`) + `nvidia-resiliency-ext`. |
| `sync_free_hybridep_b300_1node.yaml` | The recipe (ENV_VARS + ARGS) — Qwen3-30B-style MoE scaled to 8 GPUs (TP1/PP1/**EP8**, 64 experts, top-8, SL 4096), mock data, meepo tokenizer, full attention. |
| `run_sync_free_hybridep_b300.sh` | Launcher: converts the YAML → CLI, runs `pretrain_gpt.py` under `torchrun` inside the image, with optional nsys profiling. |
| `yaml_to_shell.py` | Recipe YAML → shell args/env converter. |
| `check_no_cpu_gpu_sync.py` | Parses an nsys sqlite export and reports the CPU-GPU sync verdict. |

## Image

```
megatron-sync-free-hybridep:b300
```

A portable copy is saved next to these files as
`megatron-sync-free-hybridep-b300.tar.gz` (~11 GB). Load it with:

```bash
gunzip -c megatron-sync-free-hybridep-b300.tar.gz | sudo docker load
```

Build (host has docker; base image `nvcr.io/nvidia/pytorch:26.04-py3`):

```bash
# Path A
sudo docker build --target hybridep \
  --build-arg FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:26.04-py3 \
  -f Dockerfile -t megatron-sync-free-hybridep:b300 .

# Path B (TransformerEngine PR #3224 built from source)
sudo docker build --target hybridep \
  --build-arg FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:26.04-py3 \
  -f Dockerfile.pathB -t megatron-sync-free-grouped-tensor:b300 .
```

Path B image is also saved as `megatron-sync-free-grouped-tensor-b300.tar.gz`.

## Run

```bash
# Path A — normal training
./run_sync_free_hybridep_b300.sh

# Path A — nsys profiling of steady state (time-window; robust under full CUDA graphs)
NSYS=2 NSYS_DELAY=125 NSYS_DURATION=6 TRAIN_ITERS=200 ./run_sync_free_hybridep_b300.sh

# Path B — normal training (device-init GroupedTensor, no full CG)
IMAGE=megatron-sync-free-grouped-tensor:b300 \
  RECIPE=$PWD/sync_free_grouped_tensor_b300_1node.yaml \
  ./run_sync_free_hybridep_b300.sh

# Path B — nsys profiling
IMAGE=megatron-sync-free-grouped-tensor:b300 \
  RECIPE=$PWD/sync_free_grouped_tensor_b300_1node.yaml \
  NSYS=2 NSYS_DELAY=130 NSYS_DURATION=6 TRAIN_ITERS=200 ./run_sync_free_hybridep_b300.sh

# NSYS=3 — single-rank (rank0) full CUDA+NVTX trace (recommended for inspection):
#   one report_rank0.nsys-rep with real CUPTI kernels/API, covering a few steady iters.
IMAGE=megatron-sync-free-grouped-tensor:b300 \
  RECIPE=$PWD/sync_free_grouped_tensor_b300_1node.yaml \
  NSYS=3 PROFILE_RANK=0 NSYS_DELAY=0 NSYS_DURATION=118 TRAIN_ITERS=60 \
  ./run_sync_free_hybridep_b300.sh
```

Overridable env: `IMAGE`, `RECIPE`, `GPUS_PER_NODE`, `MASTER_PORT`, `OUTPUT_PATH`,
`TOKENIZER_DIR`, `TRAIN_ITERS`, `NSYS` (`0`/`1`/`2`/`3`), `PROFILE_RANK`,
`NSYS_DELAY`, `NSYS_DURATION`, `PROFILE_STEP_START`, `PROFILE_STEP_END`.

> **CUDA trace needs `--cap-add=SYS_ADMIN`** (CUPTI kernel/API profiling). The
> launcher adds it automatically in any `NSYS!=0` mode. Without it nsys records
> only NVTX ranges (no `CUPTI_ACTIVITY_KIND_KERNEL`).
>
> **NSYS=3** wraps only `PROFILE_RANK` (default 0) with nsys (via
> `nsys_rank_shim.sh`), other ranks run plain python — so you get a single
> `report_rank0.nsys-rep` with a real CUDA trace. `check_no_cpu_gpu_sync.py`
> then isolates one steady-state MoE-layer window and reports blocking sync
> (`*Synchronize` + D2H `cudaMemcpy`) inside it.

## Verified results (8× B300, this node)

### Path A — CuTe fused GroupedMLP + full CG

Training (60 iters):

```
iteration   4/60 ... TFLOP/s/GPU: 266.0 ... lm loss: 1.199278E+01
...
iteration  60/60 ... TFLOP/s/GPU: 265.9 ... lm loss: 7.645992E+00
[after training is done]
```

- loss `12.6 → 7.6`, **~266 TFLOP/s/GPU**, ~737 ms/iter steady state,
  0 NaN / 0 skipped iterations / 0 paged-stash-overflow reruns.

Nsight Systems (`NSYS=2`, steady-state window, all 8 ranks):

```
PASS=8 FAIL=0 (out of 8 ranks)
  cuda runtime api total     : 17          # only cudaGetDeviceProperties
  synchronize/memcpy api     : 0           # NO cudaStreamSynchronize / cudaDeviceSynchronize / cudaMemcpy
  device->host memcpy rows   : 0
  layer NVTX ranges captured : 558         # full MoE data path incl. hybrid-ep dispatch/combine
  VERDICT                    : PASS (sync-free)
```

The captured NVTX ranges show the complete layer path
(`dispatch_preprocess → dispatch_core → dispatch_postprocess → grouped MLP →
combine_preprocess → combine_core → combine_postprocess`, plus TE attention /
RMSNorm / GEMM / router) with **no synchronization primitive** — the layer
forward/backward runs as CUDA-graph replays with zero CPU-GPU sync.

### Path B — device-init GroupedTensor/cublas (no op-fuser, no full CG)

Training (60 iters):

```
iteration   5/60 ... TFLOP/s/GPU: 222.8 ... lm loss: 1.172900E+01
...
iteration  60/60 ... TFLOP/s/GPU: 234.5 ... lm loss: 7.475336E+00
[after training is done]
```

- loss `12.6 → 7.48`, **~235 TFLOP/s/GPU**, ~830 ms/iter steady state,
  0 NaN / 0 skipped iterations.

Nsight Systems (single-rank `NSYS=3` report with full CUDA trace,
`check_no_cpu_gpu_sync.py` isolating one steady MoE-layer window):

```
mode                       : precise (steady MoE-layer window)
window length              : 6.623 ms, 275 kernels
*Synchronize in window     : 0
D2H cudaMemcpy in window   : 0
VERDICT                    : PASS (sync-free layer)
```

The only memcpy activity inside the steady MoE-layer window is 4× **D2D**
(device-to-device, pure on-GPU) copies — **no D2H and no `cudaStreamSynchronize`
/ `cudaDeviceSynchronize`**. Because Path B is eager (no CUDA graph), the trace
also shows the individual expert GEMMs as `nvte_grouped_gemm_with_discrete_inputA`
/ `nvte_grouped_gemm_with_discrete_out` + `nvte_group_quantize` — the device-init
GroupedTensor/cublas path from TE PR #3224 that consumes CUDA int64 split
metadata directly, so **no `tokens_per_expert.tolist()` / D2H sizing sync** ever
appears in the layer forward/backward. Overflow/rerun `.item()` remains only at
the step boundary (expected control-plane sync).

### nsys tips (learned here)

- Under full-iteration CUDA graphs, the reliable capture is a **time window**
  (`--delay/--duration`) with **`--kill=none`** so nsys stops collection and
  writes the report without SIGTERM'ing a rank mid-`nvcc` (HybridEP JIT).
- The Megatron `--profile` cudaProfilerApi range does not reliably drive
  `--capture-range-end=stop-shutdown` through the `torchrun` parent in this
  docker setup, so prefer `NSYS=2`.
