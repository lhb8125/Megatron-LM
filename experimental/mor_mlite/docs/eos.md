# EOS / H100 execution

This directory is the reproducible execution boundary for the first MoR-MLite
release.  It is intentionally fixed to one EOS H100 node and at most eight
ranks.  The launchers do not modify either Megatron-LM or MagiAttention; they
checkout the exact upstream revisions under a private dependency directory and
put MLite on `PYTHONPATH`.

## Fixed environment

| Item | Value |
|---|---|
| Login host | `login-eos` |
| Slurm account | `coreai_devtech_all` |
| Project | 启动脚本所在 checkout；通过 `MOR_PROJECT_ROOT` 校验实际 import 来源 |
| Partition / constraint | `batch` / `h100` |
| Nodes / ranks | one / at most eight |
| NGC image | `nvcr.io/nvidia/pytorch:26.01-py3` |
| Training Torch / CUDA | `2.10.0+cu129` / `12.9` |
| CUDA Python / bindings | `12.9.4` / `12.9.4` |
| Triton / NVRx / Transformer Engine | `3.6.0` / `0.6.0` / `2.13.0` |
| Megatron-LM | `5c8315f12a64a7279eec58896af9e74ee3351b74` |
| MagiAttention | `v1.1.1`, built for `sm90` |

Pyxis writes the same image as `nvcr.io#nvidia/pytorch:26.01-py3`; `#` is its
registry separator.  That image's bundled prerelease Torch reports CUDA 13.1;
it is used as the H100 compiler/driver bootstrap only.  `setup_env.sh` installs
the exact official `torch==2.10.0+cu129` wheel and a matching Transformer Engine
extension into `.deps/venv-torch210-cu129-v3`.  All training and acceptance tests
run from that version-keyed venv, and the post-setup manifest still rejects
anything other than the cu129 runtime.

The pinned Megatron revision's lock file points at Transformer Engine v2.14,
but the published v2.14 PyTorch sdist metadata unconditionally depends on the
CUDA-13 core package.  That conflicts with this deliberately fixed Torch/cu129
runtime.  The overlay therefore pins TE 2.13, the newest release whose PyTorch
sdist selects the CUDA-12 core and which already provides MLite's required
`moe_permute_and_pad_with_probs` API.  Startup imports that symbol and the real
MLite MoE utility module before exercising a BF16 TE `Linear` forward/backward;
an incompatible TE install fails before model construction.

The setup installs `nvidia-resiliency-ext==0.6.0` and the narrow MLite runtime
dependency set explicitly.  It does not install Megatron-LM's broad `dev`
extra, which would pull unrelated model-family packages and potentially replace
the pinned CUDA stack.  `constraints-cu129.txt` is exported through the entire
setup, including the pip processes spawned by MLite's official Magi builder;
this prevents Magi's unbounded transitive `torch` requirements from replacing
Torch 2.10/cu129.  The setup also verifies that ABI-sensitive distribution and
module origins are inside that venv, so importing a seemingly compatible copy
inherited from the base image is not accepted.  The manifest records bootstrap
`nvcc`, Torch's C++11 ABI flag, and Python SOABI.  Setup and `version_probe.sh`
also import MLite's required fused MoE permutation API and run a real
Transformer Engine BF16 `Linear` forward/backward canary.

## Sync, inspect, and submit

隔离 worktree 的本地目录与 EOS 目标目录可以不同。同步前用 `MOR_EOS_REMOTE_ROOT`
指定远程绝对路径；未设置时仍使用历史默认目录。`MOR_DEPS_ROOT` / `MOR_HF_HOME` 可复用
已经验证的依赖与 HF cache，不能在作业执行期间覆盖对应源码或升级共享依赖。
`submit_matrix.sh` / `submit_qwen30b.sh` 将本 checkout 的工作目录和日志路径传给 `sbatch`。
直接提交 `.sbatch` 时必须从项目根目录运行并预先创建 `logs/`，或明确设置 `MOR_PROJECT_ROOT`
和 Slurm 的 `--chdir/--output/--error`。使用外部依赖目录时，容器还须挂载这些目录；
toolkit `cluster-run` 的 EOS 配置挂载整个 `/lustre`，适用于本次隔离验证。

Run the sync command from the local package checkout:

```bash
scripts/eos/sync_to_eos.sh
ssh login-eos
cd /lustre/fsw/coreai_devtech_all/hongbinl/mor_mlite
```

The sync is non-destructive: it does not use `--delete`, and it preserves
remote `.deps`, `.cache`, and prior `artifacts/eos` results.  Submission is an
explicit separate action:

```bash
sbatch slurm/version_probe.sbatch
scripts/eos/submit_matrix.sh
scripts/eos/submit_qwen30b.sh
```

Neither sync nor setup submits a job.  The submission wrappers are the only
files that invoke `sbatch`.

For an allocation that is already running inside the pinned container, the
steps can also be called individually:

```bash
scripts/eos/setup_env.sh
scripts/eos/version_probe.sh 8
scripts/eos/run_tiny_matrix.sh
scripts/eos/run_qwen_smoke.sh
```

`setup_env.sh` is idempotent.  It refuses to switch a dependency checkout when
tracked or staged changes exist.  Magi compilation uses MLite's pinned official
`tests/setup_magi_attention_env.sh`; its version/architecture marker avoids a
rebuild, and its import test rejects stale binary artifacts.  The optional Magi
NVSHMEM group-collective extension is disabled because v1 uses standard
torch.distributed All-to-All; the required `magi_attn_ext` and sm90 attention
kernels remain built and checked.

## Fail-closed startup manifest

Every test runner invokes `python -m mor_mlite.env_check` before model
construction.  The full manifest is printed and written to
`artifacts/eos/<job-id>/versions.json`.  A job stops if any of these checks
fails:

- Torch is not exactly `2.10.0+cu129`, CUDA is not 12.9, or NCCL cannot be queried;
- CUDA Python/bindings, Triton, NVRx, or Transformer Engine differs from its
  exact pin or resolves outside `MOR_VENV`;
- a required visible GPU is not H100/sm90;
- Transformer Engine or `megatron.lite` cannot be imported, or TE does not
  export MLite's required `moe_permute_and_pad_with_probs` API;
- the Megatron source SHA differs from the fixed commit;
- MagiAttention is not 1.1.1 or its sm90 CUDA extension cannot be imported;
- the job uses another EOS account/partition, more than one node, more than
  eight ranks, or the project resolves outside the fixed work directory.

The manifest contains package versions, paths, GPU names/capabilities, and
Slurm identifiers.  It never dumps the process environment or authentication
tokens.

For a local CPU-only diagnostic (not an acceptance result), use:

```bash
PYTHONPATH=src python -m mor_mlite.env_check \
  --allow-non-eos --no-cuda --no-magi --megatron-root /path/to/Megatron-LM
```

## Tiny matrix

`run_tiny_matrix.sh` first runs the pinned MLite two-rank Magi operator test,
covering standard-All-to-All dispatch, BF16 attention forward/backward,
undispatch, and dQ/dK/dV parity. It then checks a single-card FP32 PyTorch
learned/replay semantic oracle and writes a single-rank BF16 MLite numerical
baseline. Every required 2/4/8-rank
topology runs both
its learned router (including near-tie classification) and exact RoutePlan
replay.  The reports compare routes, hidden states, losses, gradients,
optimizer updates, checkpoint resume state, and communication assertions.  The
CP=2 replay additionally compares `magi_canonical` with `magi_direct`.

To rerun only a subset while debugging:

```bash
MOR_TOPOLOGIES=cp,all scripts/eos/run_tiny_matrix.sh
```

This is a diagnostic shortcut; the completion result requires the unfiltered
matrix. A filtered run prints `Filtered tiny topology diagnostic passed`; only
an invocation with `MOR_TOPOLOGIES` unset prints `Complete tiny topology matrix
passed` and writes a hash-bound `reports/matrix_complete.json` after re-reading
all 20 required reports and verifying that each has `passed: true`. The extra
report exercises the process-isolated full-state save/resume protocol used by
the 30B smoke.

## Qwen3-30B-A3B smoke

`run_qwen_smoke.sh` uses `Qwen/Qwen3-30B-A3B-Base` by default. It folds the same
HF snapshot through the public converter into two model-only DCPs: EP=1 for the
single-card forward baseline and EP=4 for the eight-rank
`TP=2, CP=2, dense-DP=2, EP=4` run. This is required because pinned MLite names
grouped-expert checkpoint parameters by EP-local index; DP/TP/CP resharding is
supported, but changing EP degree is rejected before tensor loading. The two
imports use the same deterministic FP32 fold and router seed, and replay parity
checks their effective weights through logits/intermediate states. In this
comparison the one-card baseline runs the two whole-sequence dense-DP shards
serially, then canonical-merges them by global token ID. It therefore retains
one set of weights and the same global batch while matching the distributed
run's per-replica token-row shape; the artifact records this execution mode.
The forward-only replay then fixes the complete MoR depth RoutePlan (including
its selected gate value) and each native Qwen MoE expert set plus its consumed
score by `(logical layer, global token ID)`. Both gate and expert score use a
straight-through bridge—baseline value in forward, live-router Jacobian in
backward—and the unmodified live values are recorded as diagnostics. This
separates continuous TP/CP/EP numerical error from BF16
Top-K branch changes and repeated score-weight amplification at exact or
near-zero expert cutoffs. Training does not use expert-routing replay. The
distributed run then performs one forward/backward/optimizer step and a full
same-topology checkpoint continuation check. Save and resume use separate eight-rank
`torchrun` processes so all CUDA/NCCL state and Lite/MoR process groups from the
training process are gone before restore. A strict receipt binds the checkpoint
to full distributed parameter, optimizer, and RNG fingerprints; the resume
process must reproduce those hashes before running its next optimizer step.
The save process independently executes the same next step and records its
post-step model, optimizer, and RNG fingerprints; the final certificate also
requires the fresh resume process to reproduce all three exactly. The receipt
also binds every rank's RNG sidecar file hash and the data/routing contract
(seed, sequence lengths, microbatch count, replay RoutePlan digest, and CP
backend) before MLite allocates optimizer state.

Forward parity and checkpoint continuation are independent acceptance gates.
A forward-tolerance failure is recorded but does not stop the two
process-isolated training runs; the script gathers checkpoint evidence and
then exits non-zero if either gate failed.

An already downloaded checkpoint or mirror can be selected without placing a
credential in a command line:

```bash
MOR_QWEN_HF_PATH=/lustre/path/to/Qwen3-30B-A3B-Base \
  scripts/eos/submit_qwen30b.sh
```

Environment variables are passed to Slurm according to the site's normal
export policy.  The scripts never print a Hugging Face or NGC token.

## Result directories

By default results are isolated by Slurm job ID:

```text
artifacts/eos/<job-id>/
  versions.json
  tiny/
    reference_fp32/
    reference_fp32_replay/
    mlite_baseline/
    *_replay/
    reports/
  qwen30b/
    folded_init_ep1/
    folded_init_ep4/
    baseline_forward/
    all_forward/
    all_train_save/
    all_resume/
    reports/
```

Set `MOR_ARTIFACT_ROOT` only when a stable alternate result directory is
needed.  Existing dependency caches and results are never recursively removed
by these launchers.
