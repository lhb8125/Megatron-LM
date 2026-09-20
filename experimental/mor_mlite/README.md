# MoR-MLite

`mor_mlite` is an out-of-tree Mixture-of-Recursions model package for
Megatron-LM's experimental MLite runtime.  It registers `qwen3_moe_mor`
through MLite's public model registry and never patches either MLite core or
the reference MoR repository.

The implementation has two deliberately separate execution paths:

- `reference`: a small deterministic PyTorch model used as the FP32 semantic
  oracle and replay self-consistency harness for routing, recurrence,
  gradients, and optimizer updates.
- `mlite`: Qwen3-MoE layers, tensor/expert parallel primitives, Megatron
  distributed optimizer (ZeRO-1-like), and MagiAttention-backed context
  parallelism.

Pinned production dependencies:

- Megatron-LM `5c8315f12a64a7279eec58896af9e74ee3351b74`
- MagiAttention `v1.1.1`
- PyTorch `2.10.0+cu129`, BF16, H100

框架方法、架构、关键实现及主要验证结果见
[MoR-MLite 完整技术报告](docs/technical-report.zh.md)，包含 tiny 数值验收、大模型状态恢复和真实 Pile 100 步训练矩阵。

当前修复分支尚未完成 30B 验收：tiny 完整训练矩阵 20/20 通过，30B 8 卡 full-state
checkpoint 续训通过，但单卡/多卡 forward relative-L2 仍为 4.41%，高于 2% 门槛。
strict CP1 的 BF16 local FFA 适配和其他已修复项、精确源码 hash、作业证据见
[修复记录](docs/repair-results.zh.md)。未将 FP32-heavy 诊断移植为正式路径，未启用 MXFP8。

## Architecture

For architecture `(N_start, N_recur, k, N_end)`, the recurrent layers are
registered once and called `k` times.  Every recursion owns an independent
depth router.  Routing is per sample, active sets are nested, and the update is

```python
h = h_before + selected_gate * recurrent_block(h_before)
```

The default linear capacity schedule is `100% / 67% / 33%` for three rounds.
Early-exit tokens are parked and do not enter later attention or MoE traffic.
Each router forms FP32 decision logits as `linear(hidden) / temperature`; those
same pre-sigmoid logits drive both per-sample expert-choice Top-K and the BCE
auxiliary target (`1` for selected active tokens, `0` for rejected active
tokens). The BCE sum is normalized by the global active-token count before its
configured coefficient is applied.

## Quick start

```bash
python -m pip install -e '.[dev]'
python -m mor_mlite.train --preset tiny --backend reference --precision fp32 --steps 2
python -m mor_mlite.parity run --preset tiny --output artifacts/baseline
python -m mor_mlite.parity compare artifacts/baseline artifacts/candidate
```

Both CLIs load architecture, depth-router, tiny-model, and default parallel
settings from `configs/tiny.json` or `configs/qwen3_30b.json`; the files are
also bundled in wheels. `--preset-config PATH` selects a compatible JSON file.
Architecture and router experiments can be stated explicitly with
`--n-start-layers`, `--n-recurrent-layers`, `--num-recursions`,
`--n-end-layers`, `--capacity-schedule`, `--router-temperature`,
`--router-alpha`, and `--router-aux-loss-coef`.

The first-release `train` command is the deterministic acceptance trainer: it
consumes the built-in variable-length synthetic `PackedBatch` stream and emits
the complete parity artifact (routes, hidden states, gradients, updates, and
checkpoint evidence). It is not yet a general dataset/data-loader frontend.
It does accept a converted/self-describing MoR distributed checkpoint through
`--init-checkpoint`; this loads model tensors only and starts optimizer/RNG
state fresh. Arbitrary optimizer/RNG resume remains intentionally deferred
from the general training frontend. The parity CLI has a narrower validated
full-state path: `--checkpoint-save-only` writes a hash-bound checkpoint and a
later, separate `torchrun` uses `--resume-checkpoint` to restore model,
optimizer, and RNG before executing the next step. The save process also runs
that next step as an uninterrupted oracle; certification requires exact
post-step model, optimizer, and RNG fingerprints from both processes.

For MLite, put the pinned checkout on `PYTHONPATH` first:

```bash
export PYTHONPATH=/path/to/Megatron-LM/experimental/lite:$PYTHONPATH
python -m torch.distributed.run --standalone --nproc-per-node=8 \
  -m mor_mlite.train --preset qwen3-30b --backend mlite \
  --hf-path Qwen/Qwen3-30B-A3B-Base --topology all \
  --no-checkpoint-roundtrip

# Or cold-start training from `python -m mor_mlite.convert_hf` output. MLite
# DCP may reshard DP/TP/CP, but the checkpoint EP degree must match the runtime
# because Qwen grouped-expert parameter keys are EP-local.
python -m torch.distributed.run --standalone --nproc-per-node=8 \
  -m mor_mlite.train --preset qwen3-30b --backend mlite \
  --init-checkpoint /path/to/converted-dcp --topology all \
  --no-checkpoint-roundtrip
```

`--topology` names a validated entry in `configs/topologies.json`; the `all`
entry is `TP=2, CP=2, dense-DP=2, EP=4`.  Parallel degrees are deliberately
not accepted as unrelated command-line switches, so only topology combinations
covered by the first-release validation matrix can be launched accidentally.

公开的 `mor_mlite.objective.objective_scales` / `apply_objective` 负责 dense-DP
与 gradient accumulation 的 token 加权；普通 MLite runtime 可以直接使用，无需依赖
parity trainer 私有实现。它分别按 LM 有效 mask 权重和各轮 router candidate 数计算
`DP * num_microbatches * local_count / global_step_count`，抵消运行时的 microbatch
平均与优化器的 DP 平均。用法和计数边界见 [训练目标适配](docs/objective.zh.md)。

For the tiny PyTorch CP diagnostic, `static_reference` keeps selected tokens
on their existing (possibly imbalanced) CP owner. `PositionAwareGQA` then uses
the autograd-safe `gather_static_active_qkv()` oracle to assemble active Q/K/V
in canonical `(sample, original_position, token_id)` order. This path is tested
end to end through `TinyMoRModel.forward(static_cp_group=...)`: start,
recurrent, and end attention; global expert-choice depth routing; early exits;
LM/router losses; and CP-averaged parameter gradients all match the single-rank
model. It retains its original CP ownership after every shrinking boundary and
issues no hidden/gate/metadata All-to-All. The test-only path is intentionally
not exposed as a production Qwen CP backend. As with the MLite loss contract,
the returned differentiable loss assumes replicated parameter gradients are
mean-reduced over CP after backward. Run its two-rank Gloo diagnostic with
`PYTHONPATH=src pytest -q tests/test_static_cp.py`.

The static diagnostic's `communication` counters cover depth-route gathers and
active-layout transitions. They do not claim to count the Q/K/V gather
collectives themselves or the loss/gradient reductions; consequently
`collective_calls` is not a total raw-collective profiler for this test path.

EOS setup and Slurm launchers live under `scripts/eos`.  The fixed NGC 26.01
image is a bootstrap; setup creates a version-keyed venv containing the exact
cu129 Torch wheel and its matching Transformer Engine extension. `submit_matrix.sh`
submits the tiny 1/2/4/8-GPU topology matrix; `submit_qwen30b.sh` submits the
real-model smoke test.  Every job validates the dependency manifest before
touching the model. See [`docs/validation.md`](docs/validation.md) for observed
pass/fail results and artifact-bound evidence; the real-model smoke currently
has one open BF16 forward-parity acceptance failure even though the tiny matrix
and fresh-process checkpoint continuation pass.

## Correctness contract

- replay route IDs/metadata: bitwise exact
- native-MoE Top-K expert IDs and dispatch-visible selected scores: bitwise
  exact after canonicalizing by global token ID and logical layer;
  forward-only cross-topology replay fixes their forward values with a
  straight-through bridge while separately recording live router scores
- learned routes: exact outside a measured near-tie ambiguity set
- FP32 forward/loss: `rtol=2e-5, atol=2e-6`
- FP32 gradients/update: `rtol=5e-5, atol=5e-6`
- BF16 loss absolute error: at most `1e-2`
- BF16 forward relative L2: at most `2%`
- BF16 complete reconstructed gradient and FP32 master-update vectors, grouped
  by phase/global step: relative L2 at most `3%`, cosine at least `0.999`

The acceptance runner sets Adam/AdamW `eps=1e-6` explicitly on both the
single-rank oracle and MLite distributed optimizer. This keeps the first-step
update comparison numerically conditioned under BF16 without relaxing the
independent gradient or update thresholds; `--adam-eps` can override it and
the resolved value is recorded in every artifact manifest.

Per-parameter BF16 gradient/update comparisons remain in the report as
diagnostics, while malformed shape/dtype coverage and NaN/Inf are always hard
failures. Resident BF16 post-step weights are reported separately from the
actual FP32 optimizer delta.

The parity report separately records communication invariants: one active
hidden rebalance per changing round boundary, no recurrent-internal depth
dispatch, no later Q/K/V for parked exits, and one physical-bucket gradient
sync dispatch per global step. The probe observes MCore's real
`BucketGroup.start_grad_sync()` calls with `force_all_reduce=False`; it verifies
that the distributed-optimizer reduce-scatter path is dispatched once, but is
not an NCCL-profiler count of individual kernels.

Likewise, the legacy artifact field `physical_collectives` counts only MoR
routing and active-layout orchestration. It does not count collectives internal
to MagiAttention, native Qwen MoE, TP, or optimizer kernels.

Large-model checkpoint validation deliberately crosses a process boundary.
The save process exits before the resume process constructs MLite, which
releases every CUDA context and Lite/MoR process group instead of relying on a
second handle in the same interpreter. The save receipt fingerprints all
rank-local physical parameters, every ZeRO-1 FP32 master shard, Adam
`exp_avg`/`exp_avg_sq`, optimizer step/common metadata, and RNG state at the save
point, plus the uninterrupted next-step state. The fresh process must reproduce
both sets of hashes, so a reset or partially loaded optimizer cannot pass
merely because it can perform an update. Every rank's MLite RNG sidecar is
individually size/hash checked before restore, and the receipt binds strictness, seed,
sequence lengths, microbatch count, replay RoutePlan digest, and CP backend.
Rank-zero sidecar/receipt I/O failures are broadcast to all ranks instead of
leaving peers blocked at a barrier. Tiny tests additionally retain the
uninterrupted-versus-resumed tensor comparison in one run.

Here “one rebalance” means one differentiable hidden-payload migration at the
round boundary. Its variable-size dispatch transaction also exchanges counts,
gates, and token metadata; those control/side payload collectives are reported
separately and are not additional recurrent-block redispatches.

MLite acceptance artifacts also contain an opt-in native Qwen expert-route probe.
For every start layer, logical recurrent-round layer, and end layer it records
the dispatch-visible Top-K expert IDs, selected scores, and the raw-logit
K/(K+1) cutoff margin in canonical global-token order. Dummy padding is
excluded. The probe is enabled for tiny full-state acceptance and forward-only
parity (including the 30B smoke), but remains disabled for normal model
execution and production-like 30B training.

The built-in tiny HF fixture reserves two residual-stream coordinates and
constructs a deterministic four-expert gate with a cutoff-logit margin above
`0.5`. Attention output and expert down-projection rows initially preserve
that anchor. This keeps Magi-versus-unfused BF16 roundoff from turning a valid
continuous-kernel difference into an arbitrary Top-K discontinuity, while all
native Qwen Top-K, token dispatch, expert computation, EP collectives, and
router gradients remain active. Imported Qwen checkpoints are never modified
by this synthetic-only profile.

Replay freezes the complete depth `RoutePlan`: selected token IDs, placement,
padding, causal metadata, and the selected gate value keyed by global token ID.
A straight-through bridge uses the stored gate in forward and the live depth
router Jacobian in backward. For forward-only cross-topology acceptance it
additionally freezes native Qwen MoE expert IDs and their dispatch-visible
forward scores, keyed by `(logical layer, global token ID)`. The same bridge
uses the baseline expert score in forward and the live expert-router Jacobian
in backward; unmodified live depth/expert scores remain diagnostics. Dummy
padding rows retain their native expert choices and scores. Every declared
logical context and every live real token must resolve exactly or replay fails
closed. Training does not consume the native-expert routing oracle, so normal
Qwen router behavior is unchanged.

This second discrete oracle matters for folded 30B BF16 acceptance: the native
router has many exact or near ties at the Top-K boundary, so small legitimate
kernel-order differences can select a different expert set and cascade through
later recurrent/end layers. The learned path remains available and is measured
separately; expert-routing replay isolates the continuous TP/CP/EP
implementation from that discontinuity and from repeated BF16 score-weight
amplification. Expert identity and consumed scores are hard-gated. Model
outputs retain the BF16 numerical thresholds, while raw cutoff margins and
live (pre-bridge) scores remain diagnostic because they do not change replayed
execution.

Round-hidden canonicalization and recurrent QKV hooks are disabled by default
on the registered model. The parity runner opts into them to produce acceptance
evidence; ordinary MLite consumers therefore do not pay their sorting,
retention, or Python-hook overhead. Full-vocabulary logits and the native MoE
expert-route probe are likewise parity-only opt-ins.

`RoutePlan` files are strict JSON. A full-capacity round has no K+1 boundary,
so its unbounded cutoff margin is serialized as `null` and restored internally
as positive infinity; NaN and negative margins are rejected.

The FP32 reference and Qwen/MLite tiny model deliberately remain separate
implementations (the latter includes native Q/K normalization and Qwen RoPE
defaults), so they are not compared tensor-for-tensor. Every distributed
MLite topology is instead compared with the same single-card BF16 MLite
initialization and global batch; the FP32 reference is independently checked
through learned-route versus exact RoutePlan replay.

For the 30B `dense-DP=2` forward comparison, that single-card MLite baseline
executes the two whole-sequence DP partitions serially and merges them by
global token ID. The weights and global batch are unchanged, but each forward
uses the same local token count as one distributed DP replica. This prevents a
different GEMM/MoE row count in the oracle itself from being misclassified as
a TP/CP/EP numerical error. `--reference-dp-shards` is restricted to a
one-rank, learned, forward-only MLite baseline and is recorded in the artifact.

## Scope

Version 1 fixes `PP=VPP=ETP=1` and is single-node only. PP, ZeRO-2/3,
multi-node, MTP, Qwen3.5, DeepEP, FP8, LoRA, serving inference/KV cache,
`torch.compile`, and activation recomputation are outside this release. The
supported launch surfaces reject incompatible parallel/runtime knobs; features
that MLite does not expose through this package are fixed off rather than
silently enabled.

`magi_direct` is pinned more tightly than the canonical backend: MagiAttention
v1.1.1 does not expose its target token placement through a public API, so the
adapter reads that exact version's runtime-manager metadata and fails closed if
the structure changes. Any Magi upgrade therefore requires rebuilding the sm90
extension and rerunning the canary plus the complete CP acceptance matrix.
