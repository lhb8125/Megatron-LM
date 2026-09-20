# Validation status

2026-09-10 的遗留问题修复、源码绑定的新矩阵和 30B 精度隔离实验见
[repair-results.zh.md](repair-results.zh.md)。下文保留历史作业记录；尤其旧 near-tie
统计漏计了选择未改变的样本，不能再将其解释为“没有 cutoff 歧义”。

当前隔离快照 `c49e48c4d882aa7e8f0a3b3ea98d42711ba260f42cc0bbd21d8164618b870c87`：
pytest 374 passed / 1 skipped，tiny `6006390` 完整矩阵 20/20 通过；30B `6006440`
跨进程完整状态恢复及下一步指纹通过。30B `6006397` 的完整 TP2/CP2/DP2/EP4 forward
仍失败，logits relative-L2 `0.0441226644`（门槛 `0.02`），cosine `0.9990262189`。
纯 DP 和 DP+EP 的 30B forward 逐位一致，但不能替代上述失败 topology。
strict CP1 当前候选使用本地 BF16 FFA adapter；QKV、norm、RoPE、projection 和
CP>1 Magi runtime 保留原生实现。用户拒绝的 FP32-heavy 路径没有正式化，MXFP8 未实现。

最新 48 层 diff 来源诊断见 [diff-source.zh.md](diff-source.zh.md)。`6006544` 原轨迹
逐 tensor hash 复现，`6006559` 完成同输入算子隔离；两者成功只表示证据采集完成，
不改变上面的 30B 验收失败状态。

This document records observed results, not only the intended acceptance
contract. The current implementation passes the complete tiny-model topology
matrix. The Qwen3-30B smoke exercises the requested real-model path, but it
does **not** yet satisfy every release gate: its eight-GPU BF16 logits exceed
the 2% relative-L2 limit. Strict fresh-process certification of the full
parameter, ZeRO-1 optimizer, and RNG state now passes. The package must still
not be described as having completed the full 30B acceptance target.

## What the baselines mean

The FP32 `reference` model and the native Qwen/MLite tiny model are deliberately
separate implementations. The FP32 path verifies MoR routing, recurrence,
backpropagation, and replay semantics against itself. Distributed MLite BF16
runs are compared with the same single-card MLite BF16 initialization and
global batch, not with the FP32 reference tensors.

`python -m mor_mlite.train` is the deterministic synthetic acceptance trainer.
It covers variable-length packed sequences, padding, deliberately imbalanced
routing, and gradient accumulation. It is not a general dataset or production
data-loader frontend.

## Local CPU validation

The local FP32 learned-route run and its replay compare successfully, including
18/18 bitwise-exact RoutePlans:

- [`final_source_reference_compare.json`](../artifacts/local_validation/final_source_reference_compare.json)
- [`final_source_reference_train`](../artifacts/local_validation/final_source_reference_train)
- [`final_source_reference_replay`](../artifacts/local_validation/final_source_reference_replay)
- [`pytest.xml`](../artifacts/local_validation/pytest.xml): 258 passed, 26 skipped

The local suite includes a two-rank Gloo end-to-end `static_reference` CP
diagnostic. It checks start/recurrent/end attention, nested global routing,
parked exits, losses, and CP-averaged parameter gradients against the serial
FP32 model, including a final round in which one rank owns zero active tokens.

Local CPU tests do not validate CUDA, NCCL, Transformer Engine, MLite,
MagiAttention, or H100 kernels. Those results are listed separately below.

## EOS tiny full-topology acceptance

Slurm job `5997741` completed the unfiltered 1/2/4/8-GPU matrix. Its completion
receipt re-read and SHA-256 checked all 20 required reports for these eight
distributed topologies: ZeRO-1, TP, CP, EP, TP+CP+EP, TP+DP+EP, CP+DP+EP, and
DP+CP+TP+EP. Learned and replay routing passed, as did canonical-versus-direct
CP and the process-isolated checkpoint continuation check.

Primary evidence:

- [`matrix_complete.json`](../artifacts/eos/5997741/tiny/reports/matrix_complete.json)
- [`all.json`](../artifacts/eos/5997741/tiny/reports/all.json)
- [`pytest.xml`](../artifacts/eos/5997741/tiny/reports/pytest.xml): 270 passed,
  1 skipped

Worst observed hard-gated BF16 metrics across the matrix were:

| Quantity | Relative L2 | Cosine / absolute error | Limit |
|---|---:|---:|---:|
| Forward | 0.00504269 | cosine 0.99998729 | L2 <= 0.02, cosine >= 0.999 |
| LM loss | n/a | absolute 0.00036335 | absolute <= 0.01 |
| Reconstructed gradient | 0.00298666 | cosine 0.99999556 | L2 <= 0.03, cosine >= 0.999 |
| FP32 master update | 0.01242692 | cosine 0.99992279 | L2 <= 0.03, cosine >= 0.999 |
| Resident post-step weight | 0.00164567 | cosine 0.99999865 | L2 <= 0.03, cosine >= 0.999 |

All depth routes were exact. The historical zero-near-tie statistic was incomplete
and is superseded by the corrected reports above. The native Qwen MoE probe
matched all 48 logical expert contexts and all 96 expert identity tensors.
Communication and model-structure assertions also passed.

Slurm job `5997892` separately reran tiny ZeRO-1 after optimizer-fingerprint
canonicalization and passed a fresh-process full-state restore:

- [`checkpoint_external_resume.json`](../artifacts/eos/5997892/tiny/reports/checkpoint_external_resume.json)
- [`mor_parity_checkpoint.json`](../artifacts/eos/5997892/tiny/external_checkpoint_save/runtime-checkpoint/mor_parity_checkpoint.json)
- save-point parameter / optimizer / RNG fingerprints:
  `d9e8f996...6f224` / `80c77cd7...325a` / `af180db1...fcee`
- resumed-next-step parameter / optimizer fingerprints:
  `69791399...4f0b` / `d27d3061...d94e`; RNG was exact

The gradient-sync assertion observes MCore
`BucketGroup.start_grad_sync(force_all_reduce=False)` dispatches. It proves one
distributed-optimizer sync dispatch per physical bucket per global step; it is
not raw NCCL kernel profiling.

## EOS Qwen3-30B-A3B smoke

Slurm job `5998357` used `Qwen/Qwen3-30B-A3B-Base`, structure
`3 + 14 x 3 + 3`, sequence length 128, and the requested eight-GPU topology
`TP=2, CP=2, dense-DP=2, EP=4`. It completed the real HF fold, one-card MLite
baseline, eight-rank forward, one forward/backward/optimizer step, and
fresh-process checkpoint continuation. The job exited nonzero solely because
the following forward acceptance checks did not pass.

Forward evidence:

- [`all_vs_baseline.json`](../artifacts/eos/5998357/qwen30b/reports/all_vs_baseline.json),
  SHA-256 `d28900d902b6cd15ea07acda8369682363f537cf4b7a9c828f87da336d121e1a`
- 15 of 28 forward hard gates failed
- recurrent rounds 0/1/2 relative L2:
  `0.00667154 / 0.00667229 / 0.00878502`
- end layers 0/1/2 relative L2:
  `0.01276659 / 0.03311233 / 0.02238232`
- normalized hidden before the head: relative L2 `0.05566594`
- logits: relative L2 `0.05673339`, cosine `0.99840639`

The three depth RoutePlans were exact with zero near ties. All 48 native expert
contexts and 96 expert identity tensors were exact, and communication/model
structure assertions passed. The failure is therefore a continuous BF16
numerical-parity failure, not an accepted routing mismatch.

Topology-isolation reports from job `5997299` further show:

- [`zero1.json`](../artifacts/eos/5997299/qwen_isolation/reports/zero1.json):
  bitwise-exact logits
- [`cp.json`](../artifacts/eos/5997299/qwen_isolation/reports/cp.json): post-merge
  relative L2 `0.00112808`, logits `0.05816752`
- [`tp.json`](../artifacts/eos/5997299/qwen_isolation/reports/tp.json): post-merge
  relative L2 `0.00867627`, logits `0.06831570`

已有证据显示差异在后续 Qwen blocks 和 final normalization 中放大，TP-only 与 CP-only
路径也各自产生差异。但 replay 一致不能单独证明误差只来自正常 BF16 roundoff，也不能据此
彻底排除实现偏差。需要补充相同 hidden/权重的逐算子对照；任何高精度替代路径都必须单独
验证训练、显存和性能，不能用它的结果冒充原始 BF16 路径验收。

Checkpoint evidence from the same final-source job passed independently:

- [`checkpoint_external_resume.json`](../artifacts/eos/5998357/qwen30b/reports/checkpoint_external_resume.json),
  SHA-256 `2cde26097cdafc4f8bd9c1c7402d41fe9cd1b0c0caed81d7319bf79ec8966162`
- [`mor_parity_checkpoint.json`](../artifacts/eos/5998357/qwen30b/all_train_save/runtime-checkpoint/mor_parity_checkpoint.json),
  SHA-256 `0c9dfc3d70d51bdee0a3eb419fdfa38b62bd050959c1a043fec595cbff984a06`
- [`all_resume/manifest.json`](../artifacts/eos/5998357/qwen30b/all_resume/manifest.json),
  SHA-256 `f95412798efb0b35767faf0c47ced16cdb25b6f7b8822823959c990bd31eb3be`
- [`versions.json`](../artifacts/eos/5998357/versions.json)
- save-point parameter / optimizer / RNG fingerprints:
  `8a33fc72...606f` / `1f4f8aa8...4535` / `1a4d159b...933a`
- uninterrupted/resumed-next-step parameter / optimizer fingerprints:
  `fe991e56...51ef` / `c9cc8a39...dcc8`; RNG was exact and both paths
  reported gradient norm `129.93274251752328`

The first diagnostic exposed a fingerprint false positive: after restore,
MCore can materialize an additional empty optimizer parameter group. The old
metadata count included that empty group even though the owned FP32 master
weights, Adam moments, optimizer steps, group options, and all tensor bytes
were unchanged. The production fix counts a step exactly once per actual owned
parameter record while continuing to hash every state tensor and resolved step
value.

Slurm job `5998296` first isolated and confirmed the fix by rerunning the
eight-rank save and resume in separate processes through the normal production
fingerprint path; it completed with exit code 0:

- [`manifest.json`](../artifacts/eos/5998296/qwen_optimizer_diagnostic/resume/manifest.json),
  SHA-256 `45bcb2472d045e225a3e4d776d43f6d68c5ca890ddbc5bdbf3f6b54ac62d37be`
- [`mor_parity_checkpoint.json`](../artifacts/eos/5998296/qwen_optimizer_diagnostic/save/runtime-checkpoint/mor_parity_checkpoint.json),
  SHA-256 `491d89e2791c76121faaa9c220f9aeab18435011c0439f8639544552e84034fd`
- save/restored parameter, optimizer, and RNG fingerprints were respectively
  `8a33fc72...606f`, `1f4f8aa8...4535`, and `3d251190...61ba4`
- uninterrupted/resumed-next-step parameter and optimizer fingerprints were
  `fe991e56...51ef` and `c9cc8a39...dcc8`; RNG was exact and both paths
  reported gradient norm `129.93274251752328`

Both `5998296` and the final standard job `5998357` have checkpoint certificate
`status=passed`, `optimizer_loaded=true`, and `rng_loaded=true`. Neither receipt
contains a component-diagnostic manifest, so the release path uses the normal
full-state production hash.

## Release conclusion

The implementation and tiny distributed acceptance target are delivered. The
complete user-requested acceptance target remains open on one explicit Qwen
30B gate: BF16 forward parity at the 2% threshold. It remains reported as a
failure even though RoutePlan, expert identity, communication structure, and
fresh-process parameter/optimizer/RNG continuation all behave as designed.
