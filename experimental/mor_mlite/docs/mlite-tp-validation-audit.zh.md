# MLite 上游如何验证 TP 精度：固定版本源码审查

后续对照：[Megatron CI 的 TP 精度验证审查](megatron-ci-tp-validation-audit.zh.md)。
其中区分跨 TP attention/梯度检查、固定拓扑 determinism 和历史 loss golden，
并核实内部 mr 与 GitHub L0/L1 的实际选取范围。

## 范围和结论

日期：2026-09-10。按用户要求检查实际运行依赖，而非另一个本地 Megatron checkout。
上游固定 commit 为 `5c8315f12a64a7279eec58896af9e74ee3351b74`；EOS 依赖 checkout
的 `experimental/lite` 无工作区改动。本地另一个 `vendor/Megatron-LM` 为
`3bcc70b6624825aff051f02a945fb3cf45ec6438`，未将其测试内容混入此次结论。

本轮是**源码审查**：未启动 GPU 作业、未执行上游测试、未查询历史 CI 运行结果，
因此“存在某项测试”不等于“本轮已证明该测试通过”。没有修改上游代码或容差。
复制的 142 个上游测试、文档及 MLite 规范文件均与固定 commit 的 Git blob hash
逐项一致，见 [audit_manifest.json](../runtime/diagnostics/mlite_tp_test_audit/5c8315f/audit_manifest.json)。

结论：MLite 有真实 BF16 的跨 TP logits/loss 测试，但直接数值对照主要是
**2 层、hidden=16、4 个 token 的 Qwen3.5 TP2↔TP4 小模型**。其他 TP 相关测试
多验证分片语义、进程组、保存恢复和固定拓扑续训。此次检查没有找到对应我们
**预训练 Qwen3-30B/MoR、hidden=2048、48 次逻辑层调用、CP1、TP1↔TP2** 的
完整 logits/逐参数 gradient/update 数值回归。不能拿这些测试为本次 4.56%
logits relative-L2 背书，也不能仅凭覆盖不足就宣判上游 TP 实现错误。

## 1. 直接跨 TP 模型精度测试

源码：[test_qwen_lite_forward_smoke.py](../runtime/diagnostics/mlite_tp_test_audit/5c8315f/experimental/lite/tests/smoke/model/test_qwen_lite_forward_smoke.py)，
函数 `test_qwen35_tp2_tp4_mixed_attention_parity_and_backward`，第 173 行起。

配置和数据：

- Qwen3.5，2 层：`linear_attention` + `full_attention`；hidden=16，4 个 Q heads、
  2 个 KV heads、head_dim=4，vocab=128，2 个 experts、top1，MoE intermediate=8。
- 真正的 CUDA/Transformer Engine，BF16 模型；4 GPU，CP1/PP1/EP1/ETP1，
  `deterministic=True`、`fp8=False`、`use_deepep=False`。
- input IDs `[[1,2,3,4]]`，labels `[[2,3,4,5]]`。
- seed=1234 建立 TP2 模型，保存 HF 格式权重，再记录 logits/loss。
  seed=4321 建立 TP4 模型后加载刚才同一份权重，比较 logits/loss。
  因此不是依赖“相同 seed 自动得到相同跨 TP 初始化”，也不是加载完整预训练 30B。
  reference 是 **同一 MLite 的 TP2 实现**；HF 在此只承担权重保存格式，不是独立
  HuggingFace 模型 forward 或 MCore reference。

断言（第 257–268 行）：

```python
torch.testing.assert_close(tp4_logits, tp2_logits, atol=1e-2, rtol=1e-2)
torch.testing.assert_close(tp4_loss, tp2_loss, atol=1e-3, rtol=1e-3)
```

之后仅对 TP4 做 backward；检查首层 GDN 的 `dt_bias` 和 `A_log` 梯度存在，
并在 TP4 ranks 间 `atol=rtol=0`。这验证 replicated state 的梯度同步，
**没有比较 TP2/TP4 全部参数梯度，也没有 optimizer update 对照**。
TP2 reference forward 在 `torch.no_grad()` 下执行，随后删除模型。

### 这里的 1e-2 不等于全局 L2 的 1%

对 logits 的每个元素，测试要求：

```text
abs(tp4[i] - tp2[i]) <= 0.01 + 0.01 * abs(tp2[i])
```

它不是 `||delta||₂ / ||reference||₂ <= 1%`；同时带绝对与相对项，不能不看
数据分布就和我们的 2% global-L2 / 0.999 cosine 门槛比较松紧，也不能仅由
4.56% global-L2 判定这个逐元素断言会失败多少。`.float()` 是把已有 BF16 结果
转换成 FP32 做比较，不是模型 forward 使用了 FP32。

## 2. 其他 TP 相关测试实际覆盖什么

| 测试 | 实际断言 | 不应推导出的保证 |
| --- | --- | --- |
| Qwen3-MoE tiny forward/backward | TP1、1 层、hidden=16；loss/已有梯度有限，输出 shape 正确 | 没有 TP1↔TP2 数值对照 |
| Qwen3-MoE checkpoint 续训 | TP2/PP2/EP2/CP1，2 层、hidden=16，保存恢复 RNG 后，续训参数与未中断路径逐位一致 | 固定 TP2 的可复现性不等于不同 TP 的一致性 |
| 多模型 save/load/export | 小模型训练 1 步；同拓扑加载参数逐位一致；HF export 检查 dtype、有限性、key 和层覆盖 | 并未将各拓扑的训练结果互相比较 |
| Optimizer checkpoint TP1→TP2 reshard | 人造 `TinyTopologyAwareState` 及 seeded optimizer state，重新分片再保存，比较 checkpoint tensor | 没有真实 Transformer 的跨 TP forward/backward |
| Parallel topology smoke | 验证 TP/CP/EP/PP group size、rank 与相邻关系 | 分组正确不等于 BF16 GEMM+归约数值误差有界 |
| TP/GQA/linear CPU unit tests | 人工小张量验证切片、KV replication、shape、gradient reduction 语义；部分 TE/collective 被 stub/mock | 不检验真实 NCCL + BF16 row projection 的舍入误差 |
| Uneven pipeline smoke | 9 层 Qwen proxy、TP2/PP2/EP2 等，训练一步 loss 有限、导出层完整 | 不是跨 TP logits/梯度精度比较 |

关键源码：

- [Qwen3-MoE 固定拓扑续训](../runtime/diagnostics/mlite_tp_test_audit/5c8315f/experimental/lite/tests/smoke/workflows/checkpoint/test_qwen3_moe_distopt_checkpoint_smoke.py)：
  topology 第 67 行、tiny config 第 71 行、测试第 174 行；8 个 packed tokens，
  两个长度 4 的 sequence，比较一步保存后下一步续训的参数。
- [Save/load/export](../runtime/diagnostics/mlite_tp_test_audit/5c8315f/experimental/lite/tests/smoke/workflows/checkpoint/test_save_load_export_smoke.py)：
  `_topology` 第 332 行、参数比较第 458 行、测试第 572 行；正文函数为准，
  文件头中部分拓扑说明较旧，例如当前 FSDP2 分支实际使用 TP1/PP2。
- [跨拓扑 checkpoint reshard](../runtime/diagnostics/mlite_tp_test_audit/5c8315f/experimental/lite/tests/smoke/workflows/checkpoint/test_distopt_checkpoint_smoke.py)：
  `_build_sharded_model_and_dist_opt` 第 123 行、测试第 281 行。
- [拓扑 smoke](../runtime/diagnostics/mlite_tp_test_audit/5c8315f/experimental/lite/tests/smoke/primitive/parallel/test_parallel_topologies_smoke.py)：
  TP 分组测试第 73 行；文件内第 199 行的 `atol=rtol=2e-2` 是 **TP1/CP2 的 GDN**
  数值对照，不是 TP 精度容差。
- [GQA replication unit](../runtime/diagnostics/mlite_tp_test_audit/5c8315f/experimental/lite/tests/unit/primitive/modules/test_attention_moe_unit.py)：
  KV replication 与 backward 第 97、160 行附近，后者使用 fake all-gather/reduce-scatter。

## 3. “真实 HF”“Magi E2E”也不能代替本次 TP 对照

- [Qwen3.5 真实 HF 检查](../runtime/diagnostics/mlite_tp_test_audit/5c8315f/experimental/lite/tests/smoke/model/qwen3_5/lite/test_qwen35_hf_numeric_roundtrip_smoke.py)
  是 optional，需要 `QWEN35_HF_DIR`；**TP1**，最多加载前 8 层，比较 HF 导入再导出
  的参数，容差 `atol=rtol=2e-2`。没有做 logits forward，更不是完整模型跨 TP 比较。
  文件本身明确说明，互为逆变换但共同错误的 load/export 仍可能逃过这种检查。
- [MagiAttention E2E](../runtime/diagnostics/mlite_tp_test_audit/5c8315f/experimental/lite/tests/smoke/model/test_magi_attention_e2e.py)
  是 optional，固定 **TP1、CP=world_size**，检查 forward/backward 有限性和
  undispatch；Magi→TE→Magi 后的容差检查是同 Magi backend 返回后的 loss 重现，
  不是 Magi/TE 跨 backend logits 对齐，也不是跨 TP 对照。

## 4. 规范与执行入口

按上游 `skills/mcore-testing/SKILL.md` 的测试分类审查，并将规范与已存在的测试
断言分开。MLite 自身规范要求：

- [constitution](../runtime/diagnostics/mlite_tp_test_audit/5c8315f/experimental/lite/skills/basic/constitution.md)：
  尽可能逐位一致；参考优先级 Megatron→HF→Torch→第一性原理；交付前端到端验证。
- [review-threshold](../runtime/diagnostics/mlite_tp_test_audit/5c8315f/experimental/lite/skills/basic/review-threshold.md)：
  无法 bitwise 时默认 relative 0.01，但明确要求人工审核，警告容差可能掩盖真实 bug。
  **这是流程规范，不是所有测试已经采用统一 1% 指标或大模型已有 1% 保证。**
- [align-precision](../runtime/diagnostics/mlite_tp_test_audit/5c8315f/experimental/lite/skills/basic/align-precision.md)：
  固定变量、重复检查确定性、逐层/模块/primitive 定位。
- [align-e2e-precision](../runtime/diagnostics/mlite_tp_test_audit/5c8315f/experimental/lite/skills/basic/align-e2e-precision.md)：
  数据/checkpoint/seed/schedule 一致的训练比较 loss curve、grad norm、checkpoint delta；
  高成本运行需明确批准。这里没有将规范的伪代码当作已经完成的实验。

实际执行入口为 [tests/run_tests.sh](../runtime/diagnostics/mlite_tp_test_audit/5c8315f/experimental/lite/tests/run_tests.sh)，
递归发现 `unit/`、`smoke/`，按 GPU marker 分隔执行。标准无参数 Hopper profile
为 8×SM90；显式选择上述跨 TP 测试需要 4 张符合架构的 GPU。mandatory 测试意外
skip 会失败，optional 默认不进入无参数工作流。harness 设置
`CUBLAS_WORKSPACE_CONFIG=:4096:8`、`NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`、
`NCCL_NVLS_ENABLE=0`，不代表遍历了不同归约算法。

## 对当前 30B 调查的意义

我们观察的是完整 checkpoint 和 MoR 多次调用下的误差传播，而上游直接跨 TP
数值对照主要覆盖极小模型、极短输入及少量梯度同步状态。差别不仅是测试容差，
更包括模型类型、权重分布、hidden 尺寸、执行深度、输入长度及比较对象。

当前缺少的是对应实际配置的跨 TP 数值证据，不能通过引用上游 smoke 通过、
固定拓扑 bitwise 续训或规范的默认容差来替代。后续若补验证，应维持原验收门槛，
分别报告 logits 逐元素误差/global-L2、逐参数 gradients/updates 和训练指标；
本轮没有擅自启动这些额外实验。
