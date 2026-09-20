# Logits relative-L2 与同一输入上的 LM loss 差异

## 比较对象与定义

此前报告最终比较同一 MoR 模型、同一输入/checkpoint 来源/routing replay，
不同并行拓扑得到的 **原始 logits**，不是 loss，也不是 HF-vs-MCore 对照。
张量名 `forward/step_000/mb_000/logits`，shape 为 `[256,151936]`，
包含两条 128-token 序列每个位置对全部词表项的分数，dtype=BF16。

relative-L2 = `sqrt(sum((candidate-reference)^2)) / sqrt(sum(reference^2))`，
对所有 token 和 vocabulary 元素一起求和；
它等价于误差 RMS / 参考 logits RMS，不是逐元素相对误差的平均，
也不是 loss 的百分比变化、概率误差或准确率下降。
若给一个 token 的所有 logits 同加一个常数，softmax 和 CE 不变，
但原始 logits 的 L2 差可以不为零，因此不能直接将该指标翻译为训练质量损失。

## 原 forward 产物没有 LM loss

`parity/mlite.py:_make_local_batches` 在 forward-only 路径显式将
`labels=None`、`loss_mask=None`，以进入完整 vocabulary logits 分支。
核对原产物，未记录 `loss/*` tensor，之前的 logits 数字不能称为 loss diff。

作业 **6011940** 补算的是**保存的 BF16 inference logits 上的 next-token CE**：
没有执行原生训练/fused 或 vocab-parallel CE 分支，没有 backward/optimizer，
也未包含 MoR router auxiliary loss。

## 数据、标签与验证方法

- 完全按原 `_make_local_batches` 使用
  `make_synthetic_batch(seq_lens=(128,128), vocab_size=257, seed=1234,`
  `extreme_routing=True)` 恢复数据；实际输入为合成随机 token IDs，
  不是自然语言验证集。注意数据生成的 257 不是模型输出词表大小 151936。
- 调用原 `packed_lm_targets`，在每条序列内部 shift 一次；
  每条末位不计 loss，共 **254** 个有效 next-token 目标，包含预测 EOS 的位置。
- 从已验证的 TP1 loaded-parameter raw bytes 读取 embedding 权重并核验 SHA256，
  将恢复的 input IDs 查表，与原 GPU `00/block_input` 按 global token ID 配对，
  全部 bitwise 相同，直接确认数据重建与原运行一致。
- 四份 logits 的 shape、dtype、SHA256、seed、strict、sequence lengths 均核验。
- 离线用 FP64 计算 `logsumexp(z)-z[target]`，与
  `F.cross_entropy(..., reduction="none")` 逐 token 交叉检查；
  按有效 mask 求全局 token mean，单位为 nats/token。
- 同时补算 FP32 CE 作为数值检查。它与 FP64 的平均 loss 相差约 6.4–6.6e-6，
  远小于这里关心的 CP 平均差异。FP64 只用于离线统计，不改变模型运行精度。

## 平均 loss

各行相对同一个 TP1/CP1 参照：

| 配置 | 平均 LM CE | signed loss diff | 相对平均 loss 变化 |
| --- | ---: | ---: | ---: |
| TP1 CP1 baseline | 13.1793265528 | 0 | 0 |
| TP1 CP2 DP2 EP2 | 13.1379419097 | −0.0413846431 | −0.314012% |
| TP2 CP1 DP2 EP2 | 13.1703267908 | −0.0089997620 | −0.068287% |
| TP4 CP1 DP2 EP2 | 13.1799041894 | +0.0005776366 | +0.004383% |

TP4 对 TP2 的直接平均差为 **+0.0095773986（+0.072720%）**。
这里负号仅表示这一小批数据上 CE 较低，不能据此说 CP 更准确或训练效果更好。

## 平均值会抵消逐 token 差异

| 比较 | mean signed delta | mean absolute token delta | max absolute token delta |
| --- | ---: | ---: | ---: |
| CP2 / CP1，TP1 | −0.041385 | 0.162499 | 2.556865 |
| TP2 / TP1，CP1 | −0.009000 | 0.118088 | 0.961074 |
| TP4 / TP1，CP1 | +0.000578 | 0.115952 | 0.724492 |
| TP4 / TP2，CP1 | +0.009577 | 0.120631 | 0.946275 |

CP2 的最大逐 token loss 差在 global token ID 125。
报告还保留全部目标/掩码、四组逐 token loss 及差异统计。
`token_delta_std_population` 是该 batch 内跨 token 的差值标准差，
**不是 run-to-run variance**。

结论：CP 的 logits relative-L2=5.8545% 与平均 LM loss 变化 −0.3140% 同时成立，
二者测量对象不同。标量 loss 相近可以掩盖逐 token 分布和中间状态的差异；
但 raw logits L2 较大也不能直接等同于 loss 或训练质量损失同样大。
本次没有更改已有数值 gate，亦不据一个合成 batch 宣布训练通过/失败。
原生训练总 loss、gradient 和长期收敛仍未在本次验证。

## 版本与证据

EOS 6011940，节点 eos0259，CPU-only，复用原 GPU logits 与 pinned
Torch 2.10.0+cu129；原模型 MCore
`5c8315f12a64a7279eec58896af9e74ee3351b74`、TE 2.13.0、
MagiAttention 1.1.1、H100 80GB，PyTorch 26.01 sqsh 加独立 cu129 venv。
生产源码 hash：
`c49e48c4d882aa7e8f0a3b3ea98d42711ba260f42cc0bbd21d8164618b870c87`。
原 artifacts：baseline 6006397，CP2 6006410，TP2/TP4 6011796。
按 `ckpt-tools` 核验产物、`cluster-run` 调度统计，不修改生产代码。

- [完整 loss 报告](../runtime/ckpt_tools/logits-loss/6011940/report.json)
- [分析脚本](../runtime/ckpt_tools/logits-loss/scripts/analyze.py)
- [作业日志](../runtime/ckpt_tools/logits-loss/6011940/mor-logits-loss_6011940.log)
