# 真实 Pile 数据 50 步训练 loss 对比

## 实验合同

用户要求用真实数据训练 50 步，不能再用单步 synthetic forward 或离线 logits CE
替代训练轨迹。实验入口为 `runtime/real-data-50/scripts/train.py`，直接调用现有
MLite runtime 的 `forward_backward`、`optimizer_step` 和 `lr_scheduler_step`。
这是一项训练 loss 曲线实验，不签发缺少全参数逐步梯度/更新证据的完整 parity 证书。
生产模型和 dtype 路径不变。

- 模型：原 Qwen3-30B-A3B-Base folded MoR，20 个物理层、48 次逻辑调用。
- 初始化：EOS `mor_mlite/artifacts/eos/5998357/qwen30b/folded_init_ep4`，
  只加载模型，各配置从相同的全新 optimizer 状态开始。
- 组别：TP1/CP1、TP2/CP1、TP1/CP2；固定 DP4/EP4/ETP1。
- batch：每步四条等长 256-token chunk，一次 microbatch，共 50 次 optimizer update；
  每步 1,020 个有效 next-token target，共 51,000 个训练 target。
- 数据：已有 EOS Pile `datasets/pile/gpt3/my-gpt3_00_text_document` 顺序前缀；
  用同目录 GPT-2 vocab/merges 解码，严格检查编码回环，再用本地 Qwen3 vocab/merges
  编码。两文件与 Base snapshot `1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9`
  的 Git blob hash 完全相同，使用 `transformers==4.57.1` 的 Qwen2Tokenizer，
  raw text、`add_special_tokens=False`、不套聊天模板。文档之间插入 Base EOS 151643，
  连续打包为固定长度 chunk，屏蔽每条 chunk 最后的预测位置。不是直接挪用 GPT-2 ID。
- 输入证据：源 idx hash、选定源 token bytes hash、解码文本 hash、Qwen tokenizer 文件
  hash、最终 token tensor hash 和每步输入 hash。不同组别必须逐步一致。
- 优化器：MLite distributed Adam，初始 lr=1e-5、adam_eps=1e-6、clip_grad=1；
  使用同一默认 scheduler，total_training_steps=50，smoke 也保持该计划。
- 精度：native BF16（正常 optimizer FP32 states 不变）；strict=True，并显式设置
  `MAGI_ATTENTION_DETERMINISTIC_MODE=1`，不启用额外 FP32 attention/reduction 开关。
- 路由：learned，未冻结或回放 native expert/depth routing；自然分支分歧属于训练结果。
- 加载后、首次 forward 前，全部 rank（包括所有 CP 副本）对既有 TP1 原始参数基准
  做完整 raw-byte 比较及 TP/EP 名称与分片覆盖检查。
- Loss：native vocab-parallel CE 的有效 token mean，单独列出 depth-router aux loss
  和其相加 objective。后续 100 步矩阵准备时核对 pinned MLite 的
  `model/qwen3_moe/lite/model.py`，其 MoELayer 设置 `compute_aux_loss=False`；
  原文“原生 MoE 辅助梯度保持开启”不准确。50 步原始 manifest 的相应描述字段也有
  此文字错误，原 artifact 保留不改；实际 LM/depth aux/objective 数值不受影响。
  每步还记录 finite gradient norm、成功更新标志和所有 rank 的峰值显存。

固定环境：Megatron-LM `5c8315f12a64a7279eec58896af9e74ee3351b74`，Torch
2.10.0+cu129、TE 2.13.0、MagiAttention 1.1.1，H100；NGC PyTorch 26.01 container
及既有 cu129 venv。生产源码 hash 仍为
`c49e48c4d882aa7e8f0a3b3ea98d42711ba260f42cc0bbd21d8164618b870c87`。
tokenizer 依赖独立安装到本实验的 `prep-deps-<job_id>`，不会进入训练 PYTHONPATH。

## 准备阶段

- `6012186`：CPU 数据准备失败，固定训练 venv 没有 `transformers`；没有生成可用数据。
- `6012188`：独立 tokenizer 依赖安装成功，但原模型 snapshot 未缓存 tokenizer，准备失败。
- `6012191`：离线准备成功，原始 index 0–47 共 48 条记录，最终 tensor shape `[50,4,256]`。
  GPT-2 文本编码回环全部通过；选定 token tensor SHA256：
  `023722b1986701b90601de4b1f68ccf96332c515cca2ca3ed01eac3affd89216`。
  缓存 Qwen3-30B-A3B 的 vocab/merges 依据官方 Base
  [Git tree 元数据](https://huggingface.co/api/models/Qwen/Qwen3-30B-A3B-Base/tree/1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9)
  验证：vocab blob `4783fe10ac3adce15ac8f358ef5462739852c569`，merges blob
  `31349551d90c7606f325fe0f11bbb8bd5fa0d7c7`。未沿用聊天模型 EOS 151645。
- `6012198`：eos0331，三组各 2 次 optimizer update 的 GPU smoke 完成；objective 单测
  5 passed，环境和 TE canary 通过。三组加载后全部 5,266 个 canonical parameter、
  13,084,750,848 个 unique element 均与原 TP1 raw-byte 基准一致，different_bytes=0。
- `6012211`：eos0331，三组各 50 步正式任务完成，`COMPLETED / 0:0`，耗时 8 分 33 秒。
  继续使用 smoke 的原训练脚本，不改模型源码。三组每个 step 的 input hash 和 target
  数一致，150 次 optimizer update 全部成功，全部 loss 和 gradient norm 有限。
- loss 比较器的三项独立 CPU 测试通过：有符号差正确；拒绝缺步；拒绝同一步不同输入。
- 本地主机的轻量 venv 没有 torch，不能运行 objective 单测；改在实际训练环境执行。

smoke 仅用于确认训练可执行，不作为 50 步结果：

| 配置 | Step 1 LM loss | Step 2 LM loss | 最大 allocated GiB |
|---|---:|---:|---:|
| TP1/CP1 | 31.53304434 | 20.20613384 | 65.04 |
| TP2/CP1 | 31.36828947 | 20.66204071 | 43.15 |
| TP1/CP2 | 31.35017014 | 20.37223959 | 46.18 |

三组每一步均更新成功且梯度范数有限。此处起始 loss 较高是实验观测，不能声称是
未折叠原始 Qwen checkpoint 的表现，也尚未定位高起始 loss 的独立成因。

## 正式 50 步结果

下表的 loss 为第 50 步 **更新前** 当前训练 batch 的 native LM loss，不是第 50 次更新
后的固定验证集评估。Δ 定义为候选组减去 TP1/CP1；平均绝对差是 50 个逐步 Δ 的
绝对值平均，不是两个运行平均 loss 之差的绝对值。

| 配置 | 第 50 步 LM loss | 最终 Δ | 最终相对 Δ | 50 步平均绝对 Δ | 最大绝对 Δ（step） |
|---|---:|---:|---:|---:|---:|
| TP1/CP1 | 5.995144367 | 0 | 0 | 0 | 0 |
| TP2/CP1 | 6.021071434 | +0.025927067 | +0.432468% | 0.076557217 | 0.455906868（2） |
| TP1/CP2 | 5.957888484 | −0.037255883 | −0.621434% | 0.052357447 | 0.325671196（4） |

TP 的 50 步有符号平均差为 +0.024298096，RMS diff 为 0.126050312；
CP 的有符号平均差为 −0.023612549，RMS diff 为 0.081919427。符号相消使有符号平均
明显小于平均绝对差。CP 最后 loss 略低不代表它的计算更准确或长期收敛更好。

| 区间 | TP2 平均绝对 Δ | CP2 平均绝对 Δ |
|---|---:|---:|
| 1–10 | 0.211240232 | 0.133693731 |
| 11–20 | 0.030614138 | 0.018441463 |
| 21–30 | 0.049790752 | 0.019385982 |
| 31–40 | 0.053589618 | 0.033551061 |
| 41–50 | 0.037551343 | 0.056715000 |

本配置下最大差出现在早期，后续没有单调放大或 loss 发散，但存在持续、非零的轨迹
差异；不能称为 bitwise 对齐，也不能由此证明之前的 logits 数值门槛已经通过。

三组正式运行加载后的全量参数检查都覆盖 5,266 个 canonical parameter、
13,084,750,848 个 unique element，所有 rank 的 different_bytes=0；CP 的所有副本
也检查过。正式运行与独立 smoke 的前两步 LM loss、depth aux loss、gradient norm
逐项完全相同；这只是每组各两次、前两步的复现证据，没有测量重复 50 步的方差。

三组最大 allocated 显存分别为 65.0517、43.1553、46.1941 GiB；这些是当前
短序列真实训练记录，不是吞吐/显存最优配置结论。

![50 步训练 loss 与逐步差值](/home/scratch.hongbinl_sw/work/fsdp/agentic-mcore-dev/.claude/worktrees/mor-mlite-validation/vendor/mor_mlite/runtime/real-data-50/6012211/loss-curves.png)

产物：

- [完整逐步 CSV](../runtime/real-data-50/6012211/loss-comparison.csv)
- [比较汇总 JSON](../runtime/real-data-50/6012211/comparison.json)
- [可缩放曲线 SVG](../runtime/real-data-50/6012211/loss-curves.svg)
- 各组 `manifest.json`、`loss.jsonl`、`complete.json` 和 `loaded-rank-*.json`
  位于 `runtime/real-data-50/6012211/{baseline_real,tp2_real,cp2_real}/`。
- 数据与来源 manifest 位于 `runtime/real-data-50/data-6012191/`。
- 训练、准备、比较和绘图脚本位于 `runtime/real-data-50/scripts/`。

## 解释限制

50 步短轨迹只能检验此数据前缀/长度/batch/lr 下的损失差异，不证明长期收敛等价。
对比报告同时提供逐步有符号差、平均绝对差、最大绝对差和最终 loss；不能仅凭
平均差相消或最后一步接近就宣称 TP/CP 误差已修复。
