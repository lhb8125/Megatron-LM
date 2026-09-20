# CP=1：TP 输出投影扰动在单层、末端层和 final norm 中的传播

2026-09-10，继续 [diff 来源调查](diff-source.zh.md)，回答三个问题：同一个 layer 内
`o_proj` 首差如何变化、逻辑层 45→46 为什么跳升、47→final norm 为什么再次升高。
层号均从 0 开始；所有百分数均为 relative-L2，而非元素相对误差或准确率差异。

## 配置与控制

- Qwen3-30B-A3B-Base，MoR 20 个物理层、48 次逻辑调用；hidden=2048，
  两条 128-token 序列，1 step / 1 microbatch，native BF16 forward-only。
- 只对比 TP1 baseline 与 TP2/CP1/DP2/EP2；TP2 开启 SP。depth routing、expert IDs
  和实际 dispatch 使用的 expert scores 均 replay，不存在换专家导致的差异。
- 依赖和 checkpoint 沿用原调查：Torch 2.10.0+cu129、TE 2.13.0、Magi 1.1.1、
  MCore `5c8315f12a64a7279eec58896af9e74ee3351b74`；H100 80GB，NGC PyTorch
  26.01 容器 `pytorch_26.01-py3_4a7dd6b5c237.sqsh`，独立 venv。
  HF snapshot `1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9`，真实 EP2 checkpoint `6006410`。
- 本包 60 个 Python 源文件合并 SHA256 保持
  `c49e48c4d882aa7e8f0a3b3ea98d42711ba260f42cc0bbd21d8164618b870c87`。
  本轮只增加诊断代码和文档，生产路径、dtype、阈值未变。
- 自然轨迹取自 `6006544`，与 `6006397` / `6006410` 主 artifact 已逐 tensor hash
  复现。新任务 **6006744**（`eos0514`，2:06，COMPLETED / exit 0）仅在每个 block
  开始注入 baseline hidden，随后允许该层的
  原生 TP 投影误差经过完整层内路径；与 `6006559` 每个算子独立重置输入的实验不同。
  48/48 层的 block input、QKV、core output 均逐位相同，首差仍在 `o_proj`。
- final norm 和 LM head 分别注入其自己的 baseline 输入，各自输出 tensor hash
  都与 TP1 baseline **完全相同**；这不是一次自然端到端对齐，也不取得验收证书。

## 1. 只观察本层投影误差的自然传播

下表每层输入均 exact，层内没有再次重置 hidden：

| 边界 | 第 0 层 | 第 45 层 | 第 46 层 | 第 47 层 |
| --- | ---: | ---: | ---: | ---: |
| Core output | 0 | 0 | 0 | 0 |
| `o_proj` 输出 | 0.2633% | 0.2960% | 0.2877% | 0.1657% |
| Attention residual 后 | 0.2332% | 0.0597% | 0.0688% | 0.2027% |
| MLP norm 后 | 0.3027% | 0.3286% | 0.2436% | 0.3154% |
| MoE 分支输出 | 0.4519% | 0.7230% | 0.2714% | 0.5140% |
| Block residual 后 / 层输出 | 0.3274% | 0.1615% | 0.3341% | 0.4107% |

扰动并不逐边界单调增大。残差改变 reference 尺度并有 BF16 舍入，RMSNorm 重加权
token/通道，MoE 对不同输入响应；因此不能将表内相邻百分数的比值直接称为算子增益。
同输入下 48 层 MLP norm 与 MoE 独立对照均 exact；本表它们出现非零 diff，是因为
这里刻意保留了投影首差传播后的不同输入，而非证明 MoE 新增独立 TP 错误。

本层新误差与自然累积误差的对照：

| 层输出 | 相同层输入，仅本层投影扰动 | 自然输入，保留前层扰动 |
| --- | ---: | ---: |
| 第 45 层 | 0.1615% | 1.1344% |
| 第 46 层 | 0.3341% | 3.4605% |
| 第 47 层 | 0.4107% | 2.4134% |

若 baseline 为 `B(x)`，TP 算子为 `T(x)`，自然 TP 输入为 `x+dx`，使用精确路径分解：

```text
T(x+dx) - B(x) = [T(x) - B(x)] + [T(x+dx) - T(x)]
                    本层差异            输入扰动传播
```

这是向量恒等式，两个分量可以相关或相消；不能用范数相减构造独立贡献百分比。
第 46 层的大跳变主要涉及前层扰动传播，不能解释为该层 `o_proj` 自身误差突然变大。

特别注意第一处残差的“相对下降”并非绝对误差消失。第 46 层本地投影的 delta norm
是 **0.7911**，attention residual 后反而为 **1.0255**；只是 reference norm 从
**274.94** 增至 **1490.99**。随后本地扰动的 delta norm 依次为 MLP norm **2.3576**、
MoE **2.9280**、block output **3.3338**。这一条真实传播链比只看百分数更清楚。

GPU 分析任务 **6006849** 给出层输出的路径分量：

| 层 | 总 delta norm | 本层差异 norm | 输入扰动传播 norm | 两分量 cosine |
| --- | ---: | ---: | ---: | ---: |
| 45 | 16.5774 | 2.3597 | 16.5913 | −0.0770 |
| 46 | 34.5321 | 3.3338 | 35.1174 | −0.2216 |
| 47 | 56.6588 | 9.6416 | 57.3713 | −0.1575 |

可见本层差异甚至与输入扰动传播部分反向，不能用“移除本层误差必然改善总误差”推断
端到端修复结果。这里没有给自然路径做任何纠正。

## 2. 第 45→46 层：两种效应，以及大范数 token/通道

自然轨迹第 46 层的实际边界：

| 边界 | Reference norm | Delta norm | Relative-L2 |
| --- | ---: | ---: | ---: |
| 层输入 / 第 45 层输出 | 1461.35 | 16.58 | 1.1344% |
| Core attention 输出 | 265.26 | 9.15 | 3.4494% |
| `o_proj` 输出 | 274.94 | 9.56 | 3.4784% |
| Attention residual 后 | 1490.99 | 18.93 | 1.2698% |
| MLP norm 后 | 967.79 | 30.16 | 3.1169% |
| MoE 分支输出 | 1078.72 | 30.39 | 2.8168% |
| Block residual 后 | 997.90 | 34.53 | 3.4605% |

### 输入的 1.13% 掩盖了典型 token 的 2.45%

本层输入的 channel **940** 占 reference 平方范数的 **92.4840%**。
global token IDs **0、128** 合计占 input reference 平方范数的 **90.5849%**。
因此原始 hidden 的全局 relative-L2 极度偏向少数大幅值分量，而非均匀衡量各 token。
实际逐 token relative-L2 中位数在第 45 层输出已经是 **2.4515%**，第 46 层输出为
**3.0761%**，不是所有 token 都经历 1.13%→3.46% 的三倍增长。

使用 checkpoint 中真实 input norm gamma 做 FP64 数学分析：原始 1.1344%，仅每个
token 自身 RMS 缩放后为 **2.4322%**，再乘 gamma 后为 **2.3323%**。
这处 input norm 与 QKV 融合，未直接捕获独立 norm 输出；此处明确是对真实输入和
权重的数学分析，不冒充 fused kernel 内部观测值。自然 core output 已有 3.4494%，
而 `o_proj` 输出为 3.4784%，也不支持将整个 attention 分支差异归因于当前投影首差。

MLP norm 的分解则有真实前后 tensor 对照：

```text
真实 norm 输入      1.269751%
仅每个 token 的 RMS  2.580239%
再乘 learned gamma  3.107312%  （未舍入的 FP64 数学值）
实际 BF16 norm 输出 3.116901%
```

差异主要已经由输入扰动、token 尺度和 gamma 的作用解释，而非同输入 TP norm 新增了
约 2% 误差。此处 gamma 后的误差向量范数为 30.0729，真实为 30.1650；两种误差向量
之差的范数是 **2.2709**，不能将两个范数之差 0.0921 称为舍入误差范数。

### 残差相消：reference 抵消，误差向量却没有同比抵消

在第二处残差中，reference residual 与 MoE 输出的 cosine 是 **−0.743289**，
但两条分支的误差向量 cosine 只有 **−0.082192**。也就是信号明显相消，误差接近
正交，没有同样强的相消；reference norm 因而降到 997.90，delta norm 升到 34.53。

从整层输入到输出，relative-L2 的 **3.0505 倍**增长可以精确写成：

```text
(34.5321 / 16.5774) × (1461.3528 / 997.9001)
       2.0831      ×          1.4644
    绝对误差增长           reference 分母缩小
```

将已捕获两个分支的差向量在 FP64 中相加，norm 为 **34.4548**；实际 BF16 残差输出
的 delta norm 为 **34.5321**。定义差分舍入向量
`q = delta_output - delta_residual - delta_moe`，其 norm 为 **1.9113**。
所以“最后一次残差加法舍入凭空制造整个跳升”不成立，但不能把非零的舍入项忽略。

大范数 token 的真实相消尤其明显：

| Global token ID | 输入 ref norm | 输出 ref norm | 输入 delta norm | 输出 delta norm | Ref residual/MoE cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 1016.92 | 472.27 | 0.1270 | 0.9058 | −0.9525 |
| 128 | 948.88 | 396.43 | 12.0006 | 16.1102 | −0.9709 |
| 85 | 193.56 | 231.26 | 4.0695 | 19.0815 | −0.5769 |
| 156 | 93.15 | 468.12 | 1.9022 | 12.8590 | −0.3823 |

输出误差能量也并不均匀：IDs **85、128、156** 分别占 `||delta||²` 的
**30.5337%、21.7650%、13.8665%**，合计 **66.1651%**。这些是本批次 ID，
没有把数值特征推断为 token 的语义类别。尚未继续展开单个 expert 的 FC1/SiLU/FC2
灵敏度，也不声称解释每个误差分量的唯一来源。

## 3. 第 47 层→final norm：token 重加权与 gamma 通道抑制

第 47 层首先不是“修复了”第 46 层的误差：delta norm 从 **34.53** 继续增至
**56.66**，但 reference norm 从 **997.90** 增至 **2347.63**，所以 relative-L2
反而降到 **2.4134%**。其后 final norm 的实际变化可由真实权重的数学分解复现：

| 处理 | Relative-L2 |
| --- | ---: |
| 原始第 47 层输出 | 2.413443% |
| 双方各自按 token 的 RMS 缩放，不乘 gamma | 3.426665% |
| 再乘真实 learned gamma，尚未舍入 | 4.332014% |
| 实际 BF16 final norm 输出 | 4.338377% |
| 实际 LM head / logits | 4.558517% |

RMSNorm 使用 `y = gamma * x / sqrt(mean(x²) + eps)`。公式核对
[HF Qwen3-MoE v4.57.1](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py#L241-L254)；
HF eager 在乘 gamma 前转换回输入 dtype，TE fused 路径并非同一舍入顺序。上表
明确使用未舍入的数学值，不将其称为 HF eager BF16 的逐位复制。

### 真实 gamma 揭示了被原始 hidden 指标掩盖的差异

在第 47 层原始 hidden 中，channels **940、1992** 合计占 reference 平方范数的
**73.8230%**，但只占原始误差平方范数的 **27.0294%**。final norm 的 gamma 为：

| Channel | Gamma | 原始 reference 能量占比 | Norm 后 reference 能量占比（数学值） |
| --- | ---: | ---: | ---: |
| 940 | −0.0143433 | 31.1878% | 0.0016143% |
| 1992 | 0.0114746 | 42.6352% | 0.0002642% |

相比之下 gamma 的中位数是 **2.84375**，最大值 **8.1875**（channel 1465）。
这两个大幅值通道在 final norm 后几乎不再主导 reference 范数；原来被大分母稀释的
其他通道误差变得更显著。token 的 RMS 缩放与这种强烈的通道重加权共同解释了
2.4134%→4.3384%，不是“final norm 的 TP kernel 又算错了 2%”。

令 `a` 是 baseline 输入，`b` 是 TP 输入，`ra/rb` 是各自 inverse RMS，则逐元素有：

```text
delta_y = gamma * ra * (b-a) + gamma * b * (rb-ra)
            固定尺度的输入差异       RMS 尺度对扰动的响应
```

这两个向量的 norm 分别为 **70.6596、17.0920**，cosine **−0.192825**；合并后
未舍入的 delta norm 为 **69.4202**，实际为 **69.5212**。实际误差向量减去数学误差
向量的 norm 为 **3.7220**，即有舍入/有限精度影响，但主要跳升在未舍入公式中已出现。

### 同输入与独立 GPU 检查排除了什么

- `6006744` 给 final norm 相同 `end_hidden_2`，TP2 的 `hidden_for_head` hash
  与 baseline 完全相同；给 head 相同 `hidden_for_head`，logits hash 也完全相同。
  因此本批次这两个模块没有检出同输入下独立的 TP 数值差异。
- `6006849` 使用实际 **256×2048** BF16 输入和 live gamma，单独构造 TE RMSNorm
  与独立 PyTorch GPU RMSNorm。参考使用 FP32 中间量并在乘 gamma 后一次舍入 BF16；
  不调用 TE fused kernel，也不重写正式路径。
- 第 46 层 MLP norm / final norm 的 forward relative-L2 分别为
  **0.001142% / 0.001104%**（524288 元素中仅 2 / 4 个不同）；input-gradient
  为 **0.265536% / 0.265768%**；gamma-gradient 为 **exact / 0.000342%**。
  所有 cosine 和 tensor similarity 均大于 **0.999**，最低约 **0.99999647**。
- HF eager 中间 BF16 cast 版本与 baseline fused norm 的 forward 差异约
  **0.2758% / 0.2789%**，不能将这个已知舍入顺序差异隐去，冒充逐位相同参考。

结论限于这组 shape/输入/权重与前述对照；独立 norm 的 backward 通过不是 30B
全网 backward、optimizer 或跨 TP 训练验收通过。原生 TP2/CP1 logits 仍为
**4.5585%**，原先 2% / 0.999 门槛未通过，本轮没有提交生产修复或 BF16+FP32 路径。

## 任务与复现材料

- `6006744`：TP2/CP1/DP2/EP2 单层链和 final operator 控制；`eos0514`，2:06。
- `6006849`：GPU 分析、独立 norm forward/backward；`eos0569`，1:08，COMPLETED / exit 0。
- `6006798` / `6006837` 均在 PENDING 时取消，未消耗计算时间；分别是普通队列和
  interactive-only 队列尝试。最终通过公共 `cluster-run slurm` 提交至
  `batch,interactive`，仍遵循调度配额。CPU 备用统计脚本未提交，不作为结果证据。
- 本地摘要：`runtime/diagnostics/diff_source/6006744/layer_chain.json`、
  `layer_chain.manifest.json`、`layer_chain.norm_weights.pt`；以及
  `6006849/chain_analysis.json`。完整 raw trace 留在 EOS，同原调查项目。
- `isolate_layer_chain.py` SHA256：
  `ca9e4c56c62526532d09671a7865db619627dbaad9866d6bb21f500033b34423`。
- `analyze_layer_chain.py` SHA256：
  `498d656204bad9b1a709976e8b7d11dafc9adc4319b013b25896e154bd281058`。
- 分析所绑定 `layer_chain.json` SHA256：
  `8edcf74479776719e66a8de0793a40900fdbb50012b68f9aa8632db8989f0d40`。
- 启动脚本 `run_layer_chain.sh` / `run_analyze_chain.sh`；依赖仍通过
  `scripts/eos/run_bf16_environment.sh` 验证，不跳过原环境 guard。

## FP32 partial 对照续查

FP32 partial/reduction 的完整 48×15 边界对照、token 156/channel 940 的 BF16
舍入分界、L45 norm 指标优势消失与正确 final gamma 分解，见
[fp32-partial-propagation.zh.md](fp32-partial-propagation.zh.md)。该续查说明为何
仅改善本层 projection 误差不能保证最终 logits 改善，并非正式 BF16+FP32 路径。
