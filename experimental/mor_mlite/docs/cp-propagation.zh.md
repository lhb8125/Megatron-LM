# TP1 下 CP2 的首差、传播和末端放大

后续 [logits 与 loss 对照](logits-vs-loss.zh.md)：原 forward-only 未记录 LM loss；
6011940 在相同 logits/labels 上补算 CE，CP1=13.179327、CP2=13.137942，
平均差 −0.041385（−0.3140%），不等于 logits relative-L2 的 5.8545%。

## 结论与适用范围

本次固定 TP=1，调查原 Qwen3-30B-A3B-Base folded MoR 的 BF16 forward：
20 个物理层、48 次逻辑调用；CP2/DP2/EP2/ETP1，
两条 128-token 输入，seed=1234，相同 depth/expert routing replay。
层号 L0–L47 从 0 开始；L45/L46/L47 为三个 end layers，对应物理层 17/18/19。

**首个差异出现在 L0 attention core；主要后续放大来自输入扰动的传播、
RMSNorm 的重新加权、L46 expert 对特定方向的敏感性，以及残差相消。**
CP 不只是复现 TP 的 BF16 partial-reduce 差异，不能把两者根因混为一谈。
最终 logits relative-L2 为 **5.854544%**，cosine **0.9982971352**，
原 2% / 0.999 门槛仍未通过；256 个位置中 32 个 argmax 不同。

本次复用原 GPU 自然轨迹 6006544、逐算子同输入隔离 6006559，
新跑 CPU 统计 6011881、独立 MoE FP64 数学分析 6011910，
以及 GPU Magi deterministic 隔离/重复实验 6011900。
没有修改生产计算代码或精度路径，没有运行 backward/optimizer。

## 先纠正 deterministic 状态，再做隔离

上一轮“strict 已覆盖 attention deterministic”的说法对 **CP1 local FFA** 成立，
不能直接推广到 CP>1：

- 框架 strict=true，PyTorch deterministic algorithms 已启用，TE 禁用非确定性算法。
- CP1 `LocalMagiAttention` 显式传 `deterministic=True`。
- CP2 的 MLite `MagiDotProductAttention` 调 `calc_attn`；
  Magi `DistAttnRuntime.deterministic` 单独读取
  `MAGI_ATTENTION_DETERMINISTIC_MODE`，不读取 PyTorch 开关。
- 原 CP2 manifest 中该环境变量为 **0**。这是 strict 配置传递上的缺口，
  不能以有限次重跑一致代替模式真正启用的证明。

GPU 作业 **6011900** 仅在运行专用脚本中将该变量改为 **1**，
沿用同一 CP2 checkpoint、输入和 replay，独立 torchrun 两次：

| 对比 | logits relative-L2 | 268 项 tensor hash |
| --- | ---: | --- |
| Magi deterministic=1 / 原 CP2=0 | 0 | 全部相同 |
| deterministic=1 第 2 次 / 第 1 次 | 0 | 全部相同 |
| deterministic=1 / CP1 baseline | 5.854544% | 不相同 |

所以此配置下该缺口**不是已观测 logits diff 的解释**；开启它不改变结果。
这仅限当前 forward，不为 backward 或其他输入/规模的确定性作保证。
生产 strict→Magi 开关尚未正式修复：本轮是诊断请求，不实施生产变更。

## 首差来自什么

原自然 GPU 轨迹按 global token ID 重组，每组 48×15=720 个边界；
主 artifact 268 项 hash 与原运行逐位一致，探针没有改变数值轨迹。
CP1 参照为串行模拟相同 DP2 样本划分的单 rank baseline；
既有 TP1/CP1/DP2/EP2 的全部 28 项硬 forward 张量均与该 baseline 零差异。

L0 的 block input、QKV、QK norm、RoPE 后 Q/K、V 全部 exact。
第一次不同是 `core_output`：

- relative-L2 **0.09900278%**；
- max-abs **0.00048828125**；
- 187315 / 1048576 个元素不同。

给每个算子喂同一 baseline 输入，48 层结果如下：

| 算子 | CP2 同输入结果 |
| --- | --- |
| attention core | 0/48 exact；relative-L2 0.061913%–0.156029% |
| QKV | 37/48 层 exact；Q/K/V 最大分别 0.007899% / 0.010318% / 0.012625% |
| QK norm | 48/48 exact |
| output projection | 14/48 exact；最大 0.013112% |
| MLP norm | 36/48 exact；最大 0.003036% |
| MoE | 48/48 exact |

因此 core 是主要持续独立差异源，但不是唯一有差异的算子。
QKV/projection/norm 的较小差异尚未细分到具体 fused kernel/算法选择。
同输入隔离中的 RoPE/core inputs、residual inputs 是人为控制量；
不拿它们的 exact 或被干预后的 block/logits 当自然端到端验收。

### CP core 并不是 BF16 partial output 归约

核对固定 Magi 1.1.1 实际安装源码：

- CP1 通过 public local `flex_flash_attn_func` 执行完整 local ranges。
- CP2 通过分布式 runtime 的 host/remote 计算范围、KV 通信及 partial out/LSE 路径。
- 当前 `MAGI_ATTENTION_QO_COMM=0`、FFA backend；
  分布式 FFA 显式指定 high-precision output，使用 FP32 output/LSE 累加缓冲，
  最后转回 Q 的 BF16 dtype。CP1 public 路径默认也先产生 FP32 output，再转 BF16。

这说明“把 CP partial 从 BF16 改成 FP32”不是本配置现状的正确描述。
分块、计算顺序、softmax/LSE 合并与最终舍入位置的实现差异是待细分的数值来源；
目前已定位到不同 attention core 路径，**尚未用 kernel 内部干预证明某条指令贡献占比**。
旧作业 6006454 将 overlap degree 设为 0，logits 仍完全不变，亦不支持把异步 overlap
直接当成根因。注意 `FORWARD_HIGH_PRECISION_REDUCE=0` 在 QO_COMM=0 时不能
被误读为本路径的 partial accumulator 是 BF16。

独立 FP32 causal GQA 数学参考在同一 L0 Q/K/V 上得到：
CP1 core 的误差 **0.172698%**，CP2 core **0.171171%**。
CP2 没有明显比 CP1 更偏离该数学参考；两条数值实现不一致并不自动等于 CP 算法错误。
这里不是 HF eager BF16 的 bitwise reference，也不能据此签发整网正确性。

## 自然误差如何传播

每层同时包含“传入误差的响应”和“当前算子独立新增差异”；
两种误差是向量，可能增强或抵消，不能把 relative-L2 直接累加或相减。
例如 L46 core 自然输入下差 **3.8820%**，相同 Q/K/V 对照只差 **0.1310%**：
自然 core 的大部分差异不能再全部记作这一层新产生的 CP 误差。

| 边界 | CP2 / CP1 relative-L2 |
| --- | ---: |
| L0 core | 0.099003% |
| L0 output projection | 0.108246% |
| L0 post-attention residual | 0.102513% |
| L0 MLP norm | 0.144468% |
| L0 MoE | 0.309193% |
| L0 block output | 0.209654% |
| L2 block output | 0.829847% |
| L45 input（post-merge） | 0.733324% |
| L45 output | 1.053357% |
| L46 post-attention residual | 1.235655% |
| L46 MLP norm | 3.648769% |
| L46 MoE | 6.166882% |
| L46 output | 6.218280% |
| L47 output | 5.308622% |
| final norm 后 | 5.545923% |
| logits | 5.854544% |

中间 recurrent 轮次的 token 集合会改变。L44 只含该轮 active tokens，
L45 post-merge 恢复完整 token 集合；不能把两者相对范数变化归因于单个算子的放大。
本次 CPU 分析重新计算全部 720 个自然边界并核验旧 JSON 指标与 token identity。

## 为什么 L46 特别敏感

### RMSNorm 改变 token / channel 的权重，不只是新增舍入

对真实输入和实际 gamma，用 FP64 做离线数学分解：

| L46 MLP norm 处理阶段 | relative-L2 |
| --- | ---: |
| norm 输入 | 1.235655% |
| 只按每个 token 的 RMS 归一化，gamma=1 | 3.047145% |
| 乘实际 gamma，仍不做 BF16 舍入 | 3.642689% |
| 实际 GPU norm 输出 | 3.648769% |

数学计算已经重现主要变化，不需要用“norm kernel 算错”解释。
全张量 relative-L2 原先按 token 的幅度隐式加权；
RMSNorm 将各 token 的幅度拉到相近尺度，原先被大幅度 token 掩盖的误差更显著。
输入逐 token relative-L2 中位数为 **2.8757%**，本就高于全张量的 1.2357%。
gamma 又重新加权 channel（即 hidden 向量中的特征位置）。
例如输入 channel 940 占参考平方范数 88.84%，却只占误差平方范数 31.25%。
这些权重变化说明，不能把前后的百分数之比解释成每个元素误差都放大了约三倍。

### MoE 的高敏感方向：token 85 / 156

L46 输出误差能量（平方范数）中，token 85 占 **63.07%**，
token 156 占 **22.44%**，两者合计 **85.51%**；top10 占 88.92%。
global token ID=85 是第一条序列第 86 个位置，156 是第二条序列第 29 个位置。

| token | MoE 输入 diff | MoE 输出 diff | L46 block 输出 diff |
| --- | ---: | ---: | ---: |
| 85 | 0.970531% | 20.286852% | 21.309673% |
| 156 | 2.173549% | 5.793855% | 6.278579% |

depth route、expert ID 和 selected scores 均固定；同输入 MoE 48 层 exact，
所以此处不是 router top-k 翻转。已有扰动进入相同 expert 后仍可大幅改变输出。

为深入该原因，作业 6011910 从已验证的 TP1 loaded-parameter raw bytes
读取物理层 18 的相关 FC1/FC2 权重并核验每个 hash；
对保存的两个输入分别按
`sum_e W2_e [score_e * SiLU(Wgate_e x) * (Wup_e x)]`
做独立 FP64 数学计算，不调用/替代 GPU 模型 kernel：

| token | 原生 GPU MoE relative-L2 | 独立 FP64 relative-L2 | 两者误差向量 cosine |
| --- | ---: | ---: | ---: |
| 85 | 20.286852% | 20.253502% | 0.9998293 |
| 156 | 5.793855% | 5.618240% | 0.9967590 |

两条原生输出各自相对 FP64 数学值的差异约 0.32%–0.41%；
此数学参考不是 TE grouped GEMM / BF16 激活舍入的逐位模拟。
但它已经复现 token 85 的主要放大，因此并不需要额外 BF16 舍入噪声才会出现。

token 85 的权重主要给 expert 96（score=0.55078125）和 74（0.439453125）。
expert 96 的 FC1 输出全向量只差 **0.7706%**，
经加权 SwiGLU 差 **15.2316%**，经 FC2 后该 expert 输出差 **27.9806%**。
具体到它的 intermediate channel 147（768 维 expert 中间向量的第 148 个位置）：

- gate：3.94413 → 4.66540；
- up：33.56040 → 33.47543；
- `score * SiLU(gate) * up`：71.51989 → 85.21660。

全 FC1 向量很接近不意味着每个关键分量都只差 0.77%；
这个 gate 分量的变化乘上较大的 up 分量，产生明显的激活差，再由 FC2 投影。
这里的 gate 是 SwiGLU 的连续分支，**不是 expert 选择开关**。
expert 74 也有相似敏感方向：channel 578 的 up 从 1.22181 变到 1.60487，
而 gate 约 46.7，导致加权激活从 25.0800 变到 32.9099。

FP64 在 baseline 点的一阶响应与实际有限差分方向高度一致：
token 85 cosine=0.999658，token 156 cosine=0.999954。
所以更准确的描述是 expert 对这些扰动方向有较大的局部响应，
而不是未经证明地声称出现了离散跳变或“非线性失控”。

### 残差相消使相对误差进一步显著

L46 MoE residual 的直接测量：

| 量 | L2 范数 |
| --- | ---: |
| 参考 residual 输入 | 1490.989 |
| 参考 MoE 输出 | 1078.716 |
| 相加后的参考 block 输出 | 997.900 |
| residual 输入误差 | 18.423 |
| MoE 输出误差 | 66.523 |
| 相加前的总误差 | 62.065 |
| BF16 相加后的总误差 | 62.052 |

参考 residual 与 MoE 输出的 cosine 为 **−0.74329**，有强相消；
输出参考范数比两个输入分支小，而误差未同比缩小。
因此 `62.052 / 997.900 = 6.21828%`。
本处末次 BF16 residual 加法的差异舍入项 L2 为 1.893，
且舍入前后总误差几乎相同：这次主要放大不是最后一次 residual 加法的舍入。

## L47 与 final norm 不要照搬 TP 的解释

L47 输出 relative-L2 从 6.2183% 降至 5.3086%，但绝对误差从 62.052
升至 124.627；参考输出范数从 997.900 升至 2347.631。
所以相对指标下降不等于绝对误差被修复，只是参考幅度增大得更快。

final norm 的分解：

| 阶段 | relative-L2 |
| --- | ---: |
| 原始 final hidden | 5.308622% |
| RMS-only，gamma=1 | 4.381038% |
| 乘实际 gamma，FP64 数学 | 5.540690% |
| 实际 GPU final norm | 5.545923% |
| 实际 logits | 5.854544% |

这里 RMS-only 反而减小 relative-L2；实际 gamma 重新加权后升高。
channel 940/1992 合计占原始参考能量约 73.82%，但 final gamma 分别只有
−0.0143433 / 0.0114746，被显著压低，其他 channel 在最终相对指标中变得重要。
CP 的 final norm 只带来较温和的净增加，**主要跳升已发生于 L46**。
head 的剩余增幅没有在本轮拆成“输入响应”和“同输入 CP head 差异”，不声称已解释全部。

## 证据、版本和边界

- 原自然轨迹：6006544 / eos0544；同输入隔离：6006559 / eos0196。
- CPU 全轨迹统计：6011881 / eos0080 / 1:02 / COMPLETED 0:0。
- CPU MoE 独立数学：6011910 / eos0117 / 0:58 / COMPLETED 0:0。
- GPU deterministic 对照：6011900 / eos0273 / 4:16 / COMPLETED 0:0。
  SLURM exit 0 表示报告生成完成；对 CP1 的数值门槛仍 failed。
- Torch 2.10.0+cu129、TE 2.13.0、MagiAttention 1.1.1、H100 80GB。
- MCore `5c8315f12a64a7279eec58896af9e74ee3351b74`。
- NGC `nvcr.io/nvidia/pytorch:26.01-py3` 的既有 sqsh，加原独立 cu129 venv。
- 生产 60 个 Python 文件 hash：
  `c49e48c4d882aa7e8f0a3b3ea98d42711ba260f42cc0bbd21d8164618b870c87`，未修改。
- 原 EP2 checkpoint：`mor-bf16-axes/6006410/folded_init_ep2`；
  TP1 数学权重：`loaded-tp-bitwise/6011449/baseline`。

本轮按 `ckpt-tools` 纪律将自然 GPU 证据、输入隔离、CPU 数学参考分开，
通过 `cluster-run` 调度。未重新宣称 CP2 全量加载后参数已经逐字节审计；
已有 TP1/TP2/TP4 参数报告不能冒充 CP2 全量参数实测。
本轮 norm 数学采用已验证 baseline gamma，算子同输入 GPU 对照另有 6006559 证据。
数学误差解释不等于数值误差可接受，也没有给出训练收敛或 backward 的结论。

- [全轨迹统计、norm 分解、residual 与 token 贡献](../runtime/ckpt_tools/cp-propagation/6011881/analysis.json)
- [MoE 独立数学及 per-expert/channel 分解](../runtime/ckpt_tools/cp-propagation/6011910/moe_math.json)
- [deterministic 对照汇总](../runtime/ckpt_tools/cp-propagation/6011900/deterministic_summary.json)
- [原同输入算子对照](../runtime/diagnostics/diff_source/6006559/cp_dp_ep.json)
- [CPU 分析脚本](../runtime/ckpt_tools/cp-propagation/scripts/analyze.py)
- [MoE 数学脚本](../runtime/ckpt_tools/cp-propagation/scripts/analyze_moe.py)
- [GPU 运行脚本](../runtime/ckpt_tools/cp-propagation/scripts/run_deterministic.sh)
- [GPU 实际环境](../artifacts/eos/6011900/versions.json)
- [固定 Magi distributed core 源码副本](../runtime/ckpt_tools/cp-propagation/dist_attn.py)
- [固定 Magi local core 源码副本](../runtime/ckpt_tools/cp-propagation/flex_flash_attn.py)
- [固定 Magi deterministic 环境开关源码副本](../runtime/ckpt_tools/cp-propagation/general.py)

三个新作业的日志位于 `runtime/ckpt_tools/cp-propagation/`。
原始 GPU traces、参数 bytes、完整新 forward tensors 保留 EOS，没有删除旧产物。
