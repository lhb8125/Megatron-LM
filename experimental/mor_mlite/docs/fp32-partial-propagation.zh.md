# CP=1：FP32 partial/reduction 为何没有改善最终 logits

## 范围与结论

2026-09-10，承接 [TP 单层传播分析](tp-layer-chain.zh.md)。本次不是新精度方案：
仅复现 `6006454` 的 forward-only projection 诊断，生产源码、正式 BF16 路径和
2% relative-L2 / 0.999 cosine 门槛不变。本文层号均为 **0-based logical layer**。

FP32 partial/reduction 确实显著减小了每层同输入 projection 误差，但没有使其逐位
相同；projection 返回 BF16，后续 norm、MoE 和残差也没有改成全 FP32。剩余扰动
经 attention 混入其他 token、norm 重缩放及 BF16 舍入、MoE 输入响应与残差传播，
形成了另一条误差轨迹，不是原生 BF16 误差的等比例缩小。

关键反例是 token 156：第 0 层输出在 FP32 诊断中 exact，第 1 层自身 Q/K/V 也
exact，但 attention 使用的上下文已不完全相同；第 2 层 MLP norm 的误差明显变大，
随后主导递归结束时的误差。早期 global-L2 改善主要来自大幅值 token 128；第 45
层 MLP norm 后优势基本消失，第 46、47 层进一步传播。最终 logits 两个 error
norm 接近，但 error vector cosine 仅 **0.162708**，不能称为同一误差或已证明的精度下限。

## 实验合同与指标

- Qwen30B MoR，20 个物理 block、48 次逻辑调用，hidden=2048；两个长度 128 的
  packed sequence，global token ID 0–255；seed=1234。递归阶段保留的 token 数不同，
  不能将第 16→17、30→31 层的全局指标当作同一 token 集合的连续增益。
- baseline TP1/CP1；对照 TP2/CP1/DP2/EP2，ETP1。routing replay 固定 depth/expert
  选择和 dispatch-visible scores，不能用 Top-K 换专家解释本次结果。
- Torch `2.10.0+cu129`、TE `2.13.0`、Magi `1.1.1`，MCore
  `5c8315f12a64a7279eec58896af9e74ee3351b74`，EOS H100 80GB。
- 固定容器 `pytorch_26.01-py3_4a7dd6b5c237.sqsh`；生产 `src/` 的 60 个 Python
  文件内容快照 SHA256：
  `c49e48c4d882aa7e8f0a3b3ea98d42711ba260f42cc0bbd21d8164618b870c87`。
- 自然路径补采 48×15 个边界，并要求主 artifact 的全部 **268** 项 tensor hash
  复现原实验。另做 block-only reset：每层从相同 baseline block input 开始，
  block 内不重置，观察本层 projection 新扰动的完整传播。
- `relative-L2 = ||candidate-baseline||₂ / ||baseline||₂`。表中百分比均为此值，
  除非显式写明 token 中位数。不同 op 的 reference 尺度不同，相对数的比值不是
  扰动的 Jacobian 增益；绝对误差也必须结合所在空间和 reference 范数理解。
- 15 个边界为 block input、Q/K/V、Q/K norm、RoPE Q/K、core V、core output、
  projection output、attention residual、MLP norm、MoE output、block output。
  未捕获 MoE 内部 FC1/SwiGLU/FC2 张量，不能把 MoE 分支输出增长完全归给其中一个 op。

## 1. FP32 partial 到底改了什么

冻结诊断的 `projection_accumulator` 对 BF16 输入、BF16 权重执行：

```text
torch.mm(BF16 input, BF16 weight.T, out_dtype=FP32)
    → FP32 reduce_scatter_tensor
    → output.to(BF16)
```

它去掉了各 TP rank 的 partial 提前落 BF16 及 BF16 归约这一组误差，但没有统一
TP1 整矩阵 GEMM 与 TP2 分片 GEMM 的运算分组；诊断还将 native TE GEMM 换成了
`torch.mm`。因此不能声称唯一改变的变量只是 NCCL 的加法顺序，也不能把余下每一位
差异精确归因到某个 kernel。BF16 输出舍入仍保留。

block-only 同输入对照中，全部 48 层 projection 前的 block input、QKV、core output
均 exact。native projection relative-L2 为 **0.165681%–0.296045%**；FP32 为
**0.002542%–0.013112%**。所有 48 层仍存在非零 projection 差异。
按层配对的改善倍数范围为 **21.9047–65.1774**。

第 0 层最直观：projection 从 **0.263258% 降到 0.005288%**，约改善 50 倍；
但 524288 个输出元素仍有 **861** 个不同，不能等同于消除扰动。

| 第 0 层边界 | Native BF16 | FP32 partial/reduction |
| --- | ---: | ---: |
| Core output | 0 | 0 |
| Projection | 0.263258% | 0.005288% |
| Attention residual | 0.233185% | 0.005248% |
| MLP norm | 0.302665% | 0.009426% |
| MoE output | 0.451909% | 0.082768% |
| Block output | 0.327427% | 0.057314% |

FP32 第 0 层的 MLP norm delta norm 为 0.012221，MoE 为 0.012141；这里不能
因为 relative-L2 从 0.009426% 变为 0.082768%，就说绝对扰动增大 9 倍。
第二次 residual 的分支 delta 在未舍入相加时 norm 为 0.012188，实际 BF16 输出
为 0.015815；两条路径的差分舍入向量 norm 为 0.010218。后续 BF16 运算仍会改变误差。

## 2. Attention 把极小局部扰动传播到原先 exact 的 token

追踪 global token **156**（第二个 sequence 的 position 28）：

- 第 0 层 FP32 projection 自身 delta norm 仅 0.00000769，经过 residual 后
  舍入到相同 BF16；该 token 的 MLP norm、MoE、block output 都 exact。
- 第 1 层该 token 的 block input、QKV、Q/K norm、RoPE 和 core V 仍 exact。
- 但 core attention output 已有 **0.001186993** 的差异，因为它使用的是整个
  causal context 的 K/V，不是仅使用 token 156 自己的 K/V。
  CPU 核对发现其 causal context 中 token **128、131、134、136、144、145、151、
  152、155** 的 K/V 已不同；不是 token 156 自身 Q/K/V 的新生差异。
- 后续 projection → attention residual → MLP norm → MoE → block output 的
  delta norm 依次为 **0.003164 → 0.004146 → 0.027476 → 0.063846 → 0.064435**。

第 2 层出现更显著的非单调性：FP32 输入比 native 更接近，但 norm 输出却更差。

| Token 156 / 第 2 层 | Reference norm | Native delta norm | FP32 delta norm |
| --- | ---: | ---: | ---: |
| Block input | 12.156324 | 0.068482 | 0.064435 |
| Attention residual / MLP norm input | 12.109144 | 0.069118 | 0.064950 |
| MLP norm output | 142.172804 | 0.057031 | 1.000954 |
| MoE output | 55.665563 | 1.012938 | 3.002610 |
| Block output | 66.722437 | 1.014504 | 3.502503 |

### Channel 940 跨过 BF16 舍入分界

`6008118` 使用 forward 中捕获并核对 hash 的真实 gamma，得到该 token / channel
的下列值。gamma 为 **3.40625**；所有实际 norm 输出均与数学值舍入 BF16 后一致：

| Channel 940 | Baseline | Native TP | FP32 partial TP |
| --- | ---: | ---: | ---: |
| Norm input | 11.1875 | 11.125 | 11.25 |
| Token inverse RMS | 3.737218587 | 3.755565979 | 3.719141179 |
| 未舍入 `gamma × input × inverse RMS` | 142.415765336 | 142.315412341 | 142.518652218 |
| 实际 BF16 norm output | 142 | 142 | 143 |

在这一数值区间，相邻 BF16 值为 142 和 143，分界是 **142.5**。FP32 的未舍入
差异仅 **+0.102886882**，却使最终输出相差 **1**；native 的未舍入差异为
**−0.100352995**，反而被舍入到与 baseline 相同的 142。

这一通道的平方误差 1 几乎解释了 token 156 整个 norm 输出的误差范数 1.000954。
注意两种 TP 输入在该通道分别为 baseline 的 −1 / +1 BF16 ULP：FP32 改善的是
整体投影误差，不保证每个后续标量或扰动方向更好。这里是实测的量化分界跨越，
不是仅凭“浮点加法不结合”作出的泛化猜测。随后固定专家 MoE 的输出 delta norm
达 3.002610，第二次残差后为 3.502503；尚未做单通道反事实注入，因此不声称
channel 940 单独解释了之后全部 MoE 和最终 logits 误差。

## 3. 早期 global hidden 改善并非所有 token 等比例改善

递归结束、进入第 45 层前，global relative-L2 从 **0.904273% 降到 0.356309%**。
但误差平方范数分布为：

| Token 组 | Reference 能量 | Native error 能量 | FP32 error 能量 |
| --- | ---: | ---: | ---: |
| 0、128 | 1941414.62 | 144.01805 | 0.002591 |
| 其他 254 个 | 49028.72 | 18.74253 | 25.26724 |

也就是说，大范数组几乎完全对齐，**其他组的误差能量反而增大**。token 128 的
delta norm 从 **12.00037 降至 0.04865**；token 156 从 **1.51840 升至 5.00357**，
占 FP32 总误差能量的 **99.07335%**。FP32 改善了 235 个 token、恶化 21 个，
并不是每个 token 都变差，但少数 token 的误差足以主导全局 numerator。

递归中期的 global-L2 也掩盖了多数 token 和 norm 后功能分支的相近误差：

| 边界 | Native global | FP32 global | Native token 中位数 | FP32 token 中位数 |
| --- | ---: | ---: | ---: | ---: |
| L16 block output | 0.529469% | 0.451543% | 1.748550% | 1.700488% |
| L30 block output | 0.655658% | 0.352792% | 1.689490% | 1.609400% |
| L44 block output | 0.653185% | 0.375360% | 1.596936% | 1.509437% |

例如 L16 MLP norm 是 2.101493% / 2.025432%，MoE 是 2.991637% / 2.907358%，
早就没有 global hidden 指标看上去那么悬殊。

## 4. 第 45、46 层如何丢失优势

RMSNorm 数学分解使用实际 checkpoint gamma，在 FP64 中计算：

```text
y = gamma * x * r(x),  r(x) = 1 / sqrt(mean(x²) + eps)
delta_y = gamma * r(a) * (b-a) + gamma * b * (r(b)-r(a))
```

这分别是固定尺度的输入扰动、RMS 尺度响应；最后单独比较实际 BF16 输出的差分
舍入。不是把 HF eager 的中间 BF16 cast 当作 TE fused 的逐位参考。

第 45 层 MLP norm 最清楚地显示了优势消失的位置：

| 同一 norm 的处理阶段 | Native | FP32 |
| --- | ---: | ---: |
| 输入 / attention residual | 0.987642% | 0.561508% |
| 各 token 除以自身 RMS，不乘 gamma | 1.749162% | 1.751625% |
| 再乘真实 gamma，未舍入 | 2.059573% | 2.054429% |
| 实际 BF16 norm output | 2.073196% | 2.067290% |

尚未做 BF16 输出舍入，两条误差已基本相等；不能将此处主要归咎于 norm kernel
又注入了独立 TP 误差。RMS 的 token 尺度重加权已移除了原始 hidden 的指标优势。

| 末端 op | Native | FP32 |
| --- | ---: | ---: |
| L45 input | 0.904273% | 0.356309% |
| L45 core attention output | 1.932853% | 2.015986% |
| L45 projection | 2.286046% | 2.311660% |
| L45 attention residual | 0.987642% | 0.561508% |
| L45 MLP norm | 2.073196% | 2.067290% |
| L45 MoE | 3.133116% | 3.148098% |
| L45 block output | 1.134388% | 0.804798% |
| L46 input norm（未舍入 gamma 数学值） | 2.332350% | 2.412392% |
| L46 core attention output | 3.449400% | 3.515501% |
| L46 projection | 3.478415% | 3.537364% |
| L46 attention residual | 1.269751% | 0.999887% |
| L46 MLP norm | 3.116901% | 3.247655% |
| L46 MoE | 2.816823% | 2.964373% |
| L46 block output | 3.460478% | 3.465530% |

L46 原始 residual 的 FP32 误差较小，但 norm 后反而更差，MoE 输入响应进一步
消耗优势。FP32 的 attention residual delta norm **14.90820**、MoE delta norm
**31.97717**，相加前的 delta sum norm 已是 **34.53379**，BF16 相加后是
**34.58253**；不是 residual 最后一次舍入凭空制造了全部跳变。

同时 baseline residual 和 MoE 分支 cosine 为 **−0.743289**，reference norm
从 residual 的 **1490.99** 经相加降至 **997.90**。实际误差增长与 reference
相消缩小分母同时发生。FP32 的 L46 block-only 同输入 output 仅差 **0.077514%**，
自然轨迹却差 **3.465530%**，说明主要差异来自前面不同输入的传播，而非本层新生
projection 扰动本身。此分解是路径恒等式，不是可随意相加的独立误差百分比。

## 5. 第 47 层、final norm 和 logits

| 边界 | Native relative-L2 | FP32 relative-L2 | Native delta norm | FP32 delta norm |
| --- | ---: | ---: | ---: | ---: |
| L46 block output | 3.460478% | 3.465530% | 34.53211 | 34.58253 |
| L47 block output | 2.413443% | 3.088367% | 56.65875 | 72.50348 |
| Final norm | 4.338377% | 4.449212% | 69.52118 | 71.29728 |
| Logits | 4.558517% | 4.561992% | 981.44295 | 982.19112 |

L47 projection 的 FP32 误差确实更小（**3.260257% → 1.673851%**），但经过
attention residual 后又接近（3.430059% / 3.433558%）；MLP norm 为
3.982121% / 4.214469%，MoE 为 2.419026% / 3.408828%，最终 FP32 hidden 更差。
这再次表明“投影更准”不能替代对残差与不同输入后续响应的检查。

Final norm 做 token 尺度与 learned gamma 的通道重加权，先前大幅值 hidden 通道
不再主导指标。`6008118` 用 `6006744` 已验证的真实 loaded gamma 离线分解：

| Final norm 处理阶段 | Native | FP32 |
| --- | ---: | ---: |
| 原始 hidden | 2.413443% | 3.088367% |
| RMS-only | 3.426665% | 3.508480% |
| Learned gamma，未舍入 | 4.332014% | 4.444160% |
| 实际 BF16 output | 4.338377% | 4.449212% |

FP32 的固定尺度输入扰动 / RMS 响应向量 norm 分别为 **72.270239 / 19.789754**，
cosine 为 **−0.189732**，合成的数学 delta norm **71.217324**，实际 **71.297276**；
差分舍入向量 norm 为 **3.744797**。这里的大部分相对误差变化在未舍入数学值中
已出现，与第 2 层 channel 940 的单 ULP 跳变是不同机制。

Final norm 后，FP32 的 tokens 0/128 error 能量仍更小（7.5577→0.22069），
但其他 token 从 4825.637 增至 5083.081，已主导最终差异。token 156 在 L47
占 FP32 error 能量 69.78%，到 logits 仅占约 0.196%；不能把所有最终 logits
误差都归到这个单一 token，它是早期非单调传播的具体反例。

两个最终 error norm 约 4.56% 相近，**不是同一向量**：native/FP32 error vector
cosine 为 **0.162708**；FP32 logits 与 native logits 彼此的差异，相对 baseline
norm 达 **5.901231%**。FP32 改善 131 个 token、恶化 125 个；token relative-L2
中位数 3.957975% / 3.955924%，最坏值 14.8694% / 20.2401%。

## 6. 归因边界

1. 已证明同输入下 TP 首差在 row projection，也已证明 FP32 partial 大幅降低本层
   首差；没有证明剩余差异完全由 NCCL all-reduce/reduce-scatter 加法顺序造成。
2. 后续 op 在相同输入下可以 exact，但对不同输入经 norm、BF16 舍入、attention
   混合和 MoE 网络产生明显差异，这两件事不矛盾。
3. final norm/head 的同输入 exact 控制来自 `6006744`；真实形状 norm 独立 GPU
   forward/backward 检查来自 `6006849`。当前自然轨迹不能替代整个 30B backward、
   optimizer、显存和吞吐验收。
4. 所有本轮诊断标记 `acceptance_eligible=false`。当前 logits 仍未通过原门槛，
   不宣称修复完成，也不引入用户拒绝的正式 BF16+FP32 路径。

上游 MLite 的 TP 验证覆盖另见 [mlite-tp-validation-audit.zh.md](mlite-tp-validation-audit.zh.md)：
固定版本直接跨 TP logits 对照是极小 Qwen3.5 TP2↔TP4；固定拓扑续训 bitwise、
checkpoint 分片正确及流程规范不能替代本次完整 30B/MoR 的跨 TP 精度验收。

## 任务和产物审计

- `6007959` / `eos0005` / 2:48：FP32 自然及 block-only trace；全部 268 项主
  tensor exact 复现 `6006454`。`6007990` / `eos0130` / 1:19 完成 720 边界分析。
- 核对时发现新增 final gamma 探针在 checkpoint load 前读取初始全 1 权重。
  96 个 block norm 权重在 forward 内采集，不受影响；自然输出张量仍正确。
  **`6007990` 的 final norm gamma 数学分解不可用**，其余实际张量统计不受此问题影响。
- 已将 final gamma 捕获移到实际 forward pre-hook，加入与 `6006744` 加载后已验证
  9 项权重 hash 对照、跨 rank 权重 hash 对照，以及 gamma 数学输出对实际 BF16 norm
  输出的量化误差界检查；新增回归要求错误初始 gamma 必须被拒绝。
- `6007952` 因 `batch,interactive` 输入触发项目原有 batch-only guard，改正提交参数
  为 `batch`；未跳过检查。`6007997` 在 `eos0365` 报 CUDA peer/NVLink contained
  error，Slurm 随后标记 `DOWN+DRAIN+REBOOT_ISSUED`，reason 为 `Xids 63`。
  保持代码、依赖、精度及 guard 不变，通过公共 `cluster-run slurm` 重试。
- `6008104` 的 GPU 重试在 PENDING 时取消，未消耗计算时间；CPU 数学核对已补齐
  本次诊断所需证据。修正后的整网 gamma 捕获 hook 和新增 2 个 GPU utility tests
  尚未在该重试中跑完，不能写作通过。首次 trace 原有 4 个 probe tests 已通过。
- `6008118` / `eos0040` / 0:58 / COMPLETED / exit 0：CPU-only 离线 norm 核对，
  不执行模型。final gamma **显式读取 `6006744` 已验证的加载后权重**，核对其全部
  9 项 hash；L0/2/45/46/47 的 gamma 读取 `6007959` forward 捕获结果，核对记录 hash
  及与已有 loaded control 的交集。已证明错误初始全 1 gamma 会被数学输出检查拒绝。
  没有改写原始 trace、旧 metadata 或训练参数，也不将初始 gamma 冒充真实权重。
- CPU 与 `6007990` GPU 结果的 5 层 × 2 路径 × 4 个 norm 数学阶段逐项一致
  （relative tolerance 1e-8）；native final norm 分解复现上一轮 `6006849`。
  本文最终 norm 数学分解和 channel 940 值以 `6008118` 为准；720 个 block 边界
  统计以 `6007990` 为准，不依赖其错误的 final gamma。
- 本地完整逐层逐 op 表：[all_operators.csv](../runtime/diagnostics/diff_source/6007990/all_operators.csv)，
  包含 720 行以及自然/本层同输入误差、绝对范数、逐 token 中位数、误差方向、
  改善/恶化 token 数。详细分布与 residual 分解在同目录 `fp32_analysis.json`，
  **其中 `norms.final_norm` 的旧 gamma 分解无效，使用下一个文件替代**。
- 已校验的 norm / scalar 审计：[fp32_norm_cpu.json](../runtime/diagnostics/diff_source/6008118/fp32_norm_cpu.json)。
  原始张量留在 EOS worktree `mor_mlite_bf16_train` 的同相对路径，未搬运大文件。
- 冻结 `6006454` partial 实现 SHA256：
  `e8a6b76544c4a45fce4ed07cac60df8326799a53278a253dcb50cd31a422afa9`。
  已执行 CPU audit 脚本 SHA256：
  `2659756f008f1c8ea791d1d518640d348cc6077ebfe35166102fc067b263eea0`。
  新版 gamma 捕获 probe SHA256：
  `466b89751d2389dbc1857a4e94b9e317dec441a48619bff30f1d40c216270fe9`
  （GPU 重跑未完成，不作为新的整网执行证据）。Ruff、Python 编译和 shell 语法
  检查通过；最终生产 `src/` hash 仍与本节开头一致。

## 附录：48 层 block output 总表

单位为 relative-L2 百分比。FP32 block-only 列只重置该层 block input，不重置内部 op。
完整 15 个边界见上面的 720 行 CSV。自然轨迹的前层扰动不能由 block-only 列代替。

| Layer | Native 自然 | FP32 自然 | FP32 block-only |
| --- | ---: | ---: | ---: |
| 0 | 0.327427% | 0.057314% | 0.057314% |
| 1 | 0.525475% | 0.060946% | 0.012827% |
| 2 | 0.493389% | 0.300287% | 0.003091% |
| 3 | 0.493643% | 0.300644% | 0.001616% |
| 4 | 0.494047% | 0.301196% | 0.001885% |
| 5 | 0.491056% | 0.301576% | 0.001655% |
| 6 | 0.491620% | 0.281273% | 0.002245% |
| 7 | 0.492269% | 0.303387% | 0.002046% |
| 8 | 0.493580% | 0.326299% | 0.002965% |
| 9 | 0.495006% | 0.349245% | 0.002617% |
| 10 | 0.497145% | 0.352332% | 0.002871% |
| 11 | 0.499671% | 0.376164% | 0.003116% |
| 12 | 0.501964% | 0.399535% | 0.002820% |
| 13 | 0.505433% | 0.403823% | 0.002850% |
| 14 | 0.513397% | 0.432825% | 0.004648% |
| 15 | 0.520610% | 0.441677% | 0.004940% |
| 16 | 0.529469% | 0.451543% | 0.005119% |
| 17 | 0.636281% | 0.316446% | 0.001131% |
| 18 | 0.636532% | 0.316863% | 0.001636% |
| 19 | 0.636666% | 0.317089% | 0.001088% |
| 20 | 0.636958% | 0.317635% | 0.001808% |
| 21 | 0.637282% | 0.318242% | 0.001861% |
| 22 | 0.637852% | 0.319415% | 0.002116% |
| 23 | 0.638461% | 0.320748% | 0.002423% |
| 24 | 0.639493% | 0.322897% | 0.001939% |
| 25 | 0.640717% | 0.325127% | 0.002536% |
| 26 | 0.641963% | 0.327457% | 0.002421% |
| 27 | 0.643610% | 0.330558% | 0.002036% |
| 28 | 0.647722% | 0.337730% | 0.003336% |
| 29 | 0.651161% | 0.345209% | 0.004694% |
| 30 | 0.655658% | 0.352792% | 0.003068% |
| 31 | 0.642332% | 0.355768% | 0.000718% |
| 32 | 0.642515% | 0.356077% | 0.001090% |
| 33 | 0.642606% | 0.356217% | 0.000852% |
| 34 | 0.642823% | 0.356567% | 0.001191% |
| 35 | 0.643044% | 0.356900% | 0.001111% |
| 36 | 0.643378% | 0.357494% | 0.001787% |
| 37 | 0.643721% | 0.358195% | 0.001530% |
| 38 | 0.644350% | 0.359422% | 0.001624% |
| 39 | 0.644999% | 0.360670% | 0.001696% |
| 40 | 0.645705% | 0.361983% | 0.002009% |
| 41 | 0.646619% | 0.363676% | 0.001233% |
| 42 | 0.648693% | 0.367279% | 0.002786% |
| 43 | 0.650671% | 0.370841% | 0.003243% |
| 44 | 0.653185% | 0.375360% | 0.002464% |
| 45 | 1.134388% | 0.804798% | 0.054735% |
| 46 | 3.460478% | 3.465530% | 0.077514% |
| 47 | 2.413443% | 3.088367% | 0.032726% |
