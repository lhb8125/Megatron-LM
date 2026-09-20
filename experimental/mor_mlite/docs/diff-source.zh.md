# 30B 单卡/多卡 forward diff 来源

后续 [CP2 来源及放大分析](cp-propagation.zh.md)：补齐 L46 token/expert 数学、
norm/residual 分解，并发现 strict 未传至 Magi 分布式 deterministic 环境开关；
6011900 打开该开关重跑两次，结果与原 CP2 全部 268 项 hash 相同，仍未对齐 CP1。

后续 [TP2 / TP4 对比](tp2-vs-tp4.zh.md)：作业 6011796 证明加载后全部参数
bitwise 一致，但 CP1 下 logits relative-L2 仍为 4.5088%；原数值门槛未通过。

后续 CP=1 细查见 [单层投影扰动、45→46 与 47→final norm](tp-layer-chain.zh.md)：
`6006744` 完成 block-only 输入隔离和 final norm/head 同输入 exact 对照；
`6006849` 完成真实 gamma 数学分解与独立 GPU norm forward/backward。

## 范围与复现基线

### 补充：全量加载后参数一致性已补齐

后续 GPU 作业 **6011449** 已完成直接检查：原 TP1/EP1 baseline 与 TP1/EP2、
TP2/EP2 的全部 **5,266** 个完整参数均 bitwise 一致，差异参数/元素/字节为 0。
覆盖全部 20 个物理层、128 experts/层、norms、routers 和 head；没有裁剪 padding。
详见 [加载后参数 bitwise 检查](loaded-tp-parameters-bitwise.zh.md)。
下段保留此前缺口的审计背景，不再代表当前验证状态。

2026-09-10 用户追问后核对：不能将“同一 HF checkpoint 来源”表述为
“已经验证 TP1/TP2 全部加载后参数逐位一致”。`parity/mlite.py` 的
`capture_full_state` 仅对 tiny preset/architecture 启用；只有该分支采集
`initial/*`，`parity/compare.py` 才会对这些参数做 shape、dtype、原始 bytes
一致性检查。30B 报告 `6006397/all_vs_baseline.json` 和
`6006410/tp_dp_ep.json` 都没有 `initial/` 记录。因此目前没有整模型、加载后、
首次 forward 前，将 TP2 分片还原并与 TP1 全量参数逐位比较的通过证据。

已有 norm 权重 hash、同输入算子对照、EP-only forward exact 和同拓扑
checkpoint 恢复指纹属于局部或间接证据，不能替代这项检查。
尤其 baseline 与 EP2 candidate 使用同一 HF snapshot 的不同 folded DCP
转换产物，并不是同一组分片文件。此前逐 op 观测仍成立，但排除全量加载/分片
差异需要补齐此项直接验证；本次核对没有运行新 GPU 作业或修改生产代码。

2026-09-10，承接任务 `01a0799b-f8a1-7171-ae82-809b58c678c1`，按用户要求只诊断
数值 diff，不继续修改生产计算路径。不是 git diff 调查，也不是训练验收通过声明。

- 模型：Qwen3-30B-A3B-Base，MoR folded 20 个物理层、48 次逻辑层调用
  （3 start + 14 recurrent × 3 + 3 end）；BF16 参数/激活。
- 输入：两条 128-token 序列，1 step、1 microbatch、forward-only；baseline 串行模拟
  DP=2，候选 replay 相同 depth/expert 选择。验收阈值仍为 relative-L2 ≤ 0.02、cosine ≥ 0.999。
- 源码：本包 60 个 Python 文件 sha256
  `c49e48c4d882aa7e8f0a3b3ea98d42711ba260f42cc0bbd21d8164618b870c87`，本轮未改。
- 上游：MCore `5c8315f12a64a7279eec58896af9e74ee3351b74`，MagiAttention 1.1.1，
  Torch 2.10.0+cu129，TE 2.13.0；H100 80GB，NGC PyTorch 26.01 容器，
  独立 `.deps-cu129-clean/venv-torch210-cu129-v3`。实际编译工具链 NVCC 13.1，
  不宣称所有动态库均为 CUDA 12。
- HF snapshot：`1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9`；EP1/EP4 使用 `5998357`
  的 folded checkpoint；EP2 使用 `6006410` 同一 HF 源的真实转换，不重标 EP4。
- Attention policy：`native-bf16-local-ffa-strict-cp1-magi-cp-v1`。

## 无输入注入的完整轨迹

EOS 任务 **6006544**，节点 `eos0544`，4:49，COMPLETED / exit 0。
探针 4 个测试通过，包括稀疏 ID、padding、TP head 重建、重复冲突/覆盖缺失拒绝。
每组记录 48 × 15 = 720 个边界；按 global token ID 和 TP head 重组，不按 rank 顺序比较。
四组各自的 **268 项主 artifact tensor index/hash 全部与旧运行逐位一致**：
baseline/all 对照 `6006397`，TP/CP 对照 `6006410`。因此下述不是探针改变调度后产生的新轨迹。

下表数字为 relative-L2 的百分数，层号从 0 开始：

| 拓扑 | 首个不同边界 | 该边界误差 | 首个边界之前 |
| --- | --- | ---: | --- |
| TP2 CP1 DP2 EP2 | 第 0 层 `attention_output`（`o_proj` 输出） | 0.2632578% | input、QKV、QK norm、RoPE、core output 全部 bitwise exact |
| TP1 CP2 DP2 EP2 | 第 0 层 `core_output` | 0.0990028% | input、QKV、QK norm、RoPE Q/K、V 全部 bitwise exact |
| TP2 CP2 DP2 EP4 | 第 0 层 `core_output` | 0.0990028% | 同上；投影输出误差进一步为 0.260241% |

这证明最早的 diff 已在 start 层出现，早于 MoR 的首轮 depth routing/递归转换。
既有 `6006431` 的纯 DP 和 EP2 对照所有硬 forward tensor bitwise exact；该组输入下
它们不是首个 diff 来源。不能由此推广为所有形状、输入和拓扑都不产生误差。

### TP：row-parallel 投影的舍入位置不同

MLite `primitive/parallel/linear.py` 的 `RowParallelLinear` 使用 TE row mode +
sequence parallel。固定 TE 2.13.0 `pytorch/module/linear.py:324` 的 forward GEMM
显式 `out_dtype=activation_dtype`（这里是 BF16），随后在 :354–358 对 `gemm_out`
做 `reduce_scatter_along_first_dim`。

因此数学顺序是：单卡完整 dot product 后舍入；TP2 则每个输入维分片先产生 BF16
partial output，再相加/舍入。浮点舍入不满足分配律。第 0 层 core output 完全相同，
投影输出却不同，已将 TP 首差定位到该投影，而不是其上游 attention。

### CP：相同 Q/K/V 的 core attention 输出不同

CP1 为 `qwen3_moe_mor/local_attention.py` 的官方 local FFA functional adapter；
CP2 为 MLite `MagiDotProductAttention` → Magi `calc_attn` 的分布式 core。
两者第 0 层输入完全一致，但输出有 187315 / 1048576 个元素不同，max-abs 为
0.00048828125。尚不能将这直接命名为某个 CUDA 指令 bug 或 token/mask 错排。

Magi `functional/dist_attn.py` 分布式 FFA 路径使用 FP32 partial output/accumulator；
当前 QO_COMM=0，不能错误归因于“CP 一律 BF16 partial 通信”。既有 `6006454`
将 overlap degree 设为 0 后整个 CP forward 数值不变，也不支持将 overlap 本身当作根因。

## 差异在哪里增大

不同边界的相对范数使用各自的参考分母，不能把百分数简单相减当成误差贡献分解。
以下列出原始自然轨迹中相同 block hidden 边界：

| 位置 | TP2 CP1 | TP1 CP2 | TP2 CP2（8 卡） |
| --- | ---: | ---: | ---: |
| 第 0 层输出 | 0.32743% | 0.20965% | 0.33031% |
| 第 45 层输入（post-merge） | 0.90427% | 0.73332% | 0.92932% |
| 第 45 层输出 | 1.13439% | 1.05336% | 1.14296% |
| 第 46 层输出 | 3.46048% | 6.21828% | 3.74687% |
| 第 47 层输出 | 2.41344% | 5.30862% | 2.55261% |
| 最终 logits（旧运行已逐 hash 复现） | 4.55852% | 5.85454% | 4.41227% |

8 卡第 46 层：输入 hidden 1.14296%，attention 后残差 1.26606%，MLP norm 输出
3.03628%，MoE 输出 3.24730%，block 输出 3.74687%。这里是显著放大区，
但自然轨迹的 MoE 输入已经不同，不能仅凭其输出 diff 宣称 MoE 自身实现错误。

## 同输入算子隔离与独立数学参考

第二组诊断任务 **6006559**，节点 `eos0196`，2:53，COMPLETED / exit 0。
独立运行各算子输入控制，以及纯 PyTorch attention/投影数学参考。注入只用于诊断，
不运行 backward/optimizer，不冒充端到端验收，也不形成 BF16+FP32 正式路径。

分别给 block、Q/K norm、core Q/K/V、projection、MLP norm、MoE 注入 baseline
的对应输入；保持当前算子、权重、dtype 不变。RoPE/core 输入和残差输入是被注入的
控制量，不以其“相同”证明该操作实现正确；最终 block/logits 也不是完整自然轨迹。
两组 block input 的 48 层 identity/value 均逐位相同，覆盖仍为完整 720 个边界。

| 相同输入下的算子输出 | TP2 CP1（48 层） | TP1 CP2（48 层） |
| --- | --- | --- |
| QKV | 48/48 层 exact | 37/48 层 exact；最大误差 Q 0.007899%、K 0.010318%、V 0.012625% |
| QK norm | 48/48 层 exact | 48/48 层 exact |
| Core attention | 48/48 层 exact | 0/48 层 exact；误差 0.061913%–0.156029% |
| 输出投影 | 0/48 层 exact；误差 0.165681%–0.296045% | 14/48 层 exact；最大 0.013112% |
| MLP norm | 48/48 层 exact | 36/48 层 exact；最大 0.003036% |
| MoE | **48/48 层 exact** | **48/48 层 exact** |

因此 TP 的独立数值误差在本实验中集中于 row projection。CP 的主要独立差异在 core，
但还存在较小的 QKV、projection 和 norm 差异，不能将它们抹去后宣称“只有 core”。
CP 的 QKV 非 exact 层为 17–21、23–24、26–29；MLP norm 为 17–18、20–23、25–30。
这些 CP 小误差的 fused norm/GEMM 内部算子或算法选择尚未继续细分；分片形状变化是
待验证解释，不写成已证实根因。MoE 同输入完全一致，说明自然 forward 中它的明显
差异来自不同输入的传播/非线性响应，而不是这两组配置下 MoE 自身的独立误差。

### 独立数学对照

第 0 层使用真实 baseline core output 和 HF `model.layers.0.self_attn.o_proj.weight`，
纯 CPU FP64 计算，再模拟 BF16 舍入：

- 完整 dot product 后舍入一次，与实际单卡输出有 853 / 524288 个元素不同，
  relative-L2 为 0.008243%。
- 两个 input-dimension 分片分别算 dot product、各自舍入 BF16 后相加，与实际 TP2
  输出仅 **296 / 524288** 个元素不同，即 **99.9435% 元素逐位相同**；relative-L2
  为 **0.028461%**。相反完整 dot product 的模型与 TP 输出差异为 0.263252%。
- 该实验支持“partial output 的 BF16 舍入/归约顺序是 TP 投影主要差异来源”，但
  296 个剩余不同元素仍说明高精度模拟不是 TE GEMM 的逐位复制，不能声称解释 100%。

第 0 层 CP1/CP2 的输入 Q/K/V 相同，相对独立 FP32 causal GQA 数学参考，core output
误差分别 **0.172698% / 0.171171%**。CP2 并不比单卡参考明显偏离数学值；两者
相互的 0.099003% diff 是不同实现的数值结果差异，不能仅凭此宣称 CP 算错。
另外抽查逻辑层 1、2、16、30、44、45、46、47，两种路径相对各自真实输入的数学参考
误差均约 0.16%–0.19%，与自然整网最终 4%–6% 的差异不同量级。

参考公式核对 [HF Qwen3-MoE v4.57.1](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py)
的 GQA、head-dim scaling、causal softmax 和输出投影；诊断显式保留 FP32/FP64
中间量，不把该高精度数学参考称为 HF eager BF16 的逐位实现。

## 固定 CP=1：为什么 TP 最终 logits 差异仍达 4.56%

用户明确要求后续只关注 TP。本节不使用 CP2 的结果解释 CP1；对照是相同输入、
folded checkpoint 与 replay，CP=1 下 TP1 baseline 与 TP2/DP2/EP2。
既有 TP1/DP2/EP2 与 baseline bitwise exact，且 48 层同输入 attention core 全部 exact，
因此这里没有 CP 分布式 attention 的数值差异混入。TP=2 同时启用 SP，所以 row
projection 的 `reduce_scatter` 属于本次 TP/SP 分析范围。

**6006662**（EOS `eos0063`，CPU-only，1:01，COMPLETED / exit 0）只读取已有
`6006397`、`6006410`、`6006454` tensors，逐项校验所用 tensor hash，未执行模型或训练。
分析源码和输入 manifest/source hash 均记录在 `6006662/tp_growth.json`。

### 全局 relative-L2 与典型 token 的误差不是一回事

relative-L2 是 `||candidate - reference||₂ / ||reference||₂`，不是每个 logit 的
百分比误差，也不是概率或准确率差异。将全批 token 展平计算，相当于按 reference
token 的平方范数给每个 token 的相对误差平方加权。范数分布改变时，不能把边界间的
百分数比值全部解释为同一扰动被算子放大的倍数。

以下均为 CP1 / native TP2；层号从 0 开始：

| 位置 | 全局 relative-L2 | 逐 token relative-L2 中位数 |
| --- | ---: | ---: |
| Post-merge / 第 45 层输入 | 0.9043% | 0.7281% |
| 第 45 层输出 | 1.1344% | 2.4515% |
| 第 46 层输出 | 3.4605% | 3.0761% |
| 第 47 层输出 | 2.4134% | 3.2057% |
| Final norm 后 / `hidden_for_head` | 4.3384% | 3.9205% |
| Logits | 4.5585% | 3.9580% |

尤其第 45 层输出的 reference token norm 中位数为 17.99、最大为 1016.92，
并非所有 token 都只差 1.13%。到第 47 层，中位数/最大为 53.41 / 1468.49；
final norm 后变成 97.49 / 174.52，token 在全局指标中的权重显著改变。

离线仅把第 47 层的两个已存在 hidden 各 token 除以自身向量范数，不执行任何模型
kernel，relative-L2 就从 2.4134% 变为 **3.4267%**；若双方都除以 baseline 的每个
token 范数，结果为 **3.6202%**。真实 learned RMSNorm 后为 4.3384%。这些控制
显示 token 尺度重加权的重要性，但不是 learned RMSNorm 的精确模拟。
当时尚未排除 final norm/head 的同输入独立小差异；后续 `6006744` 已补齐该控制，
两者均逐位相同，真实 gamma 分析与独立 GPU norm 检查见上述细查文档。

### 第 46 层：真实绝对误差增长与 reference 残差相消同时发生

该层 input 的 `||reference|| / ||delta||` 为 **1461.35 / 16.58**。
Attention 后 residual 为 **1490.99 / 18.93**，MoE 分支为 **1078.72 / 30.39**。
baseline residual 与 MoE 分支的全局 cosine 为 **−0.743289**：二者相加后 reference
范数降至 **997.90**，而 delta 范数增至 **34.53**，使全局 relative-L2 达 3.4605%。

将两个分支的已捕获差向量在 FP64 中相加，delta norm 为 **34.4548**，实际 BF16
残差相加后的 delta norm 为 **34.5321**。因此该大跳变不是最后一次残差加法凭空制造
全部误差：大部分差异已存在于两个分支，且 reference 相消使分母缩小。第 47 层
delta norm 又增至 56.66，但 reference norm 增至 2347.63，所以相对数反而降至 2.4134%。

这与同输入下 48 层 MoE 全部 exact 不矛盾：被测自然路径的 MoE 输入已经不同，
固定专家的非线性网络仍可对这些不同输入产生不同输出。当前 replay 还固定了
dispatch-visible expert scores，不能用 Top-K 换专家来解释这组结果。

### 为什么 FP32 partial 归约诊断也没有让 logits 对齐

`6006454` 的 TP-only 诊断保留 BF16 GEMM 输入/权重，以 FP32 partial 输出做 TP 归约，
再舍入为 BF16；无 hidden 注入，不是正式训练路径。
它把 post-merge 全局误差从 0.9043% 降到 0.3563%，但 token 中位数只从
**0.7281% 降到 0.5961%**，最坏 token 反而从 **2.0935% 升到 5.7030%**。
第 45 层输出的 token 中位数仍为 2.3638%，最终 logits 为 4.5620%。

因此“early global-L2 降低很多”不等于每个 token 都改善同样倍数，也不保证后续
非线性路径更接近 baseline。这条反证禁止把“已找到 TP 首差”扩大成“改一次归约精度
即可消除整个 4.56%”。当前仍需针对 TP 投影及其扰动传播闭环；不能拿 CP2 解释、
放宽阈值，或称此误差天然可接受。生产代码、dtype、阈值在本轮均未改变。

后续 `6007959` / `6007990` / `6008118` 已完成 FP32 路径逐层逐 op 及真实 gamma
细查：token 156 在 L2 MLP norm/channel 940 跨 BF16 142.5 分界，输出 142→143；
L45 MLP norm 后早期 global hidden 优势基本消失；最终两条 error vector cosine
仅 0.162708。完整数据和对“归约顺序是全部原因”的限制见
[fp32-partial-propagation.zh.md](fp32-partial-propagation.zh.md)。

## 复现材料

- 本地相对目录：`runtime/diagnostics/diff_source/`，`probe.py` 为无注入原探针，
  sha256 `de3de92bbe37a705f50b692cbba0604ef23ab4a81d097442d8ae333781073571`。
- 自然轨迹摘要：`6006544/{baseline,tp_dp_ep,cp_dp_ep,all}.json`。
- 同输入隔离摘要及 manifest：`6006559/{tp_dp_ep,cp_dp_ep}.json` 和同名子目录；
  `isolate_operators.py` sha256 `55495910946fef40542f37f84e64485e82812721087e1ef4bf761ec09e241815`。
- 独立数学结果：`6006544/independent_math.json`（由第二个任务读取第一个任务的 trace 产生）。
- 原始 trace `.pt` 和全部 rank shard 留在 EOS，同相对目录，未将大文件复制到本地。
- EOS 项目：`/lustre/fsw/coreai_devtech_all/hongbinl/agentic-mcore-dev/.claude/worktrees/mor-mlite-validation/vendor/mor_mlite_bf16_train`。
- 启动通过公共 `cluster-run slurm` → `scripts/eos/run_bf16_environment.sh`，
  `run.sh` / `run_isolated.sh` 保留原环境与 checkpoint guard。
- 原阈值仍未通过；诊断任务 exit 0 只表示证据采集成功。
