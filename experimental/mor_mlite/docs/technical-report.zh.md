# MoR-MLite 技术报告：从 Mixture-of-Recursions 到分布式 Recurrent Transformer 训练

实验数据截至 2026-09-11；报告修订于 2026-09-16。本文按“方法—架构—实现—验证—结果”的主线组织，仅保留正式实现及主要验证结论，不展开中间数值定位和调试过程。

## 摘要

本项目以 Mixture-of-Recursions（MoR）为方法起点，在 Megatron-LM 的 experimental MLite runtime 上实现了一个支持参数共享、逐 token 动态递归深度及多种并行组合的训练扩展，包名为 `mor_mlite`，注册模型为 `qwen3_moe_mor`。

框架将 Transformer 分为前置层、共享递归层和后置层；通过每轮独立的 depth router 选择继续计算的 token，退出 token 保留其最后状态，递归结束后恢复完整序列。训练后端复用 MLite 的 Qwen3-MoE、TP/EP primitives、分布式优化器和 checkpoint API，并通过 MagiAttention 接入 CP。新增实现集中在递归执行、路由、活跃 token 布局变换、位置语义、全局训练目标和权重转换，不修改上游框架源码。

主要结果如下：

- 小模型完整分布式验收矩阵 **20/20 项报告通过**，覆盖 1/2/4/8 GPU、learned/replay 路由、前向、反向、优化器更新、CP 布局及 checkpoint 续训。
- 基于 Qwen3-30B-A3B-Base 构建的 folded MoR，具有 **20 个物理 Transformer 层、最多 48 次逻辑层调用、约 13.085B 个实际参数**；完成大模型跨进程完整状态恢复，恢复后继续一步与不中断训练的参数、优化器和 RNG 指纹一致。
- 真实 Pile 数据 100 步实验共尝试 **16 组配置**，其中 **7 组各完成 100 次 optimizer update**，第 100 步 LM loss 为 **6.24007–6.27442**；其余 9 组受到 H100 80GB 显存限制。
- 完成训练不等于跨拓扑数值完全一致：100 步实验观察到跨 TP、CP、EP、DP 的非零 loss 轨迹差异；本实验报告这些差异，不将完成训练视作数值 parity 通过。

因此，当前交付是一个具有分层验证证据的 recurrent transformer 分布式训练实现，而不是论文所有实验指标、长期收敛或全配置大模型数值一致性的完整复现。

## 1. 方法起点与复现范围

方法来源为 Bae 等人的论文 [Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation](https://arxiv.org/abs/2507.10524)。其核心是将跨深度参数共享与 token 级自适应计算结合：重复使用共享层，并让不同 token 经过不同数量的递归轮次。

本项目选取其中的 expert-choice、逐轮缩减活跃 token 的训练路线，工程目标是把这类动态递归计算接入现有大模型分布式训练栈。这里的 recurrent 指**深度方向上重复调用共享 Transformer 层**，不是跨数据 batch 保留 RNN 状态。

具体边界是：

| 维度 | 本项目实现范围 |
|---|---|
| 递归结构 | 可配置前置层、共享层、递归次数、后置层 |
| 深度分配 | 每轮独立 router，按样本选择 Top-K token，活跃集合逐轮嵌套 |
| Attention 语义 | 每轮仅对本轮活跃 token 计算；保留样本隔离和原始位置 |
| 模型后端 | 独立 PyTorch tiny reference；原生 Qwen3-MoE/MLite 分布式实现 |
| 并行 | 单节点 TP、CP、dense-DP、EP 与分布式优化器，PP=VPP=ETP=1 |
| 初始化 | 将既有 HF Qwen3-MoE checkpoint 折叠为共享层，新增 depth router |
| 验证目标 | 结构、路由、梯度与更新、状态恢复及真实数据短程训练 |
| 不在本次范围 | Token-choice 另一分支、推理 KV cache/KV sharing、论文的等 FLOPs 质量对比和推理吞吐复现 |

论文同时讨论了其他路由和 KV 策略；不能把本项目选定分支的训练实现描述为论文全部机制均已实现。[论文方法章节](https://arxiv.org/html/2507.10524v3#S2)

## 2. 从方法到训练框架的实现流程

实现按五个阶段推进，各阶段解决不同问题，而不是直接用大模型 loss 判断所有模块是否正确。

1. **定义可执行语义。** 固定物理层和逻辑层的关系、逐轮容量、Top-K 规则、gate 更新、退出 token 的状态，以及 packed sequence 的位置和 label/mask 合同。
2. **建立小规模语义实现。** 用独立 PyTorch 模型验证参数共享、嵌套路由、退出与合并、autograd 和 route replay，形成容易检查的参考路径。
3. **接入原生分布式算子。** 通过 MLite model registry 注册模型，保留原生 Qwen3-MoE 层和 optimizer，只替换其外层执行调度及必要适配。
4. **构建分布式训练与持久化。** 实现活跃 token 的可微重分布、全局目标归一化、HF 权重折叠，以及完整 checkpoint 恢复。
5. **逐级扩大验证。** 从 CPU/小模型测试进入 H100 全拓扑验收，再验证真实权重的大模型、跨进程续训，最后用同一真实数据训练 100 步并比较 loss 轨迹。

这里的独立 PyTorch reference 用于检查 MoR 语义与 learned/replay 自一致性；它和原生 Qwen/MLite 并非完全相同的网络实现，**没有将二者直接逐张量比较，也没有据此宣称与论文官方实现完成逐张量等价验证**。Tiny 跨拓扑数值比较使用同一 MLite 模型的单卡版本作为参考；大模型真实数据实验使用 TP1/CP1/DP4/EP4 训练作为主参考。

## 3. 框架架构

### 3.1 模型执行结构

以大模型配置 `3 + 14 × 3 + 3` 为例：

```text
Token IDs / packed metadata
           │
       Embedding
           │
   Start layers × 3                         所有 token
           │
   Router 0 → Shared layers × 14           容量 100%
           │
   Router 1 → Shared layers × 14           容量约 2/3
           │                └──退出 token → 保存最后 hidden
   Router 2 → Shared layers × 14           容量约 1/3
           │                └──退出 token → 保存最后 hidden
           └───────────┬─────────────────────────────┘
                       │
          按 global token ID 恢复完整序列
                       │
               End layers × 3             所有 token
                       │
             Final norm → LM head → Loss
```

图中三次 `Shared layers × 14` 调用的是**同一组 14 层及其同一批 Parameter**，不是三份权重。退出仅指退出后续共享递归轮次；退出 token 仍参与最后的后置层与语言模型预测。

若结构为 `(N_start, N_recur, R, N_end)`：

![物理层数与最大逻辑层数公式](/home/scratch.hongbinl_sw/work/fsdp/agentic-mcore-dev/.claude/worktrees/mor-mlite-validation/vendor/mor_mlite/docs/assets/technical-report/formula-1.png)

文字版：物理层数 = 前置层数 + 共享层数 + 后置层数；最大逻辑层数 = 前置层数 + 递归次数 × 共享层数 + 后置层数。

当前大模型为 20 个物理层、最大逻辑深度 48；根据退出轮次，每个 token 经过的 Transformer 层数为 20、34 或 48。层内仍有原生 attention/MLP residual，递归块外另有 depth-gated residual。

### 3.2 软件职责划分

| 层次 | 主要模块 | 职责 |
|---|---|---|
| 配置与数据 | `config.py`、`config_loader.py`、`data.py` | 结构/并行配置，packed token、labels、mask 和原始位置 |
| 模型接入 | `register.py`、`qwen3_moe_mor/protocol.py` | MLite registry、模型构建、batch/forward/checkpoint 协议 |
| 递归计算 | `qwen3_moe_mor/model.py` | 物理共享层、逐轮执行、退出合并、最终 LM loss |
| 深度路由 | `routing/depth_router.py`、`routing/plan.py` | 打分、容量、Top-K、gate、BCE auxiliary、RoutePlan |
| 活跃布局 | `distributed/`、`qwen3_moe_mor/execution.py` | CP/TP-SP token 布局、可微迁移、parking、inverse merge |
| 训练目标 | `objective.py` | DP 与 microbatch 的全局 token 加权归一化 |
| 权重与状态 | `convert_hf.py`、`checkpoint_io.py`、`qwen3_moe_mor/checkpoint.py` | HF folding、分片加载、模型及训练状态保存/恢复 |
| 验证 | `tiny/`、`parity/`、`provenance.py` | 参考实现、逐项比较、完整性检查及实验源码绑定 |

底层 Qwen attention/MoE、TP/EP 通信和 optimizer 复用 MLite/MCore。CP>1 使用 MagiAttention；当前 strict CP=1 路径使用本包的本地 BF16 FFA attention adapter，QKV、QK norm、RoPE 和 output projection 仍沿用原生模块。

该实现是 out-of-tree 扩展，不要求给 Megatron-LM 或论文参考仓库打源码补丁。但 `magi_direct` 的目标布局解码依赖 MagiAttention 1.1.1 的内部 runtime metadata，因此版本升级需要重新验证，并非任意版本即插即用。

## 4. 关键实现

### 4.1 参数共享与递归更新

构建时只实例化 `start + recurrent + end` 的物理层。`recurrent_layers` 是同一 `ModuleList` 的切片视图，每轮重复调用；optimizer 和 checkpoint 都按物理参数管理，不按逻辑调用次数复制状态。

对继续计算的 token，本实现使用：

![路由打分、gate 与递归残差更新公式](/home/scratch.hongbinl_sw/work/fsdp/agentic-mcore-dev/.claude/worktrees/mor-mlite-validation/vendor/mor_mlite/docs/assets/technical-report/formula-2.png)

文字版：`z[r,t] = dot(w[r], h[r,t]) / T`；`g[r,t] = alpha × sigmoid(z[r,t])`；`h[r+1,t] = h[r,t] + g[r,t] × F_theta(h[r])[t]`。

`F_theta` 表示包含 attention 和 MoE 的整个共享层栈。其权重在轮次间共享；`w_r` 是每轮独立的 depth-router 投影。默认 `T=1`、`alpha=0.1`。

router 打分使用 FP32 计算，更新前将 gate 转为模型激活 dtype；BF16 模型参数和 hidden 保持 BF16。标准 FP32 gradient/master/Adam state 保留，未引入额外的 FP32-heavy forward 或 FP8 训练路径。

共享参数接收所有实际调用路径的 autograd 梯度贡献。完成一个 optimizer step 所需的 microbatches 后，由原生 distributed optimizer 进行同步和更新；不在每个递归轮次单独更新共享权重。

### 4.2 嵌套的 token 深度路由

设一个样本原始长度为 `L`，递归总轮数为 `R`，轮次从 0 开始。线性容量预算为：

![每轮活跃 token 容量公式](/home/scratch.hongbinl_sw/work/fsdp/agentic-mcore-dev/.claude/worktrees/mor-mlite-validation/vendor/mor_mlite/docs/assets/technical-report/formula-3.png)

文字版：`K[r](L) = max(1, floor((R − r) × L / R))`，适用于 `L > 0`；`floor` 表示向下取整。

预算相对于原始长度，而不是上一轮长度。三轮、长度 256 的样本分别选择 **256、170、85** 个 token；后轮仅能从前轮留下的 token 中选择。模型因此具有严格的 early-exit 行为，退出 token 不会再次进入更深递归轮。

选择按样本独立进行，以 FP32 decision logit 排序，完全同分时按 global token ID 决定顺序。选中后再按样本和原始位置排序，恢复 attention 所需的因果顺序。`RoutePlan` 显式保存 token 身份、原始位置、容量、gate 和分布式目标布局。

这有两个不同层次的路由，不能混淆：

- **Depth router** 决定某 token 是否再经过一轮共享层。
- **原生 MoE expert router** 决定该 token 在某一 Transformer 层中进入哪些 FFN experts。

当前大模型每个 MoE 层有 128 个 experts、Top-8；递归层被重复调用时，原生 expert routing 正常重新执行。100 步真实数据训练没有冻结或回放这两类路由。

### 4.3 活跃 token 与 CP 重分布

动态路由改变了各 rank 的 token 数及分布。仅修改 attention mask 而保留所有 token 参与矩阵运算，不能实现这里的执行语义；本框架会真正构造更小的活跃 token batch。

每个 token 始终携带 `sample_id / original_position / global_token_id / source_rank / source_row`。深度路由在同一 dense-DP replica 的 TP-SP×CP 范围汇总标量分数与必要元数据，统一形成按样本的选择结果；不把不同 DP 样本混在一起竞争容量。

当活跃集合缩减时，系统将退出 token 的 hidden 保存在其当前位置，并把继续计算的 hidden、gate 和元数据送往下一轮的目标布局。迁移使用支持 autograd 的 variable-size All-to-All，反向能够沿对应逆映射传回梯度。

实现约束是：

- 第一轮容量为 100%，不额外触发一次深度路由 hidden 重平衡。
- 每次活跃集合变化的轮次边界只进行一次 hidden payload 重平衡事务。
- 同一共享层栈的内部不再重复做深度路由 dispatch；原生 TP、MoE、attention 通信仍正常存在。
- 已退出 token 不进入后续递归轮的 Q/K/V 或 MoE 计算。
- 递归结束后，将退出和仍活跃的 token 按原始身份合并，恢复后置层的完整输入。

“一次重平衡”描述的是 hidden 数据迁移，不是整个事务只调用一个 collective；计数、gate、metadata，以及原生 attention/TP/EP/optimizer 的通信需要分别理解。

### 4.4 Packed sequence、位置与训练目标

token 被压缩和换 rank 后，其原始位置不能重编号为新的连续位置。框架保留原始 position IDs 供 RoPE 使用，并为当前活跃集合重新构造 packed sequence 边界，防止不同样本互相 attention。dummy padding 不参与有效 loss、路由容量或真实 token 身份比较。

当前训练目标为 native token-level LM cross entropy 加各递归轮的 depth-router BCE auxiliary。BCE 以本轮被选中/未选中的候选 token 为目标，按候选数归一化，系数为 `0.001`。固定版本的 native Qwen MoE 使用 `compute_aux_loss=False`，所以这里没有额外的普通 MoE load-balancing auxiliary 项。

不同 DP replica 和 microbatch 可能含有不同数量的有效 token。公共 `objective_scales` / `apply_objective` 按完整 optimizer step 计数，为各局部均值施加：

![全局训练目标的局部缩放系数公式](/home/scratch.hongbinl_sw/work/fsdp/agentic-mcore-dev/.claude/worktrees/mor-mlite-validation/vendor/mor_mlite/docs/assets/technical-report/formula-4.png)

文字版：`s[d,m] = D × M × n[d,m] / N_total`；`N_total` 是完整 optimizer step 内所有 dense-DP replica、所有 microbatch 的该项计数之和。

其中 `D` 为 dense-DP 度数，`M` 为 microbatch 数，`n` 为该项真实分母。LM 使用 shifted mask 的有效权重和；每轮 router auxiliary 使用该轮候选 token 数。该缩放抵消 runtime 的 microbatch 平均和 optimizer 的 DP 平均，得到所需的全局加权目标。CP 的局部贡献由模型内部处理，不重复把 TP/CP 副本计入样本数。

该机制已通过不等长 DP/microbatch、局部零权重及 loss/gradient 对照测试。真实数据 100 步采用等长全局 batch，但复用同一实现。

### 4.5 HF 权重折叠与 checkpoint

初始化输入为 `Qwen/Qwen3-30B-A3B-Base` 的 48 层 checkpoint。保留最前 3 层和最后 3 层，将中间 42 层折叠为 14 个共享层。对共享层索引 `i=0…13`：

![三个逻辑层折叠为一个共享层的权重公式](/home/scratch.hongbinl_sw/work/fsdp/agentic-mcore-dev/.claude/worktrees/mor-mlite-validation/vendor/mor_mlite/docs/assets/technical-report/formula-5.png)

文字版：`W_shared[i] = cast_BF16((W[3+i] + W[17+i] + W[31+i]) / 3)`。

对应张量以 FP32 求均值，再按目标 dtype 和原生 Qwen WeightSpec 处理 QKV packing、TP 切片及 expert 映射。新增的 3 个 depth routers 用固定 seed 初始化。读取和折叠采用流式方式，不要求在每张 GPU 同时物化完整源模型。

折叠会改变网络函数，因此该初始化不是与原始 48 层 Qwen 完全等价的无损变换。报告中的约 13.085B 是折叠后模型的实际参数量；“30B”说明源 checkpoint，而不是宣称当前训练了 30B 个互不共享的参数。

checkpoint 使用 MLite 的 distributed checkpoint 路径，并保存递归结构、folding、router、并行和运行策略元数据。验证覆盖两类入口：

- **模型冷启动：** 加载模型权重，optimizer/RNG 从新状态开始；100 步比较采用此方式。
- **完整状态续训：** 恢复模型、FP32 master weights、Adam moments/step 和各 rank RNG，再执行下一步，与不中断路径比较。

EP-local expert key 的布局使当前实现不能直接跨 EP 度数加载同一 folded DCP。跨 EP 实验分别从同一源权重生成相应 checkpoint，再对实际加载后的完整参数做逐字节检查，不能仅以文件名判断初始化一致。

## 5. 验证设计与配置

### 5.1 三组实验：目的、比较对象与计数

本报告的主线是三组**独立实验**，不是同一矩阵逐步加大模型或延长步数。先区分它们，再看具体超参数和结果：

| 实验 | A：Tiny 完整训练验收 | B：大模型 checkpoint 续训 | C：真实 Pile 100 步 |
|---|---|---|---|
| 要回答的问题 | 并行化是否保持递归语义、梯度和参数更新 | 进程退出后是否完整恢复训练状态 | 大模型各拓扑能否持续训练，loss 轨迹相差多少 |
| 路由模式 | 8 个分布式组合各测 learned 和 replay；另有 4 项专项 | **Depth replay**，受控续训比较 | **全部 learned**，depth/native expert 均正常路由 |
| 比较对象 | 同初始化、同全局输入的单卡与多卡 MLite；专项另列 | **同一拓扑**下，不中断路径与独立进程恢复路径 | 同初始化、同逐步全局输入的不同拓扑 |
| 主参考 | TP1/CP1/DP1/EP1 | TP2/CP2/DP2/EP4 的不中断路径 | TP1/CP1/DP4/EP4；另做单轴配对 |
| 更新长度 | 1 个主更新；另比较保存后下一步 | 第 1 次更新后保存，再比较第 2 次更新 | 每个配置从冷启动训练 100 次更新 |
| 数量如何计算 | `8×2+4=20` 项**验收报告** | 1 个拓扑的**恢复实验**，不计入 A 的 20 项 | 16 个**拓扑配置**：7 个完成、9 个 OOM |
| 判定方式 | 张量数值门槛及结构/状态检查 | 恢复点及继续一步的模型、optimizer、RNG 指纹一致 | 完成 100 次更新、记录完整且数值有限；另报告 loss 差，不设 MAE 通过线 |
| 作业 | `6006390` | `6006440` | `6012240 / 6012241 / 6012301` |

**Learned** 指运行时由模型自行计算深度分数并选择 token；**replay** 指消费已记录的深度路由，用于控制离散执行路径。B 是状态恢复实验，不是 learned 模式的真实数据训练；C 不回放 A 或 B 的路由。三组结果不能相互替代，也不能把 20 和 16 相加作为“通过的并行配置数”。

各类检查的职责如下：

| 层次 | 核心问题 | 比较对象 |
|---|---|---|
| 语义与单测 | 共享、选择、退出、位置和目标是否符合实现合同 | 独立 PyTorch 路径、显式数学目标及输入不变量 |
| Tiny 分布式数值 | 并行实现是否保持前向、梯度和更新 | 同初始化、同全局 batch 的单卡 MLite 与多卡 MLite |
| Loaded 参数 | 不同分片布局是否实际加载相同权重 | 所有 Parameter 的原始 bytes 及 TP/EP 覆盖 |
| 大模型状态恢复 | 能否准确恢复并继续训练 | 独立进程恢复路径与不中断下一步 |
| 真实数据训练 | 不同拓扑能否持续更新，loss 轨迹差多大 | 同 checkpoint、同逐步输入的 100 步训练 |

相对 L2 定义为 `||candidate-reference||₂ / ||reference||₂`。它衡量张量差异，不是 loss 差。
Tiny BF16 forward 门槛为 relative-L2≤2%、cosine≥0.999；完整梯度及 FP32 master-update 向量为 relative-L2≤3%、cosine≥0.999；loss 绝对差≤0.01。梯度和更新按阶段/step 重建完整向量，不能解释为每个参数分别都满足同一相对误差。

上述数值门槛仅用于 A 的 tiny 验收，不用于 B 的状态指纹检查，也不直接用于 C 的 100 步训练轨迹。C 中模型从相同参数出发，但每次更新后各自演化，因此它比较的是训练轨迹，而非每步重置到相同参数的单步算子测试。

### 5.2 模型与运行环境

| 项目 | A：Tiny 数值验收 | B：大模型 checkpoint 续训 | C：真实数据 100 步 |
|---|---|---|---|
| 模型 | 原生 Qwen-MoE tiny fixture | Qwen3-30B-A3B-Base folded MoR | 同一 folded MoR |
| 结构 | `1 + 2×3 + 1` | `3 + 14×3 + 3` | `3 + 14×3 + 3` |
| 物理/最大逻辑层 | 4 / 8 | 20 / 48 | 20 / 48 |
| Hidden size | 256 | 2048 | 2048 |
| Experts / Top-K | 4 / 2 | 128 / 8 | 128 / 8 |
| 输入 | synthetic，序列长度 `[9,6,3]` | synthetic，`[128,128]` | Pile，4 条 256-token chunk/step |
| 更新设置 | 1 个主训练 step；另测保存、不中断/恢复下一步 | 1 步保存 + 下一步续训 | 每组 100 次更新 |
| Microbatches | 2 | 1 | 1 |
| 初始 LR / Adam eps | `1e-3 / 1e-6` | `1e-3 / 1e-6` | `1e-5 / 1e-6` |
| Seed / clip_grad | `1234 / 1` | `1234 / 1` | `1234 / 1` |
| 模型精度 | BF16；另有独立 FP32 reference | BF16 | BF16 |

Tiny fixture 用合成的明确 expert cutoff 保持原生 Top-K 的可检查性；它保留 expert 计算、路由梯度和 EP 通信，但不代表真实 checkpoint 的路由分布。

GPU 环境为 EOS 单节点 H100 80GB，按实验使用 1/2/4/8 ranks；NGC PyTorch 26.01 容器内使用固定 cu129 运行环境：Torch 2.10.0+cu129、CUDA runtime 12.9、TE 2.13.0、NCCL 2.27.5、MagiAttention 1.1.1。Megatron-LM 固定为 `5c8315f12a64a7279eec58896af9e74ee3351b74`。

各实验保留自身 determinism 配置，不把所有历史任务视为同一模式。Tiny `6006390` 及大模型续训 `6006440` 为 `strict=True`，记录的 Magi distributed deterministic 环境值为 0；100 步实验为 `strict=True` 且显式设置 `MAGI_ATTENTION_DETERMINISTIC_MODE=1`。开启该模式不意味着不同并行拓扑必须 bitwise 相同，也不替代同配置重复运行的方差实验。

### 5.3 并行拓扑如何理解

固定 PP=ETP=1 时，总 ranks 按 `TP×CP×dense-DP` 计算。EP 是在同一 world 上的另一种分组视图，不应再乘一次。例如 TP2/CP2/DP2/EP4 使用 8 ranks，而不是 32 ranks。

原始 tiny 矩阵包括以下参考及组合：

| 名称 | TP | CP | dense-DP | EP | Ranks |
|---|---:|---:|---:|---:|---:|
| baseline | 1 | 1 | 1 | 1 | 1 |
| zero1 | 1 | 1 | 2 | 1 | 2 |
| tp | 2 | 1 | 1 | 1 | 2 |
| cp | 1 | 2 | 1 | 1 | 2 |
| ep | 1 | 1 | 2 | 2 | 2 |
| tp_cp_ep | 2 | 2 | 1 | 2 | 4 |
| tp_dp_ep | 2 | 1 | 2 | 2 | 4 |
| cp_dp_ep | 1 | 2 | 2 | 2 | 4 |
| all | 2 | 2 | 2 | 4 | 8 |

“20/20”统计的是 **tiny 验收报告数量**：上述 8 个分布式组合，每个都执行 learned 和 replay 两种模式，共 16 项，再加 4 项专项验证。baseline 是这些比较的参考，不单独计作一个分布式比较项。

- **Learned：** 候选模型自己计算 depth-router 分数并选择继续递归的 token，检查正常路由下的训练行为。
- **Replay：** 候选模型消费单卡参考记录的深度 RoutePlan（包含 token 选择和 gate 值），固定递归路径，检查相同路径下的分布式计算、梯度和更新。Replay 是受控验收工具，不是本项目真实数据训练采用的模式。

因此，20 既不是不同拓扑的数量，也不是训练步数。它与后文真实 Pile 的“7 组完成、9 组 OOM”属于两套实验，统计单位不同：

| 实验 | 模型/数据 | 路由模式 | 计数单位与结果 |
|---|---|---|---|
| Tiny 完整验收 | 小模型、短序列 synthetic 数据 | 8 个分布式组合各测 learned/replay；另有专项 | `8×2+4=20` 项报告，20 项通过 |
| 真实 Pile 100 步 | 约 13.085B 参数的 folded 模型、真实文本 | **全部计划使用 learned**，depth 和 native expert 路由均不冻结 | 16 个拓扑配置：7 个完成 100 步，9 个 OOM |

两套实验共有一部分拓扑，但 Pile 并不是把 tiny 的 20 项报告逐项延长到 100 步，也没有再运行那 4 项专项。OOM 的配置未完成首次更新，不计作训练通过；7 个完成组也仅表示完成所测训练及记录检查，不表示取得完整数值 parity 证书。

### 5.4 为什么需要额外的 4 项专项验证

前面的 16 项主要回答“改变并行拓扑后，正常路由及固定路由的训练计算是否对齐”。它们不能单独充分验证 replay 工具自身、另一种 CP 布局路径，以及进程退出后的训练状态恢复，因此补充下列检查：

| 专项 | 具体怎么比较 | 为什么需要：补充覆盖的风险 |
|---|---|---|
| FP32 reference replay | 在 PyTorch 小型参考模型中先运行 learned，再回放同一路径，比较输出、梯度和更新 | Replay 是其他比较的控制工具，需先确认回放没有遗漏轮次、错配 gate 或改变应有的梯度路径。它检验参考路径的自一致性，不是与官方论文实现的独立等价证明。 |
| canonical CP | 先把活跃 token 整理为统一的 canonical batch，再通过 Magi 公共接口分发；与单卡 MLite 参考比较 | 常规 CP 组合使用 direct 路径，不能据此认为 canonical 路径也正确。需要单独检查其打包、样本边界、token 顺序和 CP 计算。 |
| canonical/direct CP 对照 | 在相同模型、输入和受控路由下，直接比较两种布局执行方式 | 检查 direct 直接迁移到目标 rank/slot 的实现是否保持 canonical 路径的语义。这是两种布局实现的等价检查；与上一项的“canonical 对单卡参考”比较对象不同，两项形成交叉约束。 |
| 外部进程 checkpoint | 保存完整状态，退出进程；新建独立进程恢复并继续一步，对比不中断路径 | 单次训练对齐不能证明状态被完整保存。新进程可排除内存中残留状态的影响，并检查参数、FP32 master、Adam moments/step 和 RNG 是否真正恢复。 |

这四项不是为了增加并行配置数量，而是分别检查**验收工具、CP 两种布局实现、训练持久化**。它们与 8 个拓扑的 learned/replay 比较互补，但不能替代大模型和真实数据验证。

## 6. 主线结果

### 6.1 Tiny：完整训练验收通过

作业 `6006390` 的完整矩阵 20/20 报告通过；同一源码快照的测试套件为 **374 passed / 1 skipped**。唯一 skip 是原有 opt-in CLI smoke，实际 GPU MLite 训练由矩阵独立执行。本报告生成时重新校验了全部 20 份报告的 SHA256 与矩阵凭据一致。

全矩阵最坏的硬门槛指标如下：

| 指标 | 最坏观测值 | 门槛 | 结果 |
|---|---:|---:|---|
| Forward relative-L2 | 0.569257% | ≤2% | 通过 |
| Loss tensor 最大绝对差 | 0.000263214 | ≤0.01 | 通过 |
| 完整 gradient 向量 relative-L2 | 0.300522% | ≤3% | 通过 |
| 完整 gradient 向量最低 cosine | 0.999995503 | ≥0.999 | 通过 |
| FP32 master-update 向量 relative-L2 | 1.192856% | ≤3% | 通过 |
| FP32 master-update 最低 cosine | 0.999928863 | ≥0.999 | 通过 |

同时通过物理层共享、路由、活跃 token 覆盖、退出语义、CP 布局及 gradient-sync 调度检查。这里的同步检查观测原生 bucket 的同步调度，不宣称测量了全部 NCCL kernel 数量。

这组证据支持：所实现的递归执行、路由与分布式训练机制在 tiny 验收范围内正确接入，且梯度/更新并非只凭 loss 接近推断。

### 6.2 大模型：初始化一致与完整状态续训通过

大模型加载后参数检查覆盖 **5,266 个 canonical parameter tensors、13,084,750,848 个元素**。参数按真实 TP 切片和全局 expert ID 对应，所有副本参与检查，结果为 **different_bytes=0**。100 步实验的 7 个完成组均在第一次 forward 前通过该检查，包含所有 CP 副本。

作业 `6006440` 使用 synthetic `[128,128]` 输入、1 个 microbatch、depth replay，在 TP2/CP2/DP2/EP4 上完成首次 forward/backward/optimizer update，保存完整状态。不中断路径和恢复路径使用相同 replay 合同；下一步按记录的 `exact-step-or-final-selection-with-current-gates` 策略回放深度选择，不应理解为所有后续 gate 永远固定。原生 MoE expert routing 正常执行。这不是后文 Pile 的 learned 训练实验。

另启独立 8-rank 进程恢复后：

- 保存点的模型、优化器和 RNG 指纹一致。
- 继续执行下一步后，与不中断路径的模型、优化器和 RNG 指纹仍一致。
- 下一步 optimizer 更新成功。

这证明该拓扑的 checkpoint 确实恢复了训练状态，而非仅重新加载模型后使用新 optimizer 继续运行。它不证明不同拓扑之间每个梯度或 optimizer update 都相同。

### 6.3 真实 Pile：7 组完成 100 步

数据使用现有 Pile indexed shard 的顺序文本前缀，先按原 GPT-2 tokenizer 解码并检查回环，再用与目标 Base 模型词表一致的 Qwen tokenizer 编码；没有直接把 GPT-2 token IDs 输入 Qwen。使用 raw text、不套聊天模板、Base EOS=151643，文档串接后切为固定长度 chunk。

100 步共取源记录 0–93，冻结 token tensor `[100,4,256]`：每步 1,024 个输入 token、1,020 个有效 next-token targets，总计 102,400 个输入 token、102,000 个 targets。前 50 步输入与上一轮 50 步实验逐位相同。

所有组从相同实际权重和全新 optimizer 开始；GBS=4、每步一个 global microbatch、LR=1e-5、Adam eps=1e-6、clip_grad=1、scheduler horizon=100。没有从旧 50 步末尾续跑。DP=1/2/4 时，每个 replica 每步分别处理 4/2/1 条序列；TP 和 CP 不增加独立训练样本数，因此所有拓扑逐步看到的全局 4 条输入及有效 targets 完全一致。

作业 `6012240 / 6012241 / 6012301` 的 16 个拓扑来源如下，均使用 learned 训练配置，不使用 replay：

- 原始矩阵 9 组：单卡 baseline 加 8 个分布式组合。
- 既有 TP4 组 1 组：TP4/CP1/DP2/EP2。
- 此前真实数据实验 3 组：固定 DP4/EP4 的 TP1/CP1、TP2/CP1、TP1/CP2。
- 新增单轴对照 3 组：TP1/CP1/DP4/EP2、TP2/CP1/DP2/EP4、TP4/CP1/DP2/EP4。

合计 `9+1+3+3=16` 个配置。其中 7 组完成各 100 步，共 700 次总更新，全部记录连续、loss/grad norm 有限、每步 update 成功；另外 9 组 OOM，未完成首次更新。因此这里的 **7+9 是完成/未完成的配置数量划分，与 tiny 的 20 项报告不是同一计数**。

下表主参考为 TP1/CP1/DP4/EP4，**不是无并行单卡参考**。
**MAE（Mean Absolute Error）在这里指“逐步 loss 差的平均绝对值”**，不是候选运行本身的平均 loss，也不是有符号差的平均值。

计算方式为：先取相同步数、相同输入上的 `Δ_t = loss_candidate,t − loss_reference,t`，对每个差取绝对值，再将 100 个绝对值相加除以 100，即 `MAE = (|Δ_1| + … + |Δ_100|) / 100`。例如两步差分别为 `+0.1` 和 `−0.1`，有符号平均差为 0，但 MAE 为 0.1，不会因正负抵消而掩盖差异。

因此表中 `MAE=0.059937` 表示该配对在 100 步中，平均每步 LM loss 相差约 0.059937；这是 loss 的绝对差单位，**不是 5.9937%**。它概括整条训练轨迹，而“最终 Δ”仅描述第 100 步。本实验报告观测差异，没有为 100 步 MAE 另外设定通过阈值，也不把 tiny 的单步 loss 门槛直接套到独立演化的训练轨迹上。

第 100 步 loss 是该步更新前的训练 batch loss，该次更新随后成功；不是固定验证集的最终 loss。

| TP/CP/DP/EP | 第 100 步 LM loss | 最终 Δ | 100 步 MAE | 最大绝对 Δ（step） | 峰值 allocated GiB |
|---|---:|---:|---:|---:|---:|
| 1/1/4/4（参考） | 6.240067 | 0 | 0 | 0 | 65.05 |
| 2/1/4/4 | 6.274415 | +0.034348 | 0.059937 | 0.455907（2） | 43.16 |
| 1/2/4/4 | 6.241386 | +0.001319 | 0.042937 | 0.325671（4） | 46.19 |
| 2/2/2/4 | 6.252738 | +0.012671 | 0.047335 | 0.421156（4） | 43.17 |
| 4/1/2/2 | 6.254544 | +0.014477 | 0.041387 | 0.456071（2） | 62.64 |
| 2/1/2/4 | 6.257241 | +0.017174 | 0.045158 | 0.458784（2） | 62.04 |
| 4/1/2/4 | 6.242172 | +0.002105 | 0.056502 | 0.353976（3） | 41.67 |

显存列为训练阶段各 rank、各 step 的 PyTorch allocated 峰值，不是整卡占用，不包含全部通信/driver 内存，也不覆盖模型加载峰值。本实验不是正式吞吐 benchmark。

![真实 Pile 100 步训练 loss 及相对参考的逐步差值](/home/scratch.hongbinl_sw/work/fsdp/agentic-mcore-dev/.claude/worktrees/mor-mlite-validation/vendor/mor_mlite/runtime/ckpt_tools/real-data-100/report/loss-curves.png)

为区分并行维度，另按只改变一个配置轴比较：

| 改变的轴 | 固定配置 | 最终 Δ（右减左） | 100 步 MAE |
|---|---|---:|---:|
| TP1 → TP2 | CP1 DP4 EP4 | +0.034348 | 0.059937 |
| CP1 → CP2 | TP1 DP4 EP4 | +0.001319 | 0.042937 |
| CP1 → CP2 | TP2 DP2 EP4 | −0.004503 | 0.044313 |
| TP2 → TP4 | CP1 DP2 EP4 | −0.015069 | 0.051254 |
| EP4 → EP2 | TP4 CP1 DP2 | +0.012372 | 0.030587 |
| DP4 → DP2 | TP2 CP1 EP4 | −0.017174 | 0.051205 |

这些结果表明：7 组均能完成所测短程训练，但 TP、CP、EP、DP 的受控配对均存在非零训练轨迹差异。以 CP1→CP2、TP1/DP4/EP4 为例，终点相对差只有约 0.0211%，但整段 MAE 为 0.04294，不能只看终点判定一致。

六个单轴配对的后 50 步 MAE 均低于前 10 步，本次短程实验未呈现误差随步数单调扩大的现象；这不等价于长期收敛保证。每步使用不同文本，因此 loss 曲线整体下降也不能单独量化模型质量改进；还缺少固定 held-out 集、多 seed 和等计算预算基线。

### 6.4 完整覆盖中的未完成项

其余 9 组均实际尝试，但在首次更新完成前遇到显存不足，没有产生可报告的 100 步 loss：

| 配置名称 | TP/CP/DP/EP | 本轮状态 |
|---|---|---|
| baseline | 1/1/1/1 | OOM，0 次完成更新 |
| zero1 | 1/1/2/1 | OOM，0 次完成更新 |
| tp | 2/1/1/1 | OOM，0 次完成更新 |
| cp | 1/2/1/1 | OOM，0 次完成更新 |
| ep | 1/1/2/2 | OOM，0 次完成更新 |
| tp_cp_ep | 2/2/1/2 | OOM，0 次完成更新 |
| tp_dp_ep | 2/1/2/2 | OOM，0 次完成更新 |
| cp_dp_ep | 1/2/2/2 | OOM，0 次完成更新 |
| ep2_dp4 | 1/1/4/2 | OOM，0 次完成更新 |

Tiny 上某拓扑数值通过，不意味着完整模型同拓扑能够装入 H100 80GB。未通过缩模型、修改精度或改用其他并行度替代这些原配置；它们仍属于完整大模型 100 步矩阵的未完成项。

## 7. 已验证能力与交付边界

当前最有证据支持的结论是：

1. 已实现一个可配置的、真正共享物理参数的 recurrent transformer 训练扩展，并支持 token 级深度选择、退出和恢复。
2. 已将该执行方式接入单节点 MLite 的 TP/CP/DP/EP 和 distributed optimizer；小模型完整训练验收通过。
3. 已完成真实 Qwen 来源权重的 folding 和全量加载一致性检查，并验证大模型完整状态的跨进程续训。
4. 已用真实数据在 7 个大模型并行配置上完成各 100 步训练，并报告跨配置 loss 差异及其适用范围。

尚不能宣称的能力包括：全部大模型 topology 数值验收通过、全部 16 组均完成训练、论文质量/等 FLOPs 优势复现、长期收敛、长序列或多节点性能、推理 KV cache，以及 FP8/MXFP8 等低精度路径。

当前 `mor_mlite.train` 主要是 synthetic acceptance frontend。真实 Pile 训练通过独立实验入口调用同一模型/runtime 和公共 objective adapter，尚未交付通用数据加载与生产训练 frontend。PP/VPP>1、ETP>1、ZeRO-2/3、activation recomputation 和多节点等也不在当前验证范围。

Attention 的因果 mask 与整条样本上的 Top-K 深度选择是两个不同问题：后者在训练时使用样本内候选分数，不能直接外推为已验证的严格在线自回归路由。当前没有通过生成任务或推理吞吐证明这方面能力。

## 8. 复现与证据索引

以下证据绑定同一生产源码内容 SHA256：
`c49e48c4d882aa7e8f0a3b3ea98d42711ba260f42cc0bbd21d8164618b870c87`。
该内容 hash 覆盖实际 Python 源码，包括未提交改动，比仅记录一个历史 git commit 更能准确描述本次实验实现。

| 主验证 | 作业 / 证据 |
|---|---|
| Tiny 完整矩阵 | `6006390`；[20 项报告凭据](../artifacts/eos/6006390/tiny/reports/matrix_complete.json)、[训练数值报告](../artifacts/eos/6006390/tiny/reports/all.json) |
| 测试套件 | [pytest XML](../artifacts/eos/6006390/tiny/reports/pytest.xml) |
| 大模型完整状态续训 | `6006440`；[checkpoint certificate](../artifacts/bf16-6006440/qwen30b_train/reports/checkpoint_external_resume.json) |
| 加载后参数一致性 | `6011449` 及 100 步各组 manifest；[参数覆盖记录](loaded-tp-parameters-bitwise.zh.md) |
| 真实数据 100 步 | `6012240 / 6012241 / 6012301`；[汇总 JSON](../runtime/ckpt_tools/real-data-100/report/report.json)、[700 行逐步 CSV](../runtime/ckpt_tools/real-data-100/report/loss-comparison.csv) |
| 实验配置与原始产物说明 | [100 步实验记录](real-data-100steps.zh.md) |

关键实现入口：[递归模型](../src/mor_mlite/qwen3_moe_mor/model.py)、[深度路由](../src/mor_mlite/routing/depth_router.py)、[活跃状态转换](../src/mor_mlite/distributed/transition.py)、[目标适配](../src/mor_mlite/objective.py)、[HF 转换](../src/mor_mlite/convert_hf.py)。

100 步训练实际执行脚本 SHA256 为 `2583265c46c4b28cd87071ceac07f8e9d53e7dd4a19d1e56a21e25a02f12062e`，token tensor 内容 SHA256 为 `3e5122e662650ee8432165b6a139f354092367669d218a813706935da8d904b6`。每个完成组保存执行脚本副本、输入 hash、参数检查、loss、gradient norm、更新状态和完成凭据。

## 9. 总结

本项目完成了从论文方法到可执行分布式训练系统的主要工程链条：**明确递归语义 → 实现共享层与 token 路由 → 处理动态活跃布局 → 接入原生训练和 checkpoint → 从 tiny 数值验收到大模型真实数据训练**。

成果的核心不是某一次 loss 看起来接近，而是同一套实现具备了结构、数值、状态恢复和短程训练四类相互区分的证据。当前可以把 MoR-MLite 作为 recurrent transformer 的研究训练基础继续迭代；完整大模型数值验收、显存受限配置补齐，以及质量与性能评估，仍是后续明确的工作范围。
