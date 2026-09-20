# MoR MLite 遗留问题修复设计

本轮承接任务 `01a0799b-f8a1-7171-ae82-809b58c678c1` 的审查。代码基线来自 EOS
`/lustre/fsw/coreai_devtech_all/hongbinl/mor_mlite`，上游固定为 Megatron-LM
`5c8315f12a64a7279eec58896af9e74ee3351b74`、MagiAttention `v1.1.1`。

## 行为约束

- 验收器根据运行配置和初始化参数集合推导必需证据；完整训练验收必须包含每个
  step/microbatch/round 的 forward、loss、RoutePlan 和每个参数的 gradient/update。
  局部诊断必须显式声明，不能取得完整矩阵证书。
- replay 必须完整覆盖所有递归轮；选择必须满足本轮容量，真实 token 的 ID、sample、
  original position 必须与输入一致。错误 oracle 在执行递归前失败。
- reference PackedBatch 与原生协议使用一致的 unshifted label/mask 合同；None mask
  包含所有真实 token，显式 mask 按序列左移并屏蔽最后目标。sample ID 不充当数组下标。
- linear capacity 用整数预算；自定义首轮必须精确为 1。near-tie 与选择是否改变分别统计。
- 新产物记录本包源码内容 hash、配置与输入证据；receipt 核对 artifact 来源与 topology。
- direct placement 批量处理 metadata，避免逐 token CUDA 标量读回。
- EOS 启动器以自身所在项目为 `MOR_PROJECT_ROOT`，环境校验比较显式指定路径与实际
  import 路径，支持隔离 worktree，同时拒绝错误 PYTHONPATH；依赖与集群约束不变。
- 公共 objective adapter 按完整 optimizer step 的各 DP/microbatch 分母计算缩放，
  parity trainer 复用此 API；使用约束见 `objective.zh.md`。
  mask 权重总和可以小于 1；仅当总和为零才使用安全分母 1，不能用 clamp_min(1)
  改变非空 weighted mean。reference 与 native CP loss 共用此分母规则。
- artifact 记录启动时源码快照，写出时检查源码未改变；比较报告绑定 manifest、routes、
  tensors 三个文件哈希。schema v2 matrix receipt 检查同源、同配置、同初始化、拓扑和模式。
  inference-only 证据不要求不存在的 LM loss，但必须有完整 forward 与 routing；训练
  验收仍要求全部初始化、梯度、更新、loss 和 checkpoint step 证据。

## 验证与边界

加载后跨 TP 参数检查使用独立运行产物，不改变 tiny-only 训练状态采集策略：
先流式保存 TP1 实际加载后的原始参数 bytes，再让每个候选 rank 比较其完整对应切片，
同时核验 TP 区间、全局 expert ID 和每个 rank 的参数清单，保留 dtype 与 padding。
捕获点为原 checkpoint loader 返回后、首次 forward 前；生产加载 guard 原样执行。
作业 6011449 已验证当前 30B/MoR TP1/EP1、TP1/EP2、TP2/EP2 全量 bitwise 一致，
见 [结果与限制](loaded-tp-parameters-bitwise.zh.md)。这不替代计算路径的数值验收。

先运行故障注入回归和完整 CPU/Gloo 测试，再在 EOS 固定 Torch 2.10.0+cu129 / TE 2.13.0
环境验证 tiny 全拓扑。30B 继续保留 2% relative-L2 / 0.999 cosine 门槛；逐算子实验用于
定位误差，checkpoint 通过不能替代精度验收。具体结果、job ID 与未解决限制在修复后更新。

## 30B 原生 BF16 续查

用户拒绝 FP32-heavy 正式路径不表示结束原生 BF16 修复。续查保持参数、激活 dtype 和
原验收阈值，先对相同 end-block 输入记录 QKV、QK norm、RoPE、core attention 和
projection 边界；按原 global token ID 与 TP head 分片重建，禁止按 rank 拼接后直接比较。
单卡 strict 当前使用 TE unfused，CP 使用 Magi FFA，需要对完全相同 Q/K/V 另用独立
FP32 数学 attention 检查两者误差，区分被测实现缺陷与 baseline 数值缺陷。
FP32 仅用作离线诊断基准，不构成用户拒绝的正式混合精度训练路径。输入注入的所有报告
继续明确标记为诊断，不能作为端到端验收。

`6006269` 已证实三个末端层的 QKV/QK norm/RoPE bitwise exact，首次差异在 core。
据此在隔离分支试修 strict CP=1 backend policy：选择原生 BF16 fused attention，而不是
将 scaled QK scores 提前舍入到 BF16 的 unfused。strict/deterministic 约束全部保留；
CP>1 仍为 Magi，参数/activation dtype 不变。manifest 显式记录统一 policy 与实际 backend，
比较器及 receipt 绑定 policy 与 strict。必须通过独立 GPU forward/backward、30B 原阈值
完整 forward 和 tiny 全训练矩阵才能视为验证完成，不能以局部注入结果宣告修复。

`6006321/6006334` 发现 TE 2.13 内置 cuDNN frontend 会主动 dlopen CUDA12 和 CUDA13，
即使开始时只加载 CUDA12 也会拒绝执行；该二进制没有新版显式 runtime selector。
不删除或屏蔽检查、不修改原环境。改为验证 Magi v1.1.1 官方明确支持无 CP 的
`functional.flex_flash_attn_func`：在本包添加 CP=1、THD、causal、BF16 的无参数 core
adapter，通过原有 native QKV/norm/RoPE/projection，CP>1 仍使用原分布式 Magi API。
这不放宽原 distributed Magi 的 CP>1 guard，也不改变模型训练 dtype。adapter 显式验证
packed metadata，处理序列间 padding gap，并将 dummy output/gradient 置零；strict 传给
FFA deterministic。正式默认的变更须重新完成上述同等验证，TE fused 试修不冒充已验证路径。

full-training checkpoint 的 run contract 同时要求非空 `attention_policy`；producer、
schema 和外部 resume 比较使用相同字段，旧无 policy 的 training receipt 不可冒充
新路径续训。该约束不禁止加载原有 model-only folded DCP。`6006390` 已完成新路径
tiny 全矩阵，30B TP/CP 的端到端数值门槛仍未通过，不能签发整体完成结论。

## 全轨迹 diff 来源诊断

用户要求先定位而非继续修改计算路径。诊断扩展到 48 次逻辑层调用的完整边界：
block input、QKV、QK norm、RoPE、core attention、row projection、两次残差、
MLP norm 与 MoE。自然 forward 不注入输入；另设明确标记的逐层及逐算子同输入隔离实验。
按真实 global token ID 和 TP head 重组，支持递归中稀疏 ID，检查重复及缺失覆盖。
被动记录的主 artifact 与既有对应运行逐 tensor hash 对照，检查观察行为是否改变轨迹。
原生 dtype、固定依赖、checkpoint、routing oracle 和原验收门槛全部保留；诊断报告
不能替代正式验收。代码放在独立 runtime 诊断目录，不修改生产源码。

`6006544` 的 4 组自然轨迹各 268 项主 tensor hash 完全复现旧结果；`6006559` 完成
TP/CP 单轴的全部 48 层逐算子同输入对照和独立数学参考。TP 首差在第 0 层 row
projection，CP 首差在第 0 层 core；CP 另有小幅 QKV/projection/norm 独立差异。
两组所有 MoE 同输入输出 exact。详细归因、指标及未定位的 kernel 内部边界见
[diff-source.zh.md](diff-source.zh.md)，本轮不修改正式算法或 dtype。

后续按用户要求固定 CP=1，仅分析 TP/SP。`6006662` 用 CPU 对已有 tensor 验证绝对
误差、逐 token 相对误差、残差相消和归一化重加权：第 46 层分支相消与 delta 增长
同时存在，final norm 改变 token 尺度在全局 L2 中的权重。不能把不同边界的 global
relative-L2 比值直接称为扰动增益，也不能用只改善 early global-L2 的 FP32 partial
实验宣称端到端已修复。详见 diff 来源文档的 CP=1 专节。

进一步按用户要求闭环单层内传播：仅重置每个 block 的输入，不再重置 block 内部
operator 输入，使本层 TP 投影扰动自然经过两次残差、MLP norm 和 MoE；对照已保存的
自然轨迹与 baseline。另给 final norm/head 各自相同 baseline 输入，排除新的独立误差。
使用实时 checkpoint 权重、实际 2048 hidden 形状在 GPU 做独立 RMSNorm 数学分解与
forward/backward 验证；将 token 尺度、learned gamma、舍入及残差相消分别记录。
这些仍是 CP=1 forward 诊断和独立 norm 测试，不修改正式训练路径或验收阈值。

`6006744` / `6006849` 已完成该闭环：48 层投影前相同，本层投影扰动自然传播后
第 46 层只差 0.3341%，自然累积轨迹为 3.4605%；后者含输入扰动传播与 reference
残差相消。final norm/head 同输入 hash exact；final norm 的真实 gamma 近乎抑制
原始 hidden 中占 73.8230% reference 能量的两个通道。独立真实形状 norm 的
forward/input-gradient/gamma-gradient GPU 检查通过，但不代替整网训练验收。
完整指标、数值解释和限制记录于 [tp-layer-chain.zh.md](tp-layer-chain.zh.md)。

继续追查 FP32 partial/reduction 诊断为何没有改善 logits：复用 `6006454` 的冻结
forward-only 实现，补采全部 48×15 边界并要求主 artifact 逐 tensor hash 复现旧结果；
另做 FP32 partial 的 block-only 同输入控制，比较本地误差和自然输入扰动传播。
按真实 token ID 统计 BF16/FP32 两条自然轨迹的逐层误差、方向、逐 token 分布、
残差舍入和真实 gamma 作用。FP32 只作用于 projection partial 与归约，返回 hidden
仍为 BF16；不能把该诊断叫作全网 FP32，也不产生正式混合精度训练路径。

`6007959` 采集 FP32 自然/block-only 48×15 边界，全部 268 个自然主 artifact tensor
复现原诊断；`6007990` 分析证实本层同输入 projection 改善约 22–65 倍并不意味着
后续误差单调改善。token 156 的 L2 MLP norm/channel 940 跨 BF16 142.5 舍入边界，
实际 norm output 142→143；L45 MLP norm 已消除 early hidden 的全局指标优势。
L46 输入传播与 reference residual 相消共同推高相对误差，最终 logits 两条误差向量
cosine 仅 0.162708，不能称为相同误差或已证明的精度下限。

新增 final gamma 探针的首次实现误在 load 前捕获初始权重；修正为实际 forward
pre-hook，并加上加载后参考 hash、跨 rank hash、数学值对 BF16 实际输出的检查。
该修正的整网 GPU 重跑先遇 Xid 63，后因排队取消，不能记作验证通过。`6008118`
改以明确绑定 `6006744` 已验证 loaded gamma 的 CPU 离线分析，拒绝初始全 1 gamma，
完成正确 final norm 分解及 token 156 通道核对。没有改写旧证据或绕过训练 guard；
`6007990` 的旧 final gamma 数学分解明确失效，其他逐层张量统计保持有效。
详细指标、适用范围、完整 48 层表、720 边界 CSV 和审计信息见
[fp32-partial-propagation.zh.md](fp32-partial-propagation.zh.md)。本轮仍无生产精度变更。

## 真实数据训练轨迹验证

用户要求将单步 synthetic 诊断扩展为真实数据 50 次 optimizer update。
独立 runtime 实验直接调用已有 MLite training API 和公共 objective adapter，
不改变原 synthetic parity 入口，不绕过全参数 parity 证据门槛，也不签发完整矩阵证书。
采用经过 tokenizer 来源核验的 Pile 文本，冻结 `[50,4,256]` token tensor，
固定 DP4/EP4/ETP1，仅分别改变 TP 或 CP；所有 rank 加载后参数先过 raw-byte 基准检查。
保存各 step 的输入 hash、native LM/depth auxiliary loss、更新状态、gradient norm
和显存；比较器要求每组恰好 50 个成功 step，且逐步输入/目标数量相同。
`6012211` 已完成三组 50 步，测得非零 loss diff；这不是旧 logits 门槛的替代验收。
合同、数据来源、结果和限制见 [real-data-50steps.zh.md](real-data-50steps.zh.md)。

进一步按用户要求将真实数据扩展到 100 步，并覆盖此前原始矩阵与 TP4/真实数据配置。
冻结 `[100,4,256]` 输入，前 50 步 token bitwise 等于旧数据；scheduler horizon=100，
各组从相同 loaded model-only checkpoint 权重和全新 optimizer 开始，GBS4 不变。
公共输入分片/objective 支持 DP1/2/4，全部 CP 副本纳入 raw-byte 检查。
runtime driver 对每组独立 torchrun，失败继续遍历但保留非通过状态，
coverage-aware comparator 拒绝遗漏、短跑、输入/依赖合同不一致或初始参数不一致。

`6012240 / 6012241 / 6012301` 共尝试 16 组，7 组完成各 100 次更新、9 组首次更新前
因 H100 80GB 显存不足失败（gradient buffer / backward / Adam state），没有改动
生产计算路径或用新增可运行组替代原失败组。6 对固定其余轴的 TP/CP/EP/DP 对照
均有非零 loss 差异：TP1→2 MAE=0.05994，CP1→2（TP1）MAE=0.04294，
EP4→2（TP4/DP2）MAE=0.03059；不能据此签发完整 parity 证书或归因到单个算子。
本轮还核实 pinned native Qwen MoE 明确 `compute_aux_loss=False`，更正旧 50 步文档的
aux 描述但保留原数据证据；实际目标包含 LM 与 depth-router auxiliary。
详细 16 组范围、逐步指标、OOM 根因边界及产物见
[real-data-100steps.zh.md](real-data-100steps.zh.md)。
