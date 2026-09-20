# MoR / MLite 训练框架独立审查报告

审查日期：2026-09-09～10。对象：`work/mor_mlite` 当前文件快照。审查目标：对照用户确认的第一版方案，评估实现完成度、训练与分布式语义、精度证据和可维护性。

## 1. 结论

**这是一个已经跑通主要功能、并且具有较强 tiny 分布式验收证据的 correctness 原型；还不能判定原计划全部完成，也还不是经过性能和通用数据接口验收的训练框架。**

- 已有真实实现：原生 Qwen3-MoE 适配、物理层复用、逐样本 depth routing、active-only CP、TP、EP、ZeRO-1、HF 流式折叠、checkpoint，以及单卡/分布式比较工具。不是仅有接口或伪代码。
- tiny 的指定 BF16 全拓扑矩阵有通过记录。本次独立核验了 20 份报告及其 receipt hash，报告包含完整训练 tensor，不是空跑。
- **30B 精度门槛仍未通过，是原计划验收的阻塞项。** 最新 logits relative L2 为 **5.673%**，超过 2%；cosine 为 **0.998406**，低于 0.999。训练和 checkpoint 能执行，不等于单卡精度已对齐。
- 新复现了比较器证据缺失仍通过、near-tie 漏计、reference replay 缺轮回退、reference 数据合同不一致、capacity 浮点边界等问题。这些主要损害边界条件及 oracle 的可信度，**不等于已证明默认生产路径或已有 tiny 结果错误**。
- 最大工程欠账是：训练执行与验收采集耦合、损失归一化公共接口不完整、对上游内部结构依赖较深，以及缺少长序列性能测量。

不采用“完成 90%”这种简单百分比：真实模型精度是硬门槛，不能由其他功能数量抵消。更准确的状态是：**功能主干已实现；tiny 限定配置已验收；30B 精度未验收；泛化与性能尚待验证。**

## 2. 审查方法与证据边界

本次由模型语义、分布式实现、验收基础设施三个独立审查方向交叉核对，并由主审重复运行关键故障复现。没有修改训练源码、没有提交新 GPU 作业；只新增本报告、诊断脚本和 Archify 图件。

| 项目 | 本次实际工作 / 结果 |
|---|---|
| 本地完整测试 | `PYTHONPATH=src .venv/bin/python -m pytest -q`：**258 passed, 26 skipped，21.85 秒** |
| 分布式相关子集 | dispatch / routing / static CP / checkpoint / optimizer fingerprint / communication 共 **59 passed**；属于上面测试的子集，不能重复相加 |
| Lint | `.venv/bin/ruff check .`：通过 |
| 格式 | `ruff format --check src tests`：4 个文件需要格式化，76 个已符合 |
| Shell 语法 | `bash -n scripts/eos/*.sh slurm/*.sbatch`：通过；不是 Slurm 执行验证 |
| 故障注入 | CPU 复现 replay、数据合同、capacity、near-tie 和缺失验收证据问题；原 artifacts 未改动 |
| EOS | SSH 只读查询 Slurm 状态、固定依赖版本与上游工作树；检查已下载报告和 checksum |
| 未执行 | 本次没有重新运行 H100 矩阵、没有 profiler、没有长训练或真实数据集收敛试验 |

本机为 macOS arm64 / Torch 2.10.0，无 CUDA。26 个 skip 中，25 个与缺少 `megatron.lite` 及其 checkpoint 模块有关，另 1 个为显式 opt-in 的 MLite CLI smoke；不能把这 26 项算成通过。

### 快照与可复现性

该目录不是 Git 工作树，不能给当前实现附上可靠的本包 commit/branch。上游 EOS Megatron-LM 则确认在 `5c8315f12a64a7279eec58896af9e74ee3351b74`，MagiAttention 为 `v1.1.1`，两者 `git status --porcelain` 均为空。

- 当前 `src/tests/configs/scripts/slurm` 中非缓存文件，加 `pyproject.toml`，共 106 个文件，审查内容 SHA-256：`490d4fe91f83c51540a317c8317067ead1f3b987d80f15908c498f57bfcae8be`。算法：各目录按路径排序，依次拼接路径、NUL、文件字节、NUL，最后加入 pyproject；不含本次新增 docs。
- 当前 wheel SHA-256：`3af2d0a5ba98396c029e859937190eee234419298ab22215a154723d706b7397`。
- 当前 56 个 Python source 文件与该 wheel 中对应文件逐字节一致。
- **历史 EOS receipt 没有记录本包 source/wheel hash**，所以“当前 wheel 与源码一致”不能自动证明旧全矩阵就是当前快照执行的。详见问题 F7。

## 3. 实际逻辑结构

### 3.1 两套执行路径及职责边界

| 层次 | 实现入口 | 负责什么 |
|---|---|---|
| CLI 与配置 | `train.py`、`convert_hf.py`、`parity/__main__.py`、`config.py` | 结构、router、topology、依赖、输入与运行参数校验 |
| MLite 注册 | `register.py`、`qwen3_moe_mor/protocol.py` | 使用公开 registry 注册 `qwen3_moe_mor`，构造原生 Qwen 物理栈和 optimizer |
| MoR 控制流 | `qwen3_moe_mor/model.py` | start / recurrent / end，router 调度、退出保留、最终恢复 |
| Routing 与布局 | `routing/`、`distributed/routing.py`、`distributed/backends/magi.py` | 全局 token identity、逐样本选择、Magi 目标布局、active dispatch |
| 迁移与反传 | `distributed/all_to_all.py`、`dispatch.py`、`parking.py` | 可微 variable All-to-All、退出状态归并、反向 owner 恢复 |
| 原生层内计算 | MLite Qwen3-MoE + position-aware attention 适配 | TP attention/projections、CP attention、native MoE routing/EP |
| 训练执行及验收 | `parity/mlite.py` | synthetic batch、DP loss 权重、runtime 调用、梯度/更新重建、checkpoint 与诊断采集 |
| 比较与证书 | `parity/compare.py`、`receipt.py`、`external_checkpoint.py` | tensor/route/通信/continuation 比较和验收报告 |
| 独立参考实现 | `tiny/model.py`、`parity/reference.py` | PyTorch FP32 控制流与小规模 CP 参考；不是原生 Qwen 实现 |

重要区别：当前 `train --backend mlite` **直接调用 `parity/mlite.py::run_mlite()`**。因此训练入口是带完整诊断的 synthetic harness，还没有把轻量训练 engine 与验收采集独立出来。[入口接线](/Users/hongbinl/Documents/Codex/2026-09-07/y/work/mor_mlite/src/mor_mlite/train.py:148)

注册使用公开 API，但后续适配并非只依赖公开 API：原生 model 的 `__class__` 被升级，attention 被安装 position-aware 适配，`sp_params` 被扩展，runtime handle 和 bucket hook 被读取或包装。固定上游版本使这条路径可控，但升级时需要专门兼容性检查。[模型适配](/Users/hongbinl/Documents/Codex/2026-09-07/y/work/mor_mlite/src/mor_mlite/qwen3_moe_mor/model.py:186)

### 3.2 模型的一次 forward

1. **输入与完整序列 start。** PackedBatch 保留 sample ID、global token ID、原 position、序列长度和 mask；先执行 embedding 与独立的 start layers。TP/CP 所需 dummy tails 不属于有效训练 token。
2. **Router r，先选后算。** 首轮也执行 router，容量为 100%；后续轮只在上一轮 active tokens 中按样本选择。默认三轮容量基于 original length，数学比例为 1、2/3、1/3，再取 floor；不是基于上轮剩余长度逐次乘比例。
3. **保留退出状态并更新布局。** 未选 token 进入 exit buffer，不再参加后续 recurrent attention/MoE。选中 token 按 RoutePlan 迁移；`magi_direct` 首轮复用 start layout，后续 active set 改变时直接做一次 TP×CP hidden variable All-to-All。
4. **复用同一物理 block。** 每轮调用相同 `N_recurrent_layers` 个原生 Qwen 层，不构造 k 份参数。每层内部仍做 attention、norm、MoE 及该层所需 TP/CP/EP 通信。
5. **门控更新。** 实际约定是 `h_after = h_before + gate * recurrent_block(h_before)`。block 内已有层残差，外层仍按确认的 MoR 语义再门控相加；不能自行解释为仅对 block 的增量进行门控。
6. **进入下一轮或结束。** 保留原 token ID 和原 RoPE position。active 序列 `[0,3,7]` 的因果顺序仍是这个子序列，但 RoPE 不能变成 `[0,1,2]`。
7. **最终完整恢复。** 将所有 exit buffers 与最后一轮输出按原 token identity 合并回完整布局，检查丢失/重复/owner；所有原始有效 token 再经过 end layers 和 LM head。
8. **损失。** LM loss 加逐轮 BCE；BCE 的候选分母是本轮 router 输入的 active tokens，标签由该轮是否选中确定。代码把 `linear(h)/temperature` 称为 decision logits；temperature=1 时与未缩放 logits 相同。该语义在 README 已说明，不把“原始 logits”的词面差异列为训练 bug。

tiny 为 `1 + 2×3 + 1`，实际只有 4 个物理 Transformer 层；30B preset 为 `3 + 14×3 + 3`，实际只有 20 个物理层、48 次逻辑层调用，另外有 3 个独立 depth routers。[物理模型构造](/Users/hongbinl/Documents/Codex/2026-09-07/y/work/mor_mlite/src/mor_mlite/qwen3_moe_mor/protocol.py:416)

“early exit”只表示退出递归，不是退出最终模型，也不是切断梯度。被保存的状态仍经最终 end/head 对 loss 产生贡献。

### 3.3 DP、CP、TP、EP：数据和参数的实际分配

分组公式为 `world = dense_DP × TP × CP = expert_DP × EP × ETP`。**EP 不是额外乘到 dense-DP 之外的新 world 维度。**

| 并行 / 对象 | token / activation | 参数、梯度及同步 |
|---|---|---|
| dense DP | 不同 replica 持有不同样本；depth selection 不跨 replica 混选 | 相同 dense TP shard 在 DP/CP peers 间复制；step 末按 optimizer group 同步 |
| TP + sequence parallel | 层间 SP 布局中 rank 可持有不同 token rows，但这些 rows 的 hidden 维完整；不是始终每个 TP rank 持有同一份 tokens | attention/projection/embedding/head 使用原生 TP；depth-router weight 是 replicated SP 参数，需 TP 梯度求和 |
| CP | 同一 DP replica 的序列被分片；router 汇聚 scores/identity 决定全样本 Top-K，然后 direct rebalance | CP 不按层切参数；每个 active layer 的 attention 仍有跨 CP 计算/交换；退出 token 不回流到后续 Q/K/V |
| EP，ETP=1 | 在每个原生 MoE 层把当前 active tokens 发给其 Top-K experts，再 combine | 每个 expert 不做 ETP 切分；不同 experts 分到 EP owners，expert replicas 的 optimizer 沿 expert-DP 分片 |
| ZeRO-1 dense | 不改变样本的 DP ownership | BF16 model TP shard 常驻复制；FP32 master/Adam states 沿 DP×CP optimizer group 分片 |
| ZeRO-1 expert | 使用 EP dispatcher 所定义的 expert ownership | FP32 master/Adam states 沿 expert-DP group 分片；不是套用 dense group |

对 8 卡组合 `TP=2, CP=2, dense_DP=2, EP=4, ETP=1`：每个 dense replica 跨 4 个 TP×CP ranks；dense optimizer group size 为 4，expert-DP size 为 2。这里不假设未经验证的物理 rank 排列。

`magi_direct` 的“一次 dispatch”只指每次 active-set 边界的 **hidden rebalance**；不包含 score/metadata collectives，也不取消层内 CP attention、TP collectives 或 EP All-to-All。实际后向也需要相应逆向通信。

`magi_canonical` 目前只支持 **TP=1** 的 CP 路径；`static_reference` 是不 rebalance、gather active Q/K/V 的 tiny 诊断路径，不是正式 Qwen backend。[CP scope](/Users/hongbinl/Documents/Codex/2026-09-07/y/work/mor_mlite/src/mor_mlite/qwen3_moe_mor/protocol.py:399)

### 3.4 一次 global step 与反向传播

1. runtime 清空本地梯度。
2. 逐 microbatch 执行完整 MoR forward/backward。autograd 将共享物理参数在所有递归调用中的贡献累加到同一参数梯度；跨 microbatch 再继续累加。
3. hard Top-K 不对离散索引求导；选中 gate 保留可微路径，BCE 提供 router 的分类监督。hidden dispatch 的 backward 交换 send/receive splits，把梯度还给原 owner。
4. 自带 trainer 对 LM 和每一轮 BCE **分别**乘 `dense_DP × local_count/global_count`，补偿 distributed optimizer 对 replica 的平均；只乘一个统一 aux 权重一般不正确。
5. 在所有本地轮次和 microbatch 累加结束后 finalize gradients。TP replicated 参数 reduction 与 ZeRO-1 的 bucket reduce-scatter 具有不同目的，不能合并成“整个训练只有一次通信”。
6. 每个物理参数 bucket 每 global step 发起一次预期的 ZeRO-1 gradient reduce-scatter；owner 更新 FP32 master/Adam state，更新后的 BF16 参数经 all-gather 同步，供下一 step 使用。当前配置关闭 overlap-grad-reduce，不按每次递归调用同步。
7. 如需 checkpoint，通过 MLite distributed checkpoint 保存，并写入 MoR 配置、折叠/来源和运行信息。full-state resume 验证 model/master/Adam/RNG 和后续一步；不是只验证 `load()` 不报错。

原方案里“step 开始 all-gather”与“上一个 step 更新后 all-gather”是相邻边界的两种叙述；关键是下一次 forward 前模型参数一致，以及不能每个 recurrent round 重复同步。[loss 权重与 step 接线](/Users/hongbinl/Documents/Codex/2026-09-07/y/work/mor_mlite/src/mor_mlite/parity/mlite.py:3520)

### 3.5 HF、checkpoint 与验收的数据流

标准 HF logical layers 先按 `start / recurrent groups / end` 映射。recurrent 对应层的 attention、norm、gate 和同 expert ID 权重在 FP32 流式求均值，再转换到目标 dtype/shard；routers 由固定 seed 新初始化。

训练使用折叠后的物理模型，不应把“HF 原 48 层”当成训练时仍保存 48 份参数。完整同拓扑 checkpoint 与 model-only initialization 也需区分：前者恢复 optimizer/RNG；后者只初始化模型。由于 native expert checkpoint keys 使用 EP-local index，**跨 EP degree 的恢复被明确拒绝**；不能把已通过的同拓扑恢复推广为任意 topology reshard。[checkpoint 限制](/Users/hongbinl/Documents/Codex/2026-09-07/y/work/mor_mlite/src/mor_mlite/checkpoint_io.py:175)

### 3.6 Archify 图件

图由 [Archify（tt-a1i/archify）](https://github.com/tt-a1i/archify) 的 typed JSON → validator → HTML renderer 生成；分析和节点语义来自上面的源码核对，没有让绘图器自行推断训练正确性。只在本地运行，没有上传本框架代码。

图件与渲染校验状态见同目录 `diagrams/`。前向图的初始候选触发可读性检查，按 Archify 有界修正规则停止，未通过的候选不作为成品；最终可交付图及 receipt 在本报告完成时补充。

## 4. 对照原计划的完成度

| 计划项 | 评估 | 依据 / 尚缺什么 |
|---|---|---|
| 独立包、不改上游 core | 已实现 | 公开 registry 接入；EOS 上游树干净；但进程内内部对象适配较多 |
| 物理层复用、gate 更新、nested routing、原 position | 默认路径实现完整 | 源码与单测；非默认 capacity 与 reference 边界见 F4/F5/F6 |
| RoutePlan learned/replay | 已实现，但 oracle 有缺陷 | production replay 有较强校验；reference 缺轮/metadata 未 fail-closed |
| DP ZeRO-1 | 限定配置已验证 | 2 卡 DP 及组合矩阵、完整梯度/更新；bucket hook 验证同步边界 |
| TP / ETP=1 | 限定配置已验证 | 原生 TP + router SP grad reduction；含 TP+DP/CP/EP 组合 |
| Magi direct CP | 已实现、tiny 有证据 | active dispatch、最终 merge、RoPE、QKV probes；长序列性能未验证 |
| canonical/direct CP oracle | **部分实现** | 只跑 CP-only canonical，对 TP+CP 没有同拓扑 canonical 对照 |
| Native MoE / EP | 已实现、tiny 有证据 | 真实 Top-K/dispatcher/experts/combine；稳定 fixture 不覆盖自然 routing 的全部敏感性 |
| HF streaming fold + DCP | 已实现并有真实模型运行证据 | FP32 累加、expert identity、metadata；不支持跨 EP checkpoint reshard |
| 四类 CLI / synthetic data | 已实现 | CLI 即诊断型 trainer；无通用数据训练入口和独立低开销 engine |
| tiny 单卡 / 全拓扑验收 | **历史矩阵通过** | 20 份报告及 hash 已核验；当前 source 与历史矩阵绑定不完整 |
| 真实 30B smoke | **部分通过，整体未通过** | forward/backward/step/restore 能运行；严格 forward 精度失败 |
| near-tie 数量与比例 | **未完整满足** | 只计发生选择改变的 near tie，漏计未改变的 near tie |
| 性能可用性 | **未验证** | 没有长序列吞吐、通信字节、峰值显存及 dispatcher 时间数据 |
| PP / ZeRO-2/3 / ETP>1 等 | 明确不在范围 | 不作为第一版缺陷计入 |

## 5. 正确性验证：哪些证据值得信任

### 5.1 EOS 作业核验

本次直接查询 `sacct -X`：

| Job | 状态 / exit | 时长 / 节点 | 能支持的结论 |
|---|---|---|---|
| 5997741 | COMPLETED / 0:0 | 18:14 / eos0326 | tiny 完整矩阵通过；pytest XML 为 270 pass、1 skip |
| 5997892 | COMPLETED / 0:0 | 06:45 / eos0135 | 后续 zero1 learned/replay 和 external checkpoint 通过；pytest 为 272 pass、1 skip，不是又一轮完整矩阵 |
| 5998296 | COMPLETED / 0:0 | 11:16 / eos0184 | optimizer 专项诊断，不能替代全矩阵 |
| 5998357 | **FAILED / 1:0** | 18:38 / eos0487 | 30B forward gate 失败；独立 checkpoint gate 通过 |

EOS manifests 为 Torch 2.10.0+cu129 / CUDA 12.9 / NCCL 2.27.5 / Transformer Engine 2.13.0 / 固定 MLite SHA / Magi 1.1.1，符合本轮声明的环境。

### 5.2 tiny 全矩阵的证据强度

[完整矩阵 receipt](/Users/hongbinl/Documents/Codex/2026-09-07/y/work/mor_mlite/artifacts/eos/5997741/tiny/reports/matrix_complete.json) 的 **20/20** 报告通过且 hash 匹配：FP32 reference replay、8 个 distributed topology 的 learned/replay、CP canonical、canonical-vs-direct，以及 external checkpoint。

BF16 每份主要比较报告包含 1,047 个 tensor 结果。以 all topology 为例：66 个 initialized physical parameter tensors，201 个 gradient tensor（含 grad norm）、198 个 update tensor、198 个 post-step tensor，另有 hidden/loss/expert-route 证据；每个完整参数向量包含 **7,217,152** 个元素。[8 卡报告](/Users/hongbinl/Documents/Codex/2026-09-07/y/work/mor_mlite/artifacts/eos/5997741/tiny/reports/all.json)

| 全矩阵汇总指标 | 最坏已记录值 | 门槛 |
|---|---:|---:|
| forward relative L2 | 0.5043% | ≤2% |
| reconstructed gradient relative L2 | 0.2987% | ≤3% |
| FP32 master update relative L2 | 1.2427% | ≤3% |
| 对应上述最坏 update 的 cosine | 0.9999228 | ≥0.999 |
| resident post-step tensor relative L2 | 0.1646% | 报告逐 tensor 检查 |

BF16 gradient/update 的硬门槛是**按阶段/step 拼接完整参数向量**的 L2/cosine，而不是要求每个单独参数 tensor 都小于 3%。局部 outlier 仍保留诊断，例如 all 的 `update/step_000/norm.weight`；报告通过不能读作“每个参数逐元素均严格对齐”。FP32 则使用配置的 allclose 容差。

值得肯定的实现包括：重建参数集合必须与初始化集合完全一致；真实执行 Gloo collective/backward 的测试；退出归并的 duplicate/missing 检查；保存完整 optimizer/RNG fingerprint；同 step 的 uninterrupted 和 restored continuation 比较。

### 5.3 30B 的通过项和失败项

[最新 forward 报告](/Users/hongbinl/Documents/Codex/2026-09-07/y/work/mor_mlite/artifacts/eos/5998357/qwen30b/reports/all_vs_baseline.json) 中 28 个 forward tensor 有 **15 个硬门槛失败**：

| 位置 | relative L2 |
|---|---:|
| 三轮 recurrent hidden | 0.667%、0.667%、0.879% |
| 三层 end hidden | 1.277%、3.311%、2.238% |
| final norm / hidden for head | 5.567% |
| logits | **5.673%** |

Depth RoutePlan 3 轮一致，native expert probe 的 48 个逻辑上下文也通过，但 **forward 比较使用了 depth replay，以及 native expert IDs/selected scores replay**。这些是固定离散选择后的执行证据，不证明真实 learned routing 在 8 卡自然与单卡一致。

[最新 checkpoint 报告](/Users/hongbinl/Documents/Codex/2026-09-07/y/work/mor_mlite/artifacts/eos/5998357/qwen30b/reports/checkpoint_external_resume.json) 通过：独立进程在保存点恢复 model/master/Adam/RNG，并与 uninterrupted 下一步的 fingerprint 相同。这比“能 load 和再 step”更强，但仍不等于单卡 30B gradient/update parity。

已有 CP-only、TP-only 隔离记录显示误差也能在各自路径出现，支持继续定位 topology-dependent BF16 运算。但现有证据**没有完成同输入、同权重、逐算子的独立数值归因**。不能只因 replay 一致就断言“必然只是正常 BF16 roundoff，其他实现已全部排除”。

### 5.4 不能外推的测试边界

- 默认 full matrix 固定 seed 1234、长度 `[9,6,3]`、2 microbatches、1 step，另执行恢复/不中断的下一步。这是短序列少步测试，不是长期训练稳定性。
- 原生 tiny MoE 特意构造了强 margin fixture：embedding 两维为 `16/±16`，gate 只读取其中一个维度，部分输出行初始化为零，使初始 expert pair 稳定为 `[0,2]` 或 `[1,3]`。仍运行真实 EP 和 expert training，因此它有价值；但不能替代普通初始化、多 seed、cutoff 敏感样本的测试。[fixture](/Users/hongbinl/Documents/Codex/2026-09-07/y/work/mor_mlite/src/mor_mlite/parity/mlite.py:674)
- FP32 reference 与 Qwen/MLite tiny 是两套实现。当前 full matrix 中 FP32 是 reference learned/replay 自身比较；BF16 是 native MLite 单卡与 native distributed 比较。**没有完整的“同一权重 FP32 reference ↔ 原生 Qwen”独立模型 oracle**。
- 通信 probe 观察 `BucketGroup.start_grad_sync()` 的实际发起点，不是 CUPTI/NCCL kernel 计数或字节分析。QKV pre-hook 会核对输入行数和布局 token ID，但还需数值 parity 来约束 hidden 与 metadata 是否始终对应。
- 对不同 recursion count、不同 layer 分配、不同跨 microbatch token 总量、自定义 mask、长序列极度不均衡，还缺系统性 GPU 矩阵覆盖。

## 6. 新发现的问题，按优先级排列

严重性约定：P1 为阻塞原计划验收；P2 为需修复的行为/验证缺陷或明确工程风险；P3 为不影响数值行为的维护问题。以下“默认不受影响”指当前默认配置和既有完整 artifacts，不是保证所有调用方式安全。

### F1 · P1 · 30B 单卡/8 卡 forward 精度未达标

证据见第 5.3 节。不是新发现的源码行级 bug，但它是仍未关闭的验收失败。影响：不能交付“精度对齐已完成”的总承诺。

建议：固定输入 hidden、权重和两类 replay，逐层拆分 attention、norm、MoE、residual、head；分别控制单卡与 TP/CP 路径的计算精度/内核选择，获得足以区分 numerical sensitivity 和实现偏差的证据，再按原门槛重跑。是否调整门槛应另作明确数值论证，不能用 checkpoint 通过抵消 forward 失败。

### F2 · P2 · 比较器允许两边同时缺失训练证据，仍给出 full pass

位置：[compare.py:555](/Users/hongbinl/Documents/Codex/2026-09-07/y/work/mor_mlite/src/mor_mlite/parity/compare.py:555)、[通过条件 :719](/Users/hongbinl/Documents/Codex/2026-09-07/y/work/mor_mlite/src/mor_mlite/parity/compare.py:719)。

比较器只比较实际存在的 tensor names，空 routes 与空 routes 相等，空 aggregate 的 `all([])` 也为真。本次在临时目录从两份真实 reference artifact 各移除 **823 个 initial/gradient/update/post-step tensors 和 18 个 RoutePlan**，只保留 78 个 forward/loss tensors，`scope=all` 仍返回 `passed=True, gradients=0, routes=0`。

影响：未来采集器回归或缺失整个 namespace 时，验收可能假通过。已有 EOS 报告证据完整，不因这个复现被自动作废。

修复验收：按 acceptance profile、step/microbatch/round 数和 initialized parameter 集合推导必需 evidence；空或缺字段直接失败。若支持 partial comparison，必须显式返回 partial，不能写 full-matrix pass。增加删整类证据的负向测试。

### F3 · P2 · near-tie 数量与比例漏计未改变选择的样本

位置：[compare.py:834](/Users/hongbinl/Documents/Codex/2026-09-07/y/work/mor_mlite/src/mor_mlite/parity/compare.py:834)、同文件 :988。

两边 canonical selection 一致时提前 continue，不调用 cutoff classification。输入 scores 都为 `[0.9,0.5,0.5]`、K=2，margin=0、error=0，按计划应是 near tie，却报告 `near_ties=0/1`。

影响：现有“zero near ties”不能解释为数据没有 cutoff 歧义；没有证据说明既有 EOS 输入实际存在近边界，只能说明统计口径不足。

修复验收：逐样本先计算 margin/error，分别统计 `near_ties` 与 `changed_near_ties`；覆盖相同选择、选择改变、非近边界改变三类。

### F4 · P2 · reference replay 缺轮会静默回退，错误位置 metadata 也被接受

位置：[tiny/model.py:606](/Users/hongbinl/Documents/Codex/2026-09-07/y/work/mor_mlite/src/mor_mlite/tiny/model.py:606)、同文件 :645；[depth_router.py:414](/Users/hongbinl/Documents/Codex/2026-09-07/y/work/mor_mlite/src/mor_mlite/routing/depth_router.py:414)。

复现一：`model(batch, route_mode="replay", replay_plans={})` 正常执行，三轮实际 mode 都为 learned。复现二：把 round-1 oracle 的全部 original positions 加 100，reference 仍执行，并用 live positions 替换错误 metadata。

影响：reference replay 不满足缺失/损坏 oracle 立即失败的合同。Production Qwen 会检查轮数和 sample/position，不共享这里两个缺陷。

修复验收：forward 入口要求 keys 恰好覆盖全部 rounds；统一检查 global ID、sample、position、count 和 padding。成功 round-trip 测试之外补缺轮/多轮/错误 metadata 负向测试。

### F5 · P2 · reference 的 PackedBatch 语义与 production 不完全一致

位置：[tiny/model.py:393](/Users/hongbinl/Documents/Codex/2026-09-07/y/work/mor_mlite/src/mor_mlite/tiny/model.py:393)、同文件 :617；对照 [protocol.py:318](/Users/hongbinl/Documents/Codex/2026-09-07/y/work/mor_mlite/src/mor_mlite/qwen3_moe_mor/protocol.py:318)。

- `loss_mask=None`：production 对所有 real tokens 赋 1，只排除 dummy；tiny 无条件屏蔽每序列末 token。长度 `(5,3)` 时 production 合同有 8 个训练目标权重，tiny 只有 6。这是目标差异，不是浮点误差。
- 合法 stable sample IDs 从 `[0,1]` 改为 `[10,11]`，tiny 把 ID 当 `seq_lens` 索引，报 `no original length is available for sample 10`；production 使用 sample-ID→length 映射。

影响：默认显式 mask、连续 ID fixture 不受影响，但 reference 不能直接覆盖通用 PackedBatch 或 DP 切片数据合同。

修复验收：明确 optional mask/末 token target 约定；抽出共享 target/mask helper 和 ID→length 映射；补 None mask、非连续 ID、DP sliced batch 对照。

### F6 · P2 · capacity 的浮点边界可少选 token

位置：[config.py:77](/Users/hongbinl/Documents/Codex/2026-09-07/y/work/mor_mlite/src/mor_mlite/config.py:77)、同文件 :124。

- 首项 `1-5e-13` 通过“首轮约等于 1”的校验，但 `top_k(8,0)=7`；production 随后会触发首轮不完整断言。
- `num_recursions=10, round=3, length=90` 时 `floor(0.7*90)=62`，而数学式 `((10-3)*90)//10=63`。

影响：不涉及当前默认 k=3 矩阵，但影响公开支持的非默认结构/预算。

修复验收：linear schedule 用整数表达式；自定义首项 exact 1 或接受后明确归一为 1。补多 k、多 L 的整数公式性质测试。

### F7 · P2 · 历史验收证据没有绑定本包源码快照

位置：[receipt.py:33](/Users/hongbinl/Documents/Codex/2026-09-07/y/work/mor_mlite/src/mor_mlite/parity/receipt.py:33)、同文件 :53；`versions.py` 记录本包版本 0.1.0，但没有 source/wheel hash。

Receipt 证明“这些 JSON 文件存在且 passed=true，字节 hash 是这些值”，没有验证每个 JSON 对应所声明的 topology、运行源码、相同初始状态或唯一输入。当前有后续源码/测试演进，却只有旧全矩阵和较新局部回归，独立审计不能建立端到端版本绑定。

修复验收：receipt 绑定本包 commit/tree/wheel、配置、输入/seed、初始参数指纹、依赖 manifest 和 artifact hashes；检查 topology/mode/scope。修复完成后对同一冻结 wheel 跑完整矩阵。

### F8 · P2 · direct 路径有逐 token GPU→CPU 标量读取，性能尚未成立

位置：[magi.py:268](/Users/hongbinl/Documents/Codex/2026-09-07/y/work/mor_mlite/src/mor_mlite/distributed/backends/magi.py:268)。

每个 TP×CP peer 遍历 canonical tokens，并对 CUDA metadata 执行 `.item()`；后面还做多次 `.cpu().tolist()`、dict lookup。由源码可确认 active boundary 存在 O(active length) 次标量 device→host 读取/同步风险。

影响：一次 hidden All-to-All 的通信结构不等于布局规划开销低；seq128 无法代表长序列。没有 profiler，不能给出实际减速倍率或声称 direct 比 canonical 更快/更慢。

建议：向量化 placement 构造，至少批量搬运 metadata；把同步、metadata collectives、Magi planning、hidden A2A 分别计时，覆盖 2K/8K/更长序列和不同 skew。

### F9 · P2 · 通用训练 API 的 loss normalization 仍藏在 parity trainer

位置：[mlite.py:745](/Users/hongbinl/Documents/Codex/2026-09-07/y/work/mor_mlite/src/mor_mlite/parity/mlite.py:745)、同文件 :3520。

自带 trainer 对不同 DP replica 有效 token 数的加权是正确的。但第三方只注册模型并直接使用 `output["loss"]`，会在长度不均时得到 replica-means 的平均，不是 global-token mean。README 已公开这个限制，所以它不是隐蔽 bug，而是尚未提炼完的框架合同。

进一步地，目前 scale 由 synthetic seq_lens 和末 token 屏蔽规则推导，不能直接推广为任意 loss mask、不同 microbatch token 数的训练器。

建议：公开 objective adapter，输入有效 LM count 与逐轮 candidate count，由 adapter 明确处理 DP 和跨 microbatch 累加；配一个不依赖 parity 私有字段的外部 MLite runtime 示例。

### F10 · P3 · 可维护性与格式问题

56 个 source 文件共 18,933 行；24 个 test 文件共 7,330 行。`parity/mlite.py` 4,148 行，`run_mlite()` 928 行；`_run_mor_backbone()` 531 行。不是“长文件必然错误”，但 bootstrap、loss、诊断、checkpoint、optimizer ownership 在同一调用链交织，使后续适配新数据或新并行策略的回归面较大。

`train.py` 还从 `parity.__main__` 导入私有 `_seq_lens`。若干测试通过 source 字符串/AST 检查接线，这类测试能防止特定实现回退，但不能替代执行行为验证。pyproject 中没有配置 type checker 或覆盖率门槛。

格式未通过的 4 个文件为 `distributed/counters.py`、`parity/mlite.py`、`tiny/model.py`、`tests/test_static_cp.py`，均为格式问题，本次未替用户改动。

建议按职责拆分 engine / objective / capture sink / state reconstruction / checkpoint certification；先锁住行为测试，再重构，避免把此次 review 顺带变成大范围修改。

## 7. 建议的后续顺序与完成条件

| 顺序 | 工作 | 关闭条件 |
|---|---|---|

