# 真实 Pile 数据 100 步全配置训练矩阵

## 范围与合同

用户要求按上一轮真实数据重验此前全部并行配置，将步数增至 100。
保持原 folded Qwen3-30B-A3B-Base MoR 模型：20 个物理层、48 次逻辑调用、
13,084,750,848 个参数；不是缩小模型的替代测试。PP=1、ETP=1，native BF16，
正常 FP32 gradient/master/Adam state 不变，不启用额外 FP32 投影或低精度路径。

重验原 `configs/topologies.json` 的 9 组、TP4 对比组、上一轮 50 步的 3 组，
另补 TP1/CP1/DP4/EP2 用于与 DP4/EP4 做 EP 单轴比较。原 EP2/DP2 配置发生 OOM 后，
再补 TP2/CP1/DP2/EP4 和 TP4/CP1/DP2/EP4，提供可运行的 TP、EP、DP 单轴对照，
共 16 组。新增组不替代原配置的失败记录。

| 名称 | TP | CP | DP | EP | GPU 数 | 来源 |
|---|---:|---:|---:|---:|---:|---|
| baseline | 1 | 1 | 1 | 1 | 1 | 原矩阵 |
| zero1 | 1 | 1 | 2 | 1 | 2 | 原矩阵 |
| tp | 2 | 1 | 1 | 1 | 2 | 原矩阵 |
| cp | 1 | 2 | 1 | 1 | 2 | 原矩阵 |
| ep | 1 | 1 | 2 | 2 | 2 | 原矩阵 |
| tp_cp_ep | 2 | 2 | 1 | 2 | 4 | 原矩阵 |
| tp_dp_ep | 2 | 1 | 2 | 2 | 4 | 原矩阵 |
| cp_dp_ep | 1 | 2 | 2 | 2 | 4 | 原矩阵 |
| all | 2 | 2 | 2 | 4 | 8 | 原矩阵 |
| tp4_dp_ep | 4 | 1 | 2 | 2 | 8 | 原 TP4 调查 |
| baseline_real | 1 | 1 | 4 | 4 | 4 | 上轮真实数据 |
| tp2_real | 2 | 1 | 4 | 4 | 8 | 上轮真实数据 |
| cp2_real | 1 | 2 | 4 | 4 | 8 | 上轮真实数据 |
| ep2_dp4 | 1 | 1 | 4 | 2 | 4 | 新增 EP 隔离对比 |
| tp2_dp2_ep4 | 2 | 1 | 2 | 4 | 4 | 新增 TP/CP/DP 隔离对比 |
| tp4_dp2_ep4 | 4 | 1 | 2 | 4 | 8 | 新增 TP/EP 隔离对比 |

所有配置 GBS=4，每步一个 global microbatch，sequence length=256。
DP1/2/4 分别在每个 dense-DP replica 打包 4/2/1 条等长序列，全局输入不变。
公共 objective adapter 按真实有效 token 和逐递归轮 candidate 数归一化。
CPU 分片检查要求 global ID 无重复/遗漏、shifted labels/mask 相同、全局均值一致。

从相同的初始完整权重重跑 100 步，非从原 50 步末尾续训。lr=1e-5、Adam eps=1e-6、
clip_grad=1，scheduler total_training_steps=100。前 50 步**输入**与旧实验相同，
但 scheduler horizon 不同，不承诺前 50 步 loss 轨迹与旧 50 步任务逐项相同。
路由为 learned，不冻结 expert/depth 选择。strict=True；显式启用
`MAGI_ATTENTION_DETERMINISTIC_MODE=1`。保留原 attention backend/dtype/环境。

各 EP 使用对应的既有 model-only folded checkpoint：

- EP1/EP4：`mor_mlite/artifacts/eos/5998357/qwen30b/folded_init_ep{1,4}`。
- EP2：实验项目 `runtime/ckpt_tools/mor-bf16-axes/6006410/folded_init_ep2`。

不能以 checkpoint 路径相等判断跨 EP 初始化相等。模型实际加载后，每个 rank
（含所有 CP 副本）对 `loaded-tp-bitwise/6011449/baseline` 的完整参数 raw bytes
逐项比对并核验 TP/EP 覆盖；只有通过才能记录可比较的训练结果。

固定环境：EOS H100 80GB，NGC PyTorch 26.01 container，Torch 2.10.0+cu129、
TE 2.13.0、MagiAttention 1.1.1，Megatron-LM
`5c8315f12a64a7279eec58896af9e74ee3351b74`。生产源码 hash：
`c49e48c4d882aa7e8f0a3b3ea98d42711ba260f42cc0bbd21d8164618b870c87`。

注意：实际 pinned Qwen MoELayer 设置 `compute_aux_loss=False`；训练目标为
native LM loss + depth-router aux，不额外包含普通 MoE load-balancing aux。
旧 50 步文档/manifest 的“普通 MoE aux 开启”描述不准确，文档已更正，原数值产物保留。
100 步脚本还检查实际构建的 router policy 并记录 module 数量。

## 数据来源

CPU 作业 `6012231` 已完成。复用已缓存的 Pile GPT-2 indexed shard，GPT-2 解码严格
回环校验后，用与目标 Base 词表/merges hash 相同的 Qwen3 BPE 重新编码，EOS=151643，
不套聊天模板。没有下载数据、没有循环旧 token，也没有更改训练依赖环境。

- 原始记录：同一 shard 的 index 0–93，共 94 条。
- 结果 shape：`[100,4,256]`，102,400 个输入 token，102,000 个有效训练 target。
- 100 步 token SHA256：`3e5122e662650ee8432165b6a139f354092367669d218a813706935da8d904b6`。
- 前 50 步 bitwise 等于旧 `data-6012191`，prefix SHA256：
  `023722b1986701b90601de4b1f68ccf96332c515cca2ca3ed01eac3affd89216`。
- 数据目录：`runtime/ckpt_tools/real-data-100/data-6012231/`，保存 tokenizer/source
  manifest 与独立 `prefix-check.json`。

## 作业与最终状态

- `6012240`，eos0050：`baseline_real / tp2_real / cp2_real / all / tp4_dp_ep / ep2_dp4`。
- `6012241`，eos0386：`baseline / zero1 / tp / cp / ep / tp_cp_ep / tp_dp_ep / cp_dp_ep`。
- `6012301`，eos0361：`tp2_dp2_ep4 / tp4_dp2_ep4`，在第二批结束后提交。
- 三批均先执行 objective 单测、DP1/2/4 分片检查、固定环境检查和 TE canary。
- 每组独立 torchrun，失败原始 traceback 保留。driver 继续遍历并记录子进程失败；
  SLURM driver 正常结束不表示每组都完成训练、更不表示 numerical parity 通过。

三批 driver 均已结束，SLURM elapsed 分别为 20:19、09:10、07:06。
16/16 组均实际尝试，7 组各完成 100 次 optimizer update，共 700 次；
另外 9 组 CUDA OOM，均未完成/记录首次更新。没有仍在排队或运行的本轮任务。

### 完成的 7 组

以下 Δ 和 MAE 均相对 TP1/CP1/DP4/EP4；MAE 是 100 个逐步绝对差的平均值，
不是 loss 的平均值，也不是 logits relative-L2。
第 100 步 loss 在该次更新前计算；其后第 100 次更新已成功，未额外运行更新后 validation。

| 名称 | TP/CP/DP/EP | 第 100 步 LM loss | 最终 Δ | 100 步 MAE | 最大绝对差（step） | 峰值 allocated GiB |
|---|---|---:|---:|---:|---:|---:|
| baseline_real | 1/1/4/4 | 6.240067005 | 0 | 0 | 0 | 65.05 |
| tp2_real | 2/1/4/4 | 6.274415255 | +0.034348249 | 0.059937295 | 0.455906868（2） | 43.16 |
| cp2_real | 1/2/4/4 | 6.241385818 | +0.001318812 | 0.042936823 | 0.325671196（4） | 46.19 |
| all | 2/2/2/4 | 6.252737999 | +0.012670994 | 0.047334817 | 0.421155691（4） | 43.17 |
| tp4_dp_ep | 4/1/2/2 | 6.254544258 | +0.014477253 | 0.041386835 | 0.456070900（2） | 62.64 |
| tp2_dp2_ep4 | 2/1/2/4 | 6.257241488 | +0.017174482 | 0.045157802 | 0.458784103（2） | 62.04 |
| tp4_dp2_ep4 | 4/1/2/4 | 6.242172480 | +0.002105474 | 0.056502170 | 0.353976488（3） | 41.67 |

显存为训练循环内各 rank、各 step 的 PyTorch `max_memory_allocated` 最大值，
不是设备总占用，不包含全部 NCCL/driver 分配，也不覆盖模型加载前的峰值。
因此不能拿它与 OOM 日志里的整卡 79.11 GiB 直接相减估算可用空间。

### 只改变一个并行轴的结果

Δ 始终为“箭头右侧减左侧”的第 100 步 LM loss；其他列比较同一配对，
不是全部相对主参考。每组保持 GBS=4、相同逐步 global input、相同初始化和 optimizer。

| 改变量 | 固定的轴 | 最终 Δ | 100 步 MAE | 最大绝对差（step） | 前 10 步 MAE | 后 50 步 MAE |
|---|---|---:|---:|---:|---:|---:|
| TP1 → TP2 | CP1 DP4 EP4 | +0.034348249 | 0.059937295 | 0.455906868（2） | 0.211240232 | 0.043317374 |
| CP1 → CP2 | TP1 DP4 EP4 | +0.001318812 | 0.042936823 | 0.325671196（4） | 0.133693731 | 0.033516200 |
| CP1 → CP2 | TP2 DP2 EP4 | −0.004503489 | 0.044313438 | 0.284018993（4） | 0.128305936 | 0.033109150 |
| TP2 → TP4 | CP1 DP2 EP4 | −0.015069008 | 0.051253554 | 0.274883270（1） | 0.168636012 | 0.028080347 |
| EP4 → EP2 | TP4 CP1 DP2 | +0.012371778 | 0.030587251 | 0.204269409（2） | 0.057802463 | 0.032310581 |
| DP4 → DP2 | TP2 CP1 EP4 | −0.017173767 | 0.051204590 | 0.348808050（6） | 0.137613571 | 0.036337104 |

结论：TP、CP、EP、DP 的受控配对均存在非零训练 loss 差异，不是仅 TP 才有。
例如 CP2 终点相对差仅 +0.02113%，但整段 MAE=0.04294、最大差=0.32567，
不能凭终点接近宣称轨迹对齐。所有配对的后 50 步 MAE 都低于前 10 步，
本次 100 步没有呈现差异随步数单调扩大的现象；这不证明长期稳定或精度合格。

这里的“单轴”指仅改变一个**配置维度**。实际通信分组、局部 GEMM/token 形状、
expert 数据并行副本数等可以随之变化；learned routing 及优化器状态也可逐步分歧。
本轮没有保存完整逐算子/逐参数梯度轨迹，不能把表中差异全部归因于某一个 reduce，
也没有测量 100 步同拓扑重复运行的方差，不能据此给出统计显著性结论。

### 未完成的 9 组：显存阻塞，不是数值通过

| 名称 | TP/CP/DP/EP | 首个 OOM 阶段 | 已完成更新 |
|---|---|---|---:|
| baseline | 1/1/1/1 | DDP FP32 gradient buffer 初始化 | 0 |
| zero1 | 1/1/2/1 | DDP FP32 gradient buffer 初始化 | 0 |
| tp | 2/1/1/1 | DDP FP32 gradient buffer 初始化 | 0 |
| cp | 1/2/1/1 | DDP FP32 gradient buffer 初始化 | 0 |
| ep | 1/1/2/2 | 首次 backward 的 TE LayerNormLinear wgrad GEMM | 0 |
| tp_cp_ep | 2/2/1/2 | 首次 optimizer step 的 Adam state 初始化 | 0 |
| tp_dp_ep | 2/1/2/2 | 首次 optimizer step 的 Adam state 初始化 | 0 |
| cp_dp_ep | 1/2/2/2 | 首次 optimizer step 的 Adam state 初始化 | 0 |
| ep2_dp4 | 1/1/4/2 | 首次 optimizer step 的 Adam state 初始化 | 0 |

单卡 baseline 在 `_ParamAndGradBuffer.grad_data = torch.zeros(...)` 申请 45 GiB，
GPU 总容量 79.11 GiB、只剩 25.37 GiB 时失败；该配置尚未加载 checkpoint 或进入 forward。
`ep` 在 TE backward 分配 20 MiB 时 GPU 仅余 11.56 MiB。
`ep2_dp4` 已完成加载检查和 forward/backward，但 TE FusedAdam 首次初始化
`exp_avg_sq` 分配 20 MiB 时仅余 17.56 MiB；相应进程整卡占用已达 79.08 GiB。
其余 Adam OOM 同样发生在 `initialize_state` / `_initialize_state`。
它们的最后一条 instrumented stage 虽然是 `auditing_loaded_parameters`，
但实际 OOM 阶段必须依据 traceback，不能误报为参数检查 OOM。

当前结论仅是这些原配置在当前 H100 80GB、完整模型、既有训练状态布局下无法完成。
尚未验证保持数值语义的显存优化或更大显存设备能否补齐；没有使用缩模型、改 dtype、
换 optimizer/offload 或改并行度来伪装原配置通过。补齐这 9 组仍是未完成项。

## 结果解释规则

每组必须有连续 step 1–100、每次 optimizer update 成功、loss/gradient norm 有限、
全量初始参数 byte gate 通过，才计为“完成 100 步”。比较器同时要求源代码、训练
脚本、数据、optimizer/routing/precision、依赖版本与 Magi 环境合同一致，
且逐步输入 hash/target 数相同。
未完成组单列 stage、已记录更新数和 OOM/错误，不给不存在的最终 loss。

可运行的主参考是 TP1/CP1/DP4/EP4 (`baseline_real`)，不能称其为单卡无并行参考。
跨多轴结果列出完整拓扑；只有其他轴固定的配对才称 TP/CP/EP 单轴比较。
报告逐步 CSV、最终差、平均绝对差、最大绝对差及所在 step，而非只看有符号均值。
100 步 loss 对比不替代旧 logits/逐参数 gradient-update parity 验收，也不证明长期收敛。

## 验证与产物

7 组完成结果均通过：5266 个 canonical parameter、13,084,750,848 个 unique element、
所有 CP 副本的 loaded-parameter byte 检查，`different_bytes=0`；初始权重差异已排除。
每组有恰好 100 条连续记录，全部 update 成功且 loss/grad norm 有限，
逐步 global input SHA256 一致，每步 valid_targets=1020。
实际构建的 20 个 native MoE router 均确认 `compute_aux_loss=False`。
执行脚本 SHA256：`2583265c46c4b28cd87071ceac07f8e9d53e7dd4a19d1e56a21e25a02f12062e`。

每批 GPU 作业均通过 5 项公共 objective 单测、DP1/2/4 输入/目标/scaling 检查和
版本/TE canary。离线比较器 4 项单测通过，涵盖符号/绝对差计算、输入错配、
optimizer/版本合同错配及不足 100 步拒绝。源代码本轮未改变。

- [最终机器可读报告](../runtime/ckpt_tools/real-data-100/report/report.json)：16 组覆盖、7 组完整结果、6 对单轴结果、9 组原始 OOM 摘要。
- [700 行逐步比较 CSV](../runtime/ckpt_tools/real-data-100/report/loss-comparison.csv)。
- [完整 loss 图](/home/scratch.hongbinl_sw/work/fsdp/agentic-mcore-dev/.claude/worktrees/mor-mlite-validation/vendor/mor_mlite/runtime/ckpt_tools/real-data-100/report/loss-curves.png)。
- [单轴差值图](/home/scratch.hongbinl_sw/work/fsdp/agentic-mcore-dev/.claude/worktrees/mor-mlite-validation/vendor/mor_mlite/runtime/ckpt_tools/real-data-100/report/single-axis-diffs.png)。
- 原始分组目录：`runtime/ckpt_tools/real-data-100/{6012240,6012241,6012301}/`，
  包含 plan/status、完整 subprocess 日志、逐 rank loaded audit、manifest、执行脚本副本和 loss.jsonl。

![真实 Pile 100 步训练 loss 与逐步差值](/home/scratch.hongbinl_sw/work/fsdp/agentic-mcore-dev/.claude/worktrees/mor-mlite-validation/vendor/mor_mlite/runtime/ckpt_tools/real-data-100/report/loss-curves.png)
