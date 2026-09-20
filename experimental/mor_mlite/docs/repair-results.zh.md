# 2026-09-10 遗留问题修复与验证

承接 Codex task `01a0799b-f8a1-7171-ae82-809b58c678c1`，以 EOS 原项目源码为基线。
修改位于 `codex/fix-mor-mlite-validation` 对应的独立 `vendor/mor_mlite` checkout；
原 EOS `/lustre/fsw/coreai_devtech_all/hongbinl/mor_mlite`、Megatron-LM 和 MagiAttention
源码均未覆盖。按 `dev-setup` 隔离开发，按 `cluster-run` 提交本包脚本。

用户随后要求先 debug 数值 diff 来源。新增 `6006544` 完整 48 层自然轨迹和
`6006559` 逐算子同输入/独立数学对照，结论见 [diff-source.zh.md](diff-source.zh.md)：
TP 首差是第 0 层 row projection 的 BF16 partial 舍入/归约；CP 首差在第 0 层 core，
另有更小的 QKV/projection/norm 差异。两组 48 层 MoE 同输入全部 exact。
这是诊断结果，不是 F1 已修复，也没有将 FP32-heavy 路径正式化。

## 修复范围

| 审查项 | 本轮处理 |
|---|---|
| F1：30B forward 超阈值 | 原生 BF16 尚未关闭；用户拒绝 FP32-heavy 正式路径后继续根因定位，已发现 strict baseline 的 unfused core attention 数值偏差，见续查 |
| F2：双方缺失训练证据仍通过 | 新增独立 evidence schema；按 step/microbatch/round 和初始化参数清单校验完整性；缺 namespace、参数或 routes 均失败 |
| F3：漏计选择未变的 near ties | 先逐样本分类，再区分 `near_ties` 与 `changed_near_ties`；重新跑 tiny 后确认旧统计确有漏计 |
| F4：replay 缺轮/错误 metadata 被接受 | reference 要求完整轮集合；本地与 distributed routing 校验 ID、sample、position、容量与 cu_seqlens |
| F5：PackedBatch 合同不一致 | 共用 target/mask shift 与 stable sample-ID→length 映射；支持 None mask、非连续 ID；修复小于 1 的非零 mask 权重分母 |
| F6：浮点预算少选 token | linear 使用整数除法；custom 首轮必须 exact 1；覆盖多个 recursion count 和长度的性质测试 |
| F7：旧报告与源码未绑定 | 源码内容快照、启动/写出一致性检查、manifest/routes/tensors 哈希；matrix receipt 核对同源/同实验/拓扑/模式 |
| F8：Magi decoder 逐 token CUDA item | 将 canonical token IDs 一次拷到 CPU，再进行 Python metadata lookup；未据此宣称长序列吞吐提升 |
| F9：目标归一化仅在私有 trainer | 提供公共 `objective_scales` / `apply_objective`，trainer 复用；不等长 DP/microbatch 和局部零权重的 loss/gradient 回归 |
| F10：维护问题 | 修复 Ruff 格式，抽出 objective/evidence/provenance；大 trainer 的全面拆分、通用 dataset frontend 仍未完成 |

完整训练比较默认失败即停止。显式 `--diagnostic` 返回 `status=partial`、
`acceptance_complete=false`，不能写完整矩阵凭据。forward-only 不要求不存在的 LM loss，
但仍要求完整 logits、各轮 hidden/scores/gates 和 RoutePlan。注入 operator oracle 的产物
只能诊断，不能成为 forward 或 training acceptance。

EOS 环境校验也修正了历史源码路径硬编码：`MOR_PROJECT_ROOT` 必须是本次启动目录，
且与实际 import 根目录一致；固定版本、venv 来源、GPU、Slurm account/partition 检查保留。
同步目标使用独立的 `MOR_EOS_REMOTE_ROOT`，提交模板不再静默跳回历史目录。

## 环境与作业证据

固定配置：单节点 H100，1/2/4/8 ranks；Torch `2.10.0+cu129`、CUDA `12.9`、
Transformer Engine `2.13.0`、NCCL `2.27.5`、MagiAttention `1.1.1`；Megatron-LM
`5c8315f12a64a7279eec58896af9e74ee3351b74`。NGC `nvcr.io/nvidia/pytorch:26.01-py3`
作为容器 bootstrap，运行时仍使用已锁定的 cu129 venv。

| EOS job | 结果 / 说明 |
|---|---|
| `6006078` | 初次新增测试：306 passed、2 failed、1 skipped；修正未使用 expert 的 None-grad 测试，并让 artifact 显式保存其数学零梯度（不改变 optimizer skip 行为） |
| `6006090` | 环境 guard 拒绝隔离路径；修复源码来源检查后重跑，未绕过 guard |
| `6006101` | `COMPLETED 0:0`，17:57；完整 tiny 矩阵 20/20 通过，pytest 320 passed、1 skipped |
| `6006102` | 两种精度诊断收集完成，但两个数值报告都失败；进程 exit 0 不是 parity 通过 |
| `6006114` | FP32 residual/norm/QKV/projection 组合诊断收集完成；数值仍失败 |
| `6006133` | 相同输入的 native end-block 诊断收集完成；只支持局部归因，不支持端到端验收 |
| `6006143` | 逐个固定 end submodule 输入，追加 output projection FP32 对照；仅诊断 |
| `6006167` | 混合精度 attention 的完整 forward 比较通过；FP32 expert 诊断仍失败；均未验证 backward |
| `6006134` | 在新增 fractional-mask 根因修复后主动取消（1:14），不计作测试通过 |
| `6006137` | `COMPLETED 0:0`，18:25；第一阶段冻结快照 pytest 329 passed、1 skipped；完整矩阵 20/20 通过；不覆盖后续 attention 修复 |

首轮完整矩阵源码 hash：
`7fcb7fe68927d68f053d0931f79c1186092d2e2910cf173232a5aa1ccd2d1ae9`。
第一阶段修复源码 hash（59 个 Python 文件）：
`49e6c23bc3d1940340ff7d932d345f0f8277ce4910f7cfb30f462a9e3d630a2b`。
hash 包含未提交源码，不依赖仅有一个旧 git SHA。源码快照完整文件清单保存在每个 artifact
manifest 内，matrix receipt 另行绑定三个 artifact 文件的 SHA-256。

首轮证据：[matrix_complete.json](../artifacts/eos/6006101/tiny/reports/matrix_complete.json)、
[all.json](../artifacts/eos/6006101/tiny/reports/all.json)、
[pytest.xml](../artifacts/eos/6006101/tiny/reports/pytest.xml)。
最坏硬门槛 relative-L2：forward `0.00504269`、全梯度向量 `0.00298666`、
全 FP32 master update 向量 `0.01242692`、resident post-step tensor `0.00164567`。
gradient/update 的门槛针对按阶段/step 重建的完整向量，不等于每个参数都逐一小于 3%。

修正统计后，8 份 learned topology 报告各有 15 个 near ties；选择仍一致，
`changed_near_ties=0`。历史“zero near ties”不能继续作为无歧义路由的证据。

第一阶段证据：[matrix_complete.json](../artifacts/eos/6006137/tiny/reports/matrix_complete.json)、
[all.json](../artifacts/eos/6006137/tiny/reports/all.json)、
[pytest.xml](../artifacts/eos/6006137/tiny/reports/pytest.xml)。已在本地复核全部 20 份报告哈希，
receipt 的 source hash 与该阶段快照一致，不覆盖当前续修源码。唯一 skip 是原有 opt-in MLite CLI smoke；
实际 MLite 1/2/4/8 卡训练、更新和 checkpoint 由完整矩阵另行执行。

## 30B：已排查什么，仍缺什么

仍使用原 `Qwen/Qwen3-30B-A3B-Base` folded checkpoint（结构 `3 + 14 x 3 + 3`），
seq_lens `[128,128]`、单卡 baseline 串行执行两个 dense-DP shards，candidate
`TP=2, CP=2, DP=2, EP=4`。depth route 与 native expert IDs/consumed scores replay 保留。
没有修改验收阈值：forward relative-L2 ≤ 0.02 且 cosine ≥ 0.999。

| forward 路径 | logits relative-L2 | cosine | 结论 |
|---|---:|---:|---|
| 原生 BF16，历史 job `5998357` | 0.05673339 | 0.99840639 | 失败 |
| 层内 FP32 residual，`6006102` | 0.07301072 | 0.99733461 | 失败 |
| 上项 + FP32 TP output projection，`6006102` | 0.05404205 | 0.99853872 | 失败 |
| FP32 residual/norm/QKV/projection，`6006114` | 0.05027049 | 0.99873654 | 失败 |
| FP16 attention + FP32 QKV/QK norm/residual/projection，`6006167` | **0.01646665** | **0.99986452** | **forward 通过，非原生 BF16 路径** |
| FP32 expert/residual/projection 等，attention 仍为 BF16，`6006167` | 0.05052144 | 0.99872647 | 失败 |

前两组诊断的比较器还误要求 inference-only 的 loss tensors；这一 schema 问题已修复。
上述数值均直接来自 tensor metrics，不依赖该 schema 失败。`6006114` 的 evidence 完整性
已通过，失败确实来自数值。所有 FP32 probe 都保留为独立 forward-only 实验脚本，没有
替换正式 BF16 训练模块，也不声称其 backward、optimizer 或性能已验证。

报告：[residual_fp32](../artifacts/precision-6006102/reports/residual_fp32.json)、
[both](../artifacts/precision-6006102/reports/both.json)、
[stable_fp32](../artifacts/precision-6006114/reports/stable_fp32.json)。

`6006133` 给三个 native end blocks 分别注入同一 baseline 的输入 hidden（按原 global
token ID 映射，dummy 除外），保持权重与 expert replay 一致。各 block 输出 relative-L2
变为 `0.00610936 / 0.00491935 / 0.00307233`；最后 logits 为 `0.00856829`。
但 end-block-0 的 MoE 输出仍有 `0.02671431` 差异。这支持累计误差被后续层放大的解释，
不等于已证明每个 primitive 的根因，更不等于端到端 logits 已修复：本实验人为重置了
每个 end block 的输入，报告明确为 partial。
证据：[end_operators.json](../artifacts/operators-6006133/reports/end_operators.json)。

`6006143` 进一步分别固定 attention、MLP norm、MoE 的输入，不只固定整个 block 输入。
三个 end layers 的 **MLP norm 输出和 MoE 输出均 bitwise exact**；此前 2.67% 的 MoE
差异来自其输入不同，不能归因于相同输入下的 EP expert 计算。

| 相同 attention 输入 | end-0 relative-L2 | end-1 | end-2 |
|---|---:|---:|---:|
| 原生 attention | 0.01851334 | 0.00674628 | 0.00283045 |
| 只将 output projection 改为 FP32 | 0.01842372 | 0.00653373 | 0.00254469 |

推断：这些末端层的主要差异在 attention 路径，且不能仅用 TP output projection 的
BF16 累加解释；仍需区分 input norm/QKV、QK norm/RoPE 和 Magi core attention。
这不是对所有 recurrent layers 的逐算子证明。
报告：[native submodules](../artifacts/submodules-6006143/reports/native.json)、
[projection FP32](../artifacts/submodules-6006143/reports/projection_fp32.json)。

### 可选数值路径的突破与边界

`6006167` 的 `fp16_attention` profile 保留全部 BF16 参数和原生 BF16 expert 计算，
使用 FP32 residual、FP32 RMSNorm/QKV/QK norm/RoPE，在进入 Magi attention 前将 Q/K/V
转为 FP16，output projection 和 TP reduction 保持 FP32；最后 norm 的输出仍转为 BF16
交给原生 head。FP16 增加 mantissa 精度但缩小动态范围，不能仅凭本次短序列通过就宣称
训练安全或性能可接受。

结果：所有 172 个 tensor hard gates 通过（包含 expert route identity/score），3 轮
depth replay 和 48 个 native expert contexts 一致；最坏 forward relative-L2 即 logits
的 `0.01646665`，原阈值不变。该 profile 不注入 baseline hidden，执行的是完整 forward
图；仍然使用与此前一致的 depth/expert replay，因此不能替代自然 learned routing 验证。

证据：[fp16_attention.json](../artifacts/precision-6006167/reports/fp16_attention.json)。
该隔离实验源码 hash 为
`3db6dc393131201666b07f584baa39f2edcc028977a253a1e8f395d229cf1e8b`（早于最后的
fractional-mask 修复；inference 无 LM loss，不涉及该修复），实验脚本 hash 另在 artifact
manifest 的 `precision_probe` 字段记录。[FP32 expert 对照](../artifacts/precision-6006167/reports/full_fp32.json)
没有达到阈值，进一步支持 attention 数值路径的重要性，但不构成所有算子的完整根因证明。

目前实现是明确拒绝 training 的独立诊断脚本。若允许采用可选混合精度正式路径，下一步
需要保留原生 BF16 默认和参数/checkpoint 布局，实现可反向的 TP 通信与 main_grad 对接，
重跑 tiny forward/gradient/update/checkpoint 全矩阵及 30B backward/optimizer，最后测显存
和吞吐。尚未实施这些训练改动，不能将这条 forward 报告描述为整个框架验收完成。

**用户决定（2026-09-10）**：不采用增加 FP32 计算的正式路径，仅对 BF16+MXFP8 方向
表示可以接受。本次通过的方案实际是 BF16+FP16+FP32，**不是 MXFP8**，因此不将它
移植或启用为正式训练路径。保留诊断结果用于复现；原生 BF16 默认不变。MXFP8 未实现、
未验证，本轮不将上述条件许可推断为已启动新的 MXFP8 开发。30B 原生 BF16 精度仍未关闭。

### 原生 BF16 续查：baseline 与 core attention 的差异

用户指出核心问题未解决不应收尾；拒绝 FP32-heavy 方案不是停止原生 BF16 排查。
继续保持 2% / 0.999 门槛，不把 tiny 验收代替 30B 验收。

`6006259` 的诊断工具单测 3/3 通过，但单卡 serial-DP baseline 误传 replay 参数，被
已有 guard 正确拒绝。修正诊断输入为 learned baseline，保留 candidate replay 后重跑。
`6006269` 在 eos0273 `COMPLETED 0:0`，用时 4:08；同样 3/3 工具回归通过。

本次逐边界记录三个 end layers。给整个 end layer 相同 hidden 后，QKV projection 输出、
QK norm 输出、RoPE 后 Q/K 和 V 在所有原 token/head 上 **bitwise exact**；差异首次出现在
core attention。再显式给 core 输入相同 Q/K/V，指标完全重复，排除了本次末端层的前置
算子/位置映射差异。单卡 strict 实际使用 TE unfused，而 CP 使用 Magi FFA。

| 相同 BF16 Q/K/V 的 core output 相对独立 FP32 数学 attention | end-0 L2 | end-1 L2 | end-2 L2 |
|---|---:|---:|---:|
| 单卡 strict TE unfused | 0.01275336 | 0.00484417 | 0.00440157 |
| 多卡 Magi FFA | 0.00185838 | 0.00175222 | 0.00167531 |

两 native core 之间 L2 为 `0.01289005 / 0.00511597 / 0.00463900`。TE 2.13 unfused
源码的 `torch.baddbmm` 把 scaled QK scores 写为 Q 的 BF16 dtype，softmax 之前已发生
一次舍入；Magi 更接近独立数学结果。这些证据支持优先纠正 baseline 的内核选择，
但尚不证明 30B 端到端差异全部来自这一处。数学基准仅用于诊断，不是正式 FP32 训练路径。

结果：[baseline.json](../artifacts/attention-6006269/baseline.json)、
[native.json](../artifacts/attention-6006269/native.json)、
[identical_core.json](../artifacts/attention-6006269/identical_core.json)。
输入注入产物均标记 `operator_probe` / `acceptance_eligible=false`，不能签发验收证书。

历史 fused job `5997449` 实际在 CUDA 12/13 库混载检查处失败，没有执行 fused 数值对照。
续查使用新建的无 system-site-packages 环境副本，原 venv 和依赖源码不修改；为 fused
core 补独立 GPU forward/backward 检验，使用真实 30B 捕获的 Q/K/V，参考公式为
[Qwen/HuggingFace transformers v4.57.1](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py)
的 `eager_attention_forward` / `repeat_kv`，将相同输入提升到 FP32 评估数学结果，而非
复刻 BF16 eager 的中间舍入。该 core 无可训练参数，检查 output 与 Q/K/V 三类梯度。

#### 环境与本地 FFA 适配

新副本 `.deps-cu129-clean/venv-torch210-cu129-v3` 禁止继承 system-site-packages，
保留原 venv 不变。`6006280` 的 pip check 正确报告缺少继承自容器的支持依赖，以及
TE cu12 wheel 的已知内部 cp310 tag 错误。按原环境精确版本安装依赖；仅在新环境
回补 [NVIDIA/TransformerEngine#2896](https://github.com/NVIDIA/TransformerEngine/issues/2896)
的 Python-agnostic core wheel metadata 修正，同步 RECORD 并记录旧/新哈希，不改变
任何二进制或 PyTorch extension ABI tag。`6006303` 的 Magi import 暴露额外 `debugpy`
依赖；补齐后 `6006321` 的 pip check、完整环境 guard、TE BF16 Linear 前后向均通过。

`6006321/6006334` 的 TE fused core 仍失败。根因不等于“进程开始时已经混载”：
开始的 `/proc/self/maps` 只有 CUDA12；TE 内嵌的 cuDNN frontend
`load_cudart_so()` 主动对 `libcudart.so.12` 和 `.13` 分别 `dlopen`，发现两者可用就抛错。
该 TE 2.13 二进制没有新版 `CUDNN_FRONTEND_CUDART_LIB_NAME` selector，故不能仅设置
新版环境变量解决，也没有关闭/替换这个检查。TE fused 试修未通过、不是当前正式候选。

Magi v1.1.1 的 `functional.flex_flash_attn_func` 明确支持无 distributed 环境的本地调用。
本包新增 `LocalMagiAttention` adapter，CP=1 strict 使用本地 FFA，CP>1 仍使用原
Magi dispatch/runtime；保留 QKV、norm、RoPE、projection、BF16 参数与激活以及原阈值。
原生 distributed Magi 的 CP>1 guard 未放宽。adapter 检查 THD/causal/GQA/lengths，
压紧有 padding gap 的序列，输出恢复物理槽位且 dummy 梯度为零；没有新增参数或 optimizer
state。CPU 模拟内核用于 metadata/gradient 合同回归，真正内核另对独立 FP32 数学参考验证。

`6006346` 暴露 FFA deterministic 变体需 JIT 编译，而诊断环境的 CUDA_HOME 误指向只有
runtime headers 的 cu129 wheel。修正为容器实际 NVCC13.1 toolkit，保留 Torch cu129，
继续 `6006352`。原 Magi 扩展本来就链接 libcudart13（readelf 已核实），故不能声称整套
Magi 依赖“只加载 CUDA12”；Torch/TE 的 cu129/cu12 pin 与版本来源检查仍保持。

当前候选源码快照（60 个 Python 文件）：
`bef48c86458b8545c3ebb09300ef091efe6cb9a9978efa495da92eaba1c1707d`。
`6006360` 对该隔离快照依次执行完整 pytest、真实 QKV 的 adapter 前后向、单卡 baseline
和 8 卡完整 30B forward 比较；尚未取得其结果时不得当作通过。

`6006352` 在 eos0130 `COMPLETED 0:0`（5:22，包含首次编译），本地 functional FFA
在三个真实 30B QKV 上的 output/dQ/dK/dV 共 12 项均通过独立数学检验；最低 cosine
`0.9999823301`、最低 tensor similarity `0.9999823294`、最大 relative-L2 `0.00594474`。
forward output 的最大 L2 `0.00193962`；BF16 输入/输出不变，FP32 只用于数学对照。
证据：[kernel.json](../artifacts/fused-6006352/kernel.json)；目录名沿用诊断入口，
报告明确 `backend=magi_local`，不是 TE fused 通过。

`6006360` 完整 pytest 为 369 passed / 1 skipped（原 opt-in CLI smoke），但此快照
尚缺 checkpoint run_contract 的 attention_policy 白名单。补齐 schema/producer 合同
回归后，当前新快照为
`c49e48c4d882aa7e8f0a3b3ea98d42711ba260f42cc0bbd21d8164618b870c87`。
旧的、不含 attention policy 的 full-training receipt 不再被当作同配置续训凭据；
model-only folded checkpoint 仍可加载。后续完整矩阵和 30B 重测绑定新快照。

#### 完整 30B 重测与拓扑拆分（尚未通过）

`6006360`（eos0497）真实 adapter 的独立 forward/backward 通过，但完整 30B 的
logits relative-L2 为 `0.0441226644`、cosine `0.9990262189`，15 个 hard gates 失败。
`6006397`（eos0512）用上述 `c49e48...` 新快照得到完全相同的数值；完整 pytest 为
374 passed / 1 skipped。相比旧 TE-unfused baseline 的 `0.05673339` 有改善，但仍高于
2% 门槛，因此不是完整修复。两次均执行无 hidden 注入的完整 forward，depth/expert
replay、metadata 和证据检查通过。

`6006410`（eos0161，4:48，数值失败退出 1）从同一 HF snapshot
`1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9` 通过 `mor_mlite.convert_hf` 重新生成真实
EP2 folded DCP；没有放宽跨 EP 加载检查。固定 `6006397` 的 serial-DP2 baseline，
得到如下完整 forward 结果：

| topology | TP / CP / DP / EP | logits relative-L2 | cosine | 结果 |
|---|---|---:|---:|---|
| all (`6006397`) | 2 / 2 / 2 / 4 | 0.04412266 | 0.99902622 | 失败 |
| tp_dp_ep (`6006410`) | 2 / 1 / 2 / 2 | 0.04558517 | 0.99896160 | 失败 |
| cp_dp_ep (`6006410`) | 1 / 2 / 2 / 2 | 0.05854544 | 0.99829714 | 失败 |

这排除了“只修 TP 就足够”的假设；两种拆分仍共用 EP2，不能凭此把差异单独归因于
TP 或 CP。下一步检查 EP-only 和相同输入的末层 attention。报告分别见
[all](../artifacts/bf16-6006397/qwen30b/reports/all_vs_baseline.json)、
[TP+DP+EP](../runtime/ckpt_tools/mor-bf16-axes/6006410/reports/tp_dp_ep.json)、
[CP+DP+EP](../runtime/ckpt_tools/mor-bf16-axes/6006410/reports/cp_dp_ep.json)。

`6006431` 的完整 forward 子实验中，`zero1`（TP1/CP1/DP2/EP1）和 `ep`
（TP1/CP1/DP2/EP2）均通过：所有 hard-gated forward tensors 与相同单卡 baseline
逐位一致，logits L2 为 0。这排除了当前输入上的 DP 切分、EP2 权重转换与 expert
dispatch 本身；并不证明任意配置的 EP 都无误差。证据：
[zero1](../runtime/ckpt_tools/mor-bf16-axes/6006431/reports/zero1.json)、
[ep](../runtime/ckpt_tools/mor-bf16-axes/6006431/reports/ep.json)。

`6006431` 最终 `COMPLETED 0:0`（eos0446，4:37）。同输入的末层边界实验使用新
baseline：三个 end layers 的 QKV、QK norm、RoPE 仍逐位一致；core attention 的
topology 间 relative-L2 为 `0.00156029 / 0.00130970 / 0.00096915`。相同 Q/K/V
下，CP1 local FFA 对独立数学参考的 L2 为 `0.00193425 / 0.00178177 / 0.00169272`，
CP Magi 为 `0.00185536 / 0.00175199 / 0.00167347`。相比旧 unfused baseline，局部
kernel 差异已明显缩小，但这些 partial/input-injection 结果不构成完整 forward 通过。
[baseline boundaries](../runtime/ckpt_tools/mor-bf16-axes/6006431/baseline_boundaries.json)、
[all boundaries](../runtime/ckpt_tools/mor-bf16-axes/6006431/all_boundaries.json)。

后续 rounding/staging 诊断不改变 package source：分别使用 BF16 输入/权重 GEMM，
只把 TP 局部 accumulator 保留到求和后再转 BF16；以及 native Magi 的 no-overlap
调度。前者通过 [PyTorch 2.10 `torch.mm(out_dtype=torch.float32)`](https://docs.pytorch.org/docs/2.10/generated/torch.mm.html)
实现，不是 FP32 输入 GEMM，但确实增加 FP32 临时 reduction buffer；**用户没有同意
把该路径正式化**。两者都标记 `operator_probe` / `acceptance_eligible=false`，无 hidden
注入，仍按原 tensor 门槛收集诊断；不能自动据此替换训练实现或签发完整验收。

`6006454`（eos0122，5:14）完成三组诊断收集，进程退出 0 **不代表数值通过**：

| 对照 | post-merge hidden L2 | logits L2 | 结论 |
|---|---:|---:|---|
| TP2/CP1，BF16 GEMM + FP32 临时 TP reduction | 0.00356309 | 0.04561992 | 未通过；改善中段误差但末端仍放大 |
| TP1/CP2，仅 no-overlap | 0.00733324 | 0.05854544 | 未通过；与原 CP2 结果相同 |
| TP2/CP2，两者组合 | 0.00733307 | 0.05982499 | 未通过 |

报告：[tp_accumulator](../runtime/ckpt_tools/mor-bf16-axes/6006454/reports/tp_accumulator.json)、
[cp_no_overlap](../runtime/ckpt_tools/mor-bf16-axes/6006454/reports/cp_no_overlap.json)、
[both](../runtime/ckpt_tools/mor-bf16-axes/6006454/reports/both.json)。这些报告保留
`status=partial`；`evidence.passed=false` 是 operator probe 的预期验收隔离，同时确有
tensor hard gates 失败。它们不支持正式化 FP32 临时 reduction，也不证明在所有
BF16 实现上不可能达到 2%。目前未找到满足原 dtype/依赖约束的完整 30B 修复。

#### 新 attention 路径的完整 tiny 训练矩阵

`6006390`（eos0241）`COMPLETED 0:0`，21:49；pytest 374 passed / 1 skipped，
完整矩阵 **20/20 通过**，包括 learned/replay 的八种 topology、FP32 独立 reference、
外部进程 checkpoint 恢复和 CP canonical/direct 比较。未设置 topology filter。
本地已核对全部 20 份报告 SHA-256，matrix receipt 的源码 hash 与当前
`c49e48c4d882aa7e8f0a3b3ea98d42711ba260f42cc0bbd21d8164618b870c87` 一致。

全矩阵最大硬门槛 relative-L2：forward `0.00569257`、完整 gradient 向量
`0.00300522`、完整 FP32 master update 向量 `0.01192856`、resident post-step tensor
`0.00155772`。FP32 master 是原生 optimizer state，不是新增的 FP32-heavy forward 路径。
该结果验证 CP1 local FFA adapter 的训练接入，但 **不覆盖失败的 30B TP/CP forward**。
证据：[matrix_complete.json](../artifacts/eos/6006390/tiny/reports/matrix_complete.json)、
[all.json](../artifacts/eos/6006390/tiny/reports/all.json)、
[pytest.xml](../artifacts/eos/6006390/tiny/reports/pytest.xml)。

#### 新快照的 30B 跨进程完整状态恢复

`6006440`（eos0288）`COMPLETED 0:0`，10:18。`c49e48...` 快照在
TP2/CP2/DP2/EP4、BF16、seq_lens `[128,128]` 上完成 backward / optimizer step，
保存 step 1 后执行不中断的下一步；再启动独立 8-rank 进程，恢复并继续一步。
恢复时以及继续一步后的全参数、全 optimizer、RNG 指纹均一致，`certify-checkpoint`
通过，receipt 包含新的 `attention_policy`。本地复核了 resume manifest、routes 和
save receipt 哈希。恢复步骤 `updated=true`，grad norm `131.986860141455`。

该结果证明同 topology 的训练续接，不是单卡/多卡 gradient/update parity：30B 没有
启用 tiny-only 的完整 gradient/master-update tensor 捕获，也没有测量吞吐和峰值显存。
证据：[checkpoint_external_resume.json](../artifacts/bf16-6006440/qwen30b_train/reports/checkpoint_external_resume.json)。

## 使用与已知边界

- 最终代码仍位于隔离 checkout，未覆盖原始项目，未 push；便于继续评审和合并。
- 普通 runtime 的 token-weighted loss callback 见 [objective.zh.md](objective.zh.md)。
- 不支持跨 EP checkpoint reshard、PP/VPP>1、ETP>1；这些不是本轮新增功能。
- tiny 是短序列 synthetic acceptance；不外推到多 seed、长序列性能或长期训练稳定性。
- 完整 30B 单卡/多卡 forward 和 gradient/update 验收仍未完成；checkpoint 通过不能替代它。
- 原生高精度/拓扑无关算子方案如需作为正式路径，必须另测 backward、显存和吞吐，
  不能把诊断脚本的结果直接作为生产训练结果。
