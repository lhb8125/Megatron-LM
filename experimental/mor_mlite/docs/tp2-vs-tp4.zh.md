# MoR BF16：CP1 下 TP2 与 TP4 的参数及 forward 对比

## 结论

EOS 作业 **6011796**，节点 **eos0297**，2026-09-10，运行 **6:25**，
SLURM `COMPLETED / 0:0`。任务 `01a07ee0-2d61-7e23-b9fc-8e9ca02c12fe`。
**作业执行成功不等于数值验收通过：三组 forward 对比均为 failed。**

- TP2 与 TP4 全部加载后参数 bitwise 一致；没有忽略 padding、experts 或副本。
- TP4 对 TP2 的 logits relative-L2 为 **4.50877883%**，仍明显超过原 2% 门槛。
- TP4 对 TP1 为 **4.23729028%**，略低于 TP2 对 TP1 的 **4.55851660%**。
  这组输入下，误差并不随 TP degree 单调增大。
- 本轮没有修复生产计算路径，也没有接受精度门槛变更。

## 控制变量

模型沿用 Qwen3-30B-A3B-Base 的 folded MoR：20 个物理层，
48 次逻辑调用（3 start + 14 recurrent × 3 + 3 end）。参数/激活保持原 BF16 路径。

| 配置 | TP | CP | DP | EP | ETP | ranks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TP2 | 2 | 1 | 2 | 2 | 1 | 4 |
| TP4 | 4 | 1 | 2 | 2 | 1 | 8 |

两组均在同一 allocation 内顺序运行，加载同一
`runtime/ckpt_tools/mor-bf16-axes/6006410/folded_init_ep2`；
两条 128-token 输入，1 step、1 microbatch、forward-only。
global batch 的完整 metadata 相同。

两组都从原 TP1 oracle `artifacts/bf16-6006397/qwen30b/baseline`
replay depth 路由以及 native expert ID / selected scores。
3 轮 depth 路由检查通过，48 次 expert context 全部 identity 一致。
没有向 attention、MLP 或 residual 注入参考 hidden。
因此这是**固定路由的计算路径对照**，不是自由学习路由的训练精度验收。

TP1 辅助参照为原单 rank TP1/EP1 oracle，串行模拟同样的 DP2 样本分区；
其 `single_rank_reference_dp_shards` metadata 与分布式候选不同。
比较器另行核验 effective DP、sample partitions、sequence lengths、
loss weighting 等数据语义完全相同；没有把 TP1 说成同一物理 DP/EP 配置。

ETP 保持 1，增加 TP 后 world size 和 expert-DP 副本数也随布局变化。
本实验控制用户配置层面的 TP 变量，不把所有底层 kernel/通信变化简化为某一次 reduce。

## 加载后参数逐字节检查

调用原 `load_mor_checkpoint` 完整返回后、首次 forward 前，读取实际 Parameter。
与作业 6011449 保留的 TP1 全量 `parameters.bin` 按 TP/EP 布局逐字节比较。
方法详见 [TP1/TP2 参数检查](loaded-tp-parameters-bitwise.zh.md)。

| 配置 | 完整张量 | rank-local 分片/副本 | 检查 bytes | 差异参数 / 元素 / bytes |
| --- | ---: | ---: | ---: | --- |
| TP2 | 5,266 | 10,824 | 52,360,355,840 | 0 / 0 / 0 |
| TP4 | 5,266 | 21,648 | 100,721,442,816 | 0 / 0 / 0 |

覆盖 13,084,750,848 个物理参数元素，全部 BF16，包括 5,120 个 expert 张量。
每个 rank 的清单、TP 完整区间覆盖和所有 replicated 副本均检查。
TP4 的 QKV 按完整 KV group 连续切分（本模型 4 KV heads）。
比较器通过 bit flip、signed zero、dtype、缺片等已有负对照，
并补充 TP4 dim0/dim1 重组及颠倒 rank 顺序必须失败的检查。

归档后独立读取全部 12 份 rank 报告，复核 raw-byte/hash equality、
5266 个名称覆盖、TP2/TP4 分片区间、BF16 dtype，以及 forward manifest
关联的脚本、拓扑配置、参数报告 SHA256，全部通过。
参数报告中的 `forward_calls: 0` 指**采样时刻尚未 forward**；
本次 wrapper 随后返回原加载结果，继续原框架 forward。

新 TP2 的全部 **268** 项 tensor index/hash 与旧作业 6006410 TP2 完全相同；
因此本轮参数审计没有改变已知的 TP2 forward 轨迹。

## Logits 结果

relative-L2 = `||candidate-reference||₂ / ||reference||₂`；
它是全 logits 的范数比，不是平均逐元素百分比，也不是任务准确率。
argmax 统计的是这 256 个 token 位置的最高 logit ID 是否相同。

| candidate / reference | relative-L2 | cosine | max-abs | argmax 相同 |
| --- | ---: | ---: | ---: | ---: |
| TP4 / TP2 | 4.50877883% | 0.9989833960 | 2.453125 | 228/256（89.0625%） |
| TP2 / TP1 | 4.55851660% | 0.9989616016 | 2.437500 | 232/256（90.6250%） |
| TP4 / TP1 | 4.23729028% | 0.9991035122 | 1.937500 | 226/256（88.28125%） |

原 forward 门槛为 relative-L2 ≤ 2% **且** cosine ≥ 0.999。
TP4/TP2 两项均未通过；TP4/TP1 虽通过 cosine，仍未通过 relative-L2。
TP4 对 TP1 的整体 L2 更小但 argmax 一致率更低，二者不能互相替代。

## 已有边界显示的传播（TP4 / TP2）

本轮沿用框架已采集的边界，没有新增 48 层 × 全 op 的追踪。
逻辑层号从 0 开始；`end_hidden_0/1/2` 对应逻辑层 45/46/47。

| 边界 | relative-L2 |
| --- | ---: |
| 第 3 轮 recurrent 结束 / post merge | 0.453892% |
| L45 输出 | 0.889279% |
| L46 post-attention residual | 1.085071% |
| L46 MoE output | 3.844267% |
| L46 输出 | 3.991310% |
| L47 输出 / final hidden | 3.164357% |
| final norm 后 / hidden for head | 4.458894% |
| logits | 4.508779% |

TP2/TP4 也重现了“L45→L46 显著增加、final norm 后相对误差再次增加”的现象。
这些边界的分母和张量尺度不同，百分数不能相减作为误差贡献，也不能仅凭此表
证明具体算子 bug 或断言所有 diff 完全来自 TP reduce 的累加顺序。
已有 TP1/TP2 的同输入干预证据见 [逐层调查](tp-layer-chain.zh.md)。

## 版本、启动与产物

### Deterministic 设置与同配置重复运行

后续核对当前 TP2、TP4 manifest：均为 `strict=true`、`seed=1234`。
`parity/mlite.py` 在构建 runtime 前调用 `configure_determinism`，strict 路径执行
`torch.use_deterministic_algorithms(True, warn_only=False)`，关闭 TF32 与
cuDNN benchmark，并向 MLite implementation 及 CP1 local FFA 传递 `deterministic=True`。
启动 wrapper 在 Python 运行前设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`、
`NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`。这不只是固定 seed。

**CP 调查纠正：不能由此宣称 CP>1 的 Magi core 也启用了 deterministic。**
固定 Magi 1.1.1 的分布式 core 从
`MAGI_ATTENTION_DETERMINISTIC_MODE` 独立读取设置，不读取 PyTorch 全局开关；
旧 CP2 manifest 中该值为 `0`。MLite `MagiDotProductAttention` 调用
`calc_attn` 时也未传递 strict。CP1 local FFA 的显式 `deterministic=True`
与此不同。下面的有限次重跑 hash 全等仍然成立，但不是第三方 deterministic
模式已正确开启的证明；补充隔离实验记录见 CP propagation 调查。

已有同配置重跑证据（相同输入/checkpoint/routing，部分重跑增加只读诊断探针）：

| 拓扑 | 对照作业 | 实际结果 |
| --- | --- | --- |
| TP2 CP1 DP2 EP2 | 6006410 → 6006544、6011796 | 每次 268 项 tensor hash 全部相同 |
| TP1 CP2 DP2 EP2 | 6006410 → 6006544 | 268 项 tensor hash 全部相同 |
| TP1 单 rank baseline | 6006397 → 6006544 | 268 项 tensor hash 全部相同 |
| TP2 CP2 DP2 EP4 | 6006397 → 6006544 | 268 项 tensor hash 全部相同 |

当前再次直接核对 6006410 与 6011796 的 TP2 manifest，268 项 shape/dtype/SHA256
完全一致；logits shape 为 `[256,151936]`，hash 为
`10d9710519731a90bb2b9be6f42561fe914cdc1d5fc63823f82f420a9775d22d`。
因此这些已观测重复运行的 logits 差异为 0，不是此前 4%–6% 跨拓扑 diff 的量级。
6006544 的逐组复现证据为 `runtime/diagnostics/diff_source/6006544/*.json`
中的 `artifact_check`，不是将跨拓扑通过误记为同拓扑重复运行。

这里报告的是有限次 forward 观察，没有做多 seed / 大样本方差统计；
TP4 目前只有一次完整 forward，不能声称已测得其 run-to-run variance 为 0。
没有覆盖 backward/optimizer 的重复性。Deterministic 设置也不保证不同 TP/CP
拓扑、软件版本或硬件之间 bitwise 一致，亦不能自动替代第三方 kernel 的实测。

- 生产源码 60 个 Python 文件 SHA256：
  `c49e48c4d882aa7e8f0a3b3ea98d42711ba260f42cc0bbd21d8164618b870c87`，本轮未改。
- MCore：`5c8315f12a64a7279eec58896af9e74ee3351b74`。
- Torch 2.10.0+cu129、TE 2.13.0、MagiAttention 1.1.1、H100 80GB。
- 容器 `nvcr.io/nvidia/pytorch:26.01-py3` 对应既有
  `pytorch_26.01-py3_4a7dd6b5c237.sqsh`，沿用独立 cu129 venv。
- HF snapshot：`1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9`。
- Attention policy：`native-bf16-local-ffa-strict-cp1-magi-cp-v1`；
  CP1 不执行 Magi CP 通信。本实验不验收 CP>1。
- 使用 `cluster-run slurm` 提交，batch、单节点 exclusive、8 GPUs、25 分钟上限。
  toolkit job ID：`20260910-223616-edb6`。allocation 已随作业完成释放。

首轮 6011791 在模型运行前因诊断 wrapper 的 argparse 缩写匹配失败：
`--topology` 错配 `--topology-config`。
修正为 `allow_abbrev=False`，并验证 topology 参数原样传递，随后重跑 6011796。
没有跳过拓扑 guard；TP4 使用运行专用 JSON，经原严格 loader 和运行时校验。

- [数值汇总](../runtime/ckpt_tools/tp2-vs-tp4/6011796/summary.json)
- [TP4 / TP2 完整对比](../runtime/ckpt_tools/tp2-vs-tp4/6011796/tp4_vs_tp2.json)
- [TP2 参数报告](../runtime/ckpt_tools/tp2-vs-tp4/6011796/tp2_parameters/summary.json)
- [TP4 参数报告](../runtime/ckpt_tools/tp2-vs-tp4/6011796/tp4_parameters/summary.json)
- [作业日志](../runtime/ckpt_tools/tp2-vs-tp4/6011796/mor-tp2-vs-tp4_6011796.log)
- [实际环境版本](../artifacts/eos/6011796/versions.json)
- [运行脚本](../runtime/ckpt_tools/tp2-vs-tp4/scripts/run.sh)
- [运行 wrapper](../runtime/ckpt_tools/tp2-vs-tp4/scripts/run_case.py)
- [分析脚本](../runtime/ckpt_tools/tp2-vs-tp4/scripts/analyze.py)

本地归档 JSON、rank 报告、log、环境记录；完整 `tensors.pt` 与原始参数 bytes
保留在 EOS 项目 `mor_mlite_bf16_train` 对应相对目录，没有删除旧产物。
本轮使用 `ckpt-tools` 和 `cluster-run`，未运行 backward、optimizer、
多 step 收敛或性能测试，不据此宣称训练框架精度通过。
