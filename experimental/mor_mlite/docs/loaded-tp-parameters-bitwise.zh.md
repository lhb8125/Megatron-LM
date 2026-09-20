# 加载完成后的 TP1 / TP2 参数 bitwise 检查

后续作业 6011796 将同一检查扩展至 TP4，全部参数也 bitwise 一致，
并继续比较原 forward。详见 [TP2 / TP4 对比](tp2-vs-tp4.zh.md)。

## 结果

EOS 作业 **6011449**，节点 **eos0331**，运行 **5 分 22 秒**，
SLURM `COMPLETED / 0:0`。任务 `01a07ee0-2d61-7e23-b9fc-8e9ca02c12fe`。
本次直接验证原 30B/MoR forward 调查使用的加载路径，而非只验证 checkpoint 文件名相同。

**TP1 和 TP2 的全部加载后参数 bitwise 一致。**
在 checkpoint 返回后、首次 forward 前读取实际模型 Parameter，
按分片布局配对，执行 `torch.uint8` 原始字节比较；不转换 FP32、不使用 allclose 容差。

| 对比 | ranks | 完整参数张量 | 检查的 rank-local 分片/副本 | 检查字节数 | 不同参数 / 元素 / 字节 |
| --- | --- | --- | --- | --- | --- |
| 原 TP1/EP1 → TP1/EP2，CP1 | 2 | 5,266 | 5,412 | 28,179,812,352 | 0 / 0 / 0 |
| 原 TP1/EP1 → TP2/EP2，CP1 | 4 | 5,266 | 10,824 | 52,360,355,840 | 0 / 0 / 0 |

两个候选均使用同一个 EP2 folded checkpoint，DP 都为 2；
TP 从 1 改为 2，world size 从 2 改为 4。
两者都与同一完整 TP1 原始字节参照相等，故也证明它们彼此的完整参数逐位相等。

## 覆盖范围

源模型为 Qwen3-30B-A3B-Base；当前 MoR 是 **20 个物理层、48 次逻辑层调用**
（3 start + 14 recurrent × 3 + 3 end），不是重新实例化 48 份物理层。

- 共 **13,084,750,848** 个物理参数元素，全部为 BF16。
- TP1 原始参数数据共 **26,169,501,696 bytes**。
- **5,120** 个 expert 参数张量：20 层 × 128 experts × FC1/FC2。
- 其余 **146** 个张量覆盖 embedding、QKV、output projection、所有 norms、
  MoE router、3 个 depth-router 投影和 LM head。
- 每个 rank 的本地参数清单单独核验，所有全局 expert ID 必须覆盖；
  所有 TP 区间必须连续、完整且无冲突；replicated 参数的每份副本都比较。
- vocabulary=151936，**未裁剪 padding、未忽略任何参数**。
- 反复调用的 recurrent layers 按物理 Parameter 计数，未将同一权重重复计为三份。

这项检查针对模型参数；不宣称覆盖非参数 buffers、optimizer state 或计算 kernel。

## 检查方法与故障注入

运行脚本复用原 `mor_mlite.parity run` 的模型构建、配置校验和
`load_mor_checkpoint`。诊断 wrapper 调用原加载函数，等待其完整返回并同步 CUDA，
才开始读取权重。全部加载 guard 原样执行；随后明确终止在 pre-forward 检查点。
没有运行此模型的 forward/backward/optimizer step，也没有修改生产参数或算法。

TP1 参数按名称顺序流式写入 `parameters.bin`，manifest 记录 shape、dtype、
offset、numel 和 SHA256。每次只处理一个参数，不将整模型 all-gather 到每张卡。
候选以 private mmap 读取 TP1 数据；先验证参照参数 bytes 的 hash，再按以下布局切片：

- embedding/head、QKV：dim 0；
- attention output projection：dim 1；
- experts：按 EP local index → global expert ID 映射；本实验 ETP1，不做 dense TP 切分；
- norms、routers 等 replicated 参数：全张量比较。

Qwen QKV 为 MCore KV-group interleaved packing，本配置 TP2 可以沿完整 group 连续切分。
布局规则已与固定上游的 Qwen WeightSpec、ColumnParallelLinear/RowParallelLinear 对照。
切片的拼接覆盖检查与逐字节 equality 联合证明完整张量 equality，
不是只比较几个统计量或仅比较 rank-local aggregate hash。

实际执行的比较器负对照包括：

- 单字节翻转必须失败；
- `+0.0` / `-0.0` 数值相等但 raw bytes 不等，必须识别；
- dtype 不同必须失败；
- dim 0/1 的 TP 切片重组正确；
- 缺少 TP 分片必须拒绝。

所有负对照通过。归档后又独立核对六份 rank 报告：
名称覆盖完整、每个元素/字节差异计数为 0、expected/actual SHA256 相同、
加载 step 为 0、生产源码和诊断脚本 hash 一致，且 baseline 文件 offset 连续。

## 版本与 checkpoint

- 生产源码 60 个 Python 文件的内容 hash：
  `c49e48c4d882aa7e8f0a3b3ea98d42711ba260f42cc0bbd21d8164618b870c87`；
  与原 forward 调查相同，运行时严格核验。
- Megatron-LM：`5c8315f12a64a7279eec58896af9e74ee3351b74`。
- Torch 2.10.0+cu129、TE 2.13.0、MagiAttention 1.1.1、H100 80GB。
- NGC `nvcr.io/nvidia/pytorch:26.01-py3` 的既有
  `pytorch_26.01-py3_4a7dd6b5c237.sqsh`，独立 cu129 venv 与原调查相同。
- HF snapshot：`1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9`。
- 原 baseline：`5998357/qwen30b/folded_init_ep1`；
  EP2 两个候选：`mor-bf16-axes/6006410/folded_init_ep2`。
- 启动器：`cluster-run slurm`，batch/account `coreai_devtech_all`，
  单节点 exclusive，最多运行 4 ranks；作业结束后 allocation 自动释放。

## 产物

- [TP2/EP2 汇总](../runtime/ckpt_tools/loaded-tp-bitwise/6011449/tp_dp_ep/summary.json)
- [TP1/EP2 汇总](../runtime/ckpt_tools/loaded-tp-bitwise/6011449/ep/summary.json)
- [TP1 全量参数 manifest](../runtime/ckpt_tools/loaded-tp-bitwise/6011449/baseline/manifest.json)
- [作业日志](../runtime/ckpt_tools/loaded-tp-bitwise/6011449/mor-loaded-tp-bitwise_6011449.log)
- [实际环境版本](../artifacts/eos/6011449/versions.json)
- [比较脚本](../runtime/ckpt_tools/loaded-tp-bitwise/scripts/check_loaded_parameters.py)
- [运行脚本](../runtime/ckpt_tools/loaded-tp-bitwise/scripts/run.sh)
- [提交记录](../runtime/ckpt_tools/loaded-tp-bitwise/launch.json)

六份完整 rank 报告位于上述 ep/tp_dp_ep 目录下的 `rank_*/rank.json`。
约 26.17 GB 的 TP1 raw bytes 保留在 EOS 原项目的
`runtime/ckpt_tools/loaded-tp-bitwise/6011449/baseline/parameters.bin`，
没有下载到本地，没有删除旧 checkpoint 或诊断产物。

## 对原 logits diff 的结论

之前“未做全量加载后参数一致性检查”的缺口现已补齐。
**可以排除本次复现配置中初始参数不同、EP2 转换数值不同、
TP 参数分片重建不一致造成此前 logits diff 的解释。**

但这并不自动证明所有误差都来自 TP reduce 累加顺序，也不使 4.56% logits
relative-L2 成为可接受结果。计算路径、GEMM/舍入、非线性传播等仍需要原逐 op 证据；
本报告不签发全模型 forward/backward/optimizer 精度验收。
