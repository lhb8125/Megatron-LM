# Megatron CI 如何验证 TP 精度：固定版本源码审查

## 范围与结论

日期：2026-09-10；任务：`01a07ee0-2d61-7e23-b9fc-8e9ca02c12fe`。
检查实际 EOS 依赖 Megatron-LM commit
`5c8315f12a64a7279eec58896af9e74ee3351b74`，
不是另一本地 checkout `3bcc70b6624825aff051f02a945fb3cf45ec6438`。
89 个复制文件与该 checkout Git index blob SHA1 逐项一致，
见 [audit_manifest.json](../runtime/diagnostics/megatron_ci_tp_audit/5c8315f/audit_manifest.json)。

本轮为源码及 CI 选取逻辑审查：未启动 GPU 作业，未执行训练/数值测试，
未查询历史 CI 成败，因此没有本轮训练 job ID、实测误差或运行镜像 digest。
recipe 的 dev/LTS build 不能代替一次真实运行的镜像版本证据。
没有修改生产代码、上游检查或容差。按 mcore-testing/mcore-cicd 技能区分
“测试存在”“CI recipe 会选取”“本轮实测通过”，并以实际入口代码为准，
不沿用技能文档中较旧的 GitHub scope 名称。

结论：Megatron 确有跨 TP 的 BF16 attention 输出/输入梯度检查，
也有从同一 4B checkpoint 出发的跨 TP 梯度检查。
但完整模型梯度检查的实际阈值约 8.29%，且属于内部 `mr` recipe，
不在 GitHub L0/L1 选择范围。常规 loss golden 和固定拓扑 determinism
不是跨 TP logits 验收。不能以这些检查为我们 30B/MoR 的约 4.56%
logits relative-L2 差异直接背书。

## 1. 单层 attention：真实跨 TP 前向与反向对照

源码：
[test_attention.py](../runtime/diagnostics/megatron_ci_tp_audit/5c8315f/tests/unit_tests/transformer/test_attention.py)，
helper 第 700 行附近、参数矩阵第 986 行、配置第 1003 行。

- BF16，单层、hidden=128、heads=4、sequence=256、micro batch=4，dropout=0。
- 先在 TP1/CP1 创建模型并保存 checkpoint，再加载相同权重到目标并行拓扑。
- 只调用首层 self-attention，不是完整 GPT logits；以输出求和做 backward。
- 对照 TP4/CP1，分别开关 SP；也覆盖 TP2/CP2、TP1/CP4、packing、RoPE fusion、
  QK norm 和 output gate。
- 比较 attention output、input gradient、存在时的 bias；检查 NaN/Inf。
- 普通 attention 测试实际传入 `atol=rtol=1e-2`；没有传 cosine fallback 参数。
  即逐元素要求 `abs(a-b) <= 0.01 + 0.01*abs(b)`，不是全局 relative-L2 ≤1%。
- helper 的可选参数梯度比较要求 TP1；此 TP4 用例并没有比较全部权重梯度。

MLA 的
[test_multi_latent_attention.py](../runtime/diagnostics/megatron_ci_tp_audit/5c8315f/tests/unit_tests/transformer/test_multi_latent_attention.py)
第 1585–1810 行附近同样比较 BF16 单层 output/bias/input gradient，
有 TP4/CP1 与 SP 开关。实际阈值为 `atol=rtol=5e-3`：
代码第 1756 行写的是 `if cp:`，而 CP1 的 `cp=1` 也为真，
不能误报成 else 分支的 `5e-4`。这里只记录实际行为，未改这个条件。
MLA 有 experimental 标记及 TE/PyTorch 版本 skip；latest unit runner
单独执行 experimental suite，legacy 不执行。

CI 接线：
[H100 unit recipe](../runtime/diagnostics/megatron_ci_tp_audit/5c8315f/tests/test_utils/recipes/h100/unit-tests.yaml)
第 126 行 transformer bucket、第 250 行 MLA 独立 bucket；
[unit runner](../runtime/diagnostics/megatron_ci_tp_audit/5c8315f/tests/unit_tests/run_ci_test.sh)
第 167–190 行执行 pytest。GB200 则额外按 `launch_on_gb200` marker 筛选，
不能把 H100 覆盖直接扩展成全部硬件覆盖。

注意：名字含 TP parity 不一定代表 CI 正在执行跨 TP 检查。
例如 Bagel 的 `test_tp_parity.py` 只有 `run_tp_parity_test` 和
`if __name__ == "__main__"` driver，没有默认 pytest 可收集的 test 函数。
DSA backend TP/SP 测试则是在既定 TP2/CP2 下比较 fused/unfused backend；
不能仅凭文件名把它算作 TP1↔TP2。

## 2. 完整 4B 模型：跨 TP 的梯度状态检查

配置：
[model_config.yaml](../runtime/diagnostics/megatron_ci_tp_audit/5c8315f/tests/functional_tests/test_cases/gpt/gpt3_mcore_reruns_resume_check_grads/model_config.yaml)。

这是 `checkpoint-consistency`：
32 层、hidden=3072、FFN=8192、heads=32、KV groups=8、
seq=512、GBS=2、BF16/TE、RMSNorm/SwiGLU。
每个配置从相同 `gpt3_4b_pyt/25.03.05_bf16_rerun-enabled_v2`
checkpoint 出发，训练 1 步；不加载旧 optimizer，dropout=0、
deterministic-mode，NCCL Ring。

| 分支 | DP | TP | CP | PP | MBS |
| --- | --- | --- | --- | --- | --- |
| MODEL_ARGS，基准 | 1 | 1 | 1 | 1 | 2 |
| MODEL_ARGS_2 | 2 | 1 | 1 | 1 | 1 |
| MODEL_ARGS_3 | 1 | 2 | 1 | 1 | 2 |
| MODEL_ARGS_4 | 1 | 1 | 2 | 1 | 2 |
| MODEL_ARGS_5 | 1 | 1 | 1 | 2 | 2 |

runner 选择顶层 `MODEL_ARGS(_\d+)?`，实际是上述五个训练分支；
MODEL_ENV_VARS 下孤立的 MODEL_ARGS_10 不会新增一次训练。
各分支保存 `iter_0000001` 后，与首个 checkpoint 比较。

### 实际比较什么

[test_optimizer_grads_match.py](../runtime/diagnostics/megatron_ci_tp_audit/5c8315f/tests/functional_tests/python_test_utils/test_optimizer_grads_match.py)
第 103–118、154–247、291–317 行：

- 只提取以 `optimizer.` 开头且含 `.exp_avg.` 的 tensor key。
- Adam beta1/beta2 均设为 0，exp_avg 用来保存进入 optimizer 的梯度。
  配置同时有 `clip-grad=1.0`，因此不能称其为“未经裁剪的原始梯度”。
- 读回完整 checkpoint tensor，对 TP row-parallel 的 FC2 和 attention projection
  做布局还原，比较 key、shape、dtype 和数值。
- 每个 checkpoint tensor key 做 Frobenius relative norm；同类参数可能沿层维
  堆叠为一个 tensor，并不保证逐层分别达标。
- 不比较 logits、exp_avg_sq、fp32 master weights 或实际 parameter update。
- embedding/output layer 的 padding/reshape 有特殊处理；若对应 reshape 失败，
  代码存在打印 FIXME 后跳过该 tensor 的分支，因此也不能声称绝无遗漏。

### 实际容差约为 8.29%

`relative_grad_diff(g_hat,g_ref)` 用 FP32 计算：

```text
||g_hat - g_ref||F / (||g_ref||F + 1e-30)
```

调用为 `assert_grads_close(lt,rt)`，其中 lt 来自首个 TP1 checkpoint，
rt 来自待比较分支；因此这份实现的分母实际是第二个分支的范数，
不是默认以 TP1 为分母。

`assert_grads_close` 对所有 tensor 固定传 `l=0,dtype=BF16`：

```text
bound = k * C**(L+1-l) * eps_BF16
      = 4 * 1.03**33 * (1/128)
      = 0.08288547619861457
      ≈ 8.2885%
```

虽然 helper 有按层深计算的接口，实际 gate 并未传真实 layer index。
源码称 l=0 是最宽松情况。失败后的 assert_close 和 rolled tensor
只用于诊断，不是另一条通过路径。
这是代码采用的参数化容差，不能当作任意模型的理论误差保证，
更不能移用为 logits 的 8.29% 验收阈值。

### 哪条 CI 选中它

[H100 gpt-grads recipe](../runtime/diagnostics/megatron_ci_tp_audit/5c8315f/tests/test_utils/recipes/h100/gpt-grads.yaml)
第 62–68 行仅配置 `scope:[mr]`、dev、dgx_h100；脚本固定 N_REPEAT=1。

GitHub
[cicd-main.yml](../runtime/diagnostics/megatron_ci_tp_audit/5c8315f/.github/workflows/cicd-main.yml)
第 250–270 行选 L0 或 L1；
[recipe_parser.py](../runtime/diagnostics/megatron_ci_tp_audit/5c8315f/tests/test_utils/python_scripts/recipe_parser.py)
第 18–42、239–252 行明确保留 GitLab 的 mr，不映射到 GitHub L1。

本轮只执行无 GPU、无外部写入的 recipe flatten/filter 检查，结果：

```json
{"mr": 1, "L0": 0, "L1": 0, "mr-github": 0}
```

说明这个 recipe 在 mr 范围可被选中，不代表查到了某次 CI 已运行并通过。

## 3. 常规 functional CI：各配置对自己的历史标量

[test_pretraining_regular_pipeline.py](../runtime/diagnostics/megatron_ci_tp_audit/5c8315f/tests/functional_tests/python_test_utils/test_pretraining_regular_pipeline.py)
第 14–40 行以及
[common.py](../runtime/diagnostics/megatron_ci_tp_audit/5c8315f/tests/functional_tests/python_test_utils/common.py)
第 164、209–285 行：

- golden 路径按 test case、environment、platform 区分。
  TP1 case 对自己的 golden，TP2 case 对自己的 golden，不是自动互比。
- 默认检查 lm loss 和 num-zeros；grad-norm 只有配置 METRICS 选中才检查。
- 确定性模式要求记录的 scalar 零容差一致；TensorBoard 提取已 round 到 5 位小数，
  不是完整 hidden/logits/weights 的 bitwise 比较。
- 允许非确定性时，lm loss / grad-norm 的 approximate check 为 atol=0、rtol=0.05；
  num-zeros 为 rtol=0.20。多步比较允许少量超限点，确定性检查不允许。
- GitHub “Run tests” 的 lightweight 分支会令 SKIP_PYTEST=1、exit-interval=4，
  runner 提前返回训练退出码，不执行后续 golden 数值断言。
  “Run functional tests” 是另一条非 lightweight 选择。

所以不能说“Megatron 的 TP logits 允许差 5%”。

## 4. Determinism：固定拓扑重复运行，不是跨 TP 等价

[test_gpt_model.py](../runtime/diagnostics/megatron_ci_tp_audit/5c8315f/tests/unit_tests/determinism/correctness/test_gpt_model.py)
与
[bit_exact_runner.py](../runtime/diagnostics/megatron_ci_tp_audit/5c8315f/tests/unit_tests/determinism/bit_exact_runner.py)
第 113–176 行：
分别在 TP4、TP8、TP/EP 等配置中，恢复 RNG、清零梯度、重置 quantizer，
对同一模型做两次 forward/backward，比较 output 和 gradient bit-exact。
两次之间不切换 TP，因此证明的是“TP4 重跑可复现”，不是“TP4 等于 TP1”。

## 5. 对当前 30B/MoR 调查的意义

这些测试说明上游并不一律要求跨 TP BF16 逐位一致，同时确实检查真实数值，
不能简单归纳为只看训练是否能跑或 loss 是否下降。
但单层随机初始化 attention、4B dense 梯度状态、历史训练标量，
都不能替代我们 48 次逻辑层调用、预训练 30B/MoR、CP1、TP1↔TP2 的
全模型 logits、梯度和参数更新对照。

可借鉴的是同 checkpoint / 同数据 / 单独改变 TP 的实验设计，
以及前向、反向、更新分别验证；不是直接照搬 1%、5% 或 8.29%。
本次审查既没有证明此前 4.56% logits diff 合理，也没有证明它一定来自实现 bug；
数值根因仍以此前逐 op 诊断和后续针对本模型的验证为准。

