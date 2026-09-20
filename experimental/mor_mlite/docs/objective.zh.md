# 普通 MLite runtime 的训练目标适配

模型的 LM loss 是当前 dense-DP replica 内的 token mean；每轮 depth-router BCE
使用各自的 candidate denominator。模型内部已处理 CP，不应重复把 TP/CP 副本计数。
仅注册 `qwen3_moe_mor` 并直接返回局部 `output["loss"]`，在 DP 或 microbatch 长度不等时
得到的是局部均值的平均，不是全局 token mean。

## 公共 API

```python
from mor_mlite.objective import apply_objective, objective_scales

# 整个 optimizer step 的计数：[microbatch][dense_dp_rank]。
# lm_counts 是经过 packed label/mask shift 后的有效权重和，可为小数或局部零。
lm_counts = [[2, 7], [0, 3]]
# 最后一维为各递归轮选择前的 candidate 数，不是本轮 selected 数。
router_counts = [[[4, 3], [8, 6]], [[1, 1], [5, 2]]]
scales = objective_scales(lm_counts, router_counts)

def loss_fn(output, microbatch_index, dense_dp_rank):
    return apply_objective(output, scales[microbatch_index][dense_dp_rank])
```

将 callback 接入 MLite runtime；runtime 仍然负责一次 microbatch 平均、优化器负责一次
dense-DP 平均。不要在 callback 里再次除以 microbatch 数。若外部框架使用梯度求和而非
这两个平均，则不能直接套用本适配器的补偿系数。

调用者在 optimizer step 开始前汇总整个 step 的计数，可通过数据计划或在 dense-DP
group 上汇总 metadata 获得。`packed_lm_targets(batch)` 返回 reference 合同下已经左移
一次的 labels/mask，mask.sum() 才是 LM denominator；不能用 seq_len-1 代替任意 mask。
None source mask 包含每个真实 token（包括 label=0 的最后目标）；显式 mask 会左移并
排除最后目标。dummy padding 永远不计数。

第 0 轮 candidate 是真实输入数；之后是上一轮的实际 selected 数。各轮 BCE 的 coefficient
已经由模型计入 `mor_router_aux_losses`，`apply_objective` 不重复乘 coefficient。
全局 LM 或任意全局 router denominator 为零、非有限数、负数、轴不对齐时明确报错。
这不是通用 dataset CLI；现有 `train` 仍是 synthetic acceptance trainer。

## 验证

`tests/test_objective.py` 用不等长 DP、两个不等长 microbatch 和局部零 LM 权重，比较
适配器的 loss 与 autograd 梯度是否等于直接计算的全局 token mean。
现有 parity trainer 的 scale 构造和 loss callback 同样调用公共 API，避免两套归一化公式。
