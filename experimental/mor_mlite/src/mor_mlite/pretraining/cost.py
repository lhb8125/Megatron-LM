"""Dominant-matmul FLOP accounting; not a hardware instruction counter."""

from __future__ import annotations

FLOP_METHOD = (
    "QKV/output/MoE top-k/expert-router/head GEMMs plus causal QK/AV; "
    "D uses observed active sequence lengths including discarded dummy work; "
    "training approximates forward+backward as 3x forward. "
    "Excludes norms, nonlinearities, softmax, auxiliary reductions, optimizer, "
    "communication, and kernel padding; not measured executed hardware FLOPs."
)


def layer_flops(cfg, lengths):
    tokens = sum(lengths)
    hidden, heads, dim = cfg.hidden_size, cfg.num_attention_heads, cfg.head_dim
    qkv = (heads + 2 * cfg.num_key_value_heads) * dim
    attention_projections = 2 * tokens * hidden * (qkv + heads * dim)
    attention_products = 4 * heads * dim * sum(n * (n + 1) // 2 for n in lengths)
    expert_router = 2 * tokens * hidden * cfg.num_experts
    experts = 6 * tokens * hidden * cfg.moe_intermediate_size * cfg.num_experts_per_tok
    return attention_projections + attention_products + expert_router + experts


def forward_flops(arm, cfg, lengths, round_lengths=None):
    full = layer_flops(cfg, lengths)
    head = 2 * sum(lengths) * cfg.hidden_size * cfg.vocab_size
    if arm in ("A", "B", "C"):
        return (20 if arm == "C" else 48) * full + head
    if arm != "D" or round_lengths is None or len(round_lengths) != 3:
        raise ValueError("D FLOPs require all three observed recursion layouts")
    total = 6 * full + head
    candidates = sum(lengths)
    for active in round_lengths:
        total += 2 * candidates * cfg.hidden_size
        total += 14 * layer_flops(cfg, active if sum(active) else [1])
        candidates = sum(active)
    return total


def observed_flops(model, arm, lengths):
    rounds = None
    if arm == "D":
        if model.training:
            rounds = [
                (plan.active_cu_seqlens[1:] - plan.active_cu_seqlens[:-1]).tolist()
                for plan in model.last_route_plans
            ]
        else:
            # Evaluation uses one full sequence per rank, including tail dummies.
            if len(lengths) != 1:
                raise ValueError("causal eval accounting expects one sequence per call")
            rounds = [[trace["selected_rows"].numel()] for trace in model.causal_route_traces]
    return forward_flops(arm, model.config, lengths, rounds)


def parameter_counts(model):
    import torch
    import torch.distributed as dist
    from megatron.lite.model.qwen3_moe.common import is_expert_param

    counts = [0, 0]
    for name, parameter in model.named_parameters():
        owner = model.ps.expert_dp_rank == 0 if is_expert_param(name) else dist.get_rank() == 0
        if owner:
            counts[0] += parameter.numel()
            if not name.startswith("depth_routers."):
                counts[1] += parameter.numel()
    tensor = torch.tensor(counts, dtype=torch.int64, device=torch.cuda.current_device())
    dist.all_reduce(tensor)
    return {"independent_parameters": int(tensor[0]), "backbone_parameters": int(tensor[1])}
