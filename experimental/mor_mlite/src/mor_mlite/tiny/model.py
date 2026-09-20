"""A small dependency-free PyTorch oracle for the MoR execution semantics.

This model is intentionally not a performance backend.  Its attention and MoE
implementations favor transparent, deterministic math so distributed runs can
be canonicalized and compared against it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from mor_mlite.config import DepthRouterConfig, MoRArchitectureConfig
from mor_mlite.data import PackedBatch, original_sample_lengths, packed_lm_targets
from mor_mlite.distributed import (
    ActiveTokenBatch,
    ActiveTokenDispatcher,
    ActiveTokenLayout,
    CommunicationCounters,
    StaticReferenceBackend,
    TPxCPRouteGroup,
    count_unexpected_real_token_ids,
    distributed_depth_route,
    gather_static_active_qkv,
)
from mor_mlite.objective import nonzero_weight_denominator
from mor_mlite.routing import DepthRouter, RoutePlan, apply_recurrent_update


@dataclass(slots=True)
class TinyMoRConfig:
    vocab_size: int = 257
    hidden_size: int = 32
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    intermediate_size: int = 64
    num_experts: int = 4
    num_experts_per_tok: int = 2
    max_position_embeddings: int = 512
    rope_theta: float = 10_000.0
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    architecture: MoRArchitectureConfig = field(default_factory=MoRArchitectureConfig.tiny)
    depth_router: DepthRouterConfig = field(default_factory=DepthRouterConfig)

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if not 1 <= self.num_experts_per_tok <= self.num_experts:
            raise ValueError("num_experts_per_tok must be in [1, num_experts]")
        if self.hidden_size // self.num_attention_heads % 2:
            raise ValueError("attention head dimension must be even for RoPE")
        if self.vocab_size <= 3 or self.intermediate_size <= 0:
            raise ValueError("vocab_size and intermediate_size must be positive")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["architecture"] = self.architecture.to_dict()
        result["depth_router"] = self.depth_router.to_dict()
        return result


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        variance = hidden.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = hidden.float() * torch.rsqrt(variance + self.eps)
        return normalized.to(hidden.dtype) * self.weight.to(hidden.dtype)


def _apply_rope(value: torch.Tensor, positions: torch.Tensor, theta: float) -> torch.Tensor:
    # value: [tokens, heads, head_dim]
    head_dim = value.shape[-1]
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=value.device, dtype=torch.float32) / head_dim)
    )
    angles = positions.float().unsqueeze(-1) * inv_freq.unsqueeze(0)
    cos = angles.cos().unsqueeze(1).to(value.dtype)
    sin = angles.sin().unsqueeze(1).to(value.dtype)
    even, odd = value[..., 0::2], value[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


class PositionAwareGQA(nn.Module):
    def __init__(self, config: TinyMoRConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.rope_theta = config.rope_theta
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        sample_ids: torch.Tensor,
        original_positions: torch.Tensor,
        global_token_ids: torch.Tensor | None = None,
        static_cp_group: TPxCPRouteGroup | None = None,
    ) -> torch.Tensor:
        local_token_count = hidden.shape[0]
        q = self.q_proj(hidden).view(local_token_count, self.num_heads, self.head_dim)
        k = self.k_proj(hidden).view(local_token_count, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden).view(local_token_count, self.num_kv_heads, self.head_dim)
        q = _apply_rope(q, original_positions, self.rope_theta)
        k = _apply_rope(k, original_positions, self.rope_theta)
        local_output_rows: torch.Tensor | None = None
        if static_cp_group is not None and static_cp_group.world_size > 1:
            if global_token_ids is None:
                raise ValueError("static CP attention requires global_token_ids")
            gathered = gather_static_active_qkv(
                q,
                k,
                v,
                sample_ids=sample_ids,
                position_ids=original_positions,
                global_token_ids=global_token_ids,
                route_group=static_cp_group,
            )
            q, k, v = gathered.q, gathered.k, gathered.v
            sample_ids = gathered.sample_ids
            original_positions = gathered.position_ids
            local_output_rows = gathered.local_rows
        repeat = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)

        output = torch.empty_like(q)
        for sample_id in sorted({int(x) for x in sample_ids.detach().cpu().tolist()}):
            rows = torch.nonzero(sample_ids == sample_id, as_tuple=False).flatten()
            # Routers promise this ordering, but sorting here makes the oracle
            # reject neither a different physical shard layout nor padding removal.
            rows = rows[torch.argsort(original_positions[rows], stable=True)]
            q_s = q[rows].transpose(0, 1)
            k_s = k[rows].transpose(0, 1)
            v_s = v[rows].transpose(0, 1)
            scores = torch.matmul(q_s.float(), k_s.float().transpose(-2, -1))
            scores = scores / math.sqrt(self.head_dim)
            causal = torch.ones(
                rows.numel(), rows.numel(), dtype=torch.bool, device=hidden.device
            ).triu(diagonal=1)
            scores.masked_fill_(causal.unsqueeze(0), float("-inf"))
            probs = torch.softmax(scores, dim=-1).to(v_s.dtype)
            output[rows] = torch.matmul(probs, v_s).transpose(0, 1)
        if local_output_rows is not None:
            output = output.index_select(0, local_output_rows)
        return self.o_proj(output.reshape(local_token_count, self.num_heads * self.head_dim))


class TinyExpert(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(hidden)) * self.up(hidden))


class NoDropTopKMoE(nn.Module):
    """Token-choice expert routing with no capacity drop."""

    def __init__(self, config: TinyMoRConfig) -> None:
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                TinyExpert(config.hidden_size, config.intermediate_size)
                for _ in range(config.num_experts)
            ]
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = self.router(hidden).float()
        # Expert IDs are already ascending, so stable score sort gives the
        # deterministic secondary key for exact ties.
        expert_ids = torch.argsort(logits, dim=-1, descending=True, stable=True)[:, : self.top_k]
        weights = torch.softmax(torch.gather(logits, 1, expert_ids), dim=-1).to(hidden.dtype)
        output = torch.zeros_like(hidden)
        for slot in range(self.top_k):
            ids = expert_ids[:, slot]
            for expert_id, expert in enumerate(self.experts):
                rows = torch.nonzero(ids == expert_id, as_tuple=False).flatten()
                if rows.numel():
                    contribution = expert(hidden[rows]) * weights[rows, slot].unsqueeze(-1)
                    output = output.index_add(0, rows, contribution)
        return output


class TinyTransformerLayer(nn.Module):
    def __init__(self, config: TinyMoRConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attn = PositionAwareGQA(config)
        self.moe_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.moe = NoDropTopKMoE(config)

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        sample_ids: torch.Tensor,
        original_positions: torch.Tensor,
        global_token_ids: torch.Tensor | None = None,
        static_cp_group: TPxCPRouteGroup | None = None,
    ) -> torch.Tensor:
        hidden = hidden + self.attn(
            self.attn_norm(hidden),
            sample_ids=sample_ids,
            original_positions=original_positions,
            global_token_ids=global_token_ids,
            static_cp_group=static_cp_group,
        )
        return hidden + self.moe(self.moe_norm(hidden))


@dataclass(slots=True)
class TinyMoROutput:
    logits: torch.Tensor
    lm_loss: torch.Tensor | None
    aux_loss: torch.Tensor
    total_loss: torch.Tensor | None
    route_plans: list[RoutePlan]
    hidden_by_round: list[torch.Tensor]
    router_logits: list[torch.Tensor]
    router_candidate_ids: list[torch.Tensor]
    communication: dict[str, int]
    global_token_ids: torch.Tensor


class _StaticCPGlobalValue(torch.autograd.Function):
    """Expose one global scalar while keeping the CP-scaled local gradient."""

    @staticmethod
    def forward(
        _ctx: object,
        differentiable_local_value: torch.Tensor,
        detached_global_value: torch.Tensor,
    ) -> torch.Tensor:
        del differentiable_local_value
        return detached_global_value.clone()

    @staticmethod
    def backward(_ctx: object, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return grad_output, None


def _static_cp_global_mean(
    local_sum: torch.Tensor,
    local_count: torch.Tensor,
    route_group: TPxCPRouteGroup,
) -> torch.Tensor:
    """Match a global token mean under a later replicated-parameter CP average."""

    if local_sum.ndim or local_count.ndim or local_count.numel() != 1:
        raise ValueError("static CP loss inputs must be scalar tensors")
    if route_group.world_size == 1:
        return local_sum / nonzero_weight_denominator(local_count).to(local_sum.dtype)
    if route_group.tp_size != 1 or route_group.process_group is None:
        raise ValueError("static CP loss is restricted to an explicit TP=1 process group")
    global_sum = local_sum.detach().clone()
    global_count = local_count.detach().clone()
    dist.all_reduce(global_sum, op=dist.ReduceOp.SUM, group=route_group.process_group)
    dist.all_reduce(global_count, op=dist.ReduceOp.SUM, group=route_group.process_group)
    denominator = nonzero_weight_denominator(global_count).to(local_sum.dtype)
    differentiable = local_sum * route_group.cp_size / denominator
    return _StaticCPGlobalValue.apply(differentiable, global_sum / denominator)


class TinyMoRModel(nn.Module):
    """Tiny semantic oracle with one physical recurrent block.

    Ordinary calls are single-rank. Passing a TP=1/CP>1 ``static_cp_group``
    enables the fixed-ownership distributed diagnostic path.
    """

    def __init__(self, config: TinyMoRConfig | None = None, *, seed: int = 1234) -> None:
        super().__init__()
        self.config = config or TinyMoRConfig()
        torch.manual_seed(seed)
        self.embedding = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.start_layers = nn.ModuleList(
            [
                TinyTransformerLayer(self.config)
                for _ in range(self.config.architecture.n_start_layers)
            ]
        )
        # Registered exactly once; forward invokes these modules in every round.
        self.recurrent_layers = nn.ModuleList(
            [
                TinyTransformerLayer(self.config)
                for _ in range(self.config.architecture.n_recurrent_layers)
            ]
        )
        self.depth_routers = nn.ModuleList(
            [
                DepthRouter(
                    self.config.hidden_size,
                    self.config.architecture,
                    self.config.depth_router,
                    initializer_range=self.config.initializer_range,
                    seed=seed + 10_000 + round_index,
                    round_index=round_index,
                )
                for round_index in range(self.config.architecture.num_recursions)
            ]
        )
        self.end_layers = nn.ModuleList(
            [
                TinyTransformerLayer(self.config)
                for _ in range(self.config.architecture.n_end_layers)
            ]
        )
        self.final_norm = RMSNorm(self.config.hidden_size, self.config.rms_norm_eps)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.apply(self._init_weights)
        # Router seeds/config are part of the architecture contract and must
        # not be overwritten by the general module initialization above.
        for round_index, router in enumerate(self.depth_routers):
            generator = torch.Generator(device="cpu").manual_seed(seed + 10_000 + round_index)
            initialized = torch.empty_like(router.proj.weight, device="cpu").normal_(
                mean=0.0, std=self.config.initializer_range, generator=generator
            )
            with torch.no_grad():
                router.proj.weight.copy_(initialized.to(router.proj.weight.device))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    @property
    def logical_num_layers(self) -> int:
        return self.config.architecture.logical_num_layers

    @property
    def physical_num_layers(self) -> int:
        return self.config.architecture.physical_num_layers

    def _metadata(self, batch: PackedBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = batch.input_ids.device
        positions = batch.make_position_ids().to(device=device, dtype=torch.long)
        samples = batch.extras.get("sample_ids")
        if samples is None:
            samples = torch.repeat_interleave(
                torch.arange(len(batch), device=device), batch.seq_lens.to(device=device).long()
            )
        global_ids = batch.extras.get("global_token_ids")
        if global_ids is None:
            global_ids = torch.arange(batch.total_tokens, device=device)
        return samples.to(device), positions, global_ids.to(device)

    @staticmethod
    def _lm_targets(batch: PackedBatch) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply MLite's per-sequence left roll exactly once.

        ``PackedBatch.labels`` and ``loss_mask`` are unshifted source rows.
        THD packing rolls both within sequence boundaries. An explicit mask
        excludes the synthetic final target; None includes every real token.
        """

        return packed_lm_targets(batch)

    @staticmethod
    def _static_cp_source_rows(
        sample_ids: torch.Tensor,
        positions: torch.Tensor,
        global_token_ids: torch.Tensor,
        route_group: TPxCPRouteGroup,
    ) -> torch.Tensor:
        """Assign canonical contiguous per-sample chunks without rebalancing later."""

        if route_group.tp_size != 1 or route_group.cp_size <= 1:
            raise ValueError("static_reference requires TP=1 and CP>1")
        owners = torch.empty_like(global_token_ids, dtype=torch.long)
        for sample_id in sorted(set(sample_ids.detach().cpu().tolist())):
            rows = torch.nonzero(sample_ids == sample_id, as_tuple=False).flatten()
            by_id = torch.argsort(global_token_ids.index_select(0, rows), stable=True)
            rows = rows.index_select(0, by_id)
            by_position = torch.argsort(positions.index_select(0, rows), stable=True)
            rows = rows.index_select(0, by_position)
            base, extra = divmod(rows.numel(), route_group.cp_size)
            offset = 0
            for cp_rank in range(route_group.cp_size):
                take = base + int(cp_rank < extra)
                if take:
                    owners[rows[offset : offset + take]] = cp_rank
                offset += take
        return torch.nonzero(owners == route_group.cp_rank, as_tuple=False).flatten()

    def _forward_static_cp(
        self,
        batch: PackedBatch,
        *,
        route_group: TPxCPRouteGroup,
        route_mode: str,
        replay_plans: Mapping[int, RoutePlan] | None,
        capture_hidden: bool,
    ) -> TinyMoROutput:
        """Run the complete tiny MoR graph with fixed, potentially imbalanced CP owners."""

        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("static_reference CP requires initialized torch.distributed")
        if route_group.process_group is None:
            raise ValueError("static_reference CP requires an explicit process group")
        sample_ids, positions, global_ids = self._metadata(batch)
        original_lengths = original_sample_lengths(batch.seq_lens, sample_ids)
        source_rows = self._static_cp_source_rows(sample_ids, positions, global_ids, route_group)
        local_samples = sample_ids.index_select(0, source_rows)
        local_positions = positions.index_select(0, source_rows)
        local_ids = global_ids.index_select(0, source_rows)
        hidden = self.embedding(batch.input_ids.index_select(0, source_rows))
        for layer in self.start_layers:
            hidden = layer(
                hidden,
                sample_ids=local_samples,
                original_positions=local_positions,
                global_token_ids=local_ids,
                static_cp_group=route_group,
            )

        layout = ActiveTokenLayout.from_local(
            sample_ids=local_samples,
            position_ids=local_positions,
            route_rank=route_group.rank,
            global_token_ids=local_ids,
        )
        active = ActiveTokenBatch(hidden=hidden, layout=layout)
        full_hidden = hidden
        aux_loss = hidden.float().sum() * 0.0
        plans: list[RoutePlan] = []
        hidden_by_round: list[torch.Tensor] = []
        router_logits: list[torch.Tensor] = []
        router_candidate_ids: list[torch.Tensor] = []
        counters = CommunicationCounters()
        dispatcher = ActiveTokenDispatcher(route_group, counters=counters)
        backend = StaticReferenceBackend()
        global_active_count = global_ids.numel()
        local_routing_bias = None
        routing_bias = batch.extras.get("routing_bias")
        if routing_bias is not None and bool(batch.extras.get("apply_routing_bias", False)):
            local_routing_bias = routing_bias.index_select(0, source_rows)

        for round_index, router in enumerate(self.depth_routers):
            candidate_ids = active.layout.global_token_ids
            active_logit_bias = (
                None
                if local_routing_bias is None
                else local_routing_bias.index_select(0, active.layout.source_local_rows)
            )
            replay = None if replay_plans is None else replay_plans.get(round_index)
            routed = distributed_depth_route(
                router,
                active,
                original_lengths=original_lengths,
                route_group=route_group,
                replay_plan=replay if route_mode == "replay" else None,
                counters=counters,
                logit_bias=active_logit_bias,
            )
            aux_loss = aux_loss + routed.weighted_aux_loss
            selected = routed.selected_batch
            selected_global_count = routed.plan.global_token_ids.numel()
            if round_index == 0:
                if selected_global_count != global_active_count:
                    raise AssertionError("static CP first recurrent round must retain every token")
                counters.skipped_full_first_round += 1
            elif selected_global_count < global_active_count:
                before = counters.snapshot()
                selected = backend.rebalance(selected, dispatcher)
                counters.assert_hidden_rebalance_delta(
                    before,
                    1,
                    context=f"static CP boundary entering round {round_index}",
                )
                counters.record_active_set_change()
            elif selected_global_count == global_active_count:
                counters.skipped_unchanged_boundaries += 1
            else:
                raise AssertionError("static CP active sets must be nested")
            global_active_count = selected_global_count

            block_input = selected.hidden
            block_output = block_input
            real_count, early_exit_count = count_unexpected_real_token_ids(
                selected.layout.global_token_ids,
                selected.layout.padding_mask,
                routed.plan.global_token_ids,
            )
            with counters.recurrent_block_scope():
                for layer_index, layer in enumerate(self.recurrent_layers):
                    with counters.recurrent_qkv_scope(
                        layer.attn.q_proj,
                        expected_input_rows=block_output.size(0),
                        real_token_count=real_count,
                        early_exit_token_count=early_exit_count,
                        context=f"tiny static CP recurrent layer {layer_index}",
                    ):
                        block_output = layer(
                            block_output,
                            sample_ids=selected.layout.sample_ids,
                            original_positions=selected.layout.position_ids,
                            global_token_ids=selected.layout.global_token_ids,
                            static_cp_group=route_group,
                        )
            if selected.gates is None:
                raise AssertionError("distributed depth routing returned no selected gates")
            updated = apply_recurrent_update(block_input, block_output, selected.gates)
            full_hidden = full_hidden.index_copy(
                0, selected.layout.source_local_rows.to(dtype=torch.long), updated
            )
            active = ActiveTokenBatch(hidden=updated, layout=selected.layout)
            plans.append(routed.plan)
            router_logits.append(routed.raw_logits)
            router_candidate_ids.append(candidate_ids)
            if capture_hidden:
                hidden_by_round.append(full_hidden)

        hidden = full_hidden
        for layer in self.end_layers:
            hidden = layer(
                hidden,
                sample_ids=local_samples,
                original_positions=local_positions,
                global_token_ids=local_ids,
                static_cp_group=route_group,
            )
        logits = self.lm_head(self.final_norm(hidden))
        lm_loss: torch.Tensor | None = None
        total_loss: torch.Tensor | None = None
        if batch.labels is not None:
            labels, mask = self._lm_targets(batch)
            local_labels = labels.index_select(0, source_rows)
            local_mask = mask.index_select(0, source_rows)
            token_losses = F.cross_entropy(logits.float(), local_labels, reduction="none")
            lm_loss = _static_cp_global_mean(
                (token_losses * local_mask).sum(), local_mask.sum(), route_group
            )
            total_loss = lm_loss + aux_loss
        counters.assert_execution_contract(
            expected_recurrent_blocks=self.config.architecture.num_recursions,
            expected_recurrent_qkv_checks=(
                self.config.architecture.num_recursions
                * self.config.architecture.n_recurrent_layers
            ),
        )
        return TinyMoROutput(
            logits=logits,
            lm_loss=lm_loss,
            aux_loss=aux_loss,
            total_loss=total_loss,
            route_plans=plans,
            hidden_by_round=hidden_by_round,
            router_logits=router_logits,
            router_candidate_ids=router_candidate_ids,
            communication=asdict(counters.snapshot()),
            global_token_ids=local_ids,
        )

    def forward(
        self,
        batch: PackedBatch,
        *,
        route_mode: str = "learned",
        replay_plans: Mapping[int, RoutePlan] | None = None,
        capture_hidden: bool = True,
        static_cp_group: TPxCPRouteGroup | None = None,
    ) -> TinyMoROutput:
        if route_mode not in {"learned", "replay"}:
            raise ValueError("route_mode must be 'learned' or 'replay'")
        if route_mode == "replay":
            expected_rounds = set(range(self.config.architecture.num_recursions))
            if replay_plans is None or set(replay_plans) != expected_rounds:
                raise ValueError("replay mode requires exactly one RoutePlan for every round")
            if any(
                isinstance(key, bool)
                or not isinstance(key, int)
                or not isinstance(plan, RoutePlan)
                or plan.round_index != key
                for key, plan in replay_plans.items()
            ):
                raise ValueError("replay keys must match their RoutePlan round indices")
        if static_cp_group is not None and static_cp_group.world_size > 1:
            return self._forward_static_cp(
                batch,
                route_group=static_cp_group,
                route_mode=route_mode,
                replay_plans=replay_plans,
                capture_hidden=capture_hidden,
            )
        sample_ids, positions, global_ids = self._metadata(batch)
        original_lengths = original_sample_lengths(batch.seq_lens, sample_ids)
        hidden = self.embedding(batch.input_ids)
        for layer in self.start_layers:
            hidden = layer(
                hidden,
                sample_ids=sample_ids,
                original_positions=positions,
                global_token_ids=global_ids,
            )

        active_rows = torch.arange(hidden.shape[0], device=hidden.device)
        full_hidden = hidden
        aux_loss = hidden.float().sum() * 0.0
        plans: list[RoutePlan] = []
        hidden_by_round: list[torch.Tensor] = []
        router_logits: list[torch.Tensor] = []
        router_candidate_ids: list[torch.Tensor] = []
        counters = CommunicationCounters()

        for round_index, router in enumerate(self.depth_routers):
            active_hidden = full_hidden.index_select(0, active_rows)
            active_samples = sample_ids.index_select(0, active_rows)
            active_positions = positions.index_select(0, active_rows)
            active_ids = global_ids.index_select(0, active_rows)
            routing_bias = batch.extras.get("routing_bias")
            active_logit_bias = None
            if routing_bias is not None and bool(batch.extras.get("apply_routing_bias", False)):
                active_logit_bias = routing_bias.index_select(0, active_rows)
            replay = None if replay_plans is None else replay_plans.get(round_index)
            routed = router(
                active_hidden,
                sample_ids=active_samples,
                original_positions=active_positions,
                global_token_ids=active_ids,
                original_lengths=original_lengths,
                replay_plan=replay if route_mode == "replay" else None,
                logit_bias=active_logit_bias,
            )
            aux_loss = aux_loss + routed.weighted_aux_loss
            selected_local = routed.selected_indices
            selected_rows = active_rows.index_select(0, selected_local)
            if selected_rows.numel() != active_rows.numel():
                # A single-rank oracle has no physical communication, but it
                # still installs one new logical active layout at a shrinking
                # boundary.  Keep this source independent from route choice so
                # parity cannot pass through a copied/static counter.
                counters.record_active_set_change()
                counters.record_backend_hidden_rebalance()
            block_input = full_hidden.index_select(0, selected_rows)
            block_output = block_input
            selected_samples = sample_ids.index_select(0, selected_rows)
            selected_positions = positions.index_select(0, selected_rows)
            selected_ids = global_ids.index_select(0, selected_rows)
            real_count, early_exit_count = count_unexpected_real_token_ids(
                selected_ids,
                torch.zeros_like(selected_ids, dtype=torch.bool),
                routed.plan.global_token_ids,
            )
            with counters.recurrent_block_scope():
                for layer_index, layer in enumerate(self.recurrent_layers):
                    with counters.recurrent_qkv_scope(
                        layer.attn.q_proj,
                        expected_input_rows=block_output.size(0),
                        real_token_count=real_count,
                        early_exit_token_count=early_exit_count,
                        context=f"tiny recurrent layer {layer_index}",
                    ):
                        block_output = layer(
                            block_output,
                            sample_ids=selected_samples,
                            original_positions=selected_positions,
                            global_token_ids=selected_ids,
                        )
            updated = apply_recurrent_update(block_input, block_output, routed.selected_gates)
            full_hidden = full_hidden.index_copy(0, selected_rows, updated)
            active_rows = selected_rows
            plans.append(routed.plan)
            router_logits.append(routed.raw_logits)
            router_candidate_ids.append(active_ids)
            if capture_hidden:
                hidden_by_round.append(full_hidden)

        hidden = full_hidden
        for layer in self.end_layers:
            hidden = layer(
                hidden,
                sample_ids=sample_ids,
                original_positions=positions,
                global_token_ids=global_ids,
            )
        logits = self.lm_head(self.final_norm(hidden))
        lm_loss: torch.Tensor | None = None
        total_loss: torch.Tensor | None = None
        if batch.labels is not None:
            labels, mask = self._lm_targets(batch)
            token_losses = F.cross_entropy(logits.float(), labels, reduction="none")
            lm_loss = (token_losses * mask).sum() / nonzero_weight_denominator(mask.sum())
            total_loss = lm_loss + aux_loss
        counters.assert_execution_contract(
            expected_recurrent_blocks=self.config.architecture.num_recursions,
            expected_recurrent_qkv_checks=(
                self.config.architecture.num_recursions
                * self.config.architecture.n_recurrent_layers
            ),
        )
        return TinyMoROutput(
            logits=logits,
            lm_loss=lm_loss,
            aux_loss=aux_loss,
            total_loss=total_loss,
            route_plans=plans,
            hidden_by_round=hidden_by_round,
            router_logits=router_logits,
            router_candidate_ids=router_candidate_ids,
            communication={
                "active_set_changes": counters.active_set_changes,
                "hidden_rebalances": counters.hidden_rebalances,
                "recurrent_inner_dispatches": counters.recurrent_inner_dispatches,
                "early_exit_qkv_tokens": counters.early_exit_qkv_tokens,
                "recurrent_block_calls": counters.recurrent_block_calls,
                "recurrent_qkv_checks": counters.recurrent_qkv_checks,
                "recurrent_qkv_real_tokens": counters.recurrent_qkv_real_tokens,
            },
            global_token_ids=global_ids,
        )


__all__ = ["TinyMoRConfig", "TinyMoRModel", "TinyMoROutput"]
