"""Expert-choice depth routing for nested Mixture-of-Recursions rounds."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from mor_mlite.config import DepthRouterConfig, MoRArchitectureConfig

from .plan import RoutePlan


def stable_expert_choice_indices(
    scores: torch.Tensor,
    global_token_ids: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """Choose score-descending/token-ID-ascending rows deterministically.

    Selection is discrete, so sorting detached values is intentional.  The
    two-pass stable sort avoids perturbing scores with an epsilon tie breaker.
    """

    if scores.ndim != 1 or global_token_ids.ndim != 1 or scores.shape != global_token_ids.shape:
        raise ValueError("scores and global_token_ids must be equally-sized 1D tensors")
    if isinstance(top_k, bool) or not 0 <= top_k <= scores.numel():
        raise ValueError(f"top_k must be in [0, {scores.numel()}], got {top_k!r}")
    if top_k == 0:
        return torch.empty(0, dtype=torch.int64, device=scores.device)
    if not torch.isfinite(scores.detach()).all():
        raise ValueError("router scores must all be finite")
    if len(set(global_token_ids.detach().cpu().tolist())) != global_token_ids.numel():
        raise ValueError("global_token_ids must be unique")

    # Stable ID ordering establishes the secondary key.  The second stable sort
    # by score preserves it for exact ties.
    by_id = torch.argsort(global_token_ids, stable=True)
    by_score_within_id = torch.argsort(scores.detach()[by_id], descending=True, stable=True)
    return by_id[by_score_within_id[:top_k]]


@dataclass(frozen=True)
class ExpertChoiceSelection:
    """Discrete result produced from globally gathered scalar route metadata."""

    selected_indices: torch.Tensor
    active_cu_seqlens: torch.Tensor
    cutoff_score_margins: Mapping[int, float]


def _lookup_original_length(
    sample_id: int,
    original_lengths: Mapping[int, int] | torch.Tensor,
) -> int:
    if isinstance(original_lengths, torch.Tensor):
        if original_lengths.ndim != 1 or not 0 <= sample_id < original_lengths.numel():
            raise ValueError(f"no original length is available for sample {sample_id}")
        value = int(original_lengths[sample_id].item())
    else:
        if sample_id not in original_lengths:
            raise ValueError(f"no original length is available for sample {sample_id}")
        value = original_lengths[sample_id]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"original length for sample {sample_id} must be a non-negative integer")
    return value


def select_expert_choice_per_sample(
    scores: torch.Tensor,
    *,
    sample_ids: torch.Tensor,
    original_positions: torch.Tensor,
    global_token_ids: torch.Tensor,
    original_lengths: Mapping[int, int] | torch.Tensor,
    architecture: MoRArchitectureConfig,
    round_index: int,
    padding_mask: torch.Tensor | None = None,
) -> ExpertChoiceSelection:
    """Select a nested round from gathered scalar scores and token metadata.

    Distributed callers gather detached FP32 scores and integer metadata across
    their TP-SP x CP route group, invoke this function identically on every
    peer, and then recover differentiable gates from each token's original
    local logit.
    """

    tensors = (scores, sample_ids, original_positions, global_token_ids)
    if any(tensor.ndim != 1 or tensor.shape != scores.shape for tensor in tensors):
        raise ValueError("scores and all token metadata must be equally-sized 1D tensors")
    if padding_mask is None:
        padding_mask = torch.zeros_like(scores, dtype=torch.bool)
    if padding_mask.shape != scores.shape or padding_mask.dtype != torch.bool:
        raise ValueError("padding_mask must be a bool tensor shaped like scores")
    if not torch.isfinite(scores[~padding_mask].detach()).all():
        raise ValueError("real router scores must be finite")
    real_ids = global_token_ids[~padding_mask]
    if len(set(real_ids.detach().cpu().tolist())) != real_ids.numel():
        raise ValueError("non-padding global_token_ids must be unique")
    if torch.any(sample_ids[~padding_mask] < 0) or torch.any(original_positions[~padding_mask] < 0):
        raise ValueError("real sample IDs and original positions must be non-negative")

    chosen_by_sample: list[torch.Tensor] = []
    margins: dict[int, float] = {}
    cu_values = [0]
    sample_order = sorted(set(sample_ids[~padding_mask].detach().cpu().tolist()))
    for sample_id in sample_order:
        candidates = torch.nonzero(
            (~padding_mask) & (sample_ids == sample_id), as_tuple=False
        ).flatten()
        original_length = _lookup_original_length(int(sample_id), original_lengths)
        if torch.any(original_positions[candidates] >= original_length):
            raise ValueError(f"sample {sample_id} contains a position outside its original length")
        requested = architecture.top_k(original_length, round_index)
        top_k = min(requested, candidates.numel())
        local_choice = stable_expert_choice_indices(
            scores[candidates], global_token_ids[candidates], top_k
        )
        chosen = candidates[local_choice]

        # Restore causal order after score-ranked selection, with global ID as a
        # deterministic secondary key for malformed duplicate positions.
        by_id = torch.argsort(global_token_ids[chosen], stable=True)
        by_position = torch.argsort(original_positions[chosen][by_id], stable=True)
        chosen = chosen[by_id[by_position]]
        chosen_by_sample.append(chosen)
        cu_values.append(cu_values[-1] + chosen.numel())

        if top_k < candidates.numel():
            ranked = stable_expert_choice_indices(
                scores[candidates], global_token_ids[candidates], candidates.numel()
            )
            margin = scores[candidates[ranked[top_k - 1]]] - scores[candidates[ranked[top_k]]]
            margins[int(sample_id)] = float(margin.detach().cpu().item())
        else:
            margins[int(sample_id)] = math.inf

    selected_indices = (
        torch.cat(chosen_by_sample)
        if chosen_by_sample
        else torch.empty(0, dtype=torch.int64, device=scores.device)
    )
    return ExpertChoiceSelection(
        selected_indices=selected_indices,
        active_cu_seqlens=torch.tensor(cu_values, dtype=torch.int32, device=scores.device),
        cutoff_score_margins=margins,
    )


def _dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def globally_normalized_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """BCE sum divided by the global number of active tokens.

    The collective operates only on detached value/count tensors.  The returned
    scalar has a globally identical value while retaining the local autograd
    path.  Summing replicated-router gradients across ``group`` therefore gives
    the exact gradient of the global mean.  Without initialized distributed
    state this naturally reduces to an ordinary local mean.
    """

    if logits.shape != targets.shape:
        raise ValueError("logits and targets must have the same shape")
    local_sum = F.binary_cross_entropy_with_logits(logits.float(), targets.float(), reduction="sum")
    local_count = torch.tensor(logits.numel(), dtype=torch.int64, device=logits.device)
    if not _dist_ready():
        return local_sum / local_count.clamp_min(1).to(local_sum.dtype)

    global_count = local_count.clone()
    global_sum = local_sum.detach().clone()
    dist.all_reduce(global_count, op=dist.ReduceOp.SUM, group=group)
    dist.all_reduce(global_sum, op=dist.ReduceOp.SUM, group=group)
    denominator = global_count.clamp_min(1).to(local_sum.dtype)
    local_contribution = local_sum / denominator
    global_value = global_sum / denominator
    return local_contribution + (global_value - local_contribution.detach())


def apply_recurrent_update(
    hidden_before: torch.Tensor,
    recurrent_block_output: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    """Apply the exact MoR residual update ``h + gate * block(h)``."""

    if hidden_before.shape != recurrent_block_output.shape:
        raise ValueError("hidden_before and recurrent_block_output must have equal shape")
    if gate.ndim == hidden_before.ndim - 1:
        gate = gate.unsqueeze(-1)
    try:
        torch.broadcast_shapes(hidden_before.shape, gate.shape)
    except RuntimeError as error:
        raise ValueError("gate is not broadcastable to the hidden state") from error
    # Depth routers intentionally produce FP32 gates.  Keep mixed-precision
    # hidden states in the model dtype; the cast remains differentiable and
    # sends the accumulated gradient back to the FP32 router parameter.
    gate = gate.to(device=recurrent_block_output.device, dtype=recurrent_block_output.dtype)
    return hidden_before + gate * recurrent_block_output


def validate_replay_capacity(
    plan: RoutePlan,
    *,
    candidate_samples: torch.Tensor,
    selected_samples: torch.Tensor,
    original_lengths: Mapping[int, int] | torch.Tensor,
    architecture: MoRArchitectureConfig,
    round_index: int,
) -> None:
    """Validate budgets on both local and distributed replay paths."""
    candidates = Counter(candidate_samples.detach().cpu().tolist())
    selected = Counter(selected_samples.detach().cpu().tolist())
    cumulative = [0]
    for sample, available in sorted(candidates.items()):
        expected = min(
            architecture.top_k(_lookup_original_length(sample, original_lengths), round_index),
            available,
        )
        if selected[sample] != expected:
            raise ValueError(
                f"replay capacity mismatch for sample {sample}: {selected[sample]} != {expected}"
            )
        cumulative.append(cumulative[-1] + expected)
    if set(selected) - candidates.keys():
        raise ValueError("replay contains samples absent from the candidate set")
    if plan.active_cu_seqlens.detach().cpu().tolist() != cumulative:
        raise ValueError("replay active_cu_seqlens does not match per-sample capacity")


class _ReplaySelectedGates(torch.autograd.Function):
    """Use RoutePlan gate values in forward and the live router Jacobian backward."""

    @staticmethod
    def forward(
        _ctx: object,
        live_gates: torch.Tensor,
        replay_gates: torch.Tensor,
    ) -> torch.Tensor:
        del _ctx
        return replay_gates.to(device=live_gates.device, dtype=live_gates.dtype).clone()

    @staticmethod
    def backward(
        _ctx: object,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        del _ctx
        return grad_output, None


def replay_selected_gates(
    live_gates: torch.Tensor,
    replay_gates: torch.Tensor,
) -> torch.Tensor:
    """Pin replay forward values without detaching the live depth router."""

    if live_gates.shape != replay_gates.shape:
        raise ValueError("live and replay gate tensors must have the same shape")
    if not live_gates.dtype.is_floating_point or not replay_gates.dtype.is_floating_point:
        raise TypeError("live and replay gates must use floating dtypes")
    if replay_gates.numel() and not bool(torch.isfinite(replay_gates).all().item()):
        raise ValueError("replay gates must be finite")
    return _ReplaySelectedGates.apply(live_gates, replay_gates)


@dataclass(frozen=True)
class DepthRouterOutput:
    raw_logits: torch.Tensor
    scores: torch.Tensor
    selected_indices: torch.Tensor
    selected_gates: torch.Tensor
    selected_mask: torch.Tensor
    aux_loss: torch.Tensor
    weighted_aux_loss: torch.Tensor
    plan: RoutePlan


class DepthRouter(nn.Module):
    """A scalar router followed by deterministic per-sample expert choice."""

    def __init__(
        self,
        hidden_size: int,
        architecture: MoRArchitectureConfig,
        config: DepthRouterConfig | None = None,
        *,
        initializer_range: float = 0.02,
        seed: int = 1234,
        round_index: int | None = None,
    ) -> None:
        super().__init__()
        if isinstance(hidden_size, bool) or hidden_size <= 0:
            raise ValueError("hidden_size must be a positive integer")
        if not math.isfinite(initializer_range) or initializer_range <= 0.0:
            raise ValueError("initializer_range must be finite and greater than zero")
        if round_index is not None:
            architecture.capacity_for_round(round_index)

        self.hidden_size = int(hidden_size)
        self.architecture = architecture
        self.config = config or DepthRouterConfig()
        self.round_index = round_index
        self.proj = nn.Linear(self.hidden_size, 1, bias=False, dtype=torch.float32)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        initialized = torch.empty(
            self.proj.weight.shape, dtype=torch.float32, device="cpu"
        ).normal_(mean=0.0, std=initializer_range, generator=generator)
        with torch.no_grad():
            self.proj.weight.copy_(initialized)

        # MLite/Megatron uses these attributes when deciding which replicated
        # sequence-parallel parameters require an explicit TP reduction.
        self.proj.weight.tensor_model_parallel = False
        self.proj.weight.sequence_parallel = True
        self.proj.weight.allreduce = True
        self.proj.weight.is_expert = False

    def _resolve_round(self, round_index: int | None) -> int:
        resolved = self.round_index if round_index is None else round_index
        if resolved is None:
            raise ValueError("round_index must be supplied to the router or forward call")
        self.architecture.capacity_for_round(resolved)
        return resolved

    @staticmethod
    def _metadata_tensor(
        value: int | torch.Tensor | None,
        *,
        length: int,
        device: torch.device,
        default: int,
        name: str,
    ) -> torch.Tensor:
        if value is None:
            return torch.full((length,), default, dtype=torch.int64, device=device)
        if isinstance(value, int):
            return torch.full((length,), value, dtype=torch.int64, device=device)
        if value.ndim != 1 or value.numel() != length:
            raise ValueError(f"{name} must be a scalar or a length-{length} tensor")
        return value.to(device=device, dtype=torch.int64)

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        sample_ids: torch.Tensor,
        original_positions: torch.Tensor,
        global_token_ids: torch.Tensor,
        original_lengths: Mapping[int, int] | torch.Tensor,
        round_index: int | None = None,
        padding_mask: torch.Tensor | None = None,
        source_tp_ranks: int | torch.Tensor | None = None,
        source_cp_ranks: int | torch.Tensor | None = None,
        source_local_rows: torch.Tensor | None = None,
        target_tp_ranks: int | torch.Tensor | None = None,
        target_cp_ranks: int | torch.Tensor | None = None,
        target_local_rows: torch.Tensor | None = None,
        replay_plan: RoutePlan | None = None,
        aux_process_group: dist.ProcessGroup | None = None,
        logit_bias: torch.Tensor | None = None,
    ) -> DepthRouterOutput:
        if hidden.ndim != 2 or hidden.shape[-1] != self.hidden_size:
            raise ValueError(
                f"hidden must have shape [tokens, {self.hidden_size}], got {tuple(hidden.shape)}"
            )
        token_count = hidden.shape[0]
        device = hidden.device
        metadata = (sample_ids, original_positions, global_token_ids)
        if any(tensor.ndim != 1 or tensor.numel() != token_count for tensor in metadata):
            raise ValueError(
                "sample_ids, original_positions and global_token_ids must be token-aligned"
            )
        sample_ids = sample_ids.to(device=device, dtype=torch.int64)
        original_positions = original_positions.to(device=device, dtype=torch.int64)
        global_token_ids = global_token_ids.to(device=device, dtype=torch.int64)
        if padding_mask is None:
            padding_mask = torch.zeros(token_count, dtype=torch.bool, device=device)
        elif padding_mask.shape != (token_count,) or padding_mask.dtype != torch.bool:
            raise ValueError("padding_mask must be a token-aligned bool tensor")
        else:
            padding_mask = padding_mask.to(device=device)

        real_ids = global_token_ids[~padding_mask]
        if len(set(real_ids.detach().cpu().tolist())) != real_ids.numel():
            raise ValueError("non-padding global_token_ids must be unique")
        if torch.any(sample_ids[~padding_mask] < 0) or torch.any(
            original_positions[~padding_mask] < 0
        ):
            raise ValueError("real sample IDs and original positions must be non-negative")

        resolved_round = self._resolve_round(round_index)
        # ``model.to(torch.bfloat16)`` may cast the resident router parameter.
        # Cast through an autograd-visible operation so scalar logits and router
        # math remain FP32 without severing the gradient to that parameter.
        raw_logits = F.linear(hidden.float(), self.proj.weight.float()).squeeze(-1)
        if logit_bias is not None:
            if logit_bias.shape != (token_count,):
                raise ValueError("logit_bias must be a token-aligned vector")
            logit_bias = logit_bias.to(device=device, dtype=torch.float32)
            if not torch.isfinite(logit_bias[~padding_mask]).all():
                raise ValueError("real-token logit_bias values must be finite")
            raw_logits = raw_logits + logit_bias
        # Match the paper implementation's ``router(x / temperature)``.  With
        # a bias-free linear router this is exactly ``raw_logits / T``; the
        # same pre-sigmoid decision logits drive both expert choice and BCE.
        decision_logits = raw_logits / self.config.temperature
        scores = torch.sigmoid(decision_logits) * self.config.alpha

        if replay_plan is None:
            selection = select_expert_choice_per_sample(
                scores,
                sample_ids=sample_ids,
                original_positions=original_positions,
                global_token_ids=global_token_ids,
                original_lengths=original_lengths,
                architecture=self.architecture,
                round_index=resolved_round,
                padding_mask=padding_mask,
            )
            selected_indices = selection.selected_indices
            active_cu_seqlens = selection.active_cu_seqlens
            margins = dict(selection.cutoff_score_margins)
            mode = "learned"
        else:
            if replay_plan.round_index != resolved_round:
                raise ValueError(
                    f"replay round {replay_plan.round_index} does not match {resolved_round}"
                )
            selected_indices = replay_plan.replay_indices(
                global_token_ids, padding_mask=padding_mask
            )
            real = ~replay_plan.padding_mask
            for field, live in (
                ("sample_ids", sample_ids),
                ("original_positions", original_positions),
            ):
                expected = getattr(replay_plan, field)[real].to(device=device)
                if not torch.equal(expected, live[selected_indices]):
                    raise ValueError(f"replay {field} does not match live token metadata")
            validate_replay_capacity(
                replay_plan,
                candidate_samples=sample_ids[~padding_mask],
                selected_samples=sample_ids[selected_indices],
                original_lengths=original_lengths,
                architecture=self.architecture,
                round_index=resolved_round,
            )
            margins = dict(replay_plan.cutoff_score_margins)
            mode = "replay"

        selected_mask = torch.zeros(token_count, dtype=torch.bool, device=device)
        selected_mask[selected_indices] = True
        targets = selected_mask[~padding_mask].to(raw_logits.dtype)
        aux_loss = globally_normalized_bce_with_logits(
            decision_logits[~padding_mask], targets, group=aux_process_group
        )
        selected_gates = scores[selected_indices]
        if replay_plan is not None:
            replay_gates = replay_plan.replay_gates(global_token_ids[selected_indices])
            selected_gates = replay_selected_gates(selected_gates, replay_gates)

        source_tp = self._metadata_tensor(
            source_tp_ranks,
            length=token_count,
            device=device,
            default=0,
            name="source_tp_ranks",
        )
        source_cp = self._metadata_tensor(
            source_cp_ranks,
            length=token_count,
            device=device,
            default=0,
            name="source_cp_ranks",
        )
        source_rows = self._metadata_tensor(
            source_local_rows,
            length=token_count,
            device=device,
            default=-1,
            name="source_local_rows",
        )
        if source_local_rows is None:
            source_rows = torch.arange(token_count, dtype=torch.int64, device=device)
        target_tp = self._metadata_tensor(
            target_tp_ranks,
            length=token_count,
            device=device,
            default=0,
            name="target_tp_ranks",
        )
        target_cp = self._metadata_tensor(
            target_cp_ranks,
            length=token_count,
            device=device,
            default=0,
            name="target_cp_ranks",
        )
        target_rows = self._metadata_tensor(
            target_local_rows,
            length=token_count,
            device=device,
            default=-1,
            name="target_local_rows",
        )
        if target_tp_ranks is None:
            target_tp = source_tp
        if target_cp_ranks is None:
            target_cp = source_cp
        if target_local_rows is None:
            target_rows = source_rows

        if replay_plan is not None:
            selected_samples = sample_ids[selected_indices]
            sample_order = sorted(set(selected_samples.detach().cpu().tolist()))
            counts = [
                int((selected_samples == sample_id).sum().item()) for sample_id in sample_order
            ]
            cu_values = [0]
            for count in counts:
                cu_values.append(cu_values[-1] + count)
            active_cu_seqlens = torch.tensor(cu_values, dtype=torch.int32, device=device)

        plan = RoutePlan(
            round_index=resolved_round,
            mode=mode,
            sample_ids=sample_ids[selected_indices],
            original_positions=original_positions[selected_indices],
            global_token_ids=global_token_ids[selected_indices],
            source_tp_ranks=source_tp[selected_indices],
            source_cp_ranks=source_cp[selected_indices],
            source_local_rows=source_rows[selected_indices],
            target_tp_ranks=target_tp[selected_indices],
            target_cp_ranks=target_cp[selected_indices],
            target_local_rows=target_rows[selected_indices],
            selected_gates=selected_gates.detach(),
            active_cu_seqlens=active_cu_seqlens,
            padding_mask=torch.zeros(selected_indices.numel(), dtype=torch.bool, device=device),
            cutoff_score_margins=margins,
        )
        return DepthRouterOutput(
            raw_logits=raw_logits,
            scores=scores,
            selected_indices=selected_indices,
            selected_gates=selected_gates,
            selected_mask=selected_mask,
            aux_loss=aux_loss,
            weighted_aux_loss=aux_loss * self.config.aux_loss_coef,
            plan=plan,
        )
