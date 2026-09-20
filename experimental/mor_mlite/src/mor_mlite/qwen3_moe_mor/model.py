"""Physical Qwen3-MoE stack with expert-choice depth recurrence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import replace

import torch
import torch.distributed as dist
from megatron.lite.model.qwen3_moe.config import Qwen3MoEConfig
from megatron.lite.model.qwen3_moe.lite.model import Qwen3MoEModel
from megatron.lite.primitive import transformer_engine as te
from megatron.lite.primitive.ops.cross_entropy import vocab_parallel_cross_entropy
from megatron.lite.primitive.ops.linear_cross_entropy import linear_cross_entropy
from megatron.lite.primitive.ops.logprob import vocab_parallel_entropy
from megatron.lite.primitive.parallel import (
    gather_from_sequence_parallel,
    scatter_to_sequence_parallel,
)
from megatron.lite.primitive.utils import build_fp8_recipe
from megatron.lite.primitive.utils.moe import router_gating_linear
from megatron.lite.primitive.utils.packed_seq import PackedSeqParams
from torch import nn

from mor_mlite.config import DepthRouterConfig, MoRArchitectureConfig
from mor_mlite.distributed import (
    ActiveTokenBatch,
    ActiveTokenDispatcher,
    ActiveTokenLayout,
    ActiveTokenTransition,
    CommunicationCounters,
    EarlyExitParking,
    MagiCanonicalBackend,
    MagiDirectBackend,
    count_unexpected_real_token_ids,
    decode_magi_direct_plan,
)
from mor_mlite.distributed.packing import pack_route_plan_canonical
from mor_mlite.distributed.routing import (
    assert_route_plan_consistent,
    create_dense_dp_route_group,
    distributed_depth_route,
)
from mor_mlite.objective import nonzero_weight_denominator
from mor_mlite.routing import DepthRouter, RoutePlan, apply_recurrent_update

from .execution import (
    RegularDirectBackend,
    active_packed_seq_params,
    build_regular_direct_plan,
    rewrite_route_targets,
)
from .expert_route_probe import (
    ExpertRouteContext,
    ExpertRouteProbe,
    ExpertRouteReplayPlan,
)
from .gradient_scaling import install_replicated_expert_tp_gradient_average
from .rope import install_original_position_rope


def _temperature_to_float(temperature: float | torch.Tensor) -> float:
    if isinstance(temperature, torch.Tensor):
        if temperature.numel() != 1:
            raise ValueError("Qwen3-MoE MoR currently supports scalar temperature only")
        return float(temperature.detach().float().item())
    return float(temperature)


def _flatten_token_metadata(
    value: torch.Tensor | None, *, name: str, token_count: int
) -> torch.Tensor | None:
    if value is None:
        return None
    flattened = value.reshape(-1)
    if flattened.numel() != token_count:
        raise ValueError(
            f"{name} must contain one value per local token: {flattened.numel()} != {token_count}"
        )
    return flattened


def _lookup_router_logit_bias(
    active_global_token_ids: torch.Tensor,
    bias_global_token_ids: torch.Tensor | None,
    bias_values: torch.Tensor | None,
) -> torch.Tensor | None:
    """Resolve an optional synthetic router bias after arbitrary layout moves."""

    if bias_global_token_ids is None and bias_values is None:
        return None
    if bias_global_token_ids is None or bias_values is None:
        raise ValueError("router bias token IDs and values must be supplied together")
    source_ids = bias_global_token_ids.reshape(-1).to(
        device=active_global_token_ids.device, dtype=torch.long
    )
    source_values = bias_values.reshape(-1).to(
        device=active_global_token_ids.device, dtype=torch.float32
    )
    if source_ids.numel() != source_values.numel():
        raise ValueError("router bias token IDs and values must be aligned")
    if torch.unique(source_ids).numel() != source_ids.numel():
        raise ValueError("router bias global token IDs must be unique")
    if not torch.isfinite(source_values).all():
        raise ValueError("router bias values must be finite")
    if active_global_token_ids.numel() == 0:
        return source_values.new_empty((0,))
    order = torch.argsort(source_ids)
    sorted_ids = source_ids.index_select(0, order)
    rows = torch.searchsorted(sorted_ids, active_global_token_ids.to(dtype=torch.long))
    safe_rows = rows.clamp_max(max(0, sorted_ids.numel() - 1))
    if sorted_ids.numel() == 0 or not torch.equal(
        sorted_ids.index_select(0, safe_rows), active_global_token_ids.to(dtype=torch.long)
    ):
        raise ValueError("active token IDs are missing from the router bias mapping")
    return source_values.index_select(0, order).index_select(0, rows)


def _active_packed_seq_params(plan: RoutePlan) -> PackedSeqParams:
    cu_seqlens = plan.active_cu_seqlens.to(dtype=torch.int32)
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    max_seqlen = int(lengths.max().item()) if lengths.numel() else 0
    return PackedSeqParams.from_cu_seqlens(cu_seqlens, max_seqlen=max_seqlen)


def _masked_cp_mean(
    token_loss: torch.Tensor,
    loss_mask: torch.Tensor | None,
    *,
    cp_group,
    cp_size: int,
) -> torch.Tensor:
    """Return a CP-global mean with dist-opt-correct local gradients.

    Megatron distributed optimizer averages dense gradients over ``DP x CP``.
    The loss value must be the ordinary global mean inside each DP replica,
    while each CP rank's autograd contribution must be multiplied by CP so
    that the later optimizer average forms a sum across context shards and an
    average across data replicas.
    """

    if loss_mask is None:
        mask = torch.ones_like(token_loss, dtype=torch.float32)
    else:
        mask = loss_mask.transpose(0, 1).reshape_as(token_loss).float()
    local_sum = (token_loss.float() * mask).sum()
    local_count = mask.sum(dtype=torch.float32)
    if cp_size <= 1:
        return local_sum / nonzero_weight_denominator(local_count)
    if not dist.is_initialized() or cp_group is None:
        raise RuntimeError("CP loss normalization requires an initialized CP group")
    global_sum = local_sum.detach().clone()
    global_count = local_count.detach().clone()
    dist.all_reduce(global_sum, group=cp_group)
    dist.all_reduce(global_count, group=cp_group)
    denominator = nonzero_weight_denominator(global_count)
    local_autograd = float(cp_size) * local_sum / denominator
    global_value = global_sum / denominator
    return local_autograd + (global_value - local_autograd.detach())


class Qwen3MoEMoRModel(Qwen3MoEModel):
    """A native Qwen model with one registered physical recurrent stack.

    Instances are constructed by pinned MLite's native protocol and upgraded
    in place with :meth:`adapt_native`.  This avoids allocating a second Qwen
    stack and preserves all native parameter names.  ``self.layers`` contains
    exactly ``start + recurrent + end`` modules; only the recurrent slice is
    invoked repeatedly.  A separate scalar :class:`DepthRouter` is registered
    for every logical recursion.
    """

    @classmethod
    def adapt_native(
        cls,
        model: Qwen3MoEModel,
        *,
        logical_config: Qwen3MoEConfig,
        architecture: MoRArchitectureConfig,
        router_config: DepthRouterConfig,
        router_seed: int = 1234,
        initializer_range: float = 0.02,
        cp_transition: str = "magi_direct",
        route_mode: str = "learned",
        route_peer_consensus: bool = True,
    ) -> Qwen3MoEMoRModel:
        if not isinstance(model, Qwen3MoEModel):
            raise TypeError("adapt_native expects MLite's native Qwen3MoEModel")
        if len(model.layers) != architecture.physical_num_layers:
            raise ValueError(
                "native physical layer count does not match the MoR architecture: "
                f"{len(model.layers)} != {architecture.physical_num_layers}"
            )
        if getattr(model.ps, "pp_size", 1) != 1:
            raise ValueError("Qwen3-MoE MoR requires PP=1")
        model.__class__ = cls
        model._initialize_mor(
            logical_config=logical_config,
            architecture=architecture,
            router_config=router_config,
            router_seed=router_seed,
            initializer_range=initializer_range,
            cp_transition=cp_transition,
            route_mode=route_mode,
            route_peer_consensus=route_peer_consensus,
        )
        return model

    def _initialize_mor(
        self,
        *,
        logical_config: Qwen3MoEConfig,
        architecture: MoRArchitectureConfig,
        router_config: DepthRouterConfig,
        router_seed: int,
        initializer_range: float,
        cp_transition: str,
        route_mode: str,
        route_peer_consensus: bool,
    ) -> None:
        self.physical_config = self.config
        self.config = logical_config
        self.mor_architecture = architecture
        self.mor_router_config = router_config
        self.mor_cp_transition = cp_transition
        self.mor_route_mode = route_mode
        self.mor_route_peer_consensus = bool(route_peer_consensus)
        self.mor_route_group = create_dense_dp_route_group(self.ps)

        first_parameter = next(self.parameters(), None)
        router_device = (
            first_parameter.device if first_parameter is not None else torch.device("cpu")
        )
        router_dtype = first_parameter.dtype if first_parameter is not None else torch.float32
        self.depth_routers = nn.ModuleList(
            [
                DepthRouter(
                    logical_config.hidden_size,
                    architecture,
                    router_config,
                    initializer_range=initializer_range,
                    seed=router_seed + recursion,
                    round_index=recursion,
                ).to(device=router_device, dtype=router_dtype)
                for recursion in range(architecture.num_recursions)
            ]
        )
        for layer in self.layers:
            install_original_position_rope(layer.attn)

        self._mor_tp_expert_gradient_average_count = install_replicated_expert_tp_gradient_average(
            self
        )

        router_parameters = [router.proj.weight for router in self.depth_routers]
        known_sp_parameters = {id(parameter) for parameter in self.sp_params}
        self.sp_params.extend(
            parameter for parameter in router_parameters if id(parameter) not in known_sp_parameters
        )
        self.last_route_plans: tuple[RoutePlan, ...] = ()
        self.last_router_aux_loss: torch.Tensor | None = None
        self.last_router_aux_losses: torch.Tensor | None = None
        self.last_communication: dict[str, int] = {}
        self.last_round_traces: tuple[dict[str, torch.Tensor], ...] = ()
        self.last_post_merge_hidden: torch.Tensor | None = None
        self.last_end_layer_hiddens: tuple[torch.Tensor, ...] = ()
        self.last_end_sublayer_diagnostics: dict[str, torch.Tensor] = {}
        # Canonical hidden snapshots and per-layer QKV hooks are acceptance
        # diagnostics. Registry consumers keep them disabled by default so
        # normal training does not pay their sorting, retention, or hook cost.
        self._mor_capture_diagnostics = False
        # Native expert routing is intentionally invisible in production. The
        # parity runner opts into this detached probe for tiny full-state and
        # forward-only cross-topology acceptance.
        self._moe_expert_route_probe = ExpertRouteProbe(gating_linear=router_gating_linear)
        self.last_moe_expert_route_traces: tuple[dict[str, object], ...] = ()

    def set_moe_expert_route_probe(self, enabled: bool) -> None:
        """Enable detached native-MoE route evidence for parity runs only."""

        self._moe_expert_route_probe.set_enabled(enabled)
        self.last_moe_expert_route_traces = ()

    def set_mor_diagnostic_capture(self, enabled: bool) -> None:
        """Enable detached round traces and executable recurrent-QKV assertions."""

        self._mor_capture_diagnostics = bool(enabled)
        self.last_round_traces = ()
        self.last_post_merge_hidden = None
        self.last_end_layer_hiddens = ()
        self.last_end_sublayer_diagnostics = {}

    def _moe_probe_scope(
        self,
        layer: nn.Module,
        *,
        stage: str,
        round_index: int,
        stage_layer_index: int,
        physical_layer_index: int,
        logical_layer_index: int,
        global_token_ids: torch.Tensor | None,
        padding_mask: torch.Tensor | None,
    ):
        if not (self._moe_expert_route_probe.enabled or self._moe_expert_route_probe.replay_active):
            return nullcontext()
        if global_token_ids is None or padding_mask is None:
            raise ValueError("enabled expert-route probe requires token identity metadata")
        moe = getattr(layer, "moe", None)
        router = getattr(moe, "router", None)
        if not isinstance(router, nn.Module):
            raise TypeError(f"physical Qwen layer {physical_layer_index} exposes no MoE router")
        return self._moe_expert_route_probe.capture(
            router,
            context=ExpertRouteContext(
                stage=stage,
                round_index=round_index,
                stage_layer_index=stage_layer_index,
                physical_layer_index=physical_layer_index,
                logical_layer_index=logical_layer_index,
            ),
            global_token_ids=global_token_ids,
            padding_mask=padding_mask,
        )

    @property
    def start_layers(self) -> Sequence[nn.Module]:
        return self.layers[: self.mor_architecture.n_start_layers]

    @property
    def recurrent_layers(self) -> Sequence[nn.Module]:
        start = self.mor_architecture.n_start_layers
        end = start + self.mor_architecture.n_recurrent_layers
        return self.layers[start:end]

    @property
    def end_layers(self) -> Sequence[nn.Module]:
        start = self.mor_architecture.n_start_layers + self.mor_architecture.n_recurrent_layers
        return self.layers[start:]

    def _validate_execution_scope(self, packed_seq_params) -> None:
        if packed_seq_params is None:
            raise ValueError("Qwen3-MoE MoR active recurrence requires packed THD input")
        tp_size = int(getattr(self.ps, "tp_size", 1) or 1)
        cp_size = int(getattr(self.ps, "cp_size", 1) or 1)
        if self.mor_route_group.tp_size != tp_size or self.mor_route_group.cp_size != cp_size:
            raise RuntimeError("the TPxCP route group disagrees with MLite ParallelState")
        is_magi = getattr(packed_seq_params, "qkv_format", None) == "magi"
        if cp_size > 1 and not is_magi:
            raise ValueError("Qwen3-MoE MoR CP>1 requires a Magi packed batch")
        if cp_size == 1 and is_magi:
            raise ValueError("MagiAttention requires CP>1")
        if self.mor_cp_transition == "magi_canonical" and tp_size > 1:
            raise ValueError("magi_canonical is a CP-only oracle; TP+CP uses magi_direct")
        if self.mor_cp_transition == "static_reference" and cp_size > 1:
            raise ValueError("static_reference is not a production Qwen CP backend")
        if self.mor_route_mode not in {"learned", "replay"}:
            raise ValueError("MoR route mode must be learned or replay")

    def _resolve_metadata(
        self,
        h: torch.Tensor,
        *,
        position_ids: torch.Tensor | None,
        mor_sample_ids: torch.Tensor | None,
        mor_original_positions: torch.Tensor | None,
        mor_global_token_ids: torch.Tensor | None,
        mor_original_lengths: Mapping[int, int] | torch.Tensor | None,
        mor_padding_mask: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Mapping[int, int] | torch.Tensor,
        torch.Tensor,
    ]:
        token_count = h.reshape(-1, h.size(-1)).size(0)
        sample_ids = _flatten_token_metadata(
            mor_sample_ids, name="mor_sample_ids", token_count=token_count
        )
        original_positions = _flatten_token_metadata(
            mor_original_positions,
            name="mor_original_positions",
            token_count=token_count,
        )
        global_token_ids = _flatten_token_metadata(
            mor_global_token_ids,
            name="mor_global_token_ids",
            token_count=token_count,
        )
        padding_mask = _flatten_token_metadata(
            mor_padding_mask, name="mor_padding_mask", token_count=token_count
        )

        if self.mor_route_group.world_size > 1 and any(
            value is None
            for value in (
                sample_ids,
                original_positions,
                global_token_ids,
                padding_mask,
                mor_original_lengths,
            )
        ):
            raise ValueError(
                "distributed MoR requires protocol-packed TP-local token identity "
                "metadata and global original lengths"
            )
        if sample_ids is None:
            sample_ids = torch.zeros(token_count, dtype=torch.long, device=h.device)
        if original_positions is None:
            if position_ids is None:
                original_positions = torch.arange(token_count, dtype=torch.long, device=h.device)
            else:
                original_positions = _flatten_token_metadata(
                    position_ids,
                    name="position_ids",
                    token_count=token_count,
                )
                assert original_positions is not None
        if global_token_ids is None:
            global_token_ids = torch.arange(token_count, dtype=torch.long, device=h.device)
        if padding_mask is None:
            padding_mask = torch.zeros(token_count, dtype=torch.bool, device=h.device)
        else:
            padding_mask = padding_mask.to(dtype=torch.bool)
        if mor_original_lengths is None:
            mor_original_lengths = {}
            for sample_id in sorted(set(sample_ids[~padding_mask].detach().cpu().tolist())):
                positions = original_positions[(~padding_mask) & (sample_ids == int(sample_id))]
                inferred = int(positions.max().item()) + 1 if positions.numel() else 0
                mor_original_lengths[int(sample_id)] = inferred
        return (
            sample_ids.to(device=h.device, dtype=torch.long),
            original_positions.to(device=h.device, dtype=torch.long),
            global_token_ids.to(device=h.device, dtype=torch.long),
            mor_original_lengths,
            padding_mask.to(device=h.device),
        )

    @staticmethod
    def _gates_in_padded_layout(
        padded: ActiveTokenBatch,
        selected: ActiveTokenBatch,
    ) -> torch.Tensor:
        if selected.gates is None:
            raise ValueError("a routed active batch must carry differentiable gates")
        row_by_id = {
            int(token_id): row
            for row, token_id in enumerate(padded.layout.global_token_ids.detach().cpu().tolist())
        }
        try:
            rows = torch.tensor(
                [
                    row_by_id[int(token_id)]
                    for token_id in selected.layout.global_token_ids.detach().cpu().tolist()
                ],
                dtype=torch.long,
                device=padded.hidden.device,
            )
        except KeyError as exc:
            raise ValueError(
                f"selected token ID {exc.args[0]} is absent from the current padded layout"
            ) from exc
        gates = selected.gates.new_zeros((padded.num_tokens, *selected.gates.shape[1:]))
        return gates.index_copy(0, rows, selected.gates)

    def _run_recurrent_block(
        self,
        padded: ActiveTokenBatch,
        *,
        round_index: int,
        attention_position_ids: torch.Tensor,
        packed_seq_params,
        expected_active_global_token_ids: torch.Tensor,
        counters: CommunicationCounters,
        capture_diagnostics: bool,
    ) -> ActiveTokenBatch:
        if padded.gates is None:
            raise ValueError("recurrent compute requires one gate per local token")
        expected_attention_tokens = padded.num_tokens * int(getattr(self.ps, "tp_size", 1) or 1)
        if attention_position_ids.reshape(-1).numel() != expected_attention_tokens:
            raise ValueError(
                "attention position metadata must describe the full CP-local "
                f"post-TP-gather sequence: {attention_position_ids.numel()} != "
                f"{expected_attention_tokens}"
            )
        if capture_diagnostics:
            real_token_count, early_exit_token_count = count_unexpected_real_token_ids(
                padded.layout.global_token_ids,
                padded.layout.padding_mask,
                expected_active_global_token_ids,
            )
        else:
            real_token_count = early_exit_token_count = 0

        hidden_before = padded.hidden[:, None, :]
        block_output = hidden_before
        with counters.recurrent_block_scope():
            for layer_index, layer in enumerate(self.recurrent_layers):
                if capture_diagnostics:
                    qkv = getattr(getattr(layer, "attn", None), "qkv", None)
                    if not isinstance(qkv, nn.Module):
                        raise TypeError(
                            f"recurrent layer {layer_index} exposes no instrumentable "
                            "attn.qkv module"
                        )
                    qkv_scope = counters.recurrent_qkv_scope(
                        qkv,
                        expected_input_rows=padded.num_tokens,
                        real_token_count=real_token_count,
                        early_exit_token_count=early_exit_token_count,
                        context=f"recurrent layer {layer_index}",
                    )
                else:
                    qkv_scope = nullcontext()
                with qkv_scope:
                    physical_layer_index = self.mor_architecture.n_start_layers + layer_index
                    logical_layer_index = (
                        self.mor_architecture.n_start_layers
                        + round_index * self.mor_architecture.n_recurrent_layers
                        + layer_index
                    )
                    with self._moe_probe_scope(
                        layer,
                        stage="recurrent",
                        round_index=round_index,
                        stage_layer_index=layer_index,
                        physical_layer_index=physical_layer_index,
                        logical_layer_index=logical_layer_index,
                        global_token_ids=padded.layout.global_token_ids,
                        padding_mask=padded.layout.padding_mask,
                    ):
                        block_output = layer(
                            block_output,
                            position_ids=attention_position_ids,
                            packed_seq_params=packed_seq_params,
                        )
        # ``apply_recurrent_update`` deliberately accepts the same shape used
        # by the tiny oracle: [tokens, hidden] plus one scalar gate per token.
        # Keeping the singleton sequence-batch axis here would make a [tokens]
        # gate broadcast against the hidden dimension instead of the token
        # dimension.
        updated = apply_recurrent_update(
            hidden_before[:, 0, :],
            block_output[:, 0, :],
            padded.gates.to(dtype=block_output.dtype),
        )
        return ActiveTokenBatch(hidden=updated, layout=padded.layout)

    def _build_magi_runtime_key(self, packing):
        from megatron.lite.primitive.modules.attention.magi import (
            build_magi_attention_runtime_key,
        )

        cp_group = getattr(self.ps, "cp_group", None)
        if cp_group is None:
            raise RuntimeError("Magi active recurrence requires MLite's CP group")
        tp_size = int(getattr(self.ps, "tp_size", 1) or 1)
        return build_magi_attention_runtime_key(
            packing.cu_seqlens,
            num_heads_q=self.config.num_attention_heads // tp_size,
            num_heads_kv=self.config.num_key_value_heads // tp_size,
            head_dim=self.config.head_dim,
            cp_group=cp_group,
        )

    def _canonical_magi_batch(
        self,
        current_padded: ActiveTokenBatch,
        selected: ActiveTokenBatch,
        *,
        current_runtime_key,
        packing,
    ) -> ActiveTokenBatch:
        """Restore the old Magi layout, filter, and form a replicated oracle buffer."""

        from megatron.lite.primitive.modules.attention.magi import (
            undispatch_magi_attention_tensor,
        )

        if int(getattr(self.ps, "tp_size", 1) or 1) != 1:
            raise ValueError("magi_canonical is defined only for TP=1")
        local_gates = self._gates_in_padded_layout(current_padded, selected)
        full_hidden = undispatch_magi_attention_tensor(current_padded.hidden, current_runtime_key)
        full_gates = undispatch_magi_attention_tensor(local_gates, current_runtime_key)
        local_metadata = torch.stack(
            [
                current_padded.layout.global_token_ids,
                current_padded.layout.source_route_ranks,
                current_padded.layout.source_local_rows,
            ],
            dim=1,
        )
        full_metadata = undispatch_magi_attention_tensor(local_metadata, current_runtime_key)
        row_by_id = {
            int(token_id): row
            for row, token_id in enumerate(full_metadata[:, 0].detach().cpu().tolist())
        }
        real_ids = packing.global_token_ids[~packing.padding_mask]
        try:
            old_rows = torch.tensor(
                [row_by_id[int(token_id)] for token_id in real_ids.cpu().tolist()],
                dtype=torch.long,
                device=full_hidden.device,
            )
        except KeyError as exc:
            raise ValueError(
                f"selected token ID {exc.args[0]} is absent after Magi undispatch"
            ) from exc
        new_rows = packing.real_rows
        hidden = full_hidden.new_zeros((packing.num_rows, *full_hidden.shape[1:])).index_copy(
            0, new_rows, full_hidden.index_select(0, old_rows)
        )
        gates = full_gates.new_zeros((packing.num_rows, *full_gates.shape[1:])).index_copy(
            0, new_rows, full_gates.index_select(0, old_rows)
        )
        source_ranks = torch.full(
            (packing.num_rows,), -1, dtype=torch.long, device=hidden.device
        ).index_copy(0, new_rows, full_metadata[:, 1].index_select(0, old_rows))
        source_rows = torch.full_like(source_ranks, -1).index_copy(
            0, new_rows, full_metadata[:, 2].index_select(0, old_rows)
        )
        layout = ActiveTokenLayout(
            sample_ids=packing.sample_ids,
            position_ids=packing.position_ids,
            global_token_ids=packing.global_token_ids,
            source_route_ranks=source_ranks,
            source_local_rows=source_rows,
            current_route_ranks=torch.zeros_like(source_ranks),
            padding_mask=packing.padding_mask,
            destination_slots=torch.arange(
                packing.num_rows, dtype=torch.long, device=hidden.device
            ),
            round_index=packing.round_index,
            layout_kind="magi_canonical_replicated",
            replicated=True,
        )
        return ActiveTokenBatch(hidden=hidden, gates=gates, layout=layout)

    def _run_mor_backbone(
        self,
        h: torch.Tensor,
        *,
        position_ids: torch.Tensor | None,
        packed_seq_params,
        mor_sample_ids: torch.Tensor | None,
        mor_original_positions: torch.Tensor | None,
        mor_global_token_ids: torch.Tensor | None,
        mor_original_lengths: Mapping[int, int] | torch.Tensor | None,
        mor_padding_mask: torch.Tensor | None,
        mor_replay_plans: Sequence[RoutePlan] | None,
        mor_expert_replay_plans: Mapping[int, ExpertRouteReplayPlan] | None,
        mor_router_bias_global_token_ids: torch.Tensor | None,
        mor_router_logit_bias: torch.Tensor | None,
    ) -> tuple[torch.Tensor, list[RoutePlan], torch.Tensor, torch.Tensor]:
        """Execute the physical recurrent stack over nested active token sets.

        Router collectives are confined to one dense-DP replica's TP-SP x CP
        rectangle.  A changed active set is moved exactly once at the boundary;
        all physical layers in that recurrent block consume the same padded
        layout and packed-attention runtime.  Exited activations are parked on
        their current owner and take no part in later Q/K/V computation.
        """

        self._validate_execution_scope(packed_seq_params)
        self._moe_expert_route_probe.begin_forward(replay_plans=mor_expert_replay_plans)
        self.last_moe_expert_route_traces = ()
        self.last_round_traces = ()
        capture_diagnostics = self._mor_capture_diagnostics

        # Start-layer MoE routers run after TP/CP protocol packing but before
        # the recurrent ActiveTokenLayout exists.  Resolve the same metadata
        # early only when the parity probe is enabled; normal execution keeps
        # its existing path and overhead.
        resolved_metadata = None
        if self._moe_expert_route_probe.enabled or self._moe_expert_route_probe.replay_active:
            resolved_metadata = self._resolve_metadata(
                h,
                position_ids=position_ids,
                mor_sample_ids=mor_sample_ids,
                mor_original_positions=mor_original_positions,
                mor_global_token_ids=mor_global_token_ids,
                mor_original_lengths=mor_original_lengths,
                mor_padding_mask=mor_padding_mask,
            )
            probe_global_token_ids = resolved_metadata[2]
            probe_padding_mask = resolved_metadata[4]
        else:
            probe_global_token_ids = None
            probe_padding_mask = None

        for layer_index, layer in enumerate(self.start_layers):
            with self._moe_probe_scope(
                layer,
                stage="start",
                round_index=-1,
                stage_layer_index=layer_index,
                physical_layer_index=layer_index,
                logical_layer_index=layer_index,
                global_token_ids=probe_global_token_ids,
                padding_mask=probe_padding_mask,
            ):
                h = layer(h, position_ids=position_ids, packed_seq_params=packed_seq_params)

        original_shape = h.shape
        h_flat = h.reshape(-1, h.size(-1))
        if resolved_metadata is None:
            resolved_metadata = self._resolve_metadata(
                h,
                position_ids=position_ids,
                mor_sample_ids=mor_sample_ids,
                mor_original_positions=mor_original_positions,
                mor_global_token_ids=mor_global_token_ids,
                mor_original_lengths=mor_original_lengths,
                mor_padding_mask=mor_padding_mask,
            )
        (
            sample_ids,
            original_positions,
            global_token_ids,
            original_lengths,
            padding_mask,
        ) = resolved_metadata
        route_group = self.mor_route_group
        tp_size = route_group.tp_size
        cp_size = route_group.cp_size
        source_layout = ActiveTokenLayout.from_local(
            sample_ids=sample_ids,
            position_ids=original_positions,
            route_rank=route_group.rank,
            global_token_ids=global_token_ids,
            padding_mask=padding_mask,
            round_index=0,
            drop_padding=False,
        )
        current_padded = ActiveTokenBatch(hidden=h_flat, layout=source_layout)
        active = current_padded.index_select(~padding_mask)
        dispatcher = ActiveTokenDispatcher(route_group)
        parking = EarlyExitParking(h_flat, padding_mask=padding_mask)

        if cp_size == 1:
            backend = RegularDirectBackend()
        elif self.mor_cp_transition == "magi_direct":
            backend = MagiDirectBackend()
        elif self.mor_cp_transition == "magi_canonical":
            backend = MagiCanonicalBackend()
        else:  # guarded by _validate_execution_scope; keep the failure local.
            raise ValueError("Qwen CP recurrence supports magi_direct or magi_canonical")
        transition = ActiveTokenTransition(
            dispatcher,
            parking,
            backend=backend,
        )
        active = transition.enter_first_round(active)

        plans: list[RoutePlan] = []
        weighted_aux_losses: list[torch.Tensor] = []
        round_traces: list[dict[str, torch.Tensor]] = []
        replay_plans = tuple(mor_replay_plans or ())
        if self.mor_route_mode == "replay" and not replay_plans:
            raise ValueError("route_mode='replay' requires one RoutePlan per recursion")
        if self.mor_route_mode == "learned" and replay_plans:
            raise ValueError("learned routing cannot also consume replay RoutePlans")
        if replay_plans and len(replay_plans) != len(self.depth_routers):
            raise ValueError("mor_replay_plans must contain one plan per recursion")

        current_attention_positions = position_ids
        if current_attention_positions is None:
            if tp_size != 1:
                raise ValueError(
                    "TP sequence-parallel recurrence requires full CP-local position_ids"
                )
            current_attention_positions = original_positions.clamp_min(0).unsqueeze(0)
        current_packed_seq_params = packed_seq_params
        current_runtime_key = getattr(packed_seq_params, "magi_runtime_key", None)

        for recursion, router in enumerate(self.depth_routers):
            candidate_global_token_ids = active.layout.global_token_ids.detach()
            round_logit_bias = _lookup_router_logit_bias(
                active.layout.global_token_ids,
                mor_router_bias_global_token_ids,
                mor_router_logit_bias,
            )
            replay_plan = replay_plans[recursion] if replay_plans else None
            routed = distributed_depth_route(
                router,
                active,
                original_lengths=original_lengths,
                route_group=route_group,
                round_index=recursion,
                replay_plan=replay_plan,
                # The topology-specific target placement is attached below.
                # Perform the single peer-consensus check on that final plan,
                # rather than validating a provisional current-owner target.
                verify_peer_consistency=False,
                counters=dispatcher.counters,
                logit_bias=round_logit_bias,
            )
            plan = routed.plan
            weighted_aux_losses.append(routed.weighted_aux_loss)

            if recursion == 0:
                # The first expert-choice budget is exactly 100%.  Keeping its
                # existing MLite/Magi layout avoids an otherwise useless A2A.
                first_round_incomplete = not bool(routed.selected_local_mask.all().item())
                first_round_incomplete = route_group.any(
                    first_round_incomplete, device=active.hidden.device
                )
                if route_group.world_size > 1:
                    dispatcher.counters.collective_calls += 1
                if first_round_incomplete:
                    raise AssertionError(
                        "the first recurrent router must select every real local token"
                    )
                # No layout transition occurs at 100% capacity. RoutePlan
                # targets must therefore describe the actual padded source
                # rows used by recurrent compute, not the compact router rows.
                plan = replace(
                    plan,
                    target_tp_ranks=plan.source_tp_ranks,
                    target_cp_ranks=plan.source_cp_ranks,
                    target_local_rows=plan.source_local_rows,
                )
                current_padded = ActiveTokenBatch(
                    hidden=current_padded.hidden,
                    gates=self._gates_in_padded_layout(current_padded, routed.selected_batch),
                    layout=current_padded.layout.with_round(recursion),
                )
            else:
                packing = pack_route_plan_canonical(
                    plan,
                    tp_size=tp_size,
                    cp_size=cp_size,
                    use_magi=cp_size > 1,
                    original_lengths=original_lengths,
                )

                # Gates are differentiable local values.  The global RoutePlan
                # is detached metadata and must never replace them.
                gate_vector = routed.selected_local_gates.new_zeros(
                    (active.num_tokens, *routed.selected_local_gates.shape[1:])
                )
                gate_vector = gate_vector.index_copy(
                    0,
                    routed.selected_local_indices,
                    routed.selected_local_gates,
                )
                routed_input = ActiveTokenBatch(
                    hidden=active.hidden,
                    gates=gate_vector,
                    layout=active.layout,
                )

                if cp_size == 1:
                    direct_plan = build_regular_direct_plan(
                        routed.selected_batch,
                        packing=packing,
                        route_group=route_group,
                    )
                    result = transition.advance(
                        routed_input,
                        routed.selected_local_mask,
                        round_index=recursion,
                        backend_context={"direct_plan": direct_plan},
                    )
                    plan = rewrite_route_targets(
                        plan,
                        real_token_ids=direct_plan.global_real_token_ids,
                        target_ranks=direct_plan.global_target_ranks,
                        destination_slots=direct_plan.global_destination_slots,
                        route_group=route_group,
                    )
                    if result.dispatched:
                        current_padded = result.active
                        current_attention_positions = direct_plan.attention_position_ids.unsqueeze(
                            0
                        )
                        current_packed_seq_params = active_packed_seq_params(packing)
                    else:
                        current_padded = ActiveTokenBatch(
                            hidden=current_padded.hidden,
                            gates=self._gates_in_padded_layout(
                                current_padded, routed.selected_batch
                            ),
                            layout=current_padded.layout.with_round(recursion),
                        )
                elif self.mor_cp_transition == "magi_direct":
                    runtime_key = self._build_magi_runtime_key(packing)
                    direct_plan = decode_magi_direct_plan(
                        routed.selected_batch,
                        runtime_key=runtime_key,
                        route_group=route_group,
                        **packing.magi_decode_kwargs(),
                    )
                    result = transition.advance(
                        routed_input,
                        routed.selected_local_mask,
                        round_index=recursion,
                        backend_context={"direct_plan": direct_plan},
                    )
                    plan = rewrite_route_targets(
                        plan,
                        real_token_ids=direct_plan.global_real_token_ids,
                        target_ranks=direct_plan.global_target_ranks,
                        destination_slots=direct_plan.global_destination_slots,
                        route_group=route_group,
                    )
                    if result.dispatched:
                        current_padded = result.active
                        current_runtime_key = runtime_key
                        current_attention_positions = direct_plan.attention_position_ids.unsqueeze(
                            0
                        )
                        current_packed_seq_params = active_packed_seq_params(
                            packing,
                            cp_group=getattr(self.ps, "cp_group", None),
                            cp_rank=int(getattr(self.ps, "cp_rank", 0) or 0),
                            cp_size=cp_size,
                            runtime_key=runtime_key,
                        )
                    else:
                        current_padded = ActiveTokenBatch(
                            hidden=current_padded.hidden,
                            gates=self._gates_in_padded_layout(
                                current_padded, routed.selected_batch
                            ),
                            layout=current_padded.layout.with_round(recursion),
                        )
                else:
                    # Canonical mode is a correctness oracle: restore the old
                    # CP layout, filter globally, then let Magi dispatch the new
                    # canonical buffer.  TP>1 was rejected above.
                    changed = route_group.any(
                        bool((~routed.selected_local_mask).any().item()),
                        device=active.hidden.device,
                    )
                    if route_group.world_size > 1:
                        dispatcher.counters.collective_calls += 1
                    exited = active.index_select(~routed.selected_local_mask)
                    if changed:
                        if current_runtime_key is None:
                            raise RuntimeError("magi_canonical lost the previous round runtime key")
                        runtime_key = self._build_magi_runtime_key(packing)
                        canonical = self._canonical_magi_batch(
                            current_padded,
                            routed.selected_batch,
                            current_runtime_key=current_runtime_key,
                            packing=packing,
                        )
                        parking_ticket = parking.prepare(exited)
                        before = dispatcher.counters.snapshot()
                        current_padded = backend.rebalance(
                            canonical,
                            dispatcher,
                            context={"runtime_key": runtime_key},
                        )
                        dispatcher.counters.assert_hidden_rebalance_delta(
                            before,
                            1,
                            context=f"canonical boundary entering round {recursion}",
                        )
                        parking.commit(parking_ticket)
                        dispatcher.counters.record_active_set_change()
                        current_runtime_key = runtime_key
                        placement = decode_magi_direct_plan(
                            routed.selected_batch,
                            runtime_key=runtime_key,
                            route_group=route_group,
                            **packing.magi_decode_kwargs(),
                        )
                        plan = rewrite_route_targets(
                            plan,
                            real_token_ids=placement.global_real_token_ids,
                            target_ranks=placement.global_target_ranks,
                            destination_slots=placement.global_destination_slots,
                            route_group=route_group,
                        )
                        current_attention_positions = placement.attention_position_ids.unsqueeze(0)
                        current_packed_seq_params = active_packed_seq_params(
                            packing,
                            cp_group=getattr(self.ps, "cp_group", None),
                            cp_rank=int(getattr(self.ps, "cp_rank", 0) or 0),
                            cp_size=cp_size,
                            runtime_key=runtime_key,
                        )
                    else:
                        parking.park(exited)
                        dispatcher.counters.skipped_unchanged_boundaries += 1
                        if current_runtime_key is None:
                            raise RuntimeError(
                                "magi_canonical unchanged boundary lost its runtime key"
                            )
                        # No dispatch is needed, but the provisional router plan
                        # uses compact candidate rows.  Replay metadata must name
                        # the actual padded Magi compute slots retained from the
                        # previous round.
                        placement = decode_magi_direct_plan(
                            routed.selected_batch,
                            runtime_key=current_runtime_key,
                            route_group=route_group,
                            **packing.magi_decode_kwargs(),
                        )
                        plan = rewrite_route_targets(
                            plan,
                            real_token_ids=placement.global_real_token_ids,
                            target_ranks=placement.global_target_ranks,
                            destination_slots=placement.global_destination_slots,
                            route_group=route_group,
                        )
                        current_padded = ActiveTokenBatch(
                            hidden=current_padded.hidden,
                            gates=self._gates_in_padded_layout(
                                current_padded, routed.selected_batch
                            ),
                            layout=current_padded.layout.with_round(recursion),
                        )

            if self.mor_route_peer_consensus:
                assert_route_plan_consistent(plan, route_group, device=active.hidden.device)
                if route_group.world_size > 1:
                    dispatcher.counters.collective_calls += 1

            current_padded = self._run_recurrent_block(
                current_padded,
                round_index=recursion,
                attention_position_ids=current_attention_positions,
                packed_seq_params=current_packed_seq_params,
                expected_active_global_token_ids=plan.global_token_ids,
                counters=dispatcher.counters,
                capture_diagnostics=capture_diagnostics,
            )
            active = current_padded.index_select(~current_padded.layout.padding_mask)
            active = active.with_round(recursion)
            if capture_diagnostics:
                canonical = active.canonicalized()
                round_traces.append(
                    {
                        "round": torch.tensor(
                            recursion, dtype=torch.int64, device=active.hidden.device
                        ),
                        "global_token_ids": canonical.layout.global_token_ids.detach(),
                        "sample_ids": canonical.layout.sample_ids.detach(),
                        "original_positions": canonical.layout.position_ids.detach(),
                        "hidden": canonical.hidden.detach(),
                        # ``routed`` was evaluated before this round's layout
                        # transition; retain that exact identity order beside
                        # its raw logits for cutoff-error diagnostics.
                        "candidate_global_token_ids": candidate_global_token_ids,
                        "router_logits": routed.raw_logits.detach(),
                        "selected_global_token_ids": (
                            routed.selected_batch.layout.global_token_ids.detach()
                        ),
                        "selected_gates": routed.selected_local_gates.detach(),
                    }
                )
            plans.append(plan)

        h_flat = transition.finalize(active)
        self.last_round_traces = tuple(round_traces)
        counters = dispatcher.counters
        counters.assert_execution_contract(
            expected_recurrent_blocks=len(self.depth_routers),
            expected_recurrent_qkv_checks=(
                len(self.depth_routers) * len(self.recurrent_layers) if capture_diagnostics else 0
            ),
        )
        # ``physical_collectives`` is a compatibility field for MoR routing
        # and active-layout orchestration only.  It does not count collectives
        # internal to MagiAttention, native Qwen MoE, TP, or optimizer kernels.
        self.last_communication = {
            "active_set_changes": counters.active_set_changes,
            "hidden_rebalances": counters.hidden_rebalances,
            "recurrent_inner_dispatches": counters.recurrent_inner_dispatches,
            "early_exit_qkv_tokens": counters.early_exit_qkv_tokens,
            "recurrent_block_calls": counters.recurrent_block_calls,
            "recurrent_qkv_checks": counters.recurrent_qkv_checks,
            "recurrent_qkv_real_tokens": counters.recurrent_qkv_real_tokens,
            "final_inverse_merges": counters.inverse_calls,
            "physical_collectives": counters.collective_calls,
            "hidden_all_to_all": counters.hidden_all_to_all,
            "gate_all_to_all": counters.gate_all_to_all,
            "metadata_all_to_all": counters.metadata_all_to_all,
            "magi_dispatches": counters.magi_dispatches,
            "skipped_full_first_round": counters.skipped_full_first_round,
            "skipped_unchanged_boundaries": counters.skipped_unchanged_boundaries,
        }

        h = h_flat.reshape(original_shape)
        self.last_post_merge_hidden = h.detach() if capture_diagnostics else None
        end_layer_hiddens: list[torch.Tensor] = []
        end_sublayer_diagnostics: dict[str, torch.Tensor] = {}
        end_physical_offset = (
            self.mor_architecture.n_start_layers + self.mor_architecture.n_recurrent_layers
        )
        end_logical_offset = (
            self.mor_architecture.n_start_layers
            + self.mor_architecture.num_recursions * self.mor_architecture.n_recurrent_layers
        )
        for layer_index, layer in enumerate(self.end_layers):
            hook_handles = []
            captured_sublayers: dict[str, torch.Tensor] = {}
            if capture_diagnostics:

                def capture_output(name: str, sink: dict[str, torch.Tensor] = captured_sublayers):
                    def hook(_module, _inputs, output):
                        if not isinstance(output, torch.Tensor):
                            raise TypeError(f"end-layer diagnostic {name} is not a tensor")
                        sink[name] = output.detach()

                    return hook

                def capture_input(name: str, sink: dict[str, torch.Tensor] = captured_sublayers):
                    def hook(_module, inputs):
                        if not inputs or not isinstance(inputs[0], torch.Tensor):
                            raise TypeError(f"end-layer diagnostic {name} has no tensor input")
                        sink[name] = inputs[0].detach()

                    return hook

                hook_handles.extend(
                    (
                        layer.attn.register_forward_hook(capture_output("attention_output")),
                        layer.mlp_norm.register_forward_pre_hook(
                            capture_input("post_attention_residual")
                        ),
                        layer.mlp_norm.register_forward_hook(capture_output("mlp_norm_output")),
                        layer.moe.register_forward_hook(capture_output("moe_output")),
                    )
                )
            try:
                with self._moe_probe_scope(
                    layer,
                    stage="end",
                    round_index=-1,
                    stage_layer_index=layer_index,
                    physical_layer_index=end_physical_offset + layer_index,
                    logical_layer_index=end_logical_offset + layer_index,
                    global_token_ids=global_token_ids,
                    padding_mask=padding_mask,
                ):
                    h = layer(h, position_ids=position_ids, packed_seq_params=packed_seq_params)
            finally:
                for hook_handle in hook_handles:
                    hook_handle.remove()
            if capture_diagnostics:
                expected_sublayers = {
                    "attention_output",
                    "post_attention_residual",
                    "mlp_norm_output",
                    "moe_output",
                }
                if set(captured_sublayers) != expected_sublayers:
                    raise RuntimeError(
                        "end-layer diagnostics did not observe every sublayer: "
                        f"layer={layer_index}, observed={sorted(captured_sublayers)}"
                    )
                end_sublayer_diagnostics.update(
                    {f"{layer_index}_{name}": value for name, value in captured_sublayers.items()}
                )
                end_layer_hiddens.append(h.detach())
        self.last_end_layer_hiddens = tuple(end_layer_hiddens)
        self.last_end_sublayer_diagnostics = end_sublayer_diagnostics
        self.last_moe_expert_route_traces = self._moe_expert_route_probe.finish_forward()
        if weighted_aux_losses:
            router_aux_losses = torch.stack(weighted_aux_losses)
            router_aux_loss = router_aux_losses.sum()
        else:
            router_aux_losses = h.new_empty((0,), dtype=torch.float32)
            router_aux_loss = h.new_zeros((), dtype=torch.float32)
        return h, plans, router_aux_loss, router_aux_losses

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        packed_seq_params=None,
        labels: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        temperature: float | torch.Tensor = 1.0,
        use_fused_kernels: bool = False,
        calculate_entropy: bool = False,
        return_log_probs: bool = True,
        *,
        mor_sample_ids: torch.Tensor | None = None,
        mor_original_positions: torch.Tensor | None = None,
        mor_global_token_ids: torch.Tensor | None = None,
        mor_original_lengths: Mapping[int, int] | torch.Tensor | None = None,
        mor_padding_mask: torch.Tensor | None = None,
        mor_replay_plans: Sequence[RoutePlan] | None = None,
        mor_expert_replay_plans: Mapping[int, ExpertRouteReplayPlan] | None = None,
        mor_router_bias_global_token_ids: torch.Tensor | None = None,
        mor_router_logit_bias: torch.Tensor | None = None,
        return_route_plans: bool = False,
        return_full_logits: bool = False,
    ) -> dict:
        if self.embed is not None:
            if input_ids is None:
                raise ValueError("input_ids are required on the embedding stage")
            h = self.embed(input_ids)
        else:
            if hidden_states is None:
                hidden_states = self._input_tensor
            if hidden_states is None:
                raise ValueError("hidden_states are required without an embedding")
            h = hidden_states

        fp8_context = (
            te.fp8_autocast(enabled=True, fp8_recipe=build_fp8_recipe())
            if self.fp8
            else nullcontext()
        )
        with fp8_context:
            if self.embed is not None:
                h = scatter_to_sequence_parallel(h, self.ps)
            h, route_plans, router_aux_loss, router_aux_losses = self._run_mor_backbone(
                h,
                position_ids=position_ids,
                packed_seq_params=packed_seq_params,
                mor_sample_ids=mor_sample_ids,
                mor_original_positions=mor_original_positions,
                mor_global_token_ids=mor_global_token_ids,
                mor_original_lengths=mor_original_lengths,
                mor_padding_mask=mor_padding_mask,
                mor_replay_plans=mor_replay_plans,
                mor_expert_replay_plans=mor_expert_replay_plans,
                mor_router_bias_global_token_ids=mor_router_bias_global_token_ids,
                mor_router_logit_bias=mor_router_logit_bias,
            )

        self.last_route_plans = tuple(route_plans)
        self.last_router_aux_loss = router_aux_loss
        self.last_router_aux_losses = router_aux_losses
        output: dict = {
            "hidden_states": h,
            "mor_router_aux_loss": router_aux_loss,
            # Per-round local means are required to form an exact global-batch
            # objective when dense-DP replicas contain different sequence
            # lengths.  Their candidate denominators are generally different,
            # so one aggregate aux scale is insufficient.
            "mor_router_aux_losses": router_aux_losses,
        }
        if self._mor_capture_diagnostics:
            if self.last_post_merge_hidden is None:
                raise RuntimeError("MoR diagnostic capture lost the post-merge hidden state")
            output["mor_diagnostic_post_merge_hidden"] = gather_from_sequence_parallel(
                self.last_post_merge_hidden, self.ps
            )
            for layer_index, end_hidden in enumerate(self.last_end_layer_hiddens):
                output[f"mor_diagnostic_end_hidden_{layer_index}"] = gather_from_sequence_parallel(
                    end_hidden, self.ps
                )
            for name, value in sorted(self.last_end_sublayer_diagnostics.items()):
                output[f"mor_diagnostic_end_sublayer_{name}"] = gather_from_sequence_parallel(
                    value, self.ps
                )
        if return_route_plans:
            output["mor_route_plans"] = route_plans

        if self.head is None:
            return output
        hidden_for_head = self.norm(h)
        if self._mor_capture_diagnostics:
            # The normal model output remains sequence-parallel.  Parity needs
            # a token-aligned observation on which the protocol can run its
            # ordinary THD/Magi inverse permutation, so gather only these
            # detached diagnostic tensors across TP.  This deliberately lives
            # behind the opt-in capture flag and adds no collective or retained
            # autograd state to production training.
            output["mor_diagnostic_final_hidden"] = gather_from_sequence_parallel(
                h.detach(), self.ps
            )
            output["mor_diagnostic_hidden_for_head"] = gather_from_sequence_parallel(
                hidden_for_head.detach(), self.ps
            )
        if labels is None:
            logits = self.head(hidden_for_head)
            output["logits"] = self.head.gather(logits)
            return output

        temperature_value = _temperature_to_float(temperature)
        labels_sb = labels.transpose(0, 1).contiguous()
        if use_fused_kernels:
            hidden_full = gather_from_sequence_parallel(hidden_for_head, self.ps)
            log_probs, entropy = linear_cross_entropy(
                hidden_full,
                self._head_weight_for_fused_ce(hidden_full),
                labels_sb,
                temperature_value,
                self.ps.tp_group,
            )
            token_loss = -log_probs
            output["loss"] = _masked_cp_mean(
                token_loss,
                loss_mask,
                cp_group=getattr(self.ps, "cp_group", None),
                cp_size=int(getattr(self.ps, "cp_size", 1) or 1),
            )
            if return_log_probs:
                output["log_probs"] = log_probs.transpose(0, 1).contiguous()
            if calculate_entropy:
                output["entropy"] = entropy.transpose(0, 1).contiguous()
            if return_full_logits:
                # The fused CE primitive intentionally avoids materializing
                # logits.  Parity can request them explicitly without changing
                # the production default or the loss computation.
                with torch.no_grad():
                    parity_logits = self.head(hidden_for_head)
                    if temperature_value != 1.0:
                        parity_logits = parity_logits / temperature_value
                    output["logits"] = self.head.gather(parity_logits)
        else:
            logits = self.head(hidden_for_head)
            if temperature_value != 1.0:
                logits = logits / temperature_value
            token_loss = vocab_parallel_cross_entropy(logits, labels_sb, self.ps.tp_group)
            output["loss"] = _masked_cp_mean(
                token_loss,
                loss_mask,
                cp_group=getattr(self.ps, "cp_group", None),
                cp_size=int(getattr(self.ps, "cp_size", 1) or 1),
            )
            if return_log_probs:
                output["log_probs"] = (-token_loss).transpose(0, 1).contiguous()
            if calculate_entropy:
                entropy = vocab_parallel_entropy(logits, self.ps.tp_group)
                output["entropy"] = entropy.transpose(0, 1).contiguous()
            if return_full_logits:
                output["logits"] = self.head.gather(logits)
        if self.training:
            output["loss"] = output["loss"] + router_aux_loss.to(dtype=output["loss"].dtype)
        return output


__all__ = ["Qwen3MoEMoRModel"]
