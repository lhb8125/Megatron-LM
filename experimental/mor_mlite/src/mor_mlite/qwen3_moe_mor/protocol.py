"""Megatron-Lite protocol for the external ``qwen3_moe_mor`` model."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch
from megatron.lite.model.protocol_utils import (
    add_cross_entropy_fusion,
    add_loss_context_kwargs,
    pack_magi_forward_kwargs,
    pack_thd_forward_kwargs,
    router_replay_roots,
    unpack_magi_forward_output,
    unpack_thd_forward_output,
)
from megatron.lite.model.qwen3_moe.config import Qwen3MoEConfig
from megatron.lite.model.qwen3_moe.lite import protocol as _native_protocol
from megatron.lite.model.qwen3_moe.lite.protocol import ImplConfig as _NativeImplConfig
from megatron.lite.primitive.bundle import ModelBundle
from megatron.lite.primitive.ckpt import attach_model_sharded_state_dict
from megatron.lite.primitive.config import load_hf_config_dict
from megatron.lite.primitive.parallel.thd import (
    parallel_state_from_model,
    split_packed_to_cp_local,
    thd_pack_meta,
)
from megatron.lite.runtime.contracts.data import PackedBatch
from torch import nn

from mor_mlite.checkpoint_io import read_mor_sidecar
from mor_mlite.config import (
    DepthRouterConfig,
    MoRArchitectureConfig,
    MoRParallelConfig,
)
from mor_mlite.data import original_sample_lengths

from .checkpoint import (
    EXPERT_CLASSIFIER,
    PLACEMENT_FN,
    install_mor_metadata,
    physical_model_config,
)
from .checkpoint import load_hf_weights as _load_hf_weights
from .metadata import HF_FOLDING_METADATA_FILENAME, MoRCheckpointMetadata
from .model import Qwen3MoEMoRModel


@dataclass(frozen=True)
class ImplConfig(_NativeImplConfig):
    """Native Qwen implementation knobs plus the MoR physical topology."""

    n_start_layers: int = 3
    n_recurrent_layers: int = 14
    num_recursions: int = 3
    n_end_layers: int = 3
    capacity_schedule: str | tuple[float, ...] = "linear"
    depth_router_temperature: float = 1.0
    depth_router_alpha: float = 0.1
    depth_router_aux_loss_coef: float = 0.001
    depth_router_seed: int = 1234
    hf_folding_policy: str = "mean"
    cp_transition: str = "magi_direct"
    route_mode: str = "learned"
    route_peer_consensus: bool = True
    dense_dp_size: int = 1
    local_attention_backend: str = "te"

    def mor_architecture(self) -> MoRArchitectureConfig:
        return MoRArchitectureConfig(
            self.n_start_layers,
            self.n_recurrent_layers,
            self.num_recursions,
            self.n_end_layers,
            self.capacity_schedule,
        )

    def depth_router_config(self) -> DepthRouterConfig:
        return DepthRouterConfig(
            temperature=self.depth_router_temperature,
            alpha=self.depth_router_alpha,
            aux_loss_coef=self.depth_router_aux_loss_coef,
        )


def build_model_config(source: str | Path | dict, **overrides) -> Qwen3MoEConfig:
    """Build the native architecture config and retain an optional MoR sidecar."""

    raw_source = dict(source) if isinstance(source, dict) else load_hf_config_dict(str(source))
    model_config = _native_protocol.build_model_config(raw_source, **overrides)
    model_config._mor_initializer_range = float(raw_source.get("initializer_range", 0.02))
    model_config._mor_hf_source = "<dict>" if isinstance(source, dict) else str(source)
    if not isinstance(source, dict) and str(source):
        source_path = Path(source)
        if source_path.is_dir():
            metadata_path = source_path / HF_FOLDING_METADATA_FILENAME
            if metadata_path.exists():
                metadata = read_mor_sidecar(source_path)
                model_config._mor_architecture = metadata.architecture.to_dict()
                model_config._mor_folding_policy = metadata.folding_policy
    return model_config


def _model_attention_backend(model: nn.Module) -> str:
    current = model
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        backend = getattr(current, "attention_backend", None)
        if backend is not None:
            return str(backend)
        current = getattr(current, "module", None)
    return "te"


def _unwrapped_model(model: nn.Module) -> nn.Module:
    """Return the innermost model for protocol-side packing helpers.

    Megatron-Core's DDP wrapper owns a ``config`` attribute containing a
    generic ``TransformerConfig``.  The pinned Magi packer deliberately reads
    ``model.config`` and needs Qwen-specific fields such as
    ``num_key_value_heads`` and ``head_dim``.  Passing the wrapper therefore
    resolves the wrong config before its helper has a chance to follow
    ``.module``.  Packing against the underlying Qwen model preserves the same
    ParallelState while the actual forward still goes through DDP below.
    """

    current = model
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        inner = getattr(current, "module", None)
        if not isinstance(inner, nn.Module) or inner is current:
            break
        current = inner
    return current


def _padded_mor_metadata(model: nn.Module, batch: PackedBatch, kwargs: dict) -> dict:
    """Pack token identity metadata with the exact native THD/Magi layout."""

    ps = parallel_state_from_model(model)
    tp_size = int(getattr(ps, "tp_size", 1) or 1)
    cp_size = int(getattr(ps, "cp_size", 1) or 1)
    cp_rank = int(getattr(ps, "cp_rank", 0) or 0)
    tp_rank = int(getattr(ps, "tp_rank", 0) or 0)
    meta = thd_pack_meta(
        batch.seq_lens,
        tp_size=tp_size,
        cp_size=cp_size,
        cp_group=getattr(ps, "cp_group", None),
    )
    device = batch.input_ids.device
    total_padded = int(meta.cu_seqlens_padded[-1].item())
    sample_ids = torch.full((total_padded,), -1, dtype=torch.long, device=device)
    positions = torch.full((total_padded,), -1, dtype=torch.long, device=device)
    # Every dummy row gets a unique negative ID.  Real IDs are required to be
    # non-negative, so this remains collision-free after Magi and TP slicing.
    global_ids = -torch.arange(1, total_padded + 1, dtype=torch.long, device=device)
    padding_mask = torch.ones(total_padded, dtype=torch.bool, device=device)
    extras = batch.extras
    extra_sample_ids = extras.get("sample_ids")
    extra_positions = extras.get("original_position_ids", batch.position_ids)
    extra_global_ids = extras.get("global_token_ids")
    total_real = int(batch.seq_lens.sum().item())

    def _real_metadata(value, *, name: str, fallback: torch.Tensor) -> torch.Tensor:
        if value is None:
            return fallback
        tensor = torch.as_tensor(value, dtype=torch.long, device=device).reshape(-1)
        if tensor.numel() != total_real:
            raise ValueError(
                f"PackedBatch extras[{name!r}] has {tensor.numel()} rows, expected {total_real}"
            )
        return tensor

    fallback_samples = torch.repeat_interleave(
        torch.arange(batch.seq_lens.numel(), device=device, dtype=torch.long),
        batch.seq_lens.to(device=device, dtype=torch.long),
    )
    fallback_positions = torch.cat(
        [
            torch.arange(int(length), device=device, dtype=torch.long)
            for length in batch.seq_lens.detach().cpu().tolist()
        ]
    )
    fallback_global = torch.arange(total_real, device=device, dtype=torch.long)
    real_sample_ids = _real_metadata(extra_sample_ids, name="sample_ids", fallback=fallback_samples)
    real_positions = _real_metadata(
        extra_positions, name="original_position_ids", fallback=fallback_positions
    )
    real_global_ids = _real_metadata(
        extra_global_ids, name="global_token_ids", fallback=fallback_global
    )
    if real_sample_ids.numel() and (
        int(real_sample_ids.min().item()) < 0
        or int(real_positions.min().item()) < 0
        or int(real_global_ids.min().item()) < 0
    ):
        raise ValueError("real MoR token identity metadata must be non-negative")
    if torch.unique(real_global_ids).numel() != real_global_ids.numel():
        raise ValueError("real global_token_ids must be unique inside a DP replica")

    router_logit_bias = None
    if bool(extras.get("apply_routing_bias", False)):
        raw_bias = extras.get("routing_bias")
        if raw_bias is None:
            raise ValueError("apply_routing_bias requires PackedBatch extras['routing_bias']")
        router_logit_bias = torch.as_tensor(raw_bias, dtype=torch.float32, device=device).reshape(
            -1
        )
        if router_logit_bias.numel() != total_real:
            raise ValueError(
                "PackedBatch routing_bias must contain one value per real token: "
                f"{router_logit_bias.numel()} != {total_real}"
            )
        if not torch.isfinite(router_logit_bias).all():
            raise ValueError("PackedBatch routing_bias values must be finite")

    global_offset = 0
    for sample_index, length_tensor in enumerate(batch.seq_lens):
        length = int(length_tensor.item())
        start = int(meta.cu_seqlens_padded[sample_index].item())
        stop = start + length
        source_slice = slice(global_offset, global_offset + length)
        sample_ids[start:stop] = real_sample_ids[source_slice]
        positions[start:stop] = real_positions[source_slice]
        global_ids[start:stop] = real_global_ids[source_slice]
        padding_mask[start:stop] = False
        global_offset += length

    original_lengths = original_sample_lengths(batch.seq_lens, real_sample_ids)

    packed_seq_params = kwargs["packed_seq_params"]
    if _model_attention_backend(model) == "magi":
        from megatron.lite.primitive.modules.attention.magi import (
            dispatch_magi_attention_tensor,
        )

        runtime_key = getattr(packed_seq_params, "magi_runtime_key", None)
        if runtime_key is None:
            raise ValueError("Magi MoR metadata requires the microbatch runtime key")

        def distribute(tensor: torch.Tensor, pad_value: int | bool) -> torch.Tensor:
            return dispatch_magi_attention_tensor(
                tensor, runtime_key, pad_value=float(pad_value)
            ).unsqueeze(0)

    elif cp_size > 1:

        def distribute(tensor: torch.Tensor, pad_value: int | bool) -> torch.Tensor:
            del pad_value
            return split_packed_to_cp_local(
                tensor,
                cu_seqlens_padded=meta.cu_seqlens_padded,
                cp_size=cp_size,
                cp_rank=cp_rank,
                dim=0,
            ).unsqueeze(0)

    else:

        def distribute(tensor: torch.Tensor, pad_value: int | bool) -> torch.Tensor:
            del pad_value
            return tensor.unsqueeze(0)

    def distribute_router(tensor: torch.Tensor, pad_value: int | bool) -> torch.Tensor:
        cp_local = distribute(tensor, pad_value).reshape(-1)
        if cp_local.numel() % tp_size:
            raise ValueError(
                f"CP-local metadata has {cp_local.numel()} tokens, not divisible by TP={tp_size}"
            )
        local_count = cp_local.numel() // tp_size
        start = tp_rank * local_count
        return cp_local.narrow(0, start, local_count).unsqueeze(0)

    result = {
        "mor_sample_ids": distribute_router(sample_ids, -1),
        "mor_original_positions": distribute_router(positions, -1),
        "mor_global_token_ids": distribute_router(global_ids, -1),
        "mor_padding_mask": distribute_router(padding_mask, True).bool(),
        "mor_original_lengths": original_lengths,
        "mor_replay_plans": batch.extras.get("mor_replay_plans"),
        # Cross-topology parity replays native Qwen MoE expert identities by
        # logical layer and stable global token ID.  The model still computes
        # live gate scores (and gradients); normal learned training omits this
        # mapping and retains the native Top-K path.
        "mor_expert_replay_plans": batch.extras.get("mor_expert_replay_plans"),
        # Synthetic parity can inject a deterministic logit offset to force a
        # highly skewed expert-choice selection.  Keep the source mapping
        # replicated inside one dense-DP replica; recurrent layouts resolve it
        # by stable global token ID after every transition.
        "mor_router_bias_global_token_ids": (
            real_global_ids if router_logit_bias is not None else None
        ),
        "mor_router_logit_bias": router_logit_bias,
        # Full-vocabulary logits are expensive and are therefore opt-in.  The
        # parity harness enables this flag to validate the requested end-logit
        # contract even when labels are present; ordinary training leaves it
        # disabled and keeps the native vocab-parallel loss path unchanged.
        "return_full_logits": bool(batch.extras.get("mor_return_full_logits", False)),
        # Override native regular-THD positions so the out-of-tree GQA adapter
        # receives the exact same token permutation as the identity metadata.
        "position_ids": distribute(positions.clamp_min(0), 0),
    }
    if kwargs.get("loss_mask") is None:
        # MLite permits PackedBatch.loss_mask=None, but a plain all-ones fallback
        # inside the model would include topology-dependent THD/Magi dummy tails.
        # Synthesize the physical-layout mask here, where the exact pack and
        # dispatch permutation is known.
        result["loss_mask"] = (~distribute(padding_mask, True).bool()).float()
    return result


def _forward_step(model: nn.Module, batch: PackedBatch) -> dict:
    packing_model = _unwrapped_model(model)
    if _model_attention_backend(packing_model) == "magi":
        kwargs = pack_magi_forward_kwargs(packing_model, batch)
    else:
        kwargs = pack_thd_forward_kwargs(packing_model, batch)
    kwargs.update(_padded_mor_metadata(packing_model, batch, kwargs))
    add_loss_context_kwargs(kwargs, include_return_log_probs=True)
    add_cross_entropy_fusion(kwargs, model)
    return model(**kwargs)


def unpack_forward_output(model: nn.Module, batch: PackedBatch, output) -> Any:
    packing_model = _unwrapped_model(model)
    if _model_attention_backend(packing_model) == "magi":
        return unpack_magi_forward_output(packing_model, batch, output)
    return unpack_thd_forward_output(packing_model, batch, output)


def _native_impl_config(impl_config: ImplConfig) -> _NativeImplConfig:
    kwargs = {
        field.name: getattr(impl_config, field.name)
        for field in fields(_NativeImplConfig)
        if field.init
    }
    # Native construction applies recompute/offload/LoRA/QAT, but optimizer
    # creation must wait until independent depth routers have been registered.
    kwargs["optimizer"] = None
    kwargs["mtp_enable"] = False
    kwargs["mtp_enable_train"] = False
    return _NativeImplConfig(**kwargs)


def _validate_scope(impl_config: ImplConfig) -> None:
    parallel = impl_config.parallel
    if impl_config.local_attention_backend not in {"te", "magi_ffa"}:
        raise ValueError("local_attention_backend must be te or magi_ffa")
    if impl_config.local_attention_backend == "magi_ffa" and parallel.cp != 1:
        raise ValueError("local magi_ffa requires CP=1; use distributed Magi for CP>1")
    if parallel.pp != 1 or parallel.vpp != 1:
        raise ValueError("Qwen3-MoE MoR v1 requires PP=VPP=1")
    if parallel.etp not in (None, 1):
        raise ValueError("Qwen3-MoE MoR v1 requires ETP=1")
    if isinstance(impl_config.dense_dp_size, bool) or impl_config.dense_dp_size < 1:
        raise ValueError("dense_dp_size must be a positive integer")
    if impl_config.mtp_enable or impl_config.mtp_enable_train:
        raise ValueError("Qwen3-MoE MoR v1 does not support MTP")
    if impl_config.recompute:
        raise ValueError("Qwen3-MoE MoR v1 does not support activation recomputation")
    if impl_config.offload:
        raise ValueError("Qwen3-MoE MoR v1 does not support activation offload")
    if impl_config.use_deepep:
        raise ValueError("Qwen3-MoE MoR v1 does not support DeepEP")
    if impl_config.lora is not None:
        raise ValueError("Qwen3-MoE MoR v1 does not support LoRA")
    if impl_config.qat is not None:
        raise ValueError("Qwen3-MoE MoR v1 does not support QAT")
    if not impl_config.use_thd:
        raise ValueError("Qwen3-MoE MoR requires use_thd=True")
    if impl_config.optimizer not in {None, "dist_opt"}:
        raise ValueError("Qwen3-MoE MoR supports optimizer=None or dist_opt")
    if impl_config.cp_transition not in {
        "magi_direct",
        "magi_canonical",
        "static_reference",
    }:
        raise ValueError("cp_transition must be magi_direct, magi_canonical, or static_reference")
    if impl_config.route_mode not in {"learned", "replay"}:
        raise ValueError("route_mode must be learned or replay")
    if parallel.cp > 1:
        if impl_config.attention_backend_override != "magi":
            raise ValueError("Qwen3-MoE MoR CP>1 requires the Magi attention backend")
        if impl_config.cp_transition == "static_reference":
            raise ValueError(
                "static_reference CP is a tiny diagnostic backend, not a Qwen training path"
            )
    if impl_config.cp_transition == "magi_canonical" and parallel.tp > 1:
        raise ValueError("magi_canonical v1 is the CP-only oracle; use magi_direct for TP+CP")


def _resolve_model_metadata(
    model_cfg: Qwen3MoEConfig, impl_cfg: ImplConfig
) -> tuple[MoRArchitectureConfig, str]:
    sidecar_architecture = getattr(model_cfg, "_mor_architecture", None)
    architecture = (
        MoRArchitectureConfig.from_dict(sidecar_architecture)
        if sidecar_architecture is not None
        else impl_cfg.mor_architecture()
    )
    folding_policy = str(getattr(model_cfg, "_mor_folding_policy", impl_cfg.hf_folding_policy))
    return architecture, folding_policy


def build_model(model_cfg: Qwen3MoEConfig, *, impl_cfg: ImplConfig) -> ModelBundle:
    """Build a physical native Qwen stack, attach recurrence, then the optimizer."""

    _validate_scope(impl_cfg)
    architecture, folding_policy = _resolve_model_metadata(model_cfg, impl_cfg)
    install_mor_metadata(model_cfg, architecture, folding_policy)
    physical_cfg = physical_model_config(model_cfg, architecture)
    bundle = _native_protocol.build_model(physical_cfg, impl_cfg=_native_impl_config(impl_cfg))
    if impl_cfg.local_attention_backend == "magi_ffa":
        from .local_attention import LocalMagiAttention

        for chunk in bundle.chunks:
            for layer in chunk.layers:
                if any(True for _ in layer.attn.core_attn.parameters()):
                    raise RuntimeError("local FFA adapter cannot replace a parameterized core")
                layer.attn.core_attn = LocalMagiAttention(
                    cp_size=impl_cfg.parallel.cp,
                    deterministic=impl_cfg.deterministic,
                )
    for index, chunk in enumerate(bundle.chunks):
        bundle.chunks[index] = Qwen3MoEMoRModel.adapt_native(
            chunk,
            logical_config=model_cfg,
            architecture=architecture,
            router_config=impl_cfg.depth_router_config(),
            router_seed=impl_cfg.depth_router_seed,
            initializer_range=float(getattr(model_cfg, "_mor_initializer_range", 0.02)),
            cp_transition=impl_cfg.cp_transition,
            route_mode=impl_cfg.route_mode,
            route_peer_consensus=impl_cfg.route_peer_consensus,
        )

    if impl_cfg.optimizer == "dist_opt":
        from megatron.lite.primitive.optimizers.megatron_wrap import (
            build_dist_opt_training_optimizer,
        )

        bundle.optimizer, bundle.finalize_grads = build_dist_opt_training_optimizer(
            bundle.chunks,
            model_cfg=physical_cfg,
            impl_cfg=impl_cfg,
            ps=bundle.parallel_state,
            model_name="qwen3_moe_mor",
            is_expert=EXPERT_CLASSIFIER,
            deterministic=impl_cfg.deterministic,
        )

    # Keep model serialization identical for conversion/forward-only handles
    # (optimizer=None) and ZeRO-1 training handles.  The distributed optimizer
    # replaces ``bundle.chunks`` in place with MCore DDP wrappers, so this hook
    # must be attached *after* that replacement and to the final chunk objects.
    attach_model_sharded_state_dict(
        bundle.chunks,
        bundle.parallel_state,
        get_placements=PLACEMENT_FN,
        is_expert=EXPERT_CLASSIFIER,
    )

    bundle.forward_step = _forward_step
    bundle.extras.update(
        {
            "model_cfg": model_cfg,
            "physical_model_cfg": physical_cfg,
            "mor_checkpoint_metadata": MoRCheckpointMetadata(
                architecture=architecture,
                folding_policy=folding_policy,
                depth_router=impl_cfg.depth_router_config(),
                depth_router_seed=impl_cfg.depth_router_seed,
                hf_source=str(getattr(model_cfg, "_mor_hf_source", "unknown")),
                cp_transition=impl_cfg.cp_transition,
                parallel=MoRParallelConfig(
                    dp=impl_cfg.dense_dp_size,
                    tp=impl_cfg.parallel.tp,
                    cp=impl_cfg.parallel.cp,
                    ep=impl_cfg.parallel.ep,
                    etp=impl_cfg.parallel.etp or 1,
                    zero_stage=1,
                    cp_transition=impl_cfg.cp_transition,
                ),
            ).to_dict(),
            "optimizer_backend": "dist_opt" if bundle.optimizer is not None else "none",
        }
    )
    return bundle


def load_hf_weights(chunk: nn.Module, hf_path: str, model_cfg: Qwen3MoEConfig, ps) -> None:
    if hf_path:
        _load_hf_weights(chunk, hf_path, model_cfg, ps)


def vocab_size(model_cfg: Qwen3MoEConfig) -> int | None:
    return getattr(model_cfg, "vocab_size", None)


__all__ = [
    "EXPERT_CLASSIFIER",
    "PLACEMENT_FN",
    "ImplConfig",
    "build_model",
    "build_model_config",
    "load_hf_weights",
    "router_replay_roots",
    "unpack_forward_output",
    "vocab_size",
]
