"""Explicit opt-in MLite protocol for the from-scratch four-arm experiment."""

from dataclasses import dataclass

import torch
from torch import nn

from mor_mlite.pretraining.config import parameter_seed
from mor_mlite.pretraining.execution import run_causal, run_fixed
from mor_mlite.qwen3_moe_mor import protocol as mor
from mor_mlite.qwen3_moe_mor.model import Qwen3MoEMoRModel

EXPERT_CLASSIFIER = mor.EXPERT_CLASSIFIER
PLACEMENT_FN = mor.PLACEMENT_FN
build_model_config = mor.build_model_config
vocab_size = mor.vocab_size
unpack_forward_output = mor.unpack_forward_output
router_replay_roots = mor.router_replay_roots


@dataclass(frozen=True)
class ImplConfig(mor.ImplConfig):
    experiment_arm: str = "D"
    initialization_seed: int = 1234
    pretraining_moe_aux: float = 0.001


class ExperimentModel(Qwen3MoEMoRModel):
    def _run_mor_backbone(self, h, **kwargs):
        if self.experiment_arm == "D" and self.training:
            return super()._run_mor_backbone(h, **kwargs)
        if kwargs.get("mor_replay_plans") or kwargs.get("mor_expert_replay_plans"):
            raise ValueError("from-scratch experiment never consumes replay routes")
        if self.experiment_arm == "D":
            from megatron.lite.primitive.utils.packed_seq import PackedSeqParams

            def make_packed(counts):
                cu = torch.cat((counts.new_zeros(1), counts.cumsum(0))).to(torch.int32)
                return PackedSeqParams.from_cu_seqlens(cu, max_seqlen=int(counts.max()))

            h, self.causal_route_traces = run_causal(
                h,
                layers=self.layers,
                routers=self.depth_routers,
                sample_ids=kwargs["mor_sample_ids"].reshape(-1),
                positions=kwargs["mor_original_positions"].reshape(-1),
                padding_mask=kwargs["mor_padding_mask"].reshape(-1),
                position_ids=kwargs["position_ids"],
                packed_seq_params=kwargs["packed_seq_params"],
                make_packed=make_packed,
                start=self.mor_architecture.n_start_layers,
                recurrent=self.mor_architecture.n_recurrent_layers,
                end=self.mor_architecture.n_end_layers,
            )
        else:
            a = self.mor_architecture
            order = (
                tuple(range(a.n_start_layers))
                + tuple(range(a.n_start_layers, a.n_start_layers + a.n_recurrent_layers))
                * a.num_recursions
                + tuple(range(a.n_start_layers + a.n_recurrent_layers, a.physical_num_layers))
            )
            h = run_fixed(
                h,
                self.layers,
                order,
                position_ids=kwargs["position_ids"],
                packed_seq_params=kwargs["packed_seq_params"],
            )
        return h, [], h.new_zeros((), dtype=torch.float32), h.new_empty((0,), dtype=torch.float32)


def initialize_parameters(model, seed):
    # Only TP1/ETP1 is supported, so only expert IDs need shard canonicalization.
    from mor_mlite.parity.mlite import _canonical_parameter_name

    with torch.no_grad():
        for name, parameter in model.named_parameters():
            canonical, _ = _canonical_parameter_name(
                name,
                ep_rank=model.ps.ep_rank,
                ep_size=model.ps.ep_size,
                num_experts=model.config.num_experts,
            )
            if parameter.ndim == 1:
                parameter.fill_(0 if name.endswith("bias") else 1)
            else:
                generator = torch.Generator(device=parameter.device)
                generator.manual_seed(parameter_seed(seed, canonical))
                parameter.normal_(mean=0, std=0.02, generator=generator)


def build_model(model_cfg, *, impl_cfg):
    p = impl_cfg.parallel
    if (p.tp, p.cp, p.pp, p.etp or 1, p.vpp) != (1, 1, 1, 1, 1):
        raise ValueError("pretraining protocol requires TP=CP=PP=ETP=VPP=1")
    if impl_cfg.experiment_arm not in ("A", "B", "C", "D"):
        raise ValueError("unknown experiment arm")
    # This is architecture construction, not loading/folding a 48-layer
    # checkpoint. C's logical configuration must really describe 20 layers.
    import copy

    model_cfg = copy.deepcopy(model_cfg)
    model_cfg.num_hidden_layers = impl_cfg.mor_architecture().logical_num_layers
    model_cfg.layer_types = ["full_attention"] * model_cfg.num_hidden_layers
    model_cfg._validate()
    # Construct without optimizer; parameter registration and initialization
    # must precede DDP buffers and FP32 master-weight creation.
    from dataclasses import replace

    bundle = mor.build_model(model_cfg, impl_cfg=replace(impl_cfg, optimizer=None))
    for chunk in bundle.chunks:
        chunk.__class__ = ExperimentModel
        chunk.experiment_arm = impl_cfg.experiment_arm
        chunk.causal_route_traces = []
        if impl_cfg.experiment_arm != "D":
            removed = {id(p) for p in chunk.depth_routers.parameters()}
            chunk.sp_params = [p for p in chunk.sp_params if id(p) not in removed]
            chunk.depth_routers = nn.ModuleList()
        for layer in chunk.layers:
            layer.moe.router.compute_aux_loss = True
            layer.moe.router.aux_loss_coeff = impl_cfg.pretraining_moe_aux
        initialize_parameters(chunk, impl_cfg.initialization_seed)
    if impl_cfg.optimizer == "dist_opt":
        from megatron.lite.primitive.optimizers.megatron_wrap import (
            build_dist_opt_training_optimizer,
        )

        bundle.optimizer, bundle.finalize_grads = build_dist_opt_training_optimizer(
            bundle.chunks,
            model_cfg=bundle.extras["physical_model_cfg"],
            impl_cfg=impl_cfg,
            ps=bundle.parallel_state,
            model_name="qwen3_moe_mor",
            is_expert=EXPERT_CLASSIFIER,
            deterministic=impl_cfg.deterministic,
        )
        from megatron.lite.primitive.ckpt import attach_model_sharded_state_dict

        attach_model_sharded_state_dict(
            bundle.chunks,
            bundle.parallel_state,
            get_placements=PLACEMENT_FN,
            is_expert=EXPERT_CLASSIFIER,
        )
        bundle.extras["optimizer_backend"] = "dist_opt"
    return bundle


def load_hf_weights(*args, **kwargs):
    raise ValueError("pretraining experiment forbids pretrained weight loading")
