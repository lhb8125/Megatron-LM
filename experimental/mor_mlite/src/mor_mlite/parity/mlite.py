"""Parity runner backed by Megatron-Lite's public Runtime API.

Megatron-Lite is deliberately imported only when a runtime session is built.
This keeps ``mor_mlite.parity`` and its command-line parser importable on a
login node that does not have the pinned Megatron checkout or CUDA libraries.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from mor_mlite.checkpoint_io import (
    build_checkpoint_metadata,
    load_mor_checkpoint,
    read_mor_sidecar,
    run_rank_zero_io,
    save_mor_checkpoint,
)
from mor_mlite.config import DepthRouterConfig, MoRArchitectureConfig, MoRParallelConfig
from mor_mlite.config_loader import MoRPresetConfig, load_json, load_preset_config
from mor_mlite.hf import resolve_hf_checkpoint
from mor_mlite.parity.topologies import Topology, get_topology
from mor_mlite.qwen3_moe_mor.metadata import (
    physical_to_logical_layer_map,
)
from mor_mlite.versions import MAGI_VERSION, MEGATRON_SHA

ATTENTION_POLICY = "native-bf16-local-ffa-strict-cp1-magi-cp-v1"


def _attention_backend(topology: Topology, *, strict: bool) -> str:
    # Strictness controls determinism, not an extra BF16 rounding of QK scores.
    # TE unfused materializes those scores in BF16 before FP32 softmax; it is
    # not a higher-accuracy oracle for the native fused/Magi computation.
    # Local FFA uses Magi's public no-CP functional API with deterministic=True;
    # "local" keeps TE auto-selection off for the explicitly installed core.
    return "magi" if topology.cp > 1 else ("local" if strict else "flash")


@dataclass(frozen=True, slots=True)
class MLiteRuntimeBuildConfig:
    """Inputs needed to construct one public MLite runtime session."""

    hf_path: str
    topology: Topology
    architecture: MoRArchitectureConfig
    depth_router: DepthRouterConfig = field(default_factory=DepthRouterConfig)
    load_hf_weights: bool = True
    build_optimizer: bool = True
    seed: int = 1234
    lr: float = 1e-3
    adam_eps: float = 1e-6
    clip_grad: float = 1.0
    total_training_steps: int = 1
    cp_transition: str = "magi_direct"
    route_mode: str = "learned"
    folding_policy: str = "mean"
    strict: bool = True
    cross_entropy_fusion: bool = False

    def validate(self) -> None:
        if not self.hf_path:
            raise ValueError("MLite model construction requires a non-empty hf_path")
        self.topology.validate()
        if self.topology.etp != 1:
            raise ValueError("Qwen3-MoE MoR v1 requires ETP=1")
        if self.route_mode not in {"learned", "replay"}:
            raise ValueError("route_mode must be learned or replay")
        if self.cp_transition not in {
            "magi_direct",
            "magi_canonical",
            "static_reference",
        }:
            raise ValueError("unsupported CP transition backend")
        if self.topology.cp > 1 and self.cp_transition == "static_reference":
            raise ValueError("static_reference is not a distributed Qwen CP backend")
        if self.topology.tp > 1 and self.cp_transition == "magi_canonical":
            raise ValueError("magi_canonical is defined only for TP=1")
        if (
            not math.isfinite(self.lr)
            or not math.isfinite(self.adam_eps)
            or not math.isfinite(self.clip_grad)
            or self.lr <= 0.0
            or self.adam_eps <= 0.0
            or self.clip_grad <= 0.0
        ):
            raise ValueError("lr, adam_eps, and clip_grad must be finite and positive")
        if self.total_training_steps < 1:
            raise ValueError("total_training_steps must be positive")


@dataclass(slots=True)
class MLiteRuntimeSession:
    """The opaque model handle and the Runtime that owns it."""

    runtime: Any
    handle: Any
    runtime_config: Any
    backend_config: Any


@dataclass(slots=True)
class MLiteRunConfig:
    """Configuration consumed by :func:`run_mlite`."""

    output: Path
    preset: str = "tiny"
    topology: str = "baseline"
    precision: str = "bf16"
    seed: int = 1234
    steps: int = 1
    num_microbatches: int = 2
    route_mode: str = "learned"
    replay_from: Path | None = None
    checkpoint_roundtrip: bool = True
    checkpoint_save_only: bool = False
    strict: bool = True
    lr: float = 1e-3
    adam_eps: float = 1e-6
    clip_grad: float = 1.0
    seq_lens: tuple[int, ...] = (9, 6, 3)
    hf_path: str = ""
    init_checkpoint: Path | None = None
    resume_checkpoint: Path | None = None
    cp_transition: str | None = None
    forward_only: bool = False
    reference_dp_shards: int = 1
    preset_config: Path | None = None
    architecture: MoRArchitectureConfig | None = None
    depth_router: DepthRouterConfig | None = None

    def validate(self) -> Topology:
        if (
            not math.isfinite(self.lr)
            or not math.isfinite(self.adam_eps)
            or not math.isfinite(self.clip_grad)
            or self.lr <= 0.0
            or self.adam_eps <= 0.0
            or self.clip_grad <= 0.0
        ):
            raise ValueError("lr, adam_eps, and clip_grad must be finite and positive")
        if self.preset not in {"tiny", "qwen3-30b"}:
            raise ValueError("preset must be tiny or qwen3-30b")
        if self.preset == "qwen3-30b" and self.checkpoint_roundtrip:
            raise ValueError(
                "qwen3-30b checkpoint validation must use separate "
                "--checkpoint-save-only and --resume-checkpoint torchrun processes"
            )
        # The pinned native Qwen protocol materializes model chunks in BF16.
        if self.precision != "bf16":
            raise ValueError("the pinned MLite Qwen implementation supports BF16 parity only")
        if self.steps < 1 or self.num_microbatches < 1:
            raise ValueError("steps and num_microbatches must be positive")
        if (
            isinstance(self.reference_dp_shards, bool)
            or not isinstance(self.reference_dp_shards, int)
            or self.reference_dp_shards < 1
        ):
            raise ValueError("reference_dp_shards must be a positive integer")
        if not self.seq_lens or any(length <= 0 for length in self.seq_lens):
            raise ValueError("seq_lens must contain positive lengths")
        if self.route_mode not in {"learned", "replay"}:
            raise ValueError("route_mode must be learned or replay")
        if self.route_mode == "replay" and self.replay_from is None:
            raise ValueError("replay mode requires replay_from")
        initialization_sources = sum(
            (
                bool(self.hf_path),
                self.init_checkpoint is not None,
                self.resume_checkpoint is not None,
            )
        )
        if initialization_sources > 1:
            raise ValueError(
                "--hf-path, --init-checkpoint, and --resume-checkpoint are mutually exclusive"
            )
        if self.checkpoint_save_only and self.checkpoint_roundtrip:
            raise ValueError("checkpoint_save_only and checkpoint_roundtrip are mutually exclusive")
        if self.resume_checkpoint is not None and self.checkpoint_roundtrip:
            raise ValueError("external checkpoint resume cannot start another in-process roundtrip")
        if self.resume_checkpoint is not None and self.checkpoint_save_only:
            raise ValueError("checkpoint save-only and external resume are mutually exclusive")
        if self.resume_checkpoint is not None and self.steps != 1:
            raise ValueError(
                "external checkpoint certification executes exactly one resumed optimizer step"
            )
        if (self.checkpoint_save_only or self.resume_checkpoint is not None) and self.forward_only:
            raise ValueError("process-isolated checkpoint validation requires a training run")
        preset = load_preset_config(self.preset, self.preset_config)
        topology = get_topology(self.topology)
        if self.reference_dp_shards > 1:
            if topology.name != "baseline" or topology.world_size != 1:
                raise ValueError("reference_dp_shards is valid only for the one-rank baseline")
            if not self.forward_only or self.route_mode != "learned":
                raise ValueError("reference_dp_shards requires a learned, forward-only baseline")
            if self.reference_dp_shards > len(self.seq_lens):
                raise ValueError("reference_dp_shards cannot exceed the number of global sequences")
        if self.route_mode == "replay":
            assert self.replay_from is not None
            manifest = load_json(Path(self.replay_from) / "manifest.json", expected_type=dict)
            artifact_shards = manifest.get("reference_dp_shards", 1)
            if (
                isinstance(artifact_shards, bool)
                or not isinstance(artifact_shards, int)
                or artifact_shards < 1
            ):
                raise ValueError("replay artifact has invalid reference_dp_shards metadata")
            artifact_batch = manifest.get("global_batch")
            if not isinstance(artifact_batch, Mapping):
                raise ValueError("replay artifact is missing global_batch metadata")
            if artifact_batch.get("sequence_lengths") != list(self.seq_lens):
                raise ValueError("replay artifact global batch sequence lengths differ")
            if manifest.get("preset") not in {None, self.preset}:
                raise ValueError("replay artifact preset differs from the requested preset")
            if manifest.get("precision") not in {None, self.precision}:
                raise ValueError("replay artifact precision differs from the requested precision")
            if artifact_shards > 1:
                expected_partitions = [
                    list(indices)
                    for indices in _balanced_sample_partitions(self.seq_lens, topology.dp)
                ]
                if artifact_shards != topology.dp:
                    raise ValueError(
                        "replay artifact reference_dp_shards must equal target dense-DP: "
                        f"{artifact_shards} != {topology.dp}"
                    )
                if artifact_batch.get("partition_policy") != (
                    "deterministic-longest-first-whole-sequence"
                ):
                    raise ValueError("serial replay artifact has an incompatible partition policy")
                if artifact_batch.get("sample_partitions") != expected_partitions:
                    raise ValueError("serial replay artifact sample partitions differ")
        if (
            self.preset == "qwen3-30b"
            and not self.hf_path
            and self.init_checkpoint is None
            and self.resume_checkpoint is None
            and preset.hf_source is None
        ):
            raise ValueError("qwen3-30b requires --hf-path, --init-checkpoint, or preset hf_source")
        checkpoint_source = self.resume_checkpoint or self.init_checkpoint
        if checkpoint_source is None:
            num_experts = preset.num_experts
        else:
            checkpoint = Path(checkpoint_source)
            metadata = read_mor_sidecar(checkpoint)
            if self.architecture is not None and metadata.architecture != self.architecture:
                raise ValueError(
                    "--init-checkpoint architecture conflicts with the resolved preset/CLI "
                    f"configuration: checkpoint={metadata.architecture.to_dict()}, "
                    f"requested={self.architecture.to_dict()}"
                )
            if self.depth_router is not None and metadata.depth_router != self.depth_router:
                raise ValueError(
                    "--init-checkpoint depth-router configuration conflicts with the resolved "
                    f"preset/CLI configuration: checkpoint={metadata.depth_router.to_dict()}, "
                    f"requested={self.depth_router.to_dict()}"
                )
            if metadata.parallel.ep != topology.ep:
                raise ValueError(
                    "MoR DCP loading cannot reshard local expert parameter "
                    f"keys across EP sizes: checkpoint EP={metadata.parallel.ep}, "
                    f"runtime EP={topology.ep}; convert the same HF source once per EP topology"
                )
            config_path = checkpoint / "config.json"
            if not config_path.is_file():
                raise FileNotFoundError(
                    f"MoR init checkpoint is missing its base config snapshot: {config_path}"
                )
            raw_config = load_json(config_path, expected_type=dict)
            assert isinstance(raw_config, dict)
            preset.validate_base_hf_config(raw_config, architecture=metadata.architecture)
            raw_num_experts = raw_config.get("num_experts")
            if raw_num_experts is None:
                raise ValueError(f"checkpoint config.json has no num_experts: {config_path}")
            num_experts = int(raw_num_experts)
            metadata.parallel.validate_world_size(
                metadata.parallel.expected_world_size,
                num_experts=num_experts,
            )
        topology.validate(num_experts=num_experts)
        return topology


def _optimizer_contract(config: MLiteRunConfig) -> dict[str, Any]:
    return {
        "name": "adam",
        "lr": config.lr,
        "adam_eps": config.adam_eps,
        "clip_grad": config.clip_grad,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _external_run_contract(config: MLiteRunConfig, *, cp_transition: str) -> dict[str, Any]:
    replay_routes_sha256: str | None = None
    if config.route_mode == "replay":
        assert config.replay_from is not None
        routes_path = config.replay_from.resolve() / "routes.json"
        if not routes_path.is_file():
            raise FileNotFoundError(f"replay RoutePlan file is missing: {routes_path}")
        replay_routes_sha256 = _file_sha256(routes_path)
    return {
        "preset": config.preset,
        "precision": config.precision,
        "strict": config.strict,
        "attention_policy": ATTENTION_POLICY,
        "seed": config.seed,
        "seq_lens": list(config.seq_lens),
        "num_microbatches": config.num_microbatches,
        "route_mode": config.route_mode,
        "replay_routes_sha256": replay_routes_sha256,
        "cp_transition": cp_transition,
    }


@dataclass(frozen=True, slots=True)
class MLiteCheckpointInitialization:
    """Cold-start or full-resume contract for a self-describing MoR DCP."""

    checkpoint: Path
    metadata: Any
    runtime_metadata: Any
    cp_transition: str
    full_training_state: bool


def _resolve_checkpoint_initialization(
    config: MLiteRunConfig,
    topology: Topology,
) -> MLiteCheckpointInitialization | None:
    """Resolve model-only initialization or full resume without MLite imports.

    Source parallelism is provenance, not a restriction: MLite DCP may reshard
    model tensors into the current topology.  Model semantics are copied into
    the runtime contract while the current launch owns the execution backend.
    """

    checkpoint_source = config.resume_checkpoint or config.init_checkpoint
    if checkpoint_source is None:
        return None
    checkpoint = checkpoint_source.resolve()
    metadata = read_mor_sidecar(checkpoint)
    if metadata.parallel.ep != topology.ep:
        raise ValueError(
            "MoR DCP loading cannot reshard local expert parameter keys "
            f"across EP sizes: checkpoint EP={metadata.parallel.ep}, runtime EP={topology.ep}"
        )
    resolve_hf_checkpoint(str(checkpoint), require_weights=False)
    cp_transition = config.cp_transition or metadata.cp_transition
    runtime_metadata = build_checkpoint_metadata(
        architecture=metadata.architecture,
        depth_router=metadata.depth_router,
        depth_router_seed=metadata.depth_router_seed,
        hf_source=metadata.hf_source,
        parallel=topology.to_parallel_config(cp_transition=cp_transition),
        folding_policy=metadata.folding_policy,
        cp_transition=cp_transition,
    )
    return MLiteCheckpointInitialization(
        checkpoint=checkpoint,
        metadata=metadata,
        runtime_metadata=runtime_metadata,
        cp_transition=cp_transition,
        full_training_state=config.resume_checkpoint is not None,
    )


def _load_runtime_api() -> SimpleNamespace:
    """Resolve only documented MLite runtime/config symbols, lazily."""

    try:
        from megatron.lite.runtime import RuntimeConfig, create_runtime
        from megatron.lite.runtime.contracts import (
            MegatronLiteConfig,
            OptimizerConfig,
            ParallelConfig,
        )
    except (ImportError, OSError) as exc:  # OSError covers missing CUDA/TE DSOs.
        raise RuntimeError(
            "Megatron-Lite is unavailable. Add the pinned Megatron-LM "
            "experimental/lite directory to PYTHONPATH and use its CUDA environment."
        ) from exc
    return SimpleNamespace(
        RuntimeConfig=RuntimeConfig,
        create_runtime=create_runtime,
        MegatronLiteConfig=MegatronLiteConfig,
        OptimizerConfig=OptimizerConfig,
        ParallelConfig=ParallelConfig,
    )


def _actual_world_size() -> int:
    raw = os.environ.get("WORLD_SIZE", "1")
    try:
        world_size = int(raw)
    except ValueError as exc:
        raise ValueError(f"WORLD_SIZE must be an integer, got {raw!r}") from exc
    if world_size < 1:
        raise ValueError("WORLD_SIZE must be positive")
    return world_size


def _validate_single_node_launch(world_size: int) -> None:
    """Reject torchrun/Slurm launches that span more than one physical node."""

    local_raw = os.environ.get("LOCAL_WORLD_SIZE")
    if local_raw is not None:
        try:
            local_world_size = int(local_raw)
        except ValueError as exc:
            raise ValueError(f"LOCAL_WORLD_SIZE must be an integer, got {local_raw!r}") from exc
        if local_world_size != world_size:
            raise ValueError(
                "MoR-MLite v1 is single-node only: "
                f"LOCAL_WORLD_SIZE={local_world_size} != WORLD_SIZE={world_size}"
            )
    for name in ("SLURM_NNODES", "SLURM_JOB_NUM_NODES"):
        raw = os.environ.get(name)
        if raw is None:
            continue
        try:
            node_count = int(raw)
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
        if node_count != 1:
            raise ValueError(f"MoR-MLite v1 is single-node only: {name}={node_count}")


def _ensure_single_rank_dist_environment(world_size: int) -> None:
    """Make the advertised direct one-GPU CLI valid for MLite's env:// init."""

    if world_size != 1:
        return
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")


def build_runtime_session(config: MLiteRuntimeBuildConfig) -> MLiteRuntimeSession:
    """Register, configure, and build the external model via public MLite APIs."""

    config.validate()
    actual_world_size = _actual_world_size()
    _validate_single_node_launch(actual_world_size)
    parallel_contract = config.topology.to_parallel_config(cp_transition=config.cp_transition)
    parallel_contract.validate_world_size(actual_world_size)
    _ensure_single_rank_dist_environment(actual_world_size)

    # The pinned config loader understands Hub IDs, but SafeTensorReader does
    # not. Resolve before model allocation; require only config.json when a
    # fresh DCP restore deliberately skips HF weight loading.
    resolved_hf = resolve_hf_checkpoint(config.hf_path, require_weights=config.load_hf_weights)
    hf_path = str(resolved_hf.local_path)

    from mor_mlite.register import register_with_mlite

    register_with_mlite()
    api = _load_runtime_api()
    parallel = api.ParallelConfig(
        tp=parallel_contract.tp,
        etp=parallel_contract.etp,
        ep=parallel_contract.ep,
        pp=1,
        vpp=1,
        cp=parallel_contract.cp,
    )
    optimizer = api.OptimizerConfig(
        lr=config.lr,
        adam_eps=config.adam_eps,
        clip_grad=config.clip_grad,
        total_training_steps=config.total_training_steps,
    )
    architecture = config.architecture
    router = config.depth_router
    impl_cfg = {
        "parallel": parallel,
        "optimizer": "dist_opt" if config.build_optimizer else None,
        "use_thd": True,
        "use_deepep": False,
        "cross_entropy_fusion": config.cross_entropy_fusion,
        "deterministic": config.strict,
        "local_attention_backend": "magi_ffa"
        if config.strict and config.topology.cp == 1
        else "te",
        "n_start_layers": architecture.n_start_layers,
        "n_recurrent_layers": architecture.n_recurrent_layers,
        "num_recursions": architecture.num_recursions,
        "n_end_layers": architecture.n_end_layers,
        "capacity_schedule": architecture.capacity_schedule,
        "depth_router_temperature": router.temperature,
        "depth_router_alpha": router.alpha,
        "depth_router_aux_loss_coef": router.aux_loss_coef,
        "depth_router_seed": config.seed,
        "dense_dp_size": parallel_contract.dp,
        "hf_folding_policy": config.folding_policy,
        "cp_transition": config.cp_transition,
        "route_mode": config.route_mode,
    }
    attention_backend = _attention_backend(config.topology, strict=config.strict)
    backend_config = api.MegatronLiteConfig(
        model_name="qwen3_moe_mor",
        impl="lite",
        hf_path=hf_path,
        parallel=parallel,
        optimizer=optimizer,
        attention_backend_override=attention_backend,
        load_hf_weights=config.load_hf_weights,
        impl_cfg=impl_cfg,
    )
    runtime_config = api.RuntimeConfig(
        backend="mlite",
        hf_path=hf_path,
        backend_cfg=backend_config,
    )
    runtime = api.create_runtime(runtime_config)
    handle = runtime.build_model()
    return MLiteRuntimeSession(runtime, handle, runtime_config, backend_config)


def _tiny_model_values(model: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return dict(load_preset_config("tiny").model if model is None else model)


def _tiny_hf_dict(
    architecture: MoRArchitectureConfig,
    model: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    values = _tiny_model_values(model)
    hidden = int(values["hidden_size"])
    heads = int(values["num_attention_heads"])
    return {
        "architectures": ["Qwen3MoeForCausalLM"],
        "model_type": "qwen3_moe",
        "num_hidden_layers": architecture.logical_num_layers,
        "hidden_size": hidden,
        "num_attention_heads": heads,
        "num_key_value_heads": int(values["num_key_value_heads"]),
        "head_dim": hidden // heads,
        "vocab_size": int(values["vocab_size"]),
        "num_experts": int(values["num_experts"]),
        "num_experts_per_tok": int(values["num_experts_per_tok"]),
        "moe_intermediate_size": int(values["intermediate_size"]),
        "rope_theta": float(values["rope_theta"]),
        "rms_norm_eps": float(values["rms_norm_eps"]),
        "max_position_embeddings": int(values["max_position_embeddings"]),
        "router_aux_loss_coef": 0.001,
        "num_nextn_predict_layers": 0,
        "initializer_range": float(values["initializer_range"]),
        "layer_types": ["full_attention"] * architecture.logical_num_layers,
    }


def _stable_tiny_weight(name: str, shape: tuple[int, ...], *, seed: int):
    import torch

    if name.endswith(("norm.weight", "layernorm.weight")):
        return torch.ones(shape, dtype=torch.bfloat16)
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    tensor_seed = (seed + int.from_bytes(digest[:8], "little")) % (2**63 - 1)
    generator = torch.Generator(device="cpu").manual_seed(tensor_seed)
    return (
        torch.empty(shape, dtype=torch.float32)
        .normal_(mean=0.0, std=0.02, generator=generator)
        .to(torch.bfloat16)
    )


def _tiny_hf_weights(
    architecture: MoRArchitectureConfig,
    *,
    seed: int,
    model: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic full HF state before any TP/EP partitioning."""

    values = _tiny_model_values(model)
    hidden = int(values["hidden_size"])
    heads = int(values["num_attention_heads"])
    kv_heads = int(values["num_key_value_heads"])
    head_dim = hidden // heads
    intermediate = int(values["intermediate_size"])
    experts = int(values["num_experts"])
    vocab = int(values["vocab_size"])
    layer_map = physical_to_logical_layer_map(architecture)
    physical_by_logical = {
        logical: physical
        for physical, logical_layers in layer_map.items()
        for logical in logical_layers
    }
    weights: dict[str, Any] = {}

    def add(name: str, shape: tuple[int, ...], *, canonical: str | None = None) -> None:
        weights[name] = _stable_tiny_weight(canonical or name, shape, seed=seed)

    add("model.embed_tokens.weight", (vocab, hidden))
    add("model.norm.weight", (hidden,))
    add("lm_head.weight", (vocab, hidden))
    for logical_layer in range(architecture.logical_num_layers):
        physical_layer = physical_by_logical[logical_layer]
        layer = f"model.layers.{logical_layer}"

        def layer_add(
            suffix: str,
            shape: tuple[int, ...],
            *,
            _layer: str = layer,
            _physical_layer: int = physical_layer,
        ) -> None:
            add(
                f"{_layer}.{suffix}",
                shape,
                canonical=f"physical_layer.{_physical_layer}.{suffix}",
            )

        layer_add("input_layernorm.weight", (hidden,))
        layer_add("self_attn.q_proj.weight", (heads * head_dim, hidden))
        layer_add("self_attn.k_proj.weight", (kv_heads * head_dim, hidden))
        layer_add("self_attn.v_proj.weight", (kv_heads * head_dim, hidden))
        layer_add("self_attn.q_norm.weight", (head_dim,))
        layer_add("self_attn.k_norm.weight", (head_dim,))
        layer_add("self_attn.o_proj.weight", (hidden, heads * head_dim))
        layer_add("post_attention_layernorm.weight", (hidden,))
        layer_add("mlp.gate.weight", (experts, hidden))
        for expert in range(experts):
            layer_add(f"mlp.experts.{expert}.gate_proj.weight", (intermediate, hidden))
            layer_add(f"mlp.experts.{expert}.up_proj.weight", (intermediate, hidden))
            layer_add(f"mlp.experts.{expert}.down_proj.weight", (hidden, intermediate))
    if values == _tiny_model_values():
        _apply_tiny_moe_margin_profile(
            weights,
            architecture=architecture,
            vocab_size=vocab,
            hidden_size=hidden,
            num_experts=experts,
        )
    return weights


def _apply_tiny_moe_margin_profile(
    weights: dict[str, Any],
    *,
    architecture: MoRArchitectureConfig,
    vocab_size: int,
    hidden_size: int,
    num_experts: int,
) -> None:
    """Make the default synthetic MoE router deterministic and well conditioned.

    A random BF16 gate can place a token arbitrarily close to the native MoE
    Top-K cutoff.  The expected Magi-vs-unfused attention roundoff can then
    select a different expert and create a discontinuous gradient difference,
    even though both kernels satisfy the forward tolerance.  This synthetic
    fixture reserves two residual-stream coordinates and gives the native
    Qwen router a large cutoff margin.  It does *not* bypass Top-K, token
    dispatch, expert parallelism, or expert training.

    Real Qwen checkpoints never call this helper and retain their learned gate
    weights unchanged.
    """

    import torch

    if hidden_size < 2 or num_experts != 4:
        raise ValueError("default tiny MoE margin profile requires hidden>=2 and four experts")
    embedding = weights["model.embed_tokens.weight"]
    if tuple(embedding.shape) != (vocab_size, hidden_size):
        raise ValueError("tiny embedding shape does not match its declared model config")
    embedding[:, 0] = 16.0
    token_parity = torch.arange(vocab_size, dtype=torch.long).remainder(2)
    embedding[:, 1] = torch.where(token_parity == 0, 16.0, -16.0).to(embedding.dtype)

    gate_anchor = torch.tensor([0.25, -0.25, 0.125, -0.125], dtype=torch.bfloat16)
    for logical_layer in range(architecture.logical_num_layers):
        prefix = f"model.layers.{logical_layer}"
        # These rows remain an identity-like residual anchor at initialization.
        # A training update may move them, which is intentional; the margin is
        # large enough to remain stable across the two-step acceptance run.
        weights[f"{prefix}.self_attn.o_proj.weight"][:2, :].zero_()
        gate = weights[f"{prefix}.mlp.gate.weight"]
        gate.zero_()
        gate[:, 1] = gate_anchor
        for expert in range(num_experts):
            weights[f"{prefix}.mlp.experts.{expert}.down_proj.weight"][:2, :].zero_()


def _materialize_tiny_hf(
    config: MLiteRunConfig,
    *,
    preset: MoRPresetConfig,
    architecture: MoRArchitectureConfig,
) -> str:
    """Create one identical full checkpoint per rank, then let WeightSpec shard it."""

    from safetensors.torch import save_file

    rank = int(os.environ.get("RANK", "0"))
    root = config.output / f"_tiny_hf_rank_{rank:03d}"
    root.mkdir(parents=True, exist_ok=True)
    target = root / "config.json"
    payload = json.dumps(_tiny_hf_dict(architecture, preset.model), indent=2, sort_keys=True) + "\n"
    target.write_text(payload, encoding="utf-8")
    save_file(
        _tiny_hf_weights(architecture, seed=config.seed, model=preset.model),
        str(root / "model.safetensors"),
        metadata={
            "format": "pt",
            "generator": "mor_mlite.topology_independent_tiny.margin_v2",
            "seed": str(config.seed),
        },
    )
    return str(root)


def _balanced_sample_partitions(
    seq_lens: Sequence[int], dp_size: int
) -> tuple[tuple[int, ...], ...]:
    """Assign each global sample once using deterministic longest-first packing."""

    if dp_size < 1:
        raise ValueError("dense DP size must be positive")
    lengths = tuple(int(length) for length in seq_lens)
    if len(lengths) < dp_size:
        raise ValueError(
            "global parity batch must contain at least one sequence per dense-DP "
            f"replica: {len(lengths)} sequences for DP={dp_size}"
        )
    if any(length <= 0 for length in lengths):
        raise ValueError("global parity sequence lengths must be positive")

    assignments: list[list[int]] = [[] for _ in range(dp_size)]
    token_totals = [0] * dp_size
    for sample_index in sorted(range(len(lengths)), key=lambda i: (-lengths[i], i)):
        target = min(range(dp_size), key=lambda rank: (token_totals[rank], rank))
        assignments[target].append(sample_index)
        token_totals[target] += lengths[sample_index]
    return tuple(tuple(sorted(indices)) for indices in assignments)


def _dp_objective_scales(
    seq_lens: Sequence[int],
    partitions: Sequence[Sequence[int]],
    architecture: MoRArchitectureConfig,
    *,
    training: bool,
) -> tuple[dict[str, Any], ...]:
    """Derive exact per-replica scaling for one sharded global microbatch.

    MLite's distributed optimizer averages over dense DP.  LM and depth-router
    terms therefore need ``DP * local_count / global_count``.  Each recurrent
    round has its own candidate denominator, so auxiliary scales are retained
    per round instead of being collapsed into one (generally invalid) scalar.
    """

    from mor_mlite.objective import objective_scales

    lengths = tuple(int(length) for length in seq_lens)
    valid_by_rank = [sum(max(0, lengths[index] - 1) for index in indices) for indices in partitions]
    global_valid = sum(valid_by_rank)
    if global_valid <= 0:
        raise ValueError("global parity batch has no unmasked next-token targets")

    # A round's BCE is normalized over the router's candidate buffer, before
    # that round selects its next active set.  Round zero sees the full input;
    # later rounds see the preceding round's selected capacity.
    candidates_by_round = [
        sum(
            length if round_index == 0 else architecture.top_k(length, round_index - 1)
            for length in lengths
        )
        for round_index in range(architecture.num_recursions)
    ]
    if any(count <= 0 for count in candidates_by_round):
        raise ValueError("global parity batch has an empty recurrent round")

    candidates_by_rank = [
        [
            sum(
                lengths[index]
                if round_index == 0
                else architecture.top_k(lengths[index], round_index - 1)
                for index in indices
            )
            for round_index in range(architecture.num_recursions)
        ]
        for indices in partitions
    ]
    public_scales = objective_scales([valid_by_rank], [candidates_by_rank])[0]
    result: list[dict[str, Any]] = []
    for rank, indices in enumerate(partitions):
        local_candidates = candidates_by_rank[rank]
        result.append(
            {
                "sample_indices": tuple(int(index) for index in indices),
                "global_input_tokens": sum(lengths),
                "local_valid_tokens": valid_by_rank[rank],
                "global_valid_tokens": global_valid,
                "lm_scale": public_scales[rank].lm,
                "local_router_candidates_by_round": tuple(local_candidates),
                "global_router_candidates_by_round": tuple(candidates_by_round),
                "aux_scales": public_scales[rank].auxiliary,
            }
        )
    return tuple(result)


def _slice_global_batch(batch: Any, sample_indices: Sequence[int], scales: Mapping[str, Any]):
    """Slice complete packed sequences while retaining global sample/token IDs."""

    import torch

    from mor_mlite.data import PackedBatch

    lengths = [int(value) for value in batch.seq_lens.detach().cpu().tolist()]
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    rows = torch.cat(
        [
            torch.arange(offsets[index], offsets[index + 1], dtype=torch.long)
            for index in sample_indices
        ]
    )
    rows = rows.to(device=batch.input_ids.device)
    total_tokens = offsets[-1]

    def token_slice(value: Any) -> Any:
        if isinstance(value, torch.Tensor) and value.ndim >= 1 and value.size(0) == total_tokens:
            return value.index_select(0, rows)
        return value

    extras = {key: token_slice(value) for key, value in batch.extras.items()}
    extras.update(
        {
            "mor_parity_lm_scale": float(scales["lm_scale"]),
            "mor_parity_aux_scales": tuple(float(value) for value in scales["aux_scales"]),
            "mor_parity_local_valid_tokens": int(scales["local_valid_tokens"]),
            "mor_parity_global_valid_tokens": int(scales["global_valid_tokens"]),
            "mor_parity_global_input_tokens": int(scales["global_input_tokens"]),
            # Retain the exact oracle universe on every DP shard. This is
            # token-ID metadata only (never a hidden-state replication) and
            # lets canonical merge reject a missing ID replaced by a rogue ID.
            "mor_parity_expected_global_token_ids": batch.extras["global_token_ids"]
            .detach()
            .clone(),
            "mor_parity_sample_indices": tuple(int(index) for index in sample_indices),
        }
    )
    sample_rows = torch.tensor(sample_indices, dtype=torch.long, device=batch.seq_lens.device)
    return PackedBatch(
        input_ids=batch.input_ids.index_select(0, rows),
        labels=batch.labels.index_select(0, rows),
        seq_lens=batch.seq_lens.index_select(0, sample_rows),
        loss_mask=token_slice(batch.loss_mask),
        position_ids=token_slice(batch.position_ids),
        routed_experts=token_slice(batch.routed_experts),
        r3_replay_mask=token_slice(batch.r3_replay_mask),
        extras=extras,
    )


def _localize_replay_plans(plans: Sequence[Any], sample_indices: Sequence[int]) -> tuple[Any, ...]:
    """Restrict a global oracle RoutePlan to one dense-DP sample shard."""

    from mor_mlite.routing import RoutePlan

    sample_set = {int(index) for index in sample_indices}
    token_fields = (
        "sample_ids",
        "original_positions",
        "global_token_ids",
        "source_tp_ranks",
        "source_cp_ranks",
        "source_local_rows",
        "target_tp_ranks",
        "target_cp_ranks",
        "target_local_rows",
        "selected_gates",
        "padding_mask",
    )
    localized = []
    for plan in plans:
        raw = plan.to_dict()
        keep = [
            row
            for row, (sample_id, padding) in enumerate(
                zip(raw["sample_ids"], raw["padding_mask"], strict=True)
            )
            if not bool(padding) and int(sample_id) in sample_set
        ]
        for token_field in token_fields:
            raw[token_field] = [raw[token_field][row] for row in keep]
        ordered_samples = sorted({int(value) for value in raw["sample_ids"]})
        counts = [
            sum(int(value) == sample_id for value in raw["sample_ids"])
            for sample_id in ordered_samples
        ]
        cumulative = [0]
        for count in counts:
            cumulative.append(cumulative[-1] + count)
        raw["active_cu_seqlens"] = cumulative
        raw["cutoff_score_margins"] = {
            str(sample_id): margin
            for sample_id, margin in raw["cutoff_score_margins"].items()
            if int(sample_id) in sample_set
        }
        localized.append(RoutePlan.from_dict(raw))
    return tuple(localized)


def _route_index(path: Path | None) -> dict[tuple[int, int], tuple[Any, ...]]:
    if path is None:
        return {}
    from mor_mlite.parity.artifacts import load_artifact
    from mor_mlite.routing import RoutePlan

    _, _, raw_routes = load_artifact(path)
    grouped: dict[tuple[int, int], dict[int, Any]] = {}
    for raw in raw_routes:
        # A baseline checkpoint artifact also contains its deterministic next
        # step under both ``uninterrupted`` and ``resume``.  Index those rows by
        # their real step so a replay candidate consumes the baseline's step-N
        # decision rather than incorrectly falling back to step N-1.  Duplicate
        # phases must be identical; otherwise the baseline has already failed
        # its own checkpoint-continuity contract and is not a valid replay oracle.
        phase = str(raw.get("phase", "train"))
        step = int(raw.get("step", 0))
        microbatch = int(raw.get("microbatch", 0))
        plan = RoutePlan.from_dict(raw)
        rounds = grouped.setdefault((step, microbatch), {})
        if plan.round_index in rounds:
            existing = rounds[plan.round_index]
            if existing.to_dict() != plan.to_dict():
                raise ValueError(
                    "conflicting RoutePlans across checkpoint phases for "
                    f"phase={phase}, step={step}, microbatch={microbatch}, "
                    f"round={plan.round_index}"
                )
            continue
        rounds[plan.round_index] = plan
    return {
        key: tuple(by_round[index] for index in sorted(by_round))
        for key, by_round in grouped.items()
    }


_EXPERT_ROUTE_REPLAY_RE = re.compile(
    r"^expert_route/"
    r"(?:(?P<phase>[^/]+)/)?"
    r"step_(?P<step>\d+)/mb_(?P<microbatch>\d+)/"
    r"logical_(?P<logical_layer>\d+)/"
    r"(?P<stage>start|recurrent|end)/"
    r"(?P<round>round_\d+|non_recurrent)/"
    r"physical_(?P<physical_layer>\d+)/"
    r"(?P<field>global_token_ids|topk_indices|selected_scores)$"
)


def _expert_route_replay_index(
    path: Path | None,
) -> dict[tuple[int, int], dict[int, Any]]:
    """Load canonical native-MoE expert choices and scores from an artifact.

    The depth ``RoutePlan`` and native Qwen MoE routing are separate discrete
    decisions.  Cross-topology replay fixes both: the former by recursion and
    the latter, including consumed forward scores, by logical layer plus stable
    global token ID.  Checkpoint
    artifacts can contain identical ``uninterrupted`` and ``resume`` copies of
    one step; conflicting copies are rejected instead of selecting one by
    filesystem/order accident.
    """

    if path is None:
        return {}
    import torch

    from mor_mlite.parity.artifacts import load_artifact
    from mor_mlite.qwen3_moe_mor.expert_route_probe import ExpertRouteReplayPlan

    _, tensors, _ = load_artifact(path)
    raw_contexts: dict[tuple[str, int, int, int], dict[str, torch.Tensor]] = {}
    for name, tensor in tensors.items():
        match = _EXPERT_ROUTE_REPLAY_RE.fullmatch(name)
        if match is None:
            continue
        context = (
            match.group("phase") or "train",
            int(match.group("step")),
            int(match.group("microbatch")),
            int(match.group("logical_layer")),
        )
        field = match.group("field")
        fields = raw_contexts.setdefault(context, {})
        if field in fields:
            raise ValueError(f"duplicate expert-route replay tensor: {name}")
        fields[field] = tensor

    by_step: dict[tuple[int, int], dict[int, Any]] = {}
    for (phase, step, microbatch, logical_layer), fields in sorted(raw_contexts.items()):
        expected = {"global_token_ids", "topk_indices", "selected_scores"}
        if fields.keys() != expected:
            raise ValueError(
                "incomplete expert-route replay context for "
                f"phase={phase}, step={step}, microbatch={microbatch}, "
                f"logical_layer={logical_layer}: fields={sorted(fields)}"
            )
        plan = ExpertRouteReplayPlan(
            global_token_ids=fields["global_token_ids"],
            topk_indices=fields["topk_indices"],
            selected_scores=fields["selected_scores"],
        )
        plan.validate()
        layers = by_step.setdefault((step, microbatch), {})
        existing = layers.get(logical_layer)
        if existing is not None:
            if not (
                torch.equal(existing.global_token_ids, plan.global_token_ids)
                and torch.equal(existing.topk_indices, plan.topk_indices)
                and torch.equal(existing.selected_scores, plan.selected_scores)
            ):
                raise ValueError(
                    "conflicting expert-route replay plans across checkpoint phases for "
                    f"phase={phase}, step={step}, microbatch={microbatch}, "
                    f"logical_layer={logical_layer}"
                )
            continue
        layers[logical_layer] = plan
    return by_step


def _find_model_attribute(handle: Any, name: str) -> Any:
    """Read temporary parity diagnostics through wrapper ``module`` links.

    Runtime operations never bypass the public API.  The pinned ModelHandle
    explicitly permits internal helpers to use ``_model`` while its diagnostic
    accessor surface is being stabilized, so parity capture uses that narrow
    compatibility seam for MoR-only traces not represented by ForwardResult.
    """

    roots = getattr(handle, "_model", ())
    if not isinstance(roots, (list, tuple)):
        roots = (roots,)
    for root in roots:
        current = root
        seen: set[int] = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if hasattr(current, name):
                return getattr(current, name)
            current = getattr(current, "module", None)
    return None


_LOCAL_EXPERT_RE = re.compile(r"^(?P<prefix>.*\.moe\.experts\.(?:fc1|fc2))\.weight(?P<index>\d+)$")


def _unwrapped_model_chunks(handle: Any) -> tuple[Any, ...]:
    chunks = getattr(handle, "_extras", {}).get("model_chunks")
    if chunks is None:
        chunks = getattr(handle, "_model", ())
    if not isinstance(chunks, (list, tuple)):
        chunks = (chunks,)
    result = []
    for chunk in chunks:
        current = chunk
        seen: set[int] = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            wrapped = getattr(current, "module", None)
            if wrapped is None:
                break
            current = wrapped
        if current is not None:
            result.append(current)
    return tuple(result)


def _set_moe_expert_route_probe(handle: Any, *, enabled: bool) -> bool:
    """Configure the detached MoE probe on the one PP=1 MoR model."""

    models = [
        chunk
        for chunk in _unwrapped_model_chunks(handle)
        if hasattr(chunk, "mor_architecture")
        and callable(getattr(chunk, "set_moe_expert_route_probe", None))
    ]
    if enabled and len(models) != 1:
        raise TypeError(
            "tiny MLite parity requires exactly one model with the native "
            f"expert-route probe, found {len(models)}"
        )
    for model in models:
        model.set_moe_expert_route_probe(enabled)
    return bool(enabled and models)


def _set_mor_diagnostic_capture(handle: Any, *, enabled: bool) -> bool:
    """Opt the parity runner into detached round/QKV evidence."""

    models = [
        chunk
        for chunk in _unwrapped_model_chunks(handle)
        if hasattr(chunk, "mor_architecture")
        and callable(getattr(chunk, "set_mor_diagnostic_capture", None))
    ]
    if enabled and len(models) != 1:
        raise TypeError(
            f"MLite parity requires exactly one model with diagnostic capture, found {len(models)}"
        )
    for model in models:
        model.set_mor_diagnostic_capture(enabled)
    return bool(enabled and models)


def _named_model_parameters(handle: Any) -> dict[int, tuple[str, Any]]:
    """Index physical model parameters by identity without wrapper prefixes."""

    result: dict[int, tuple[str, Any]] = {}
    for chunk_index, chunk in enumerate(_unwrapped_model_chunks(handle)):
        for name, parameter in chunk.named_parameters():
            stable_name = name if chunk_index == 0 else f"chunk_{chunk_index}.{name}"
            previous = result.get(id(parameter))
            if previous is not None and previous[0] != stable_name:
                raise RuntimeError(
                    f"one MLite parameter has multiple parity names: {previous[0]!r}, "
                    f"{stable_name!r}"
                )
            result[id(parameter)] = (stable_name, parameter)
    if not result:
        raise RuntimeError("MLite ModelHandle exposes no model parameters")
    return result


def _assert_recurrent_parameters_registered_once(
    handle: Any, architecture: MoRArchitectureConfig
) -> dict[str, Any]:
    """Prove that logical recurrence does not duplicate physical parameters.

    ``named_parameters(remove_duplicate=False)`` is intentional: the default
    iterator hides aliases, which is precisely the structural error this
    acceptance assertion must detect.
    """

    candidates = [
        chunk
        for chunk in _unwrapped_model_chunks(handle)
        if hasattr(chunk, "layers") and hasattr(chunk, "mor_architecture")
    ]
    if len(candidates) != 1:
        raise TypeError(
            "expected exactly one unwrapped Qwen3-MoE MoR model for the PP=1 "
            f"parity run, found {len(candidates)}"
        )
    model = candidates[0]
    layers = tuple(model.layers)
    if len(layers) != architecture.physical_num_layers:
        raise TypeError(
            "registered decoder layer count does not match physical MoR depth: "
            f"{len(layers)} != {architecture.physical_num_layers}"
        )
    if len({id(layer) for layer in layers}) != len(layers):
        raise RuntimeError("one decoder module is registered at multiple physical indices")

    try:
        all_entries = tuple(model.named_parameters(recurse=True, remove_duplicate=False))
    except TypeError as exc:
        raise RuntimeError(
            "PyTorch named_parameters(remove_duplicate=False) is required to "
            "audit recurrent parameter registration"
        ) from exc
    registrations: dict[int, list[str]] = {}
    for name, parameter in all_entries:
        registrations.setdefault(id(parameter), []).append(str(name))

    recurrent_start = architecture.n_start_layers
    recurrent_stop = recurrent_start + architecture.n_recurrent_layers
    recurrent_ids: list[int] = []
    for physical_index in range(recurrent_start, recurrent_stop):
        layer = layers[physical_index]
        try:
            entries = tuple(layer.named_parameters(recurse=True, remove_duplicate=False))
        except TypeError as exc:
            raise RuntimeError(
                "recurrent layer cannot expose non-deduplicated parameter names"
            ) from exc
        if not entries:
            raise RuntimeError(
                f"physical recurrent layer {physical_index} has no registered parameters"
            )
        recurrent_ids.extend(id(parameter) for _, parameter in entries)

    unique_recurrent_ids = set(recurrent_ids)
    duplicate_local = len(recurrent_ids) - len(unique_recurrent_ids)
    duplicate_global = {
        parameter_id: names
        for parameter_id, names in registrations.items()
        if parameter_id in unique_recurrent_ids and len(names) != 1
    }
    missing_global = unique_recurrent_ids - registrations.keys()
    if duplicate_local or duplicate_global or missing_global:
        details = {
            "duplicate_recurrent_entries": duplicate_local,
            "aliased_model_paths": list(duplicate_global.values()),
            "missing_model_registrations": len(missing_global),
        }
        raise RuntimeError(
            f"recurrent physical parameters are not registered exactly once: {details}"
        )
    return {
        "status": "passed",
        "physical_decoder_layers": len(layers),
        "recurrent_physical_layer_indices": list(range(recurrent_start, recurrent_stop)),
        "recurrent_physical_parameters": len(unique_recurrent_ids),
        "max_registrations_per_recurrent_parameter": 1,
        "logical_reuses_per_recurrent_parameter": architecture.num_recursions,
    }


class _GradientSyncProbe:
    """Per-handle dynamic instrumentation for the pinned dist-opt stack."""

    def __init__(self, *, required: bool) -> None:
        self.required = bool(required)
        self.finalize_status = "not_required" if not required else "unavailable"
        self.finalize_reason: str | None = None
        self.bucket_status = "not_required" if not required else "unavailable"
        self.bucket_reason: str | None = None
        self.bucket_ids: tuple[str, ...] = ()
        self._active_step: str | None = None
        self._finalize_calls = 0
        self._bucket_calls: dict[str, int] = {}
        self._out_of_step_finalize_calls = 0
        self._out_of_step_bucket_calls = 0

    def begin_step(self, global_step: str) -> None:
        if self._active_step is not None:
            raise RuntimeError(f"gradient-sync probe step {self._active_step!r} was not finished")
        self._active_step = str(global_step)
        self._finalize_calls = 0
        self._bucket_calls = {bucket_id: 0 for bucket_id in self.bucket_ids}

    def record_finalize(self) -> None:
        if self._active_step is None:
            self._out_of_step_finalize_calls += 1
            return
        self._finalize_calls += 1

    def record_bucket_dispatch(self, bucket_ids: Sequence[str]) -> None:
        if self._active_step is None:
            self._out_of_step_bucket_calls += len(bucket_ids)
            return
        for bucket_id in bucket_ids:
            self._bucket_calls[bucket_id] = self._bucket_calls.get(bucket_id, 0) + 1

    def finish_step(self, global_step: str) -> dict[str, Any]:
        if self._active_step != str(global_step):
            raise RuntimeError(
                "gradient-sync probe step mismatch: "
                f"active={self._active_step!r}, finished={global_step!r}"
            )
        report = {
            "global_step": str(global_step),
            "required": self.required,
            "finalize_grads": {
                "status": self.finalize_status,
                "reason": self.finalize_reason,
                "calls": self._finalize_calls,
            },
            "physical_buckets": {
                "status": self.bucket_status,
                "reason": self.bucket_reason,
                "count": len(self.bucket_ids),
                "sync_calls": dict(self._bucket_calls),
            },
            "out_of_step_finalize_calls": self._out_of_step_finalize_calls,
            "out_of_step_bucket_calls": self._out_of_step_bucket_calls,
        }
        self._active_step = None
        return report


def _install_gradient_sync_probe(handle: Any, *, forward_only: bool) -> _GradientSyncProbe:
    """Wrap the exact Runtime finalizer and DDP bucket-group dispatch seam.

    Pinned MLite configures Megatron-Core DDP with non-overlapped gradient
    reduction.  Its real path is ``finalize_grads -> DDP.finish_grad_sync ->
    bucket_group.finish_grad_sync -> bucket_group.start_grad_sync``; wrapping
    DDP's top-level ``start_grad_sync`` would therefore miss the collective.
    """

    probe = _GradientSyncProbe(required=not forward_only)
    if forward_only:
        return probe

    extras = getattr(handle, "_extras", None)
    if not isinstance(extras, dict):
        probe.finalize_reason = "ModelHandle._extras is not a mutable dictionary"
        probe.bucket_reason = "ModelHandle._extras is not a mutable dictionary"
        return probe

    finalize_grads = extras.get("finalize_grads")
    if callable(finalize_grads):

        def instrumented_finalize_grads(*args, **kwargs):
            result = finalize_grads(*args, **kwargs)
            probe.record_finalize()
            return result

        extras["finalize_grads"] = instrumented_finalize_grads
        probe.finalize_status = "available"
    else:
        probe.finalize_reason = "ModelHandle._extras['finalize_grads'] is not callable"

    chunks = extras.get("model_chunks")
    if not isinstance(chunks, (list, tuple)):
        chunks = (chunks,) if chunks is not None else ()
    patch_targets: list[tuple[Any, Any, tuple[str, ...]]] = []
    physical_bucket_objects: list[int] = []
    discovery_errors: list[str] = []
    for chunk_index, chunk in enumerate(chunks):
        ddp_config = getattr(chunk, "ddp_config", None)
        if ddp_config is None:
            discovery_errors.append(f"chunk {chunk_index} has no ddp_config")
        else:
            if not bool(getattr(ddp_config, "use_distributed_optimizer", False)):
                discovery_errors.append(
                    f"chunk {chunk_index} is not using the distributed optimizer"
                )
            if bool(getattr(ddp_config, "overlap_grad_reduce", False)):
                discovery_errors.append(
                    f"chunk {chunk_index} enables unsupported overlap_grad_reduce"
                )
        if bool(getattr(chunk, "force_all_reduce", False)):
            discovery_errors.append(
                f"chunk {chunk_index} forces all-reduce instead of reduce-scatter"
            )
        for family, attribute in (
            ("dense", "bucket_groups"),
            ("expert", "expert_parallel_bucket_groups"),
        ):
            groups = getattr(chunk, attribute, None)
            if groups is None:
                discovery_errors.append(f"chunk {chunk_index} has no {attribute}")
                continue
            for group_index, group in enumerate(groups):
                start_grad_sync = getattr(group, "start_grad_sync", None)
                buckets = getattr(group, "buckets", None)
                if not callable(start_grad_sync) or not isinstance(buckets, (list, tuple)):
                    discovery_errors.append(
                        f"chunk {chunk_index} {family} group {group_index} "
                        "does not expose start_grad_sync plus buckets"
                    )
                    continue
                bucket_ids = tuple(
                    f"chunk_{chunk_index:03d}/{family}/group_{group_index:03d}/"
                    f"bucket_{bucket_index:03d}"
                    for bucket_index in range(len(buckets))
                )
                physical_bucket_objects.extend(id(bucket) for bucket in buckets)
                patch_targets.append((group, start_grad_sync, bucket_ids))

    all_bucket_ids = tuple(
        bucket_id for _, _, bucket_ids in patch_targets for bucket_id in bucket_ids
    )
    if not all_bucket_ids:
        discovery_errors.append("no physical DDP buckets were discovered")
    if len(all_bucket_ids) != len(set(all_bucket_ids)):
        discovery_errors.append("physical DDP bucket identifiers are not unique")
    if len(physical_bucket_objects) != len(set(physical_bucket_objects)):
        discovery_errors.append("one physical DDP bucket appears in multiple bucket groups")
    if discovery_errors:
        probe.bucket_reason = "; ".join(discovery_errors)
        return probe

    patched: list[tuple[Any, Any]] = []
    try:
        for group, original, bucket_ids in patch_targets:

            def instrumented_start_grad_sync(
                *args,
                __group=group,
                __original=original,
                __bucket_ids=bucket_ids,
                **kwargs,
            ):
                # This is the sole no-op branch at the top of the pinned
                # BucketGroup.start_grad_sync implementation.
                already_dispatched_first_batch = bool(
                    getattr(__group, "is_first_batch", False)
                    and getattr(__group, "grad_reduce_handle", None) is not None
                )
                result = __original(*args, **kwargs)
                if not already_dispatched_first_batch:
                    probe.record_bucket_dispatch(__bucket_ids)
                return result

            group.start_grad_sync = instrumented_start_grad_sync
            if group.start_grad_sync is not instrumented_start_grad_sync:
                raise TypeError("instance start_grad_sync replacement was not retained")
            patched.append((group, original))
    except (AttributeError, TypeError) as exc:
        for group, original in patched:
            try:
                group.start_grad_sync = original
            except (AttributeError, TypeError):
                pass
        probe.bucket_reason = f"DDP bucket start_grad_sync is not patchable: {exc}"
        return probe

    probe.bucket_ids = all_bucket_ids
    probe.bucket_status = "available"
    return probe


def _merge_gradient_sync_step_reports(
    reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not reports:
        raise RuntimeError("gradient-sync instrumentation produced no rank reports")
    global_step = str(reports[0]["global_step"])
    required = bool(reports[0]["required"])
    if any(
        str(report["global_step"]) != global_step or bool(report["required"]) != required
        for report in reports
    ):
        raise RuntimeError("ranks disagree on gradient-sync global-step metadata")

    rank_rows = []
    finalize_calls = []
    bucket_calls = []
    finalize_statuses = []
    bucket_statuses = []
    reasons = []
    for rank, report in enumerate(reports):
        finalize = report["finalize_grads"]
        physical = report["physical_buckets"]
        finalize_statuses.append(str(finalize["status"]))
        bucket_statuses.append(str(physical["status"]))
        finalize_calls.append(int(finalize["calls"]))
        rank_bucket_calls = {
            str(name): int(count) for name, count in physical["sync_calls"].items()
        }
        bucket_calls.extend(rank_bucket_calls.values())
        for reason in (finalize.get("reason"), physical.get("reason")):
            if reason:
                reasons.append(f"rank {rank}: {reason}")
        rank_rows.append(
            {
                "rank": rank,
                "finalize_grads_calls": int(finalize["calls"]),
                "physical_bucket_count": int(physical["count"]),
                "physical_bucket_sync_calls": rank_bucket_calls,
                "out_of_step_finalize_calls": int(report["out_of_step_finalize_calls"]),
                "out_of_step_bucket_calls": int(report["out_of_step_bucket_calls"]),
            }
        )

    expected_status = "available" if required else "not_required"
    status = (
        expected_status
        if all(value == expected_status for value in finalize_statuses)
        and all(value == expected_status for value in bucket_statuses)
        else "unavailable"
    )
    return {
        "global_step": global_step,
        "required": required,
        "status": status,
        "reasons": reasons,
        "finalize_grads_calls_min": min(finalize_calls),
        "finalize_grads_calls_max": max(finalize_calls),
        "physical_bucket_sync_calls_min": min(bucket_calls) if bucket_calls else None,
        "physical_bucket_sync_calls_max": max(bucket_calls) if bucket_calls else None,
        "ranks": rank_rows,
    }


def _summarize_gradient_sync_steps(
    steps: Sequence[Mapping[str, Any]], *, required: bool
) -> dict[str, Any]:
    expected_status = "available" if required else "not_required"
    status = (
        expected_status
        if steps and all(step.get("status") == expected_status for step in steps)
        else "unavailable"
    )
    finalize_minima = [
        int(step["finalize_grads_calls_min"])
        for step in steps
        if step.get("finalize_grads_calls_min") is not None
    ]
    finalize_maxima = [
        int(step["finalize_grads_calls_max"])
        for step in steps
        if step.get("finalize_grads_calls_max") is not None
    ]
    bucket_minima = [
        int(step["physical_bucket_sync_calls_min"])
        for step in steps
        if step.get("physical_bucket_sync_calls_min") is not None
    ]
    bucket_maxima = [
        int(step["physical_bucket_sync_calls_max"])
        for step in steps
        if step.get("physical_bucket_sync_calls_max") is not None
    ]
    return {
        "required": required,
        "status": status,
        "steps": list(steps),
        "finalize_grads_calls_min": min(finalize_minima) if finalize_minima else None,
        "finalize_grads_calls_max": max(finalize_maxima) if finalize_maxima else None,
        "physical_bucket_sync_calls_min": min(bucket_minima) if bucket_minima else None,
        "physical_bucket_sync_calls_max": max(bucket_maxima) if bucket_maxima else None,
    }


def _canonical_parameter_name(
    name: str, *, ep_rank: int, ep_size: int, num_experts: int
) -> tuple[str, bool]:
    match = _LOCAL_EXPERT_RE.match(name)
    if match is None:
        return name, False
    if num_experts % ep_size:
        raise RuntimeError(f"num_experts={num_experts} is not divisible by EP={ep_size}")
    local_index = int(match.group("index"))
    experts_per_rank = num_experts // ep_size
    if not 0 <= local_index < experts_per_rank:
        raise RuntimeError(f"local expert index {local_index} is outside [0, {experts_per_rank})")
    global_index = ep_rank * experts_per_rank + local_index
    return f"{match.group('prefix')}.weight{global_index}", True


def _tp_shard_dim(name: str) -> int | None:
    if name in {"embed.embedding.weight", "head.col.linear.weight"}:
        return 0
    if name.endswith(".attn.qkv.linear.weight"):
        return 0
    if name.endswith(".attn.proj.linear.weight"):
        return 1
    if ".eh_proj.linear.weight" in name:
        return 0
    return None


def _parameter_context(handle: Any) -> dict[str, int]:
    ps = getattr(handle, "_parallel_state", None)
    model_cfg = getattr(handle, "_extras", {}).get("model_cfg")
    if ps is None or model_cfg is None:
        raise RuntimeError("parameter capture requires MLite parallel state and model config")
    return {
        "tp_rank": int(getattr(ps, "tp_rank", 0)),
        "tp_size": int(getattr(ps, "tp_size", 1)),
        "cp_rank": int(getattr(ps, "cp_rank", 0)),
        "dp_rank": int(getattr(ps, "dp_rank", 0)),
        "ep_rank": int(getattr(ps, "ep_rank", 0)),
        "ep_size": int(getattr(ps, "ep_size", 1)),
        "etp_rank": int(getattr(ps, "etp_rank", 0)),
        "expert_dp_rank": int(getattr(ps, "expert_dp_rank", 0)),
        "num_experts": int(model_cfg.num_experts),
        "vocab_size": int(model_cfg.vocab_size),
    }


def _optimizer_leaves(optimizer: Any) -> tuple[Any, ...]:
    children = getattr(optimizer, "chained_optimizers", None)
    if children is None:
        return (optimizer,)
    return tuple(leaf for child in children for leaf in _optimizer_leaves(child))


def _local_weight_fragments(handle: Any) -> dict[str, Any]:
    context = _parameter_context(handle)
    fragments = []
    for name, parameter in _named_model_parameters(handle).values():
        canonical, is_expert = _canonical_parameter_name(
            name,
            ep_rank=context["ep_rank"],
            ep_size=context["ep_size"],
            num_experts=context["num_experts"],
        )
        shard_dim = None if is_expert else _tp_shard_dim(canonical)
        if is_expert:
            include = context["expert_dp_rank"] == 0 and context["etp_rank"] == 0
        else:
            include = context["dp_rank"] == 0 and context["cp_rank"] == 0
            if shard_dim is None:
                include = include and context["tp_rank"] == 0
        if not include:
            continue
        value = parameter.detach().cpu().clone().contiguous()
        fragments.append(
            {
                "name": canonical,
                "is_expert": is_expert,
                "tp_rank": context["tp_rank"],
                "tp_size": context["tp_size"],
                "shard_dim": shard_dim,
                "shape": tuple(value.shape),
                "numel": value.numel(),
                "start": 0,
                "end": value.numel(),
                "value": value.reshape(-1),
            }
        )
    return {"context": context, "fragments": fragments}


def _local_gradient_fragments(handle: Any) -> dict[str, Any]:
    """Extract only this rank's valid post-reduce-scatter gradient intervals."""

    optimizer = getattr(handle, "_optimizer", None)
    if optimizer is None:
        raise RuntimeError("gradient capture requires an MLite optimizer")
    context = _parameter_context(handle)
    names = _named_model_parameters(handle)
    fragments = []
    seen: set[tuple[int, int, int]] = set()
    range_maps_found = 0
    for leaf in _optimizer_leaves(optimizer):
        gbuf_ranges = getattr(leaf, "gbuf_ranges", None)
        if gbuf_ranges is None:
            continue
        range_maps_found += 1
        for buffer_ranges in gbuf_ranges:
            for bucket_ranges in buffer_ranges.values():
                for bucket_range in bucket_ranges:
                    for parameter, ranges in bucket_range["param_map"].items():
                        parameter_range = ranges["param"]
                        start = int(parameter_range.start)
                        end = int(parameter_range.end)
                        marker = (id(parameter), start, end)
                        if marker in seen:
                            raise RuntimeError(
                                "distributed optimizer exposed a duplicate gradient interval"
                            )
                        seen.add(marker)
                        named = names.get(id(parameter))
                        if named is None:
                            raise RuntimeError(
                                "distributed optimizer range references an unnamed model parameter"
                            )
                        name, _ = named
                        canonical, is_expert = _canonical_parameter_name(
                            name,
                            ep_rank=context["ep_rank"],
                            ep_size=context["ep_size"],
                            num_experts=context["num_experts"],
                        )
                        main_grad = getattr(parameter, "main_grad", None)
                        if main_grad is None:
                            raise RuntimeError(
                                f"parameter {canonical!r} has no finalized main_grad"
                            )
                        if not 0 <= start < end <= parameter.numel():
                            raise RuntimeError(
                                f"invalid optimizer gradient range [{start}, {end}) for "
                                f"{canonical!r} with {parameter.numel()} elements"
                            )
                        fragments.append(
                            {
                                "name": canonical,
                                "is_expert": is_expert,
                                "tp_rank": context["tp_rank"],
                                "tp_size": context["tp_size"],
                                "shard_dim": (None if is_expert else _tp_shard_dim(canonical)),
                                "shape": tuple(parameter.shape),
                                "numel": parameter.numel(),
                                "start": start,
                                "end": end,
                                "value": (
                                    main_grad.reshape(-1)[start:end]
                                    .detach()
                                    .float()
                                    .cpu()
                                    .clone()
                                    .contiguous()
                                ),
                            }
                        )
    if range_maps_found == 0:
        raise RuntimeError(
            "pinned MLite optimizer exposes no gbuf_ranges for complete gradient capture"
        )
    return {"context": context, "fragments": fragments}


def _local_master_weight_fragments(handle: Any) -> dict[str, Any]:
    """Extract this rank's ZeRO-1 FP32 optimizer-owned parameter intervals.

    Megatron's distributed optimizer replaces every optimizer parameter with
    the locally owned shard of the corresponding model parameter.  For BF16
    model parameters that shard is the FP32 master weight; for an FP32 model
    parameter it is the owned view itself.  Reading the installed optimizer
    groups (rather than casting resident BF16 model weights) therefore lets the
    parity harness compare the actual update that Adam applied.
    """

    import torch

    optimizer = getattr(handle, "_optimizer", None)
    if optimizer is None:
        raise RuntimeError("master-weight capture requires an MLite optimizer")
    context = _parameter_context(handle)
    names = _named_model_parameters(handle)
    fragments = []
    seen: set[tuple[int, int, int]] = set()
    range_maps_found = 0
    for leaf in _optimizer_leaves(optimizer):
        gbuf_ranges = getattr(leaf, "gbuf_ranges", None)
        group_map = getattr(leaf, "model_param_group_index_map", None)
        inner_optimizer = getattr(leaf, "optimizer", None)
        if gbuf_ranges is None:
            continue
        if not isinstance(group_map, Mapping) or inner_optimizer is None:
            raise RuntimeError(
                "pinned MLite distributed optimizer is missing its model-to-main parameter map"
            )
        range_maps_found += 1
        for buffer_ranges in gbuf_ranges:
            for bucket_ranges in buffer_ranges.values():
                for bucket_range in bucket_ranges:
                    for parameter, ranges in bucket_range["param_map"].items():
                        parameter_range = ranges["param"]
                        start = int(parameter_range.start)
                        end = int(parameter_range.end)
                        marker = (id(parameter), start, end)
                        if marker in seen:
                            raise RuntimeError(
                                "distributed optimizer exposed a duplicate master-weight interval"
                            )
                        seen.add(marker)
                        named = names.get(id(parameter))
                        if named is None:
                            raise RuntimeError(
                                "distributed optimizer range references an unnamed model parameter"
                            )
                        name, _ = named
                        canonical, is_expert = _canonical_parameter_name(
                            name,
                            ep_rank=context["ep_rank"],
                            ep_size=context["ep_size"],
                            num_experts=context["num_experts"],
                        )
                        try:
                            group_index, group_order = group_map[parameter]
                            main_parameter = inner_optimizer.param_groups[group_index]["params"][
                                group_order
                            ]
                        except (KeyError, IndexError, TypeError) as error:
                            raise RuntimeError(
                                f"cannot resolve FP32 optimizer shard for {canonical!r}"
                            ) from error
                        if not isinstance(main_parameter, torch.Tensor):
                            raise TypeError(f"optimizer shard for {canonical!r} is not a tensor")
                        if not 0 <= start < end <= parameter.numel():
                            raise RuntimeError(
                                f"invalid optimizer master-weight range [{start}, {end}) for "
                                f"{canonical!r} with {parameter.numel()} elements"
                            )
                        if main_parameter.numel() != end - start:
                            raise RuntimeError(
                                f"optimizer shard for {canonical!r} contains "
                                f"{main_parameter.numel()} elements, expected {end - start}"
                            )
                        if main_parameter.dtype != torch.float32:
                            raise RuntimeError(
                                f"optimizer shard for {canonical!r} has dtype "
                                f"{main_parameter.dtype}, expected torch.float32"
                            )
                        fragments.append(
                            {
                                "name": canonical,
                                "is_expert": is_expert,
                                "tp_rank": context["tp_rank"],
                                "tp_size": context["tp_size"],
                                "shard_dim": (None if is_expert else _tp_shard_dim(canonical)),
                                "shape": tuple(parameter.shape),
                                "numel": parameter.numel(),
                                "start": start,
                                "end": end,
                                "value": (
                                    main_parameter.detach()
                                    .float()
                                    .reshape(-1)
                                    .cpu()
                                    .clone()
                                    .contiguous()
                                ),
                            }
                        )
    if range_maps_found == 0:
        raise RuntimeError(
            "pinned MLite optimizer exposes no gbuf_ranges for master-weight capture"
        )
    return {"context": context, "fragments": fragments}


def _gather_objects(value: Any) -> list[Any]:
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
        return [value]
    gathered: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, value)
    return gathered


def _merge_parameter_fragments(peers: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Rebuild expert-EP, dense TP, and dist-opt DP shards into full tensors."""

    import torch

    local_groups: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for peer in peers:
        for fragment in peer["fragments"]:
            group_tp = -1 if fragment["is_expert"] else int(fragment["tp_rank"])
            local_groups.setdefault((str(fragment["name"]), group_tp), []).append(fragment)

    local_tensors: dict[str, list[tuple[int, int | None, bool, Any]]] = {}
    for (name, tp_rank), fragments in local_groups.items():
        shape = tuple(fragments[0]["shape"])
        numel = int(fragments[0]["numel"])
        shard_dim = fragments[0]["shard_dim"]
        is_expert = bool(fragments[0]["is_expert"])
        if any(
            tuple(item["shape"]) != shape
            or int(item["numel"]) != numel
            or item["shard_dim"] != shard_dim
            or bool(item["is_expert"]) != is_expert
            for item in fragments
        ):
            raise RuntimeError(f"inconsistent fragment metadata for {name!r}")
        flat = torch.zeros(numel, dtype=fragments[0]["value"].dtype)
        coverage = torch.zeros(numel, dtype=torch.int16)
        for fragment in fragments:
            start, end = int(fragment["start"]), int(fragment["end"])
            value = fragment["value"].reshape(-1)
            if value.numel() != end - start:
                raise RuntimeError(f"fragment size mismatch for {name!r}")
            existing = coverage[start:end]
            if bool((existing > 0).any()):
                current = flat[start:end][existing > 0]
                candidate = value[existing > 0]
                if not torch.equal(current, candidate):
                    raise RuntimeError(f"overlapping peers disagree for {name!r}")
            uncovered = existing == 0
            target = flat[start:end]
            target[uncovered] = value[uncovered]
            coverage[start:end] += 1
        if bool((coverage == 0).any()):
            missing = int((coverage == 0).sum().item())
            raise RuntimeError(
                f"incomplete distributed reconstruction for {name!r}: "
                f"{missing}/{numel} elements missing"
            )
        local_tensors.setdefault(name, []).append(
            (tp_rank, shard_dim, is_expert, flat.reshape(shape))
        )

    if not local_tensors:
        raise RuntimeError("distributed parameter capture produced no tensors")
    tp_size = int(peers[0]["context"]["tp_size"])
    vocab_size = int(peers[0]["context"]["vocab_size"])
    result: dict[str, Any] = {}
    for name, pieces in local_tensors.items():
        shard_dim = pieces[0][1]
        is_expert = pieces[0][2]
        if is_expert or shard_dim is None:
            value = pieces[0][3]
            if any(not torch.equal(value, piece[3]) for piece in pieces[1:]):
                raise RuntimeError(f"replicated parameter peers disagree for {name!r}")
        else:
            by_tp = {piece[0]: piece[3] for piece in pieces}
            if set(by_tp) != set(range(tp_size)):
                raise RuntimeError(
                    f"TP reconstruction for {name!r} has ranks {sorted(by_tp)}, "
                    f"expected {list(range(tp_size))}"
                )
            value = torch.cat([by_tp[index] for index in range(tp_size)], dim=shard_dim)
        if name in {"embed.embedding.weight", "head.col.linear.weight"}:
            value = value.narrow(0, 0, vocab_size)
        result[name] = value.contiguous()
    return result


def _capture_full_parameter_state(handle: Any, *, gradients: bool) -> dict[str, Any]:
    local = _local_gradient_fragments(handle) if gradients else _local_weight_fragments(handle)
    return _merge_parameter_fragments(_gather_objects(local))


def _capture_full_master_weight_state(handle: Any) -> dict[str, Any]:
    """Reconstruct the canonical FP32 ZeRO-1 master weights on every rank."""

    return _merge_parameter_fragments(_gather_objects(_local_master_weight_fragments(handle)))


def _distributed_parameter_fingerprint(handle: Any) -> dict[str, Any]:
    """Hash every rank-local physical parameter without gathering model tensors.

    A same-topology checkpoint roundtrip must reproduce this manifest exactly.
    Hashing local TP/EP shards avoids materializing a 30B canonical model on
    rank zero while still detecting missing keys, wrong placements, dtypes,
    shapes, or payloads on any rank.
    """

    import torch
    import torch.distributed as dist

    entries = []
    for name, parameter in sorted(
        _named_model_parameters(handle).values(), key=lambda item: item[0]
    ):
        value = parameter.detach().cpu().contiguous()
        byte_view = value.reshape(-1).view(torch.uint8)
        entries.append(
            {
                "name": name,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "num_bytes": int(value.numel() * value.element_size()),
                "sha256": hashlib.sha256(byte_view.numpy().tobytes()).hexdigest(),
            }
        )
        del byte_view, value
    local = {
        "rank": dist.get_rank() if dist.is_available() and dist.is_initialized() else 0,
        "parameters": entries,
    }
    peers = _gather_objects(local)
    encoded = json.dumps(peers, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "rank_count": len(peers),
        "parameter_counts": [len(peer["parameters"]) for peer in peers],
        "num_bytes": [
            sum(int(entry["num_bytes"]) for entry in peer["parameters"]) for peer in peers
        ],
        "rank_manifests": peers,
    }


def _distributed_state_fingerprint(
    value: Any,
    *,
    label: str,
    local_metadata: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    # Keep parity.mlite importable on EOS login nodes that do not have Torch;
    # continuity helpers intentionally import Torch only once a runtime exists.
    from mor_mlite.parity.continuity import state_fingerprint

    """Hash rank-local nested state and gather only compact digest metadata."""

    import torch.distributed as dist

    local = {
        "rank": dist.get_rank() if dist.is_available() and dist.is_initialized() else 0,
        "label": label,
        **state_fingerprint(value),
    }
    if local_metadata is not None:
        for key, item in local_metadata.items():
            if key in local:
                raise ValueError(f"state fingerprint metadata key {key!r} is reserved")
            if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                raise ValueError(
                    f"state fingerprint metadata {key!r} must be a non-negative integer"
                )
            local[key] = item
    peers = _gather_objects(local)
    encoded = json.dumps(peers, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "rank_count": len(peers),
        "tensor_counts": [int(peer["tensor_count"]) for peer in peers],
        "tensor_bytes": [int(peer["tensor_bytes"]) for peer in peers],
        "rank_fingerprints": peers,
    }


def _tensor_statistics(value: Any) -> tuple[int, int]:
    """Count Torch tensor leaves without copying their payloads off device."""

    import torch

    if isinstance(value, torch.Tensor):
        return 1, int(value.numel() * value.element_size())
    if isinstance(value, Mapping):
        count = 0
        num_bytes = 0
        for key, item in value.items():
            key_count, key_bytes = _tensor_statistics(key)
            item_count, item_bytes = _tensor_statistics(item)
            count += key_count + item_count
            num_bytes += key_bytes + item_bytes
        return count, num_bytes
    if isinstance(value, (list, tuple)):
        count = 0
        num_bytes = 0
        for item in value:
            item_count, item_bytes = _tensor_statistics(item)
            count += item_count
            num_bytes += item_bytes
        return count, num_bytes
    return 0, 0


def _canonical_optimizer_step(value: Any, *, location: str) -> int:
    """Return one representation-independent Adam step for fingerprinting.

    MCore checkpoint loading may materialize the same optimizer step as a
    Python integer or as a scalar tensor with a different dtype/device.  Those
    are the same optimizer state, so the fingerprint must compare their
    numeric value rather than their container representation.
    """

    import torch

    if isinstance(value, torch.Tensor):
        if value.numel() != 1 or value.dtype == torch.bool or value.is_complex():
            raise TypeError(f"optimizer step at {location} must be a scalar numeric tensor")
        value = value.detach().cpu().item()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"optimizer step at {location} must be a non-negative scalar")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0 or not numeric.is_integer():
        raise TypeError(
            f"optimizer step at {location} must be a non-negative integer-valued scalar"
        )
    return int(value)


def _optimizer_fingerprint_payload(handle: Any) -> tuple[dict[str, Any], dict[str, int]]:
    """Expose the actual ZeRO-1 master shards and Adam state for hashing.

    MCore's outer ``ChainedOptimizer.state_dict()`` contains common metadata
    but can legitimately have no tensor payload.  The checkpoint tensors live
    in each distributed-optimizer leaf's native optimizer: its ``param_groups``
    hold the owned FP32 master shards and ``state`` holds Adam moments/steps.
    MCore may reorder optimizer groups while loading a checkpoint, so group
    indexes are not stable identities.  Bind each state entry instead to the
    canonical model-parameter name and this rank's owned flattened interval.
    The group options remain attached to that parameter, preserving optimizer
    semantics without relying on process-local object IDs or construction
    order.
    """

    import torch

    optimizer = getattr(handle, "_optimizer", None)
    if optimizer is None:
        raise RuntimeError("optimizer fingerprint requires a training MLite handle")

    context = _parameter_context(handle)
    names = _named_model_parameters(handle)
    parameter_records: list[dict[str, Any]] = []
    record_keys: set[tuple[Any, ...]] = set()
    master_parameter_count = 0
    master_parameter_bytes = 0
    adam_moment_tensor_count = 0
    adam_moment_tensor_bytes = 0
    optimizer_step_count = 0
    leaves = _optimizer_leaves(optimizer)
    for leaf_index, leaf in enumerate(leaves):
        inner_optimizer = getattr(leaf, "optimizer", None)
        param_groups = getattr(inner_optimizer, "param_groups", None)
        state = getattr(inner_optimizer, "state", None)
        gbuf_ranges = getattr(leaf, "gbuf_ranges", None)
        group_map = getattr(leaf, "model_param_group_index_map", None)
        if inner_optimizer is None or not isinstance(param_groups, list):
            raise TypeError("MLite distributed-optimizer leaf has no native optimizer param_groups")
        if not isinstance(state, Mapping):
            raise TypeError("MLite native optimizer state must be a mapping")
        if gbuf_ranges is None or not isinstance(group_map, Mapping):
            raise TypeError(
                "MLite distributed-optimizer leaf is missing gbuf_ranges or its "
                "model-to-main parameter map"
            )

        group_steps: dict[int, bool] = {}
        canonical_group_options: dict[int, dict[str, Any]] = {}
        inner_parameter_ids: set[int] = set()
        for group_index, group in enumerate(param_groups):
            if not isinstance(group, Mapping):
                raise TypeError("MLite native optimizer parameter group must be a mapping")
            parameters = group.get("params")
            if not isinstance(parameters, (list, tuple)):
                raise TypeError("MLite native optimizer parameter group has no parameter list")
            group_has_step = "step" in group
            group_steps[group_index] = group_has_step
            group_options = {key: item for key, item in group.items() if key != "params"}
            if group_has_step:
                group_options["step"] = _canonical_optimizer_step(
                    group["step"], location=f"leaf {leaf_index} group {group_index}"
                )
            canonical_group_options[group_index] = group_options
            for main_parameter in parameters:
                if not isinstance(main_parameter, torch.Tensor):
                    raise TypeError("MLite native optimizer parameter is not a tensor")
                marker = id(main_parameter)
                if marker in inner_parameter_ids:
                    raise RuntimeError("one optimizer leaf contains a duplicate main parameter")
                inner_parameter_ids.add(marker)

        leaf_record_count = 0
        seen_intervals: set[tuple[int, int, int]] = set()
        for buffer_ranges in gbuf_ranges:
            for bucket_ranges in buffer_ranges.values():
                for bucket_range in bucket_ranges:
                    for model_parameter, ranges in bucket_range["param_map"].items():
                        parameter_range = ranges["param"]
                        start = int(parameter_range.start)
                        end = int(parameter_range.end)
                        interval_marker = (id(model_parameter), start, end)
                        if interval_marker in seen_intervals:
                            raise RuntimeError(
                                "distributed optimizer exposed a duplicate optimizer-state interval"
                            )
                        seen_intervals.add(interval_marker)
                        named = names.get(id(model_parameter))
                        if named is None:
                            raise RuntimeError(
                                "distributed optimizer range references an unnamed model parameter"
                            )
                        name, _ = named
                        canonical, is_expert = _canonical_parameter_name(
                            name,
                            ep_rank=context["ep_rank"],
                            ep_size=context["ep_size"],
                            num_experts=context["num_experts"],
                        )
                        try:
                            group_index, group_order = group_map[model_parameter]
                            group = param_groups[group_index]
                            main_parameter = group["params"][group_order]
                        except (KeyError, IndexError, TypeError) as error:
                            raise RuntimeError(
                                f"cannot resolve optimizer state for {canonical!r}"
                            ) from error
                        if not isinstance(main_parameter, torch.Tensor):
                            raise TypeError(f"optimizer shard for {canonical!r} is not a tensor")
                        if id(main_parameter) not in inner_parameter_ids:
                            raise RuntimeError(
                                f"optimizer shard for {canonical!r} is absent from param_groups"
                            )
                        if main_parameter.dtype != torch.float32:
                            raise RuntimeError(
                                f"optimizer shard for {canonical!r} has dtype "
                                f"{main_parameter.dtype}, expected torch.float32"
                            )
                        if not 0 <= start < end <= model_parameter.numel():
                            raise RuntimeError(
                                f"invalid optimizer state range [{start}, {end}) for "
                                f"{canonical!r} with {model_parameter.numel()} elements"
                            )
                        if main_parameter.numel() != end - start:
                            raise RuntimeError(
                                f"optimizer shard for {canonical!r} contains "
                                f"{main_parameter.numel()} elements, expected {end - start}"
                            )

                        parameter_state = state.get(main_parameter, {})
                        if not isinstance(parameter_state, Mapping):
                            raise TypeError(
                                "MLite native optimizer per-parameter state is not a mapping"
                            )
                        if parameter_state:
                            for moment_name in ("exp_avg", "exp_avg_sq"):
                                moment = parameter_state.get(moment_name)
                                if not isinstance(moment, torch.Tensor):
                                    raise TypeError(
                                        f"Adam state for {canonical!r} has no {moment_name} tensor"
                                    )
                                if moment.dtype != torch.float32:
                                    raise RuntimeError(
                                        f"Adam {moment_name} for {canonical!r} has dtype "
                                        f"{moment.dtype}, expected torch.float32"
                                    )
                                if moment.shape != main_parameter.shape:
                                    raise RuntimeError(
                                        f"Adam {moment_name} for {canonical!r} has shape "
                                        f"{tuple(moment.shape)}, expected "
                                        f"{tuple(main_parameter.shape)}"
                                    )
                                adam_moment_tensor_count += 1
                                adam_moment_tensor_bytes += int(
                                    moment.numel() * moment.element_size()
                                )
                            if "step" in parameter_state:
                                _canonical_optimizer_step(
                                    parameter_state["step"], location=canonical
                                )
                            elif not group_steps[group_index]:
                                raise RuntimeError(
                                    f"Adam state for {canonical!r} has no optimizer step"
                                )

                        record_key = (
                            canonical,
                            start,
                            end,
                            bool(is_expert),
                            context["tp_rank"],
                            tuple(model_parameter.shape),
                        )
                        if record_key in record_keys:
                            raise RuntimeError(
                                f"optimizer fingerprint has duplicate canonical key {record_key}"
                            )
                        record_keys.add(record_key)
                        canonical_parameter_state = dict(parameter_state)
                        if "step" in canonical_parameter_state:
                            canonical_parameter_state["step"] = _canonical_optimizer_step(
                                canonical_parameter_state["step"], location=canonical
                            )
                        parameter_records.append(
                            {
                                "name": canonical,
                                "is_expert": is_expert,
                                "tp_rank": context["tp_rank"],
                                "shape": tuple(model_parameter.shape),
                                "numel": model_parameter.numel(),
                                "start": start,
                                "end": end,
                                "native_optimizer_type": (
                                    f"{type(inner_optimizer).__module__}."
                                    f"{type(inner_optimizer).__qualname__}"
                                ),
                                "group_options": canonical_group_options[group_index],
                                "master_parameter": main_parameter,
                                "optimizer_state": canonical_parameter_state,
                            }
                        )
                        if "step" in parameter_state or group_steps[group_index]:
                            # Count the resolved step once per owned parameter record.
                            # MCore may materialize an additional empty param group on
                            # restore; counting raw groups makes an otherwise identical
                            # optimizer fingerprint depend on that container detail.
                            optimizer_step_count += 1
                        leaf_record_count += 1
                        master_parameter_count += 1
                        master_parameter_bytes += int(
                            main_parameter.numel() * main_parameter.element_size()
                        )
        if leaf_record_count != len(inner_parameter_ids):
            raise RuntimeError(
                f"optimizer leaf {leaf_index} canonicalized {leaf_record_count} parameter "
                f"intervals but owns {len(inner_parameter_ids)} main parameters"
            )

    parameter_records.sort(
        key=lambda record: (
            record["name"],
            record["start"],
            record["end"],
            record["is_expert"],
            record["tp_rank"],
            record["shape"],
        )
    )
    if not leaves or master_parameter_count == 0 or master_parameter_bytes == 0:
        raise RuntimeError("optimizer fingerprint found no FP32 master parameter shards")
    if adam_moment_tensor_count == 0 or adam_moment_tensor_bytes == 0:
        raise RuntimeError(
            "optimizer fingerprint found no initialized Adam moment tensors; "
            "fingerprint only after a completed optimizer step"
        )
    if optimizer_step_count == 0:
        raise RuntimeError("optimizer fingerprint found no initialized optimizer steps")
    payload = {"optimizer_parameter_records": parameter_records}
    tensor_count, tensor_bytes = _tensor_statistics(payload)
    expected_minimum = master_parameter_count + adam_moment_tensor_count
    if tensor_count < expected_minimum or tensor_bytes < (
        master_parameter_bytes + adam_moment_tensor_bytes
    ):
        raise AssertionError("optimizer fingerprint payload omitted required training tensors")
    return payload, {
        "optimizer_leaf_count": len(leaves),
        "master_parameter_count": master_parameter_count,
        "master_parameter_bytes": master_parameter_bytes,
        "adam_moment_tensor_count": adam_moment_tensor_count,
        "adam_moment_tensor_bytes": adam_moment_tensor_bytes,
        "optimizer_step_count": optimizer_step_count,
    }


def _distributed_optimizer_fingerprint(handle: Any) -> dict[str, Any]:
    payload, local_metadata = _optimizer_fingerprint_payload(handle)
    result = _distributed_state_fingerprint(
        payload,
        label="optimizer",
        local_metadata=local_metadata,
    )
    array_names = {
        "optimizer_leaf_count": "optimizer_leaf_counts",
        "master_parameter_count": "master_parameter_counts",
        "master_parameter_bytes": "master_parameter_bytes",
        "adam_moment_tensor_count": "adam_moment_tensor_counts",
        "adam_moment_tensor_bytes": "adam_moment_tensor_bytes",
        "optimizer_step_count": "optimizer_step_counts",
    }
    for singular, plural in array_names.items():
        result[plural] = [int(peer[singular]) for peer in result["rank_fingerprints"]]
    return result


def _distributed_rng_fingerprint() -> dict[str, Any]:
    from mor_mlite.parity.continuity import capture_rng_state

    return _distributed_state_fingerprint(capture_rng_state(), label="rng")


def _to_cpu(value: Any) -> Any:
    try:
        import torch
    except ImportError:  # pragma: no cover - run_mlite already requires torch.
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_cpu(item) for item in value]
    return value


def _canonical_token_output(handle: Any, batch: Any, value: Any) -> Any:
    """Use the model protocol's public unpack hook before cross-rank capture."""

    import torch

    protocol = getattr(handle, "_extras", {}).get("protocol")
    unpack = getattr(protocol, "unpack_forward_output", None)
    if not callable(unpack):
        raise TypeError(
            "the pinned MLite protocol has no unpack_forward_output hook for "
            "canonical parity output"
        )
    restored = unpack(handle._model, batch, value)
    if isinstance(restored, torch.Tensor) and getattr(restored, "is_nested", False):
        rows = list(restored.unbind())
        restored = torch.cat(rows, dim=0) if rows else value.new_empty((0, *value.shape[2:]))
    if isinstance(restored, torch.Tensor) and restored.ndim >= 2 and restored.size(0) == 1:
        restored = restored.squeeze(0)
    token_ids = batch.extras.get("global_token_ids")
    if not isinstance(token_ids, torch.Tensor):
        raise TypeError("parity PackedBatch is missing tensor global_token_ids")
    if not isinstance(restored, torch.Tensor) or restored.size(0) != token_ids.numel():
        shape = None if not isinstance(restored, torch.Tensor) else tuple(restored.shape)
        raise RuntimeError(
            "protocol output did not restore one true-length row per global token: "
            f"shape={shape}, tokens={token_ids.numel()}"
        )
    return restored.detach().cpu().clone().contiguous()


def _local_diagnostics(handle: Any, output: Mapping[str, Any], batch: Any) -> dict[str, Any]:
    plans = _find_model_attribute(handle, "last_route_plans") or ()
    traces = _find_model_attribute(handle, "last_round_traces") or ()
    expert_route_traces = _find_model_attribute(handle, "last_moe_expert_route_traces") or ()
    communication = _find_model_attribute(handle, "last_communication") or {}
    ps = getattr(handle, "_parallel_state", None)
    token_ids = batch.extras.get("global_token_ids")
    sample_ids = batch.extras.get("sample_ids")
    original_positions = batch.extras.get("original_position_ids")
    if token_ids is None or sample_ids is None or original_positions is None:
        raise ValueError("parity PackedBatch is missing global token/sample/position metadata")
    captured_output = {
        key: _to_cpu(output[key])
        for key in ("loss", "mor_router_aux_loss", "mor_router_aux_losses")
        if output.get(key) is not None
    }
    # Logits are the cross-backend parity contract.  Native Qwen training also
    # returns target log-probabilities, but retaining both would add an
    # MLite-only tensor to an otherwise exact reference artifact.
    output_key = "logits" if output.get("logits") is not None else "log_probs"
    if output.get(output_key) is not None:
        captured_output[output_key] = _canonical_token_output(handle, batch, output[output_key])
    hidden_diagnostic_keys = [
        key
        for key in output
        if key == "mor_diagnostic_post_merge_hidden"
        or key == "mor_diagnostic_final_hidden"
        or key == "mor_diagnostic_hidden_for_head"
        or key.startswith("mor_diagnostic_end_")
    ]
    for key in sorted(hidden_diagnostic_keys):
        if output.get(key) is not None:
            captured_output[key] = _canonical_token_output(handle, batch, output[key])
    return {
        "dp_rank": int(getattr(handle, "dp_rank", 0)),
        "dp_size": int(getattr(handle, "dp_size", 1)),
        "tp_rank": int(getattr(ps, "tp_rank", 0)),
        "cp_rank": int(getattr(ps, "cp_rank", 0)),
        "output": captured_output,
        "plans": [plan.to_dict() for plan in plans],
        "traces": _to_cpu(traces),
        "expert_route_traces": _to_cpu(expert_route_traces),
        "communication": dict(communication),
        "batch_global_token_ids": _to_cpu(token_ids).reshape(-1),
        "batch_sample_ids": _to_cpu(sample_ids).reshape(-1),
        "batch_original_positions": _to_cpu(original_positions).reshape(-1),
        "lm_scale": float(batch.extras.get("mor_parity_lm_scale", 1.0)),
        "aux_scales": tuple(
            float(value)
            for value in batch.extras.get(
                "mor_parity_aux_scales",
                (1.0,) * len(captured_output.get("mor_router_aux_losses", ())),
            )
        ),
        "local_valid_tokens": int(
            batch.extras.get("mor_parity_local_valid_tokens", token_ids.numel())
        ),
        "global_valid_tokens": int(
            batch.extras.get("mor_parity_global_valid_tokens", token_ids.numel())
        ),
        "global_input_tokens": int(
            batch.extras.get("mor_parity_global_input_tokens", token_ids.numel())
        ),
        "expected_global_token_ids": _to_cpu(
            batch.extras.get("mor_parity_expected_global_token_ids", token_ids)
        ).reshape(-1),
    }


def _gather_diagnostics(handle: Any, output: Mapping[str, Any], batch: Any) -> list[dict[str, Any]]:
    return _gather_objects(_local_diagnostics(handle, output, batch))


def _merge_active_traces(
    peers: Sequence[Mapping[str, Any]], router: DepthRouterConfig
) -> list[dict[str, Any]]:
    import torch

    round_count = len(peers[0].get("traces", ()))
    if any(len(peer.get("traces", ())) != round_count for peer in peers):
        raise RuntimeError("MLite peers reported different recurrent trace counts")
    sample_by_token: dict[int, int] = {}
    position_by_token: dict[int, int] = {}
    for peer in peers:
        for token_id, sample_id, original_position in zip(
            peer["batch_global_token_ids"].tolist(),
            peer["batch_sample_ids"].tolist(),
            peer["batch_original_positions"].tolist(),
            strict=True,
        ):
            token_id, sample_id, original_position = (
                int(token_id),
                int(sample_id),
                int(original_position),
            )
            previous = sample_by_token.setdefault(token_id, sample_id)
            if previous != sample_id:
                raise RuntimeError(f"peers disagree on sample for token {token_id}")
            previous_position = position_by_token.setdefault(token_id, original_position)
            if previous_position != original_position:
                raise RuntimeError(f"peers disagree on original position for token {token_id}")

    merged: list[dict[str, Any]] = []
    for round_index in range(round_count):
        rows: dict[int, torch.Tensor] = {}
        metadata: dict[int, tuple[int, int]] = {}
        candidate_logits: dict[int, torch.Tensor] = {}
        selected_gates: dict[int, torch.Tensor] = {}
        for peer in peers:
            trace = peer["traces"][round_index]
            ids = trace["global_token_ids"].reshape(-1)
            raw_hidden = trace["hidden"]
            if raw_hidden.ndim < 2:
                raise RuntimeError("router trace hidden state must retain its hidden dimension")
            hidden = raw_hidden.reshape(ids.numel(), raw_hidden.shape[-1])
            sample_ids = trace["sample_ids"].reshape(-1)
            positions = trace["original_positions"].reshape(-1)
            for row, token_id_tensor in enumerate(ids):
                token_id = int(token_id_tensor.item())
                candidate = hidden[row]
                if token_id in rows and not torch.equal(rows[token_id], candidate):
                    raise RuntimeError(f"TPxCP peers disagree on hidden state for token {token_id}")
                rows[token_id] = candidate
                metadata[token_id] = (
                    int(sample_ids[row].item()),
                    int(positions[row].item()),
                )
            candidate_ids = trace["candidate_global_token_ids"].reshape(-1)
            logits = trace["router_logits"].reshape(-1)
            if candidate_ids.numel() != logits.numel():
                raise RuntimeError("router trace candidate/logit rows differ")
            for token_id_tensor, logit in zip(candidate_ids, logits, strict=True):
                token_id = int(token_id_tensor.item())
                previous = candidate_logits.get(token_id)
                if previous is not None and not torch.equal(previous, logit):
                    raise RuntimeError(f"TPxCP peers disagree on router logit for token {token_id}")
                candidate_logits[token_id] = logit
            selected_ids = trace["selected_global_token_ids"].reshape(-1)
            gates = trace["selected_gates"].reshape(-1)
            if selected_ids.numel() != gates.numel():
                raise RuntimeError("router trace selected/gate rows differ")
            for token_id_tensor, gate in zip(selected_ids, gates, strict=True):
                token_id = int(token_id_tensor.item())
                previous = selected_gates.get(token_id)
                if previous is not None and not torch.equal(previous, gate):
                    raise RuntimeError(
                        f"TPxCP peers disagree on selected gate for token {token_id}"
                    )
                selected_gates[token_id] = gate
        ordered_ids = sorted(rows)
        ordered_candidates = sorted(candidate_logits)
        ordered_selected = sorted(selected_gates)
        logits = torch.stack([candidate_logits[token_id] for token_id in ordered_candidates])
        scores = torch.sigmoid(logits.float() / router.temperature) * router.alpha
        merged.append(
            {
                "round": round_index,
                "global_token_ids": torch.tensor(ordered_ids, dtype=torch.long),
                "sample_ids": torch.tensor(
                    [metadata[token_id][0] for token_id in ordered_ids],
                    dtype=torch.long,
                ),
                "original_positions": torch.tensor(
                    [metadata[token_id][1] for token_id in ordered_ids],
                    dtype=torch.long,
                ),
                "hidden": torch.stack([rows[token_id] for token_id in ordered_ids]),
                "candidate_global_token_ids": torch.tensor(ordered_candidates, dtype=torch.long),
                "candidate_sample_ids": torch.tensor(
                    [sample_by_token[token_id] for token_id in ordered_candidates],
                    dtype=torch.long,
                ),
                "candidate_original_positions": torch.tensor(
                    [position_by_token[token_id] for token_id in ordered_candidates],
                    dtype=torch.long,
                ),
                "candidate_scores": scores,
                "selected_global_token_ids": torch.tensor(ordered_selected, dtype=torch.long),
                "selected_gates": torch.stack(
                    [selected_gates[token_id] for token_id in ordered_selected]
                ).float(),
            }
        )
    return merged


_EXPERT_ROUTE_CONTEXT_FIELDS = (
    "stage",
    "round_index",
    "stage_layer_index",
    "physical_layer_index",
    "logical_layer_index",
)


def _expert_route_context_key(trace: Mapping[str, Any]) -> tuple[str, int, int, int, int]:
    try:
        key = (
            str(trace["stage"]),
            int(trace["round_index"]),
            int(trace["stage_layer_index"]),
            int(trace["physical_layer_index"]),
            int(trace["logical_layer_index"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("native expert-route trace has invalid context metadata") from exc
    if key[0] not in {"start", "recurrent", "end"}:
        raise ValueError(f"native expert-route trace has invalid stage {key[0]!r}")
    if (key[0] == "recurrent") != (key[1] >= 0):
        raise ValueError("native expert-route trace has invalid stage/round combination")
    if min(key[2:]) < 0:
        raise ValueError("native expert-route trace layer indices must be non-negative")
    return key


def _merge_expert_route_traces(
    peers: Sequence[Mapping[str, Any]],
    active_traces: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Canonicalize native Top-K expert decisions over all topology ranks.

    A context exists on every rank even when that rank owns no real token in
    the corresponding active layout.  Real global token IDs are the join key;
    negative dummy IDs were removed by the model-side probe and are rejected
    here as a second line of defense.
    """

    import torch

    if not peers:
        raise ValueError("native expert-route trace merge requires at least one peer")
    per_peer: list[dict[tuple[str, int, int, int, int], Mapping[str, Any]]] = []
    for peer in peers:
        contexts: dict[tuple[str, int, int, int, int], Mapping[str, Any]] = {}
        for trace in peer.get("expert_route_traces", ()):
            key = _expert_route_context_key(trace)
            if key in contexts:
                raise RuntimeError(f"rank reported duplicate native expert-route context {key}")
            contexts[key] = trace
        per_peer.append(contexts)
    if all(not contexts for contexts in per_peer):
        return []
    expected_contexts = set(per_peer[0])
    if any(set(contexts) != expected_contexts for contexts in per_peer[1:]):
        raise RuntimeError("MLite peers reported different native expert-route contexts")

    all_batch_ids = {
        int(token_id)
        for peer in peers
        for token_id in peer["batch_global_token_ids"].reshape(-1).tolist()
    }
    active_ids_by_round = {
        int(trace["round"]): {
            int(token_id) for token_id in trace["global_token_ids"].reshape(-1).tolist()
        }
        for trace in active_traces
    }
    merged: list[dict[str, Any]] = []
    logical_layers: set[int] = set()
    for key in sorted(expected_contexts, key=lambda value: value[4]):
        stage, round_index, _stage_layer, _physical_layer, logical_layer = key
        if logical_layer in logical_layers:
            raise RuntimeError(f"native expert-route logical layer {logical_layer} is duplicated")
        logical_layers.add(logical_layer)
        rows: dict[
            int,
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ] = {}
        topk: int | None = None
        num_experts: int | None = None
        for contexts in per_peer:
            trace = contexts[key]
            trace_topk = int(trace["topk"])
            trace_num_experts = int(trace["num_experts"])
            if topk is None:
                topk, num_experts = trace_topk, trace_num_experts
            elif (topk, num_experts) != (trace_topk, trace_num_experts):
                raise RuntimeError(f"MLite peers disagree on native router shape for {key}")
            ids = trace["global_token_ids"].reshape(-1)
            indices = trace["topk_indices"].reshape(ids.numel(), trace_topk)
            scores = trace["selected_scores"].reshape(ids.numel(), trace_topk)
            live_scores = trace["live_selected_scores"].reshape(ids.numel(), trace_topk)
            margins = trace["cutoff_logit_margins"].reshape(-1)
            if margins.numel() != ids.numel():
                raise RuntimeError(f"native expert-route margin rows differ for {key}")
            for row, token_id_tensor in enumerate(ids):
                token_id = int(token_id_tensor.item())
                if token_id < 0:
                    raise RuntimeError("dummy token escaped native expert-route probe filtering")
                candidate = (indices[row], scores[row], live_scores[row], margins[row])
                previous = rows.get(token_id)
                if previous is not None and any(
                    not torch.equal(lhs, rhs) for lhs, rhs in zip(previous, candidate, strict=True)
                ):
                    raise RuntimeError(
                        f"topology peers disagree on native expert route for token {token_id}, "
                        f"context={key}"
                    )
                rows[token_id] = candidate
        expected_ids = (
            active_ids_by_round.get(round_index, set()) if stage == "recurrent" else all_batch_ids
        )
        if set(rows) != expected_ids:
            missing = sorted(expected_ids - set(rows))[:8]
            extra = sorted(set(rows) - expected_ids)[:8]
            raise RuntimeError(
                f"native expert-route token coverage differs for {key}: "
                f"missing={missing}, extra={extra}"
            )
        ordered_ids = sorted(rows)
        assert topk is not None and num_experts is not None
        merged.append(
            {
                **dict(zip(_EXPERT_ROUTE_CONTEXT_FIELDS, key, strict=True)),
                "topk": topk,
                "num_experts": num_experts,
                "global_token_ids": torch.tensor(ordered_ids, dtype=torch.long),
                "topk_indices": torch.stack([rows[token_id][0] for token_id in ordered_ids]),
                "selected_scores": torch.stack(
                    [rows[token_id][1] for token_id in ordered_ids]
                ).float(),
                "live_selected_scores": torch.stack(
                    [rows[token_id][2] for token_id in ordered_ids]
                ).float(),
                "cutoff_logit_margins": torch.stack(
                    [rows[token_id][3] for token_id in ordered_ids]
                ).float(),
            }
        )
    return merged


def _representative_peers(peers: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    import torch

    coordinates = {(int(peer.get("tp_rank", 0)), int(peer.get("cp_rank", 0))) for peer in peers}
    dp_size = int(peers[0].get("dp_size", 1))
    representatives = []
    for dp_rank in range(dp_size):
        group = [peer for peer in peers if int(peer["dp_rank"]) == dp_rank]
        group.sort(
            key=lambda peer: (
                int(peer.get("tp_rank", 0)),
                int(peer.get("cp_rank", 0)),
            )
        )
        group_coordinates = {
            (int(peer.get("tp_rank", 0)), int(peer.get("cp_rank", 0))) for peer in group
        }
        if group_coordinates != coordinates or len(group) != len(coordinates):
            raise RuntimeError(f"dense-DP rank {dp_rank} is missing TP/CP diagnostic peers")
        representative = next(
            (
                peer
                for peer in group
                if int(peer.get("tp_rank", 0)) == 0 and int(peer.get("cp_rank", 0)) == 0
            ),
            None,
        )
        if representative is None:
            raise RuntimeError(f"dense-DP rank {dp_rank} has no TP=0/CP=0 output representative")
        for peer in group:
            if not torch.equal(
                peer["batch_global_token_ids"],
                representative["batch_global_token_ids"],
            ):
                raise RuntimeError(f"dense-DP rank {dp_rank} TP/CP peers saw different token IDs")
            output_keys = set(peer["output"]) | set(representative["output"])
            for key in output_keys:
                lhs = peer["output"].get(key)
                rhs = representative["output"].get(key)
                if isinstance(lhs, torch.Tensor) and isinstance(rhs, torch.Tensor):
                    equal = torch.equal(lhs, rhs)
                else:
                    equal = lhs == rhs
                if not equal:
                    raise RuntimeError(f"dense-DP rank {dp_rank} TP/CP peers disagree on {key}")
        representatives.append(representative)
    global_token_owners: dict[int, int] = {}
    expected_global_tokens: int | None = None
    expected_global_token_ids: set[int] | None = None
    for representative in representatives:
        dp_rank = int(representative["dp_rank"])
        global_input_tokens = int(representative["global_input_tokens"])
        if expected_global_tokens is None:
            expected_global_tokens = global_input_tokens
        elif global_input_tokens != expected_global_tokens:
            raise RuntimeError("dense-DP replicas disagree on the global input-token count")
        peer_expected_ids = {
            int(value) for value in representative["expected_global_token_ids"].tolist()
        }
        if len(peer_expected_ids) != global_input_tokens:
            raise RuntimeError("dense-DP expected global token-ID metadata is malformed")
        if expected_global_token_ids is None:
            expected_global_token_ids = peer_expected_ids
        elif peer_expected_ids != expected_global_token_ids:
            raise RuntimeError("dense-DP replicas disagree on the expected global token IDs")
        for token_id_value in representative["batch_global_token_ids"].tolist():
            token_id = int(token_id_value)
            previous_owner = global_token_owners.get(token_id)
            if previous_owner is not None:
                raise RuntimeError(
                    "dense-DP token shards overlap: "
                    f"global token {token_id} appears on ranks {previous_owner} and {dp_rank}"
                )
            global_token_owners[token_id] = dp_rank
    if expected_global_tokens is None or len(global_token_owners) != expected_global_tokens:
        raise RuntimeError(
            "dense-DP token shards do not completely cover the global batch: "
            f"observed={len(global_token_owners)}, expected={expected_global_tokens}"
        )
    if set(global_token_owners) != expected_global_token_ids:
        missing = sorted((expected_global_token_ids or set()) - set(global_token_owners))
        unexpected = sorted(set(global_token_owners) - (expected_global_token_ids or set()))
        raise RuntimeError(
            "dense-DP token shards differ from the expected global token-ID universe: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return representatives


def _merge_token_output(representatives: Sequence[Mapping[str, Any]], key: str) -> Any | None:
    import torch

    values = [peer["output"].get(key) for peer in representatives]
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise RuntimeError(f"only some dense-DP replicas returned {key}")
    rows: dict[int, Any] = {}
    for peer, value in zip(representatives, values, strict=True):
        ids = peer["batch_global_token_ids"].tolist()
        if value.size(0) != len(ids):
            raise RuntimeError(f"canonical {key} is not token-aligned")
        for row, token_id_value in enumerate(ids):
            token_id = int(token_id_value)
            candidate = value[row]
            previous = rows.get(token_id)
            if previous is not None and not torch.equal(previous, candidate):
                raise RuntimeError(f"dense-DP replicas disagree on {key} token {token_id}")
            rows[token_id] = candidate
    return torch.stack([rows[token_id] for token_id in sorted(rows)])


def _merge_plan_rows(plans: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    token_fields = (
        "sample_ids",
        "original_positions",
        "global_token_ids",
        "source_tp_ranks",
        "source_cp_ranks",
        "source_local_rows",
        "target_tp_ranks",
        "target_cp_ranks",
        "target_local_rows",
        "selected_gates",
    )
    entries = []
    margins: dict[str, float | None] = {}
    for plan in plans:
        for sample_id, margin in plan["cutoff_score_margins"].items():
            decoded_margin = math.inf if margin is None else float(margin)
            previous = margins.get(sample_id)
            decoded_previous = math.inf if previous is None else float(previous)
            if sample_id in margins and decoded_previous != decoded_margin:
                raise RuntimeError(f"dense-DP route margins disagree for sample {sample_id}")
            margins[sample_id] = None if math.isinf(decoded_margin) else decoded_margin
        for row, padding in enumerate(plan["padding_mask"]):
            if bool(padding):
                continue
            entries.append({field: plan[field][row] for field in token_fields})
    entries.sort(
        key=lambda row: (
            int(row["sample_ids"]),
            int(row["original_positions"]),
            int(row["global_token_ids"]),
        )
    )
    global_ids = [int(row["global_token_ids"]) for row in entries]
    if len(global_ids) != len(set(global_ids)):
        raise RuntimeError("dense-DP route shards contain duplicate global token IDs")
    samples = sorted({int(row["sample_ids"]) for row in entries})
    cumulative = [0]
    for sample_id in samples:
        cumulative.append(
            cumulative[-1] + sum(int(row["sample_ids"]) == sample_id for row in entries)
        )
    first = plans[0]
    return {
        "schema_version": first["schema_version"],
        "round": first["round"],
        "mode": first["mode"],
        **{field: [row[field] for row in entries] for field in token_fields},
        "active_cu_seqlens": cumulative,
        "padding_mask": [False] * len(entries),
        "cutoff_score_margins": margins,
    }


def _merge_diagnostics(
    peers: Sequence[Mapping[str, Any]],
    *,
    router: DepthRouterConfig,
    training: bool,
) -> dict[str, Any]:
    import torch

    representatives = _representative_peers(peers)
    dp_size = len(representatives)
    traces = _merge_active_traces(peers, router)
    expert_route_traces = _merge_expert_route_traces(peers, traces)
    round_count = len(representatives[0].get("plans", ()))
    if any(len(peer.get("plans", ())) != round_count for peer in peers):
        raise RuntimeError("MLite peers reported different RoutePlan counts")
    plans = [
        _merge_plan_rows([peer["plans"][round_index] for peer in representatives])
        for round_index in range(round_count)
    ]
    for round_index, plan in enumerate(plans):
        coordinate_routes = []
        coordinates = sorted({(int(peer["tp_rank"]), int(peer["cp_rank"])) for peer in peers})
        for coordinate in coordinates:
            ids = []
            for peer in peers:
                if (int(peer["tp_rank"]), int(peer["cp_rank"])) == coordinate:
                    ids.extend(peer["plans"][round_index]["global_token_ids"])
            coordinate_routes.append(sorted(int(token_id) for token_id in ids))
        plan["peer_selected_global_token_ids"] = coordinate_routes
        trace = traces[round_index]
        plan["candidate_global_token_ids"] = trace["candidate_global_token_ids"].tolist()
        plan["candidate_sample_ids"] = trace["candidate_sample_ids"].tolist()
        plan["candidate_original_positions"] = trace["candidate_original_positions"].tolist()
        plan["candidate_scores"] = trace["candidate_scores"].tolist()

    global_lm = None
    global_aux_by_round: list[Any] | None = None
    for peer in representatives:
        local_total = peer["output"].get("loss")
        local_aux = peer["output"].get("mor_router_aux_loss")
        local_aux_by_round = peer["output"].get("mor_router_aux_losses")
        if local_total is None:
            continue
        local_lm = local_total - local_aux if training and local_aux is not None else local_total
        lm_contribution = local_lm.float() * (float(peer["lm_scale"]) / dp_size)
        global_lm = lm_contribution if global_lm is None else global_lm + lm_contribution
        if local_aux_by_round is not None:
            local_aux_by_round = local_aux_by_round.reshape(-1).float()
            aux_scales = tuple(float(value) for value in peer["aux_scales"])
            if local_aux_by_round.numel() != len(aux_scales):
                raise RuntimeError(
                    "per-round router losses and dense-DP auxiliary scales differ in length"
                )
            if global_aux_by_round is None:
                global_aux_by_round = [
                    value * (scale / dp_size)
                    for value, scale in zip(local_aux_by_round, aux_scales, strict=True)
                ]
            else:
                if len(global_aux_by_round) != local_aux_by_round.numel():
                    raise RuntimeError("dense-DP replicas reported different router round counts")
                for round_index, (value, scale) in enumerate(
                    zip(local_aux_by_round, aux_scales, strict=True)
                ):
                    global_aux_by_round[round_index] = global_aux_by_round[round_index] + value * (
                        scale / dp_size
                    )
        elif training and local_aux is not None:
            raise RuntimeError("training diagnostics are missing per-round router losses")
    global_aux_losses = (
        torch.stack(global_aux_by_round) if global_aux_by_round is not None else None
    )
    global_aux = global_aux_losses.sum() if global_aux_losses is not None else None
    output: dict[str, Any] = {}
    if global_lm is not None:
        output["lm_loss"] = global_lm
        output["loss"] = (
            global_lm + global_aux if training and global_aux is not None else global_lm
        )
    if global_aux is not None:
        output["mor_router_aux_loss"] = global_aux
        output["mor_router_aux_losses"] = global_aux_losses
    for key in ("log_probs", "logits"):
        merged_output = _merge_token_output(representatives, key)
        if merged_output is not None:
            output[key] = merged_output
    hidden_diagnostic_keys = sorted(
        {
            key
            for peer in representatives
            for key in peer["output"]
            if key == "mor_diagnostic_post_merge_hidden"
            or key == "mor_diagnostic_final_hidden"
            or key == "mor_diagnostic_hidden_for_head"
            or key.startswith("mor_diagnostic_end_")
        }
    )
    for key in hidden_diagnostic_keys:
        merged_output = _merge_token_output(representatives, key)
        if merged_output is not None:
            output[key] = merged_output
    communication: dict[str, Any] = {}
    for peer in peers:
        for key, value in peer.get("communication", {}).items():
            if isinstance(value, (int, float)):
                communication[key] = max(value, communication.get(key, value))
            else:
                communication.setdefault(key, value)
    return {
        "output": output,
        "plans": plans,
        "traces": traces,
        "expert_route_traces": expert_route_traces,
        "communication": communication,
    }


def _record_capture(
    capture: Mapping[str, Any],
    *,
    step: int,
    microbatch: int,
    prefix: str = "",
    tensors: dict[str, Any],
    routes: list[dict[str, Any]],
) -> None:
    import torch

    base = f"{prefix}step_{step:03d}/mb_{microbatch:03d}"
    output = capture["output"]
    total_loss = output.get("loss")
    lm_loss = output.get("lm_loss")
    aux_loss = output.get("mor_router_aux_loss")
    aux_losses = output.get("mor_router_aux_losses")
    if total_loss is not None:
        tensors[f"loss/{base}/total"] = total_loss.reshape(()).float()
    if lm_loss is not None:
        tensors[f"loss/{base}/lm"] = lm_loss.reshape(()).float()
    if aux_loss is not None:
        tensors[f"loss/{base}/aux"] = aux_loss.reshape(()).float()
    if aux_losses is not None:
        for round_index, round_loss in enumerate(aux_losses.reshape(-1)):
            tensors[f"loss/{base}/aux_round_{round_index}"] = round_loss.reshape(()).float()
    for key in ("log_probs", "logits"):
        if output.get(key) is not None:
            tensors[f"forward/{base}/{key}"] = output[key]
    hidden_diagnostic_names = {
        "mor_diagnostic_post_merge_hidden": "post_merge_hidden",
        "mor_diagnostic_final_hidden": "final_hidden",
        "mor_diagnostic_hidden_for_head": "hidden_for_head",
    }
    hidden_diagnostic_names.update(
        {
            key: key.removeprefix("mor_diagnostic_")
            for key in output
            if key.startswith("mor_diagnostic_end_")
        }
    )
    for source_key, artifact_key in sorted(hidden_diagnostic_names.items()):
        if output.get(source_key) is not None:
            tensors[f"forward/{base}/{artifact_key}"] = output[source_key]

    full_hidden_by_id: dict[int, Any] = {}
    first_ids: list[int] | None = None
    for round_index, trace in enumerate(capture.get("traces", ())):
        ids = [int(value) for value in trace["global_token_ids"].tolist()]
        if first_ids is None:
            first_ids = ids
        for row, token_id in enumerate(ids):
            full_hidden_by_id[token_id] = trace["hidden"][row]
        if first_ids is not None and all(token_id in full_hidden_by_id for token_id in first_ids):
            tensors[f"forward/{base}/hidden_round_{round_index}"] = torch.stack(
                [full_hidden_by_id[token_id] for token_id in first_ids]
            )
        tensors[f"forward/{base}/router_scores_round_{round_index}"] = trace["candidate_scores"]
        tensors[f"forward/{base}/selected_gates_round_{round_index}"] = trace["selected_gates"]

    for trace in capture.get("expert_route_traces", ()):
        stage = str(trace["stage"])
        logical_layer = int(trace["logical_layer_index"])
        physical_layer = int(trace["physical_layer_index"])
        round_index = int(trace["round_index"])
        round_label = f"round_{round_index:03d}" if round_index >= 0 else "non_recurrent"
        context = f"logical_{logical_layer:03d}/{stage}/{round_label}/physical_{physical_layer:03d}"
        root = f"expert_route/{base}/{context}"
        tensors[f"{root}/global_token_ids"] = trace["global_token_ids"]
        tensors[f"{root}/topk_indices"] = trace["topk_indices"]
        tensors[f"{root}/selected_scores"] = trace["selected_scores"]
        tensors[f"{root}/live_selected_scores"] = trace["live_selected_scores"]
        tensors[f"{root}/cutoff_logit_margins"] = trace["cutoff_logit_margins"]

    for plan in capture.get("plans", ()):
        raw = dict(plan)
        raw.update(
            {
                "phase": prefix.rstrip("/") or "train",
                "step": step,
                "microbatch": microbatch,
            }
        )
        routes.append(raw)


def _make_native_batches(
    config: MLiteRunConfig,
    *,
    handle: Any,
    architecture: MoRArchitectureConfig,
    step: int,
    replay: Mapping[tuple[int, int], tuple[Any, ...]],
    expert_replay: Mapping[tuple[int, int], Mapping[int, Any]],
    training: bool,
    replay_fallback_step: int | None = None,
    partition_dp_size: int | None = None,
    partition_dp_rank: int | None = None,
) -> list[Any]:
    import torch

    from mor_mlite.data import as_mlite_packed_batch, make_synthetic_batch

    actual_dp_size = int(handle.dp_size)
    actual_dp_rank = int(handle.dp_rank)
    if (partition_dp_size is None) != (partition_dp_rank is None):
        raise ValueError("partition DP size and rank overrides must be provided together")
    if partition_dp_size is not None:
        if actual_dp_size != 1 or actual_dp_rank != 0:
            raise ValueError("serial reference partitioning requires an actual one-rank DP group")
        dp_size = int(partition_dp_size)
        dp_rank = int(partition_dp_rank)
    else:
        dp_size = actual_dp_size
        dp_rank = actual_dp_rank
    if not 0 <= dp_rank < dp_size:
        raise RuntimeError(f"invalid MLite dense-DP rank {dp_rank}/{dp_size}")
    partitions = _balanced_sample_partitions(config.seq_lens, dp_size)
    scales = _dp_objective_scales(config.seq_lens, partitions, architecture, training=training)
    local_partition = partitions[dp_rank]
    local_scales = scales[dp_rank]
    device = torch.device("cuda", torch.cuda.current_device())
    batches = []
    for microbatch in range(config.num_microbatches):
        global_batch = make_synthetic_batch(
            seq_lens=config.seq_lens,
            vocab_size=257,
            seed=config.seed + step * 100 + microbatch,
            extreme_routing=microbatch == config.num_microbatches - 1,
        )
        batch = _slice_global_batch(global_batch, local_partition, local_scales).to(device)
        # Training parity needs the same canonical logits as its forward-only
        # baseline.  The protocol forwards this opt-in to the model, keeping
        # normal training behavior unchanged outside the parity runner.
        batch.extras["mor_return_full_logits"] = True
        batch.extras["mor_parity_step"] = int(step)
        batch.extras["mor_parity_microbatch"] = int(microbatch)
        if not training:
            # Inference parity compares actual vocabulary logits.  Retaining
            # labels would select the training-only log-probability branch.
            batch.labels = None
            batch.loss_mask = None
        if config.route_mode == "replay":
            plans = replay.get((step, microbatch))
            replay_source_step = step
            if plans is None and replay_fallback_step is not None:
                plans = replay.get((replay_fallback_step, microbatch))
                replay_source_step = replay_fallback_step
            if plans is None:
                raise KeyError(
                    f"replay artifact has no RoutePlan for step={step}, microbatch={microbatch}"
                )
            localized = _localize_replay_plans(plans, local_partition)
            batch.extras["mor_replay_plans"] = tuple(plan.to(device) for plan in localized)
            # Native Qwen Top-K is a second discontinuous decision boundary.
            # Freeze its expert identities only for the forward-only
            # cross-topology oracle. Training always retains the native router
            # and its load-balancing auxiliary gradient.
            if config.forward_only:
                expert_plans = expert_replay.get((replay_source_step, microbatch))
                if expert_plans is None:
                    raise KeyError(
                        "replay artifact has no native-MoE expert plan for "
                        f"step={replay_source_step}, microbatch={microbatch}"
                    )
                expected_logical_layers = set(range(architecture.logical_num_layers))
                if set(expert_plans) != expected_logical_layers:
                    missing = sorted(expected_logical_layers - set(expert_plans))
                    unexpected = sorted(set(expert_plans) - expected_logical_layers)
                    raise ValueError(
                        "native-MoE expert replay must exactly cover every logical layer: "
                        f"missing={missing}, unexpected={unexpected}"
                    )
                batch.extras["mor_expert_replay_plans"] = {
                    logical_layer: plan.to(device) for logical_layer, plan in expert_plans.items()
                }
            batch.extras["mor_parity_replay_source_step"] = replay_source_step
        batches.append(as_mlite_packed_batch(batch))
    return batches


def _claim_serial_reference_callback(
    batch: Any,
    *,
    expected_step: int,
    num_microbatches: int,
    shard: int,
    seen: set[tuple[int, int]],
) -> int:
    """Validate and claim one explicit virtual-DP microbatch callback."""

    extras = getattr(batch, "extras", None)
    if not isinstance(extras, Mapping):
        raise TypeError("serial reference callback batch has no extras mapping")
    callback_step = extras.get("mor_parity_step")
    callback_microbatch = extras.get("mor_parity_microbatch")
    if (
        isinstance(callback_step, bool)
        or not isinstance(callback_step, int)
        or callback_step != expected_step
        or isinstance(callback_microbatch, bool)
        or not isinstance(callback_microbatch, int)
        or not 0 <= callback_microbatch < num_microbatches
    ):
        raise RuntimeError("serial reference emitted invalid step/microbatch metadata")
    key = (int(shard), callback_microbatch)
    if key in seen:
        raise RuntimeError("serial reference emitted a duplicate virtual-shard microbatch")
    seen.add(key)
    return callback_microbatch


def _checkpoint_sidecar(
    architecture: MoRArchitectureConfig,
    *,
    folding_policy: str,
    hf_source: str,
    depth_router: DepthRouterConfig | None = None,
    depth_router_seed: int = 1234,
    cp_transition: str = "magi_direct",
    parallel: MoRParallelConfig | None = None,
) -> dict[str, Any]:
    parallel = parallel or MoRParallelConfig(cp_transition=cp_transition)
    payload = build_checkpoint_metadata(
        architecture=architecture,
        folding_policy=folding_policy,
        depth_router=depth_router or DepthRouterConfig(),
        depth_router_seed=depth_router_seed,
        hf_source=hf_source,
        cp_transition=cp_transition,
        parallel=parallel,
    ).to_dict()
    payload["runtime_api"] = "megatron.lite.runtime"
    return payload


def run_mlite(config: MLiteRunConfig) -> Path:
    """Run forward/backward/update/checkpoint through the pinned MLite Runtime."""

    from mor_mlite.provenance import source_snapshot

    started_source = source_snapshot()
    topology = config.validate()
    preset = load_preset_config(config.preset, config.preset_config)
    checkpoint_initialization = _resolve_checkpoint_initialization(config, topology)
    from mor_mlite.parity.artifacts import require_fresh_artifact_directory

    require_fresh_artifact_directory(config.output)
    try:
        import torch
        import torch.distributed as dist
    except ImportError as exc:  # pragma: no cover - exercised on login nodes.
        raise RuntimeError("the MLite parity backend requires PyTorch") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("the pinned MLite Runtime requires a CUDA device")

    from mor_mlite.determinism import configure_determinism
    from mor_mlite.parity.artifacts import save_artifact
    from mor_mlite.versions import collect_version_manifest

    configure_determinism(config.seed, strict=config.strict)
    init_metadata = (
        checkpoint_initialization.metadata if checkpoint_initialization is not None else None
    )
    architecture = (
        init_metadata.architecture
        if init_metadata is not None
        else (config.architecture or preset.architecture)
    )
    depth_router = (
        init_metadata.depth_router
        if init_metadata is not None
        else (config.depth_router or preset.depth_router)
    )
    depth_router_seed = (
        init_metadata.depth_router_seed if init_metadata is not None else config.seed
    )
    cp_transition = (
        checkpoint_initialization.cp_transition
        if checkpoint_initialization is not None
        else (config.cp_transition or preset.parallel.cp_transition)
    )
    external_run_contract = (
        _external_run_contract(config, cp_transition=cp_transition)
        if config.checkpoint_save_only or config.resume_checkpoint is not None
        else None
    )
    checkpoint_save_receipt: dict[str, Any] | None = None
    checkpoint_save_receipt_digest: str | None = None
    if checkpoint_initialization is not None and checkpoint_initialization.full_training_state:
        from mor_mlite.parity.external_checkpoint import (
            checkpoint_save_receipt_sha256,
            read_checkpoint_save_receipt,
        )

        checkpoint_save_receipt = read_checkpoint_save_receipt(checkpoint_initialization.checkpoint)
        checkpoint_save_receipt_digest = checkpoint_save_receipt_sha256(
            checkpoint_initialization.checkpoint
        )
        expected_contracts = {
            "topology": topology.to_dict(),
            "architecture": architecture.to_dict(),
            "depth_router": depth_router.to_dict(),
            "optimizer": _optimizer_contract(config),
            "run_contract": external_run_contract,
        }
        for field_name, expected in expected_contracts.items():
            if checkpoint_save_receipt[field_name] != expected:
                raise ValueError(
                    f"external checkpoint {field_name} contract differs from this run: "
                    f"saved={checkpoint_save_receipt[field_name]!r}, requested={expected!r}"
                )
    if checkpoint_initialization is not None:
        raw_hf_path = str(checkpoint_initialization.checkpoint)
    elif config.hf_path:
        raw_hf_path = config.hf_path
    elif config.preset == "tiny":
        raw_hf_path = _materialize_tiny_hf(
            config,
            preset=preset,
            architecture=architecture,
        )
    else:
        assert preset.hf_source is not None
        raw_hf_path = preset.hf_source
    hf_resolution = resolve_hf_checkpoint(
        raw_hf_path, require_weights=checkpoint_initialization is None
    )
    hf_path = str(hf_resolution.local_path)
    hf_source = (
        init_metadata.hf_source
        if init_metadata is not None
        else (
            os.environ.get("MOR_HF_SOURCE")
            or config.hf_path
            or preset.hf_source
            or "generated:mor_mlite.topology_independent_tiny.margin_v2"
        )
    )
    runtime_build_config = MLiteRuntimeBuildConfig(
        hf_path=hf_path,
        topology=topology,
        architecture=architecture,
        depth_router=depth_router,
        load_hf_weights=init_metadata is None,
        build_optimizer=not config.forward_only,
        seed=depth_router_seed,
        lr=config.lr,
        adam_eps=config.adam_eps,
        clip_grad=config.clip_grad,
        total_training_steps=(
            config.steps
            + int(config.checkpoint_roundtrip)
            + int(config.checkpoint_save_only)
            + (
                int(checkpoint_save_receipt["saved_step"])
                if checkpoint_save_receipt is not None
                else 0
            )
        ),
        cp_transition=cp_transition,
        route_mode=config.route_mode,
        strict=config.strict,
        # Parity always requests canonical full-vocabulary logits.  Fused CE
        # intentionally elides those logits, so it cannot be used here.
        cross_entropy_fusion=False,
    )
    session = build_runtime_session(runtime_build_config)
    runtime, handle = session.runtime, session.handle
    init_restored_step: int | None = None
    external_checkpoint_resume: dict[str, Any] | None = None
    if checkpoint_initialization is not None:
        if checkpoint_initialization.full_training_state:
            # The process boundary removes the previous MLite ParallelState,
            # NCCL groups, model, and optimizer before this allocation.  Clear
            # construction scratch immediately before DCP creates Adam shards.
            import gc

            from mor_mlite.parity.external_checkpoint import (
                verify_rng_sidecar_manifest,
            )

            assert checkpoint_save_receipt is not None
            verify_rng_sidecar_manifest(
                checkpoint_initialization.checkpoint,
                checkpoint_save_receipt["rng_sidecars"],
                saved_step=checkpoint_save_receipt["saved_step"],
                rank_count=topology.world_size,
            )
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
        init_restored_step = load_mor_checkpoint(
            runtime,
            handle,
            checkpoint_initialization.checkpoint,
            expected_metadata=checkpoint_initialization.runtime_metadata,
            strict_runtime=checkpoint_initialization.full_training_state,
            load_rng=checkpoint_initialization.full_training_state,
            load_model=True,
            load_optimizer=checkpoint_initialization.full_training_state,
        )
        if checkpoint_initialization.full_training_state:
            assert checkpoint_save_receipt is not None
            assert checkpoint_save_receipt_digest is not None
            if init_restored_step != checkpoint_save_receipt["saved_step"]:
                raise RuntimeError(
                    "external checkpoint restored an unexpected global step: "
                    f"{init_restored_step} != {checkpoint_save_receipt['saved_step']}"
                )
            restored_fingerprint = _distributed_parameter_fingerprint(handle)
            expected_fingerprint = checkpoint_save_receipt["parameter_fingerprint"]
            if restored_fingerprint["sha256"] != expected_fingerprint["sha256"]:
                raise RuntimeError(
                    "fresh-process checkpoint restore changed distributed physical parameters"
                )
            restored_optimizer_fingerprint = _distributed_optimizer_fingerprint(handle)
            expected_optimizer_sha256 = checkpoint_save_receipt["optimizer_fingerprint"]["sha256"]
            restored_optimizer_sha256 = restored_optimizer_fingerprint["sha256"]
            if restored_optimizer_sha256 != expected_optimizer_sha256:
                raise RuntimeError(
                    "fresh-process checkpoint restore changed optimizer state: "
                    f"expected {expected_optimizer_sha256}, restored "
                    f"{restored_optimizer_sha256}"
                )
            restored_rng_fingerprint = _distributed_rng_fingerprint()
            if (
                restored_rng_fingerprint["sha256"]
                != checkpoint_save_receipt["rng_fingerprint"]["sha256"]
            ):
                raise RuntimeError("fresh-process checkpoint restore changed RNG state")
            external_checkpoint_resume = {
                "status": "passed",
                "checkpoint": str(checkpoint_initialization.checkpoint),
                "save_receipt_sha256": checkpoint_save_receipt_digest,
                "restored_step": init_restored_step,
                "expected_parameter_sha256": expected_fingerprint["sha256"],
                "restored_parameter_sha256": restored_fingerprint["sha256"],
                "restored_optimizer_sha256": restored_optimizer_sha256,
                "restored_rng_sha256": restored_rng_fingerprint["sha256"],
                "optimizer_loaded": True,
                "rng_loaded": True,
                "run_contract": external_run_contract,
            }
    replay = _route_index(config.replay_from)
    expert_replay = _expert_route_replay_index(config.replay_from)
    rank = dist.get_rank() if dist.is_initialized() else 0
    tensors: dict[str, Any] = {}
    routes: list[dict[str, Any]] = []
    optimizer_steps: list[dict[str, Any]] = []
    last_communication: dict[str, Any] = {}
    gradient_sync_steps: list[dict[str, Any]] = []
    model_structure = _assert_recurrent_parameters_registered_once(handle, architecture)
    gradient_sync_probe = _install_gradient_sync_probe(handle, forward_only=config.forward_only)
    default_tiny = load_preset_config("tiny")
    tiny_model_matches = dict(preset.model) == dict(default_tiny.model)
    tiny_margin_profile_enabled = config.preset == "tiny" and tiny_model_matches
    capture_full_state = tiny_margin_profile_enabled and architecture == preset.architecture
    diagnostic_capture_enabled = _set_mor_diagnostic_capture(handle, enabled=True)
    # Forward-only parity is precisely where a topology-induced native MoE
    # Top-K branch can otherwise be mistaken for an attention/head error.
    # Capture those detached decisions for every preset; training retains the
    # smaller tiny-only policy so this oracle never inflates production-like
    # Qwen optimizer runs.
    expert_route_probe_enabled = _set_moe_expert_route_probe(
        handle, enabled=capture_full_state or config.forward_only
    )
    expected_parameter_names: frozenset[str] = frozenset()
    last_post_step_state: dict[str, Any] | None = None
    if capture_full_state:
        initial_state = _capture_full_parameter_state(handle, gradients=False)
        expected_parameter_names = frozenset(initial_state)
        if rank == 0:
            tensors.update({f"initial/{name}": value for name, value in initial_state.items()})

    mode: Callable[[Any], Any] = runtime.eval_mode if config.forward_only else runtime.train_mode

    def execute_step(
        step: int,
        *,
        prefix: str = "",
        replay_fallback_step: int | None = None,
        record_artifacts: bool = True,
    ) -> None:
        nonlocal gradient_sync_probe, last_communication, last_post_step_state
        captures: list[dict[str, Any]] = []
        serial_reference = config.reference_dp_shards > 1
        serial_peer_groups: list[list[dict[str, Any]]] = [
            [] for _ in range(config.num_microbatches)
        ]
        serial_state = {"shard": 0}
        serial_callbacks: set[tuple[int, int]] = set()

        def loss_fn(output, _batch, *_context):
            if not isinstance(output, Mapping):
                raise TypeError("Qwen3-MoE MoR protocol output must be a mapping")
            if config.forward_only:
                objective = output.get("logits")
                if not isinstance(objective, torch.Tensor):
                    raise ValueError("Qwen3-MoE MoR forward-only output is missing tensor logits")
                loss = objective.float().sum() * 0.0
            else:
                loss = output.get("loss")
                if not isinstance(loss, torch.Tensor):
                    raise ValueError("Qwen3-MoE MoR protocol output is missing tensor loss")
                if not isinstance(output.get("logits"), torch.Tensor):
                    raise ValueError(
                        "Qwen3-MoE MoR parity training output is missing full logits; "
                        "the protocol must honor batch.extras['mor_return_full_logits']"
                    )
            if record_artifacts:
                # This must run on every TP/CP rank: the protocol unpack hook
                # performs the collectives needed to recover canonical true rows.
                peers = _gather_diagnostics(handle, output, _batch)
                if serial_reference:
                    callback_microbatch = _claim_serial_reference_callback(
                        _batch,
                        expected_step=step,
                        num_microbatches=config.num_microbatches,
                        shard=serial_state["shard"],
                        seen=serial_callbacks,
                    )
                    if len(peers) != 1:
                        raise RuntimeError("serial reference requires one actual topology rank")
                    peer = dict(peers[0])
                    peer["dp_rank"] = serial_state["shard"]
                    peer["dp_size"] = config.reference_dp_shards
                    if rank == 0:
                        serial_peer_groups[callback_microbatch].append(peer)
                else:
                    merged = _merge_diagnostics(
                        peers,
                        router=depth_router,
                        training=not config.forward_only,
                    )
                    if rank == 0:
                        captures.append(merged)
            if config.forward_only:
                return loss, {}
            aux_loss = output.get("mor_router_aux_loss")
            aux_losses = output.get("mor_router_aux_losses")
            if not isinstance(aux_loss, torch.Tensor) or not isinstance(aux_losses, torch.Tensor):
                raise TypeError(
                    "Qwen3-MoE MoR training output is missing aggregate/per-round router aux loss"
                )
            aux_losses = aux_losses.reshape(-1)
            aux_scales = tuple(float(value) for value in _batch.extras["mor_parity_aux_scales"])
            if aux_losses.numel() != len(aux_scales):
                raise ValueError(
                    "router aux round count differs from the dense-DP objective scale count"
                )
            from mor_mlite.objective import ObjectiveScale, apply_objective

            return apply_objective(
                output, ObjectiveScale(float(_batch.extras["mor_parity_lm_scale"]), aux_scales)
            ), {}

        phase = prefix.rstrip("/") or "train"
        global_step = f"{phase}/step_{step:03d}"
        gradient_sync_probe.begin_step(global_step)
        runtime.zero_grad(handle)
        result = None
        reference_shards = range(config.reference_dp_shards) if serial_reference else (None,)
        for reference_shard in reference_shards:
            if reference_shard is None:
                partition_kwargs: dict[str, int] = {}
            else:
                partition_kwargs = {
                    "partition_dp_size": config.reference_dp_shards,
                    "partition_dp_rank": reference_shard,
                }
                serial_state["shard"] = reference_shard
            native_batches = _make_native_batches(
                config,
                handle=handle,
                architecture=architecture,
                step=step,
                replay=replay,
                expert_replay=expert_replay,
                training=not config.forward_only,
                replay_fallback_step=replay_fallback_step,
                **partition_kwargs,
            )
            result = runtime.forward_backward(
                handle,
                iter(native_batches),
                loss_fn,
                num_microbatches=config.num_microbatches,
                forward_only=config.forward_only,
                router_replay=None,
            )
            if (
                serial_reference
                and sum(shard == reference_shard for shard, _microbatch in serial_callbacks)
                != config.num_microbatches
            ):
                raise RuntimeError(
                    "serial reference loss callback count differs: "
                    f"shard={reference_shard}, expected={config.num_microbatches}"
                )
        gradient_sync_steps.append(
            _merge_gradient_sync_step_reports(
                _gather_objects(gradient_sync_probe.finish_step(global_step))
            )
        )
        if serial_reference and rank == 0 and record_artifacts:
            if any(len(peers) != config.reference_dp_shards for peers in serial_peer_groups):
                raise RuntimeError("serial reference did not capture every virtual DP shard")
            captures.extend(
                _merge_diagnostics(
                    peers,
                    router=depth_router,
                    training=False,
                )
                for peers in serial_peer_groups
            )
        if result is None:
            raise RuntimeError("MLite forward_backward did not execute")
        if getattr(result, "model_output", None) is None:
            raise TypeError("MLite forward_backward did not return ForwardResult")
        if rank == 0 and record_artifacts:
            if len(captures) != config.num_microbatches:
                raise RuntimeError(
                    "MLite loss callback did not run once per microbatch: "
                    f"{len(captures)} != {config.num_microbatches}"
                )
            for microbatch, capture in enumerate(captures):
                _record_capture(
                    capture,
                    step=step,
                    microbatch=microbatch,
                    prefix=prefix,
                    tensors=tensors,
                    routes=routes,
                )
            for capture in captures:
                for key, value in capture["communication"].items():
                    if isinstance(value, (int, float)):
                        last_communication[key] = max(value, last_communication.get(key, value))
                    else:
                        last_communication.setdefault(key, value)

        if config.forward_only:
            return
        master_before_state: dict[str, Any] | None = None
        if capture_full_state and record_artifacts:
            gradient_state = _capture_full_parameter_state(handle, gradients=True)
            if frozenset(gradient_state) != expected_parameter_names:
                raise RuntimeError(
                    "complete MLite gradient reconstruction did not cover exactly "
                    "the initialized physical parameter set"
                )
            if rank == 0:
                tensors.update(
                    {
                        f"gradient/{prefix}step_{step:03d}/{name}": value
                        for name, value in gradient_state.items()
                    }
                )
            master_before_state = _capture_full_master_weight_state(handle)
            if frozenset(master_before_state) != expected_parameter_names:
                raise RuntimeError(
                    "complete FP32 master-weight reconstruction did not cover exactly "
                    "the initialized physical parameter set"
                )
        updated, grad_norm, num_zeros = runtime.optimizer_step(handle)
        if not bool(updated):
            raise RuntimeError(f"MLite optimizer did not update at {prefix}step {step}")
        if not math.isfinite(float(grad_norm)):
            raise RuntimeError(
                f"MLite optimizer returned a non-finite grad norm at {prefix}step {step}: "
                f"{grad_norm}"
            )
        if updated:
            runtime.lr_scheduler_step(handle)
        optimizer_steps.append(
            {
                "phase": prefix.rstrip("/") or "train",
                "step": step,
                "updated": bool(updated),
                "grad_norm": float(grad_norm),
                "num_zeros": None if num_zeros is None else int(num_zeros),
            }
        )
        if rank == 0 and record_artifacts:
            tensors[f"gradient/{prefix}step_{step:03d}/global_norm"] = torch.tensor(
                float(grad_norm), dtype=torch.float32
            )
        if capture_full_state and record_artifacts:
            master_after_state = _capture_full_master_weight_state(handle)
            if frozenset(master_after_state) != expected_parameter_names:
                raise RuntimeError(
                    "post-step FP32 master-weight reconstruction did not cover exactly "
                    "the initialized physical parameter set"
                )
            if master_before_state is None:
                raise AssertionError("pre-step FP32 master weights were not captured")
            master_updates = {
                name: master_after_state[name] - master_before_state[name]
                for name in expected_parameter_names
            }
            if rank == 0:
                tensors.update(
                    {
                        f"update/{prefix}step_{step:03d}/{name}": value
                        for name, value in master_updates.items()
                    }
                )
            post_step_state = _capture_full_parameter_state(handle, gradients=False)
            if frozenset(post_step_state) != expected_parameter_names:
                raise RuntimeError(
                    "post-step MLite reconstruction did not cover exactly the "
                    "initialized physical parameter set"
                )
            last_post_step_state = post_step_state
            if rank == 0:
                tensors.update(
                    {
                        f"post_step/{prefix}step_{step:03d}/{name}": value
                        for name, value in post_step_state.items()
                    }
                )

    initial_step = (
        int(init_restored_step)
        if checkpoint_initialization is not None
        and checkpoint_initialization.full_training_state
        and init_restored_step is not None
        else 0
    )
    initial_prefix = (
        "resume/"
        if checkpoint_initialization is not None and checkpoint_initialization.full_training_state
        else ""
    )
    with mode(handle):
        for step_offset in range(config.steps):
            execute_step(
                initial_step + step_offset,
                prefix=initial_prefix,
                replay_fallback_step=(initial_step - 1 if initial_prefix else None),
            )

    if external_checkpoint_resume is not None:
        assert checkpoint_save_receipt is not None
        expected_next = checkpoint_save_receipt["uninterrupted_next_step"]
        continued_fingerprints = {
            "parameter": _distributed_parameter_fingerprint(handle),
            "optimizer": _distributed_optimizer_fingerprint(handle),
            "rng": _distributed_rng_fingerprint(),
        }
        for label, expected_key in (
            ("parameter", "parameter_fingerprint"),
            ("optimizer", "optimizer_fingerprint"),
            ("rng", "rng_fingerprint"),
        ):
            actual_sha256 = continued_fingerprints[label]["sha256"]
            expected_sha256 = expected_next[expected_key]["sha256"]
            if actual_sha256 != expected_sha256:
                raise RuntimeError(
                    f"fresh-process {label} state differs after the uninterrupted next "
                    f"step: expected {expected_sha256}, restored {actual_sha256}"
                )
        external_checkpoint_resume.update(
            {
                "continued_step": initial_step,
                "continued_parameter_sha256": continued_fingerprints["parameter"]["sha256"],
                "continued_optimizer_sha256": continued_fingerprints["optimizer"]["sha256"],
                "continued_rng_sha256": continued_fingerprints["rng"]["sha256"],
            }
        )

    restored_step: int | None = None
    checkpoint_next_step = False
    checkpoint_uninterrupted_step = False
    checkpoint_parameter_fingerprint: dict[str, Any] = {"status": "not_run"}
    checkpoint_continuity: dict[str, Any] | None = None
    if config.checkpoint_roundtrip or config.checkpoint_save_only:
        checkpoint = config.output / "runtime-checkpoint"
        checkpoint_metadata = build_checkpoint_metadata(
            architecture=architecture,
            depth_router=depth_router,
            depth_router_seed=depth_router_seed,
            hf_source=hf_source,
            parallel=topology.to_parallel_config(cp_transition=cp_transition),
            folding_policy="mean",
            cp_transition=cp_transition,
        )
        fingerprint_before = _distributed_parameter_fingerprint(handle)
        verify_training_continuity = capture_full_state and not config.forward_only
        saved_fingerprints: dict[str, Any] = {"parameters": fingerprint_before}
        if capture_full_state:
            save_point_state = (
                last_post_step_state if last_post_step_state is not None else initial_state
            )
            # Keep an immutable CPU snapshot: execute_step assigns the live
            # continuation result to ``last_post_step_state``.
            checkpoint_saved_full_state = {
                name: value.detach().cpu().clone() for name, value in save_point_state.items()
            }
        else:
            checkpoint_saved_full_state = None
        save_mor_checkpoint(
            runtime,
            handle,
            checkpoint,
            step=config.steps,
            metadata=checkpoint_metadata,
            base_hf_path=hf_path,
            save_rng=True,
            save_model=True,
            save_optimizer=not config.forward_only,
        )
        if config.checkpoint_save_only:
            from mor_mlite.parity.external_checkpoint import (
                collect_rng_sidecar_manifest,
                write_checkpoint_save_receipt,
            )

            rng_sidecars = collect_rng_sidecar_manifest(
                checkpoint,
                saved_step=config.steps,
            )
            saved_optimizer_fingerprint = _distributed_optimizer_fingerprint(handle)
            saved_rng_fingerprint = _distributed_rng_fingerprint()
            # Keep the live process as an independent continuation oracle.  It
            # executes the exact step that a fresh process will run after
            # loading this save point, without retaining parity tensors for the
            # 30B model.  Exact distributed fingerprints make the certificate
            # stronger than a mere successful reload/update.
            with mode(handle):
                execute_step(
                    config.steps,
                    prefix="uninterrupted/",
                    replay_fallback_step=config.steps - 1,
                    record_artifacts=False,
                )
            uninterrupted_next_step = {
                "step": config.steps,
                "parameter_fingerprint": _distributed_parameter_fingerprint(handle),
                "optimizer_fingerprint": _distributed_optimizer_fingerprint(handle),
                "rng_fingerprint": _distributed_rng_fingerprint(),
                "optimizer_step": dict(optimizer_steps[-1]),
            }
            assert external_run_contract is not None
            run_rank_zero_io(
                lambda: write_checkpoint_save_receipt(
                    checkpoint,
                    saved_step=config.steps,
                    topology=topology.to_dict(),
                    architecture=architecture.to_dict(),
                    depth_router=depth_router.to_dict(),
                    optimizer=_optimizer_contract(config),
                    run_contract=external_run_contract,
                    parameter_fingerprint=fingerprint_before,
                    optimizer_fingerprint=saved_optimizer_fingerprint,
                    rng_fingerprint=saved_rng_fingerprint,
                    rng_sidecars=rng_sidecars,
                    uninterrupted_next_step=uninterrupted_next_step,
                ),
                label="external checkpoint receipt write",
            )
            return config.output
        if verify_training_continuity:
            # The pinned DCP adapter normalizes native optimizer step fields as
            # part of save.  Fingerprint after save so the oracle describes
            # exactly the state serialized to the checkpoint.
            saved_fingerprints.update(
                {
                    "optimizer": _distributed_optimizer_fingerprint(handle),
                    "rng": _distributed_rng_fingerprint(),
                }
            )

        uninterrupted_fingerprints: dict[str, Any] = {}
        if verify_training_continuity:
            # Preserve the live branch as the oracle.  It starts from the exact
            # checkpoint save point and executes the same deterministic global
            # batch that the subsequently restored fresh session will see.
            with mode(handle):
                execute_step(
                    config.steps,
                    prefix="uninterrupted/",
                    replay_fallback_step=config.steps - 1,
                )
            checkpoint_uninterrupted_step = True
            uninterrupted_fingerprints = {
                "parameters": _distributed_parameter_fingerprint(handle),
                "optimizer": _distributed_optimizer_fingerprint(handle),
                "rng": _distributed_rng_fingerprint(),
            }

        # Prove the checkpoint is sufficient on its own.  Reusing the same
        # handle could let a missing checkpoint tensor retain its pre-save
        # value, so release the original model/optimizer and restore into a
        # freshly constructed handle that does not read the HF checkpoint.
        import gc

        del mode
        del handle
        del runtime
        del session
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        session = build_runtime_session(
            replace(
                runtime_build_config,
                hf_path=str(checkpoint),
                load_hf_weights=False,
            )
        )
        runtime, handle = session.runtime, session.handle
        # Session construction creates model, gradient-buffer, and NCCL
        # allocations of many different sizes.  Return any dead construction
        # scratch to CUDA before DCP materializes the sharded FP32 master/Adam
        # tensors.  This is required for the 30B H100 checkpoint smoke, where
        # the total state fits but a fragmented caching pool may not have the
        # next small contiguous optimizer shard available.
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        restored_diagnostics_enabled = _set_mor_diagnostic_capture(handle, enabled=True)
        if restored_diagnostics_enabled != diagnostic_capture_enabled:
            raise RuntimeError("fresh checkpoint session changed MoR diagnostic capture support")
        restored_probe_enabled = _set_moe_expert_route_probe(
            handle, enabled=expert_route_probe_enabled
        )
        if restored_probe_enabled != expert_route_probe_enabled:
            raise RuntimeError("fresh checkpoint session changed native expert-route probe support")
        restored_structure = _assert_recurrent_parameters_registered_once(handle, architecture)
        if restored_structure != model_structure:
            raise RuntimeError("fresh checkpoint session changed recurrent parameter registration")
        gradient_sync_probe = _install_gradient_sync_probe(handle, forward_only=config.forward_only)
        mode = runtime.eval_mode if config.forward_only else runtime.train_mode
        restored_step = load_mor_checkpoint(
            runtime,
            handle,
            checkpoint,
            expected_metadata=checkpoint_metadata,
            load_rng=True,
            load_model=True,
            load_optimizer=not config.forward_only,
        )
        if restored_step != config.steps:
            raise RuntimeError(
                f"MLite checkpoint restored step {restored_step}, expected {config.steps}"
            )
        fingerprint_after = _distributed_parameter_fingerprint(handle)
        if fingerprint_after["sha256"] != fingerprint_before["sha256"]:
            raise RuntimeError(
                "MLite checkpoint roundtrip changed at least one rank-local physical "
                "parameter key, shape, dtype, placement, or payload"
            )
        checkpoint_parameter_fingerprint = {
            "status": "passed",
            "sha256": fingerprint_before["sha256"],
            "rank_count": fingerprint_before["rank_count"],
            "parameter_counts": fingerprint_before["parameter_counts"],
            "num_bytes": fingerprint_before["num_bytes"],
        }
        restored_fingerprints: dict[str, Any] = {"parameters": fingerprint_after}
        if verify_training_continuity:
            restored_fingerprints.update(
                {
                    "optimizer": _distributed_optimizer_fingerprint(handle),
                    "rng": _distributed_rng_fingerprint(),
                }
            )
        del fingerprint_before, fingerprint_after
        if capture_full_state:
            restored_state = _capture_full_parameter_state(handle, gradients=False)
            expected_state = checkpoint_saved_full_state
            if expected_state is None:
                raise AssertionError("tiny checkpoint save-point state was not captured")
            if frozenset(restored_state) != frozenset(expected_state) or any(
                not torch.equal(restored_state[name], expected_state[name])
                for name in expected_state
            ):
                raise RuntimeError(
                    "MLite checkpoint roundtrip changed reconstructed physical weights"
                )
        # Exercise the restored model, optimizer, and RNG state through one
        # complete next step.  An older or forward-only replay baseline may
        # lack this step; equal sequence lengths give stable global token IDs,
        # so reuse its final selection while recomputing current gates/losses.
        with mode(handle):
            execute_step(
                config.steps,
                prefix="resume/",
                replay_fallback_step=config.steps - 1,
            )
        checkpoint_next_step = True
        if verify_training_continuity:
            resumed_fingerprints = {
                "parameters": _distributed_parameter_fingerprint(handle),
                "optimizer": _distributed_optimizer_fingerprint(handle),
                "rng": _distributed_rng_fingerprint(),
            }
            if rank == 0:
                from mor_mlite.parity.continuity import (
                    build_checkpoint_continuity_report,
                )

                checkpoint_continuity = build_checkpoint_continuity_report(
                    step=config.steps,
                    precision=config.precision,
                    tensors=tensors,
                    routes=routes,
                    optimizer_steps=optimizer_steps,
                    saved_fingerprints=saved_fingerprints,
                    restored_fingerprints=restored_fingerprints,
                    uninterrupted_fingerprints=uninterrupted_fingerprints,
                    resumed_fingerprints=resumed_fingerprints,
                )

    if dist.is_initialized():
        dist.barrier()
    gradient_sync_summary = _summarize_gradient_sync_steps(
        gradient_sync_steps, required=not config.forward_only
    )
    last_communication["grad_sync_probe"] = gradient_sync_summary
    # This is an observed call count at MCore's physical bucket dispatch seam,
    # not a claim of profiler-level visibility into individual NCCL kernels.
    # ``None`` means the real bucket hook was unavailable and fails comparison.
    last_communication["physical_bucket_sync_dispatch_max"] = gradient_sync_summary[
        "physical_bucket_sync_calls_max"
    ]

    def write_final_artifact() -> None:
        effective_dp = config.reference_dp_shards if config.reference_dp_shards > 1 else topology.dp
        sample_partitions = [
            list(indices) for indices in _balanced_sample_partitions(config.seq_lens, effective_dp)
        ]
        metadata = {
            "source_snapshot": started_source,
            "backend": "mlite",
            "runtime_api": "megatron.lite.runtime",
            "megatron_lm_sha": MEGATRON_SHA,
            "preset": config.preset,
            "preset_config_source": str(preset.source),
            "model_config": dict(preset.model),
            "precision": config.precision,
            "strict": config.strict,
            "attention_policy": ATTENTION_POLICY,
            "attention_backend": _attention_backend(topology, strict=config.strict),
            "route_mode": config.route_mode,
            "seed": config.seed,
            "steps": config.steps,
            "num_microbatches": config.num_microbatches,
            "forward_only": config.forward_only,
            "reference_dp_shards": config.reference_dp_shards,
            "reference_execution": (
                "serial-whole-sequence-dense-dp-partitions"
                if config.reference_dp_shards > 1
                else None
            ),
            "topology": topology.to_dict(),
            "architecture": architecture.to_dict(),
            "depth_router": depth_router.to_dict(),
            "depth_router_seed": depth_router_seed,
            "cp_transition": cp_transition,
            "parallel": topology.to_parallel_config(cp_transition=cp_transition).to_dict(),
            "init_checkpoint": (
                str(config.init_checkpoint.resolve())
                if config.init_checkpoint is not None
                else None
            ),
            "resume_checkpoint": (
                str(config.resume_checkpoint.resolve())
                if config.resume_checkpoint is not None
                else None
            ),
            "init_checkpoint_restored_step": init_restored_step,
            "init_checkpoint_source_metadata": (
                init_metadata.to_dict() if init_metadata is not None else None
            ),
            "checkpoint_restored_step": restored_step,
            "checkpoint_next_step": checkpoint_next_step,
            "checkpoint_uninterrupted_step": checkpoint_uninterrupted_step,
            "checkpoint_parameter_fingerprint": checkpoint_parameter_fingerprint,
            "checkpoint_continuity": checkpoint_continuity,
            "external_checkpoint_resume": external_checkpoint_resume,
            "optimizer": _optimizer_contract(config),
            "resume_replay_policy": (
                "exact-step-or-final-selection-with-current-gates"
                if (config.checkpoint_roundtrip or config.resume_checkpoint is not None)
                and config.route_mode == "replay"
                else None
            ),
            "optimizer_steps": optimizer_steps,
            "versions": collect_version_manifest(),
            "communication": last_communication,
            "model_structure": model_structure,
            "global_batch": {
                "sequence_lengths": list(config.seq_lens),
                "sharded_across_dense_dp": True,
                "partition_policy": "deterministic-longest-first-whole-sequence",
                "loss_weighting": "exact-token-count-with-dp-average-compensation",
                "single_rank_reference_dp_shards": config.reference_dp_shards,
                "effective_dense_dp": effective_dp,
                "sample_partitions": sample_partitions,
            },
            "parameter_capture": {
                "enabled": capture_full_state,
                "initial_weights": capture_full_state,
                "complete_gradients": capture_full_state and not config.forward_only,
                "fp32_master_updates": capture_full_state and not config.forward_only,
                "post_step_weights": capture_full_state and not config.forward_only,
                "gradient_source": (
                    "dist-opt-gbuf-ranges"
                    if capture_full_state and not config.forward_only
                    else None
                ),
                "update_source": (
                    "dist-opt-fp32-master-shards-after-minus-before"
                    if capture_full_state and not config.forward_only
                    else None
                ),
                "scope": "tiny-only-full-physical-state",
            },
            "initialized_parameters": sorted(expected_parameter_names),
            "diagnostic_capture": {
                "enabled": diagnostic_capture_enabled,
                "round_hidden_states": diagnostic_capture_enabled,
                "recurrent_qkv_hooks": diagnostic_capture_enabled,
                "production_default": False,
            },
            "expert_route_probe": {
                "enabled": expert_route_probe_enabled,
                "scope": "forward-parity-or-tiny-native-qwen-topk-before-token-dispatch",
                "identity_contract": "canonical-global-token-id-plus-logical-layer",
                "topk_indices": "dispatch-visible-exact",
                "cutoff_margin": "minimum-selected-minus-maximum-unselected-raw-logit",
                "dummy_padding_excluded": True,
                "captured_contexts": sum(
                    name.endswith("/topk_indices")
                    for name in tensors
                    if name.startswith("expert_route/")
                ),
            },
            "expert_route_replay": {
                "enabled": bool(config.forward_only and config.route_mode == "replay"),
                "scope": "forward-only-cross-topology-oracle",
                "identity_key": "logical-layer-plus-global-token-id",
                "expert_ids": "baseline-bitwise-exact",
                "selected_scores": "baseline-forward-values-with-live-gradient-ste",
                "live_selected_scores": "diagnostic-only-live-router-values",
                "training_native_router_unchanged": True,
                "source": (
                    str(config.replay_from.resolve())
                    if config.forward_only
                    and config.route_mode == "replay"
                    and config.replay_from is not None
                    else None
                ),
            },
            "synthetic_weight_profile": (
                "native-moe-residual-anchor-margin-v2" if tiny_margin_profile_enabled else None
            ),
            "hf_source": hf_source,
            "hf_resolved_path": hf_path,
            "magi_attention_version": MAGI_VERSION,
        }
        save_artifact(config.output, metadata=metadata, tensors=tensors, routes=routes)

    run_rank_zero_io(write_final_artifact, label="parity artifact write")
    return config.output


__all__ = [
    "MLiteRunConfig",
    "MLiteRuntimeBuildConfig",
    "MLiteRuntimeSession",
    "build_runtime_session",
    "run_mlite",
]
