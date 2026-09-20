"""HF and distributed-checkpoint hooks for folded Qwen3-MoE MoR weights.

This module is imported only by the MLite protocol.  It deliberately builds on
the pinned native Qwen3-MoE ``WeightSpec`` instead of duplicating its GQA and
expert tensor transforms.
"""

from __future__ import annotations

import copy
import json
import math
import re
from pathlib import Path
from typing import Any

import torch
from megatron.lite.model.qwen3_moe.config import Qwen3MoEConfig
from megatron.lite.model.qwen3_moe.lite.checkpoint import (
    PLACEMENT_FN as _QWEN_PLACEMENT_FN,
)
from megatron.lite.model.qwen3_moe.lite.checkpoint import Qwen3MoEWeightSpec
from megatron.lite.primitive.ckpt.hf_weights import (
    SafeTensorReader,
    split_dim,
    split_gate_up,
    unwrap_model,
)
from safetensors import safe_open
from torch import nn
from torch.distributed.tensor import DTensor

from mor_mlite.config import MoRArchitectureConfig
from mor_mlite.hf import validate_local_hf_checkpoint

from .metadata import physical_to_logical_layer_map, validate_folding_policy

_NATIVE_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
_HF_LAYER_RE = re.compile(r"model\.layers\.(\d+)\.")
_SUPPORTED_MODEL_TYPES = {"qwen3_moe"}
_SUPPORTED_TARGET_DTYPES = {torch.float32, torch.bfloat16}
_SUPPORTED_SOURCE_DTYPES = {torch.float16, torch.bfloat16, torch.float32}
_SUPPORTED_SOURCE_SAFETENSOR_DTYPES = {"BF16", "F16", "F32"}
_INTEGER_CONFIG_FIELDS = (
    "num_hidden_layers",
    "hidden_size",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "vocab_size",
    "num_experts",
    "num_experts_per_tok",
    "moe_intermediate_size",
    "max_position_embeddings",
    "num_nextn_predict_layers",
)
_FLOAT_CONFIG_FIELDS = ("rope_theta", "rms_norm_eps", "router_aux_loss_coef")


def _pad_vocab_for_tp(vocab_size: int, tp_size: int) -> int:
    """Match MLite's pure padding rule without importing its CUDA/TE linear module."""

    divisor = math.lcm(128, tp_size)
    return ((vocab_size + divisor - 1) // divisor) * divisor


def physical_model_config(
    logical_config: Qwen3MoEConfig, architecture: MoRArchitectureConfig
) -> Qwen3MoEConfig:
    """Return a Qwen config describing only registered physical layers."""

    physical = copy.deepcopy(logical_config)
    physical.num_hidden_layers = architecture.physical_num_layers
    physical.layer_types = ["full_attention"] * architecture.physical_num_layers
    physical.num_nextn_predict_layers = 0
    physical._validate()
    return physical


def install_mor_metadata(
    model_config: Qwen3MoEConfig,
    architecture: MoRArchitectureConfig,
    folding_policy: str,
) -> None:
    """Attach runtime-only metadata to the architecture config passed by MLite."""

    if model_config.num_hidden_layers != architecture.logical_num_layers:
        raise ValueError(
            "HF logical depth does not match the MoR architecture: "
            f"{model_config.num_hidden_layers} != {architecture.logical_num_layers}"
        )
    model_config._mor_architecture = architecture.to_dict()
    model_config._mor_folding_policy = validate_folding_policy(folding_policy)


def _metadata_from_config(
    model_config: Qwen3MoEConfig,
) -> tuple[MoRArchitectureConfig, str]:
    raw_architecture = getattr(model_config, "_mor_architecture", None)
    if raw_architecture is None:
        raise ValueError("Qwen3-MoE MoR config is missing _mor_architecture metadata")
    architecture = (
        raw_architecture
        if isinstance(raw_architecture, MoRArchitectureConfig)
        else MoRArchitectureConfig.from_dict(raw_architecture)
    )
    policy = validate_folding_policy(getattr(model_config, "_mor_folding_policy", "mean"))
    return architecture, policy


def _native_layer_index(name: str) -> int | None:
    match = _NATIVE_LAYER_RE.search(name)
    return int(match.group(1)) if match else None


def _replace_hf_layer(name: str, logical_index: int) -> str:
    return _HF_LAYER_RE.sub(f"model.layers.{logical_index}.", name, count=1)


class Qwen3MoEMoRWeightSpec:
    """Fold logical HF layers into one registered start/recurrent/end stack.

    ``mean`` averages corresponding recurrent rounds in FP32, matching the
    fixed first-release initialization contract.  The first release is
    deliberately import-only.  Distributed checkpoints
    keep the physical recurrent stack and depth routers; no reverse HF export
    is exposed because expanding shared weights would lose the MoR execution
    contract.
    """

    def __init__(
        self,
        logical_config: Qwen3MoEConfig,
        architecture: MoRArchitectureConfig,
        *,
        folding_policy: str = "mean",
    ) -> None:
        if logical_config.num_hidden_layers != architecture.logical_num_layers:
            raise ValueError("logical config depth must equal architecture.logical_num_layers")
        self.logical_config = logical_config
        self.architecture = architecture
        self.folding_policy = validate_folding_policy(folding_policy)
        self.physical_config = physical_model_config(logical_config, architecture)
        self._base = Qwen3MoEWeightSpec(self.physical_config)
        self._base_weight_map = self._base.weight_map()
        self._layer_map = physical_to_logical_layer_map(architecture)

    @property
    def num_experts(self) -> int:
        return self._base.num_experts

    def _load_sources(self, physical_index: int) -> tuple[int, ...]:
        return self._layer_map[physical_index]

    def weight_map(self) -> dict[str, list[str]]:
        """Return the flattened compatibility map expected by MLite tooling.

        The MoR loader below deliberately does *not* pass this flattened map to
        MLite's generic loader: doing so materializes every recurrent source at
        once.  :meth:`source_groups` is the authoritative load plan.
        """

        return {
            native_name: [name for group in self.source_groups(native_name) for name in group]
            for native_name in self._base_weight_map
        }

    def source_groups(self, native_name: str) -> tuple[tuple[str, ...], ...]:
        """Return one native-transform input group per selected logical layer.

        A QKV group contains ``(q, k, v)`` and an expert FC1 group contains
        ``(gate, up)``.  Keeping these group boundaries is what permits true
        recurrent folding with only one logical layer's inputs resident at a
        time.
        """

        try:
            hf_names = self._base_weight_map[native_name]
        except KeyError as exc:
            raise KeyError(f"unknown native Qwen weight {native_name!r}") from exc
        physical_index = _native_layer_index(native_name)
        if physical_index is None:
            return (tuple(hf_names),)
        return tuple(
            tuple(_replace_hf_layer(name, logical_index) for name in hf_names)
            for logical_index in self._load_sources(physical_index)
        )

    def load_weight_map(self, base_model, ps, logical_state_keys) -> dict[str, list[str]]:
        """Pinned generic-loader extension point; the plan has physical keys only."""

        del base_model, ps, logical_state_keys
        return self.weight_map()

    def validate_load(self, ps) -> None:
        """Prevent accidental use through MLite's all-sources generic loader."""

        del ps
        raise RuntimeError(
            "Qwen3MoEMoRWeightSpec must be loaded through "
            "mor_mlite.qwen3_moe_mor.checkpoint.load_hf_weights so recurrent "
            "sources are streamed one logical group at a time"
        )

    def hf_to_native(self, native_name: str, hf_tensors: list[torch.Tensor]) -> torch.Tensor:
        """Compatibility transform for callers that already materialized inputs.

        Production import uses :func:`load_hf_weights` and never takes this
        all-sources-at-once path.  Retaining it keeps the WeightSpec useful to
        mapping/debug tooling and makes its mathematical contract explicit.
        """

        try:
            group_size = len(self._base_weight_map[native_name])
        except KeyError as exc:
            raise KeyError(f"unknown native Qwen weight {native_name!r}") from exc
        if group_size < 1 or len(hf_tensors) % group_size:
            raise ValueError(
                f"invalid folded HF tensor group for {native_name!r}: "
                f"{len(hf_tensors)} tensors / group {group_size}"
            )
        source_count = len(hf_tensors) // group_size
        if source_count == 1:
            return self._base.hf_to_native(native_name, hf_tensors[:group_size])

        accumulator: torch.Tensor | None = None
        output_dtype: torch.dtype | None = None
        for offset in range(0, len(hf_tensors), group_size):
            converted = self._base.hf_to_native(
                native_name, hf_tensors[offset : offset + group_size]
            )
            if accumulator is None:
                output_dtype = converted.dtype
                accumulator = converted.to(dtype=torch.float32).clone()
            else:
                accumulator.add_(converted.to(dtype=torch.float32))
            del converted
        assert accumulator is not None and output_dtype is not None
        accumulator.div_(source_count)
        return accumulator.to(dtype=output_dtype)

    def native_to_hf(
        self, native_name: str, tensor: torch.Tensor
    ) -> list[tuple[str, torch.Tensor]]:
        if native_name.startswith("depth_routers."):
            # Depth routers are MoR-only parameters stored by distributed
            # checkpoints and described by mor_config.json, not vanilla Qwen HF.
            return []
        physical_index = _native_layer_index(native_name)
        base_values = self._base.native_to_hf(native_name, tensor)
        if physical_index is None:
            return base_values
        return [
            (_replace_hf_layer(hf_name, logical_index), hf_tensor)
            for logical_index in self._layer_map[physical_index]
            for hf_name, hf_tensor in base_values
        ]

    def optional_for_load(self, native_name: str) -> str | None:
        if native_name.startswith("depth_routers."):
            return "depth routers are initialized by the MoR implementation"
        return None

    def qkv_spec(self, native_name: str):
        return self._base.qkv_spec(native_name)

    def tp_spec(self, native_name: str):
        if native_name.startswith("depth_routers."):
            return None
        return self._base.tp_spec(native_name)

    def is_expert(self, native_name: str) -> bool:
        return self._base.is_expert(native_name)

    def expert_global_id(self, native_name: str) -> int | None:
        return self._base.expert_global_id(native_name)

    def expert_local_name(self, native_name: str, local_idx: int) -> str:
        return self._base.expert_local_name(native_name, local_idx)


def _spec(model_config: Qwen3MoEConfig) -> Qwen3MoEMoRWeightSpec:
    architecture, policy = _metadata_from_config(model_config)
    return Qwen3MoEMoRWeightSpec(model_config, architecture, folding_policy=policy)


def _validate_hf_checkpoint_directory(path: str, logical_config: Qwen3MoEConfig) -> Path:
    """Validate the standard, local Qwen3-MoE safetensors input contract."""

    root = validate_local_hf_checkpoint(path, require_weights=True)
    config_path = root / "config.json"
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid HF config.json: {config_path}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"HF config.json must contain an object: {config_path}")
    model_type = payload.get("model_type")
    if model_type not in _SUPPORTED_MODEL_TYPES:
        raise ValueError(f"unsupported HF model_type={model_type!r}; expected 'qwen3_moe'")

    normalized = dict(payload)
    if "head_dim" not in normalized:
        try:
            normalized["head_dim"] = int(normalized["hidden_size"]) // int(
                normalized["num_attention_heads"]
            )
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
            raise ValueError("HF config cannot derive head_dim") from exc
    normalized.setdefault("num_nextn_predict_layers", 0)

    rope_parameters = normalized.get("rope_parameters")
    if rope_parameters is not None:
        if not isinstance(rope_parameters, dict):
            raise ValueError("HF rope_parameters must be an object or null")
        unsupported_rope_keys = sorted(set(rope_parameters) - {"rope_theta"})
        if unsupported_rope_keys:
            raise ValueError(
                "HF rope_parameters enables semantics unsupported by Qwen3-MoE "
                f"MoR v1: {unsupported_rope_keys!r}; only an equivalent rope_theta "
                "is accepted"
            )
        if "rope_theta" in rope_parameters:
            nested_theta = rope_parameters["rope_theta"]
            if "rope_theta" in normalized:
                try:
                    top_level_theta = float(normalized["rope_theta"])
                    nested_theta_value = float(nested_theta)
                except (TypeError, ValueError) as exc:
                    raise ValueError("HF rope_theta must be numeric") from exc
                if not math.isclose(
                    top_level_theta, nested_theta_value, rel_tol=0.0, abs_tol=1e-12
                ):
                    raise ValueError(
                        "HF rope_parameters.rope_theta conflicts with top-level rope_theta"
                    )
            else:
                normalized["rope_theta"] = nested_theta

    for key in _INTEGER_CONFIG_FIELDS:
        if key not in normalized:
            raise ValueError(f"HF config.json is missing required field {key!r}")
        try:
            actual = int(normalized[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"HF config field {key!r} must be an integer") from exc
        expected = int(getattr(logical_config, key))
        if actual != expected:
            raise ValueError(f"HF {key}={actual} does not match runtime config {int(expected)}")

    for key in _FLOAT_CONFIG_FIELDS:
        if key not in normalized:
            raise ValueError(f"HF config.json is missing required field {key!r}")
        try:
            actual = float(normalized[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"HF config field {key!r} must be numeric") from exc
        expected = float(getattr(logical_config, key))
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"HF {key}={actual!r} does not match runtime config {expected!r}")

    source_layer_types = normalized.get(
        "layer_types", ["full_attention"] * logical_config.num_hidden_layers
    )
    if list(source_layer_types) != list(logical_config.layer_types):
        raise ValueError("HF layer_types do not match the runtime layer schedule")

    unsupported_semantics = {
        "attention_bias": (normalized.get("attention_bias", False), False),
        "attention_dropout": (float(normalized.get("attention_dropout", 0.0)), 0.0),
        "decoder_sparse_step": (int(normalized.get("decoder_sparse_step", 1)), 1),
        "hidden_act": (normalized.get("hidden_act", "silu"), "silu"),
        "mlp_only_layers": (normalized.get("mlp_only_layers", []), []),
        "norm_topk_prob": (normalized.get("norm_topk_prob", True), True),
        "rope_scaling": (normalized.get("rope_scaling"), None),
        "sliding_window": (normalized.get("sliding_window"), None),
        "tie_word_embeddings": (normalized.get("tie_word_embeddings", False), False),
        "use_sliding_window": (normalized.get("use_sliding_window", False), False),
    }
    mismatches = [
        f"{name}={actual!r} (supported value {expected!r})"
        for name, (actual, expected) in unsupported_semantics.items()
        if actual != expected
    ]
    if mismatches:
        raise ValueError(
            "HF config enables semantics unsupported by Qwen3-MoE MoR v1: " + "; ".join(mismatches)
        )
    if logical_config.num_nextn_predict_layers != 0:
        raise ValueError("Qwen3-MoE MoR v1 HF import does not support MTP weights")
    return root


def _positive_parallel_value(ps, name: str) -> int:
    value = int(getattr(ps, name, 1) or 1)
    if value < 1:
        raise ValueError(f"ParallelState.{name} must be positive, got {value}")
    return value


def _validate_streaming_load_scope(
    base_model: nn.Module,
    spec: Qwen3MoEMoRWeightSpec,
    ps,
) -> None:
    """Fail before reading payloads when the v1 loader cannot be exact."""

    pp_size = _positive_parallel_value(ps, "pp_size")
    etp_size = _positive_parallel_value(ps, "etp_size")
    tp_size = _positive_parallel_value(ps, "tp_size")
    ep_size = _positive_parallel_value(ps, "ep_size")
    if pp_size != 1:
        raise ValueError("Qwen3-MoE MoR streaming import requires PP=1")
    virtual_pipeline_size = getattr(ps, "virtual_pipeline_size", None)
    if virtual_pipeline_size not in (None, 1):
        raise ValueError("Qwen3-MoE MoR streaming import requires VPP=1")
    if etp_size != 1:
        raise ValueError("Qwen3-MoE MoR streaming import requires ETP=1")
    if int(getattr(ps, "pp_rank", 0)) != 0 or int(getattr(ps, "etp_rank", 0)) != 0:
        raise ValueError("PP and ETP ranks must both be zero when their size is one")
    for size_name, rank_name, size in (
        ("TP", "tp_rank", tp_size),
        ("EP", "ep_rank", ep_size),
    ):
        rank = int(getattr(ps, rank_name, 0))
        if not 0 <= rank < size:
            raise ValueError(f"{size_name} rank {rank} is outside [0, {size})")
    if spec.logical_config.num_nextn_predict_layers != 0:
        raise ValueError("Qwen3-MoE MoR v1 does not support MTP")
    if spec.logical_config.num_experts % ep_size:
        raise ValueError(
            f"num_experts={spec.logical_config.num_experts} must be divisible by EP={ep_size}"
        )
    if spec.logical_config.num_attention_heads % tp_size:
        raise ValueError(
            f"num_attention_heads={spec.logical_config.num_attention_heads} must be "
            f"divisible by TP={tp_size}"
        )
    # The pinned native Qwen WeightSpec packs one complete GQA group at a time;
    # its ordinary TP split therefore requires KV groups to divide across TP.
    # MLite's separate replicated-KV path is outside the v1 MoR import scope.
    if spec.logical_config.num_key_value_heads < tp_size:
        raise ValueError("Qwen3-MoE MoR v1 HF import does not support TP replicated-KV loading")
    if spec.logical_config.num_key_value_heads % tp_size:
        raise ValueError(
            f"num_key_value_heads={spec.logical_config.num_key_value_heads} must be "
            f"divisible by TP={tp_size}"
        )

    physical_layers = spec.architecture.physical_num_layers
    if not hasattr(base_model, "layers") or len(base_model.layers) != physical_layers:
        raise ValueError(
            "model does not contain the configured physical MoR layer stack: "
            f"expected {physical_layers} layers"
        )
    layer_indices = tuple(int(index) for index in getattr(base_model, "layer_indices", ()))
    if layer_indices != tuple(range(physical_layers)):
        raise ValueError(
            "PP=VPP=1 MoR import requires every physical layer on the local rank; "
            f"got layer_indices={layer_indices!r}"
        )
    depth_routers = getattr(base_model, "depth_routers", None)
    if depth_routers is None or len(depth_routers) != spec.architecture.num_recursions:
        raise ValueError("model must initialize one depth router per recursion before HF import")

    for name, parameter in base_model.named_parameters(remove_duplicate=False):
        lowered = name.lower()
        if ".parametrizations." in name:
            raise ValueError("Qwen3-MoE MoR v1 HF import does not support QAT")
        if "lora" in lowered or "adapter" in lowered:
            raise ValueError("Qwen3-MoE MoR v1 HF import does not support LoRA/adapters")
        if isinstance(parameter, DTensor):
            raise TypeError("Qwen3-MoE MoR v1 HF import does not support FSDP/DTensor")
        if parameter.is_meta:
            raise ValueError("Qwen3-MoE MoR v1 requires materialized parameters before HF import")
        if parameter.dtype not in _SUPPORTED_TARGET_DTYPES:
            raise ValueError(
                f"unsupported target dtype for {name!r}: {parameter.dtype}; "
                "v1 supports BF16 training and FP32 correctness only"
            )


def _resolve_target_name(name: str, targets: dict[str, torch.Tensor]) -> str | None:
    if name in targets:
        return name
    matches = [candidate for candidate in targets if candidate.endswith(f".{name}")]
    if len(matches) > 1:
        raise RuntimeError(f"ambiguous native target {name!r}: {matches!r}")
    return matches[0] if matches else None


def _source_scale_metadata_names(name: str) -> tuple[str, ...]:
    """Return quantized/scaled aliases recognized by the pinned HF reader."""

    return (
        f"{name}_scale_inv",
        f"{name.removesuffix('.weight')}.scale",
        f"{name}_scale",
        f"{name}_shape",
        f"{name}_packed",
    )


def _preflight_source_tensor_headers(
    checkpoint: Path,
    source_names: tuple[str, ...],
) -> None:
    """Validate every local source from safetensors headers before model mutation.

    ``safe_open`` and ``get_slice(...).get_dtype()`` inspect metadata only; no
    tensor payload is materialized.  This prevents a later FP8/quantized source
    from being discovered after earlier physical parameters were overwritten.
    """

    unique_sources = tuple(dict.fromkeys(source_names))
    if not unique_sources:
        return

    index_path = checkpoint / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = payload["weight_map"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(f"invalid HF safetensors index: {index_path}") from exc
        if not isinstance(weight_map, dict):
            raise TypeError(f"HF safetensors weight_map must be an object: {index_path}")

        missing = [name for name in unique_sources if name not in weight_map]
        if missing:
            raise KeyError(f"HF checkpoint is missing required tensor(s): {missing!r}")
        metadata_keys = set(weight_map)
        grouped: dict[str, list[str]] = {}
        for name in unique_sources:
            shard = weight_map[name]
            if not isinstance(shard, str):
                # validate_local_hf_checkpoint already rejects this; retain a
                # defensive guard because this helper is independently tested.
                raise TypeError(f"invalid shard path for HF tensor {name!r}: {shard!r}")
            grouped.setdefault(shard, []).append(name)
    else:
        grouped = {"model.safetensors": list(unique_sources)}
        metadata_keys = set()

    indexed_scale_metadata = sorted(
        metadata_name
        for source_name in unique_sources
        for metadata_name in _source_scale_metadata_names(source_name)
        if metadata_name in metadata_keys
    )
    if indexed_scale_metadata:
        raise TypeError(
            "HF checkpoint uses unsupported scale metadata for required source tensors: "
            f"{indexed_scale_metadata!r}; MoR v1 accepts unscaled BF16/FP16/FP32 only"
        )

    for shard, shard_sources in grouped.items():
        shard_path = checkpoint / shard
        with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
            shard_keys = set(handle.keys())
            missing_from_shard = [name for name in shard_sources if name not in shard_keys]
            if missing_from_shard:
                raise KeyError(
                    f"HF shard {shard!r} is missing indexed tensor(s): {missing_from_shard!r}"
                )
            scale_metadata = sorted(
                metadata_name
                for source_name in shard_sources
                for metadata_name in _source_scale_metadata_names(source_name)
                if metadata_name in shard_keys
            )
            if scale_metadata:
                raise TypeError(
                    f"HF shard {shard!r} uses unsupported scale metadata "
                    f"{scale_metadata!r}; MoR v1 accepts unscaled BF16/FP16/FP32 only"
                )
            for source_name in shard_sources:
                source_dtype = str(handle.get_slice(source_name).get_dtype())
                if source_dtype not in _SUPPORTED_SOURCE_SAFETENSOR_DTYPES:
                    raise TypeError(
                        f"HF tensor {source_name!r} has unsupported safetensors dtype "
                        f"{source_dtype}; MoR v1 accepts unscaled BF16, FP16, or FP32 only"
                    )


def _read_source_group(
    reader: SafeTensorReader,
    native_name: str,
    source_group: tuple[str, ...],
) -> list[torch.Tensor]:
    """Materialize exactly one logical layer's transform inputs on CPU."""

    tensors: list[torch.Tensor] = []
    for hf_name in source_group:
        try:
            resolved = reader.first_available((hf_name,))
            scale_metadata = tuple(
                name for name in _source_scale_metadata_names(resolved) if reader.has_tensor(name)
            )
            if scale_metadata:
                raise TypeError(
                    f"HF tensor {resolved!r} uses unsupported scale metadata "
                    f"{scale_metadata!r}; MoR v1 accepts unscaled BF16/FP16/FP32 only"
                )
            tensor = reader.get_tensor(resolved, device="cpu")
        except KeyError as exc:
            raise KeyError(
                f"required HF tensor {hf_name!r} for native target {native_name!r} is missing"
            ) from exc
        if tensor.dtype not in _SUPPORTED_SOURCE_DTYPES:
            raise TypeError(
                f"HF tensor {resolved!r} has unsupported dtype {tensor.dtype}; "
                "MoR v1 accepts unscaled BF16, FP16, or FP32 only"
            )
        tensors.append(tensor)
    return tensors


def _fold_source_groups_streaming(
    reader: SafeTensorReader,
    spec: Qwen3MoEMoRWeightSpec,
    native_name: str,
    source_groups: tuple[tuple[str, ...], ...],
) -> torch.Tensor:
    """Transform one logical source group at a time into one FP32 average."""

    if not source_groups:
        raise ValueError(f"native target {native_name!r} has no HF source groups")
    accumulator: torch.Tensor | None = None
    for source_group in source_groups:
        hf_tensors = _read_source_group(reader, native_name, source_group)
        converted = spec._base.hf_to_native(native_name, hf_tensors)
        if not converted.is_floating_point():
            raise TypeError(
                f"native transform for {native_name!r} returned non-floating "
                f"dtype {converted.dtype}"
            )
        contribution = converted.to(device="cpu", dtype=torch.float32).clone()
        if accumulator is None:
            accumulator = contribution
        else:
            if contribution.shape != accumulator.shape:
                raise ValueError(
                    f"logical sources for {native_name!r} produced inconsistent "
                    f"shapes: {tuple(accumulator.shape)} and {tuple(contribution.shape)}"
                )
            accumulator.add_(contribution)
        # Do not let one iteration retain the previous logical layer's Q/K/V or
        # gate/up group while the next layer is read.
        del hf_tensors, converted, contribution
    assert accumulator is not None
    if len(source_groups) > 1:
        accumulator.div_(len(source_groups))
    return accumulator


def _shard_dense_for_target(
    native_name: str,
    tensor: torch.Tensor,
    spec: Qwen3MoEMoRWeightSpec,
    ps,
) -> torch.Tensor:
    """Apply the pinned native Qwen TP layout after FP32 folding."""

    tp_info = spec.tp_spec(native_name)
    if tp_info is None:
        return tensor
    split_dimension, tp_or_etp = tp_info
    if tp_or_etp != 0:
        # Experts take the separate EP path and ETP is rejected above.
        return tensor
    tp_size = int(getattr(ps, "tp_size", 1))
    tp_rank = int(getattr(ps, "tp_rank", 0))
    if "embed" in native_name or "head" in native_name:
        logical_vocab = int(spec.logical_config.vocab_size)
        if tensor.ndim < 1 or tensor.size(0) != logical_vocab:
            rows = tensor.size(0) if tensor.ndim >= 1 else None
            raise ValueError(
                f"HF vocabulary tensor for {native_name!r} must have exactly "
                f"logical vocab_size={logical_vocab} rows before TP padding; got {rows}"
            )
        padded_vocab = _pad_vocab_for_tp(spec.logical_config.vocab_size, tp_size)
        if tensor.size(0) < padded_vocab:
            tensor = torch.cat(
                [
                    tensor,
                    torch.zeros(
                        padded_vocab - tensor.size(0),
                        *tensor.shape[1:],
                        dtype=tensor.dtype,
                        device=tensor.device,
                    ),
                ],
                dim=0,
            )
    if split_dimension == 0 and ("gate_up" in native_name or ".fc1." in native_name):
        return split_gate_up(tensor, tp_rank, tp_size)
    return split_dim(tensor, tp_rank, tp_size, dim=split_dimension)


def _copy_target(
    *,
    native_name: str,
    actual_name: str,
    target: torch.Tensor,
    folded: torch.Tensor,
) -> None:
    """Perform the only target-dtype/device conversion, immediately before copy."""

    if isinstance(target, DTensor):  # defensive: scope validation rejects this earlier.
        raise TypeError("Qwen3-MoE MoR v1 HF import does not support FSDP/DTensor")
    if tuple(folded.shape) != tuple(target.shape):
        raise ValueError(
            f"HF tensor for {native_name!r} has local shape {tuple(folded.shape)}, "
            f"but target {actual_name!r} expects {tuple(target.shape)}"
        )
    converted = folded.to(device=target.device, dtype=target.dtype)
    with torch.no_grad():
        target.copy_(converted)
    del converted


def load_hf_weights(model, path: str, model_config: Qwen3MoEConfig, ps) -> None:
    """Stream a vanilla Qwen3-MoE checkpoint into the physical MoR model.

    Dense parameters are TP-sharded, experts are selected by EP, and DP/CP and
    expert-DP replicas load the same deterministic local shard independently.
    No distributed communication is performed here.  For a recurrent physical
    parameter only one logical source group is resident at a time; each group is
    converted through the pinned native Qwen transform, accumulated in FP32,
    and only then sharded/cast immediately before the destination copy.
    """

    spec = _spec(model_config)
    base_model = unwrap_model(model)
    _validate_streaming_load_scope(base_model, spec, ps)
    checkpoint = _validate_hf_checkpoint_directory(path, model_config)

    targets: dict[str, torch.Tensor] = dict(base_model.named_parameters(remove_duplicate=False))
    targets.update(dict(base_model.named_buffers(remove_duplicate=False)))
    loaded_names: set[str] = set()
    experts_per_rank = spec.num_experts // int(getattr(ps, "ep_size", 1))
    local_expert_start = int(getattr(ps, "ep_rank", 0)) * experts_per_rank

    local_native_names = []
    for native_name in spec._base_weight_map:
        expert_gid = spec.expert_global_id(native_name)
        if expert_gid is None or (
            local_expert_start <= expert_gid < local_expert_start + experts_per_rank
        ):
            local_native_names.append(native_name)
    local_source_names = tuple(
        hf_name
        for native_name in local_native_names
        for source_group in spec.source_groups(native_name)
        for hf_name in source_group
    )
    _preflight_source_tensor_headers(checkpoint, local_source_names)

    with SafeTensorReader(str(checkpoint), device="cpu") as reader:
        # Metadata-only preflight prevents a late missing shard/key from leaving
        # the live model partially overwritten.
        missing_sources = [
            hf_name
            for hf_names in spec.weight_map().values()
            for hf_name in hf_names
            if not reader.has_tensor(hf_name)
        ]
        if missing_sources:
            preview = missing_sources[:8]
            suffix = " ..." if len(missing_sources) > len(preview) else ""
            raise KeyError(
                f"HF checkpoint is missing {len(missing_sources)} required tensor(s): "
                f"{preview!r}{suffix}"
            )

        for native_name in spec._base_weight_map:
            expert_gid = spec.expert_global_id(native_name)
            if expert_gid is None:
                actual_name = _resolve_target_name(native_name, targets)
                if actual_name is None:
                    raise RuntimeError(
                        f"HF native target {native_name!r} is absent from the MoR model"
                    )
            else:
                if not (local_expert_start <= expert_gid < local_expert_start + experts_per_rank):
                    continue
                local_index = expert_gid - local_expert_start
                local_name = spec.expert_local_name(native_name, local_index)
                actual_name = _resolve_target_name(local_name, targets)
                if actual_name is None:
                    raise RuntimeError(
                        f"local expert target {local_name!r} for global expert "
                        f"{expert_gid} is absent from the MoR model"
                    )

            if actual_name in loaded_names:
                raise RuntimeError(f"native target {actual_name!r} is mapped more than once")
            target = targets[actual_name]
            folded = _fold_source_groups_streaming(
                reader,
                spec,
                native_name,
                spec.source_groups(native_name),
            )
            if expert_gid is None:
                folded = _shard_dense_for_target(native_name, folded, spec, ps)
            _copy_target(
                native_name=native_name,
                actual_name=actual_name,
                target=target,
                folded=folded,
            )
            loaded_names.add(actual_name)
            del folded

    # Depth routers are intentionally initialized from initializer_range + seed,
    # not vanilla Qwen weights.  Every other physical model parameter is required.
    missing_targets = sorted(
        name
        for name, _ in base_model.named_parameters(remove_duplicate=False)
        if not name.startswith("depth_routers.") and name not in loaded_names
    )
    if missing_targets:
        raise RuntimeError(
            f"HF import did not initialize every physical MoR parameter: {missing_targets!r}"
        )


def EXPERT_CLASSIFIER(name: str) -> bool:
    return "experts" in name and "router" not in name


def PLACEMENT_FN(param_name: str) -> list[Any]:
    # Base Qwen's fallback is fully replicated, exactly what depth-router
    # weights require; routed expert placements remain unchanged.
    return _QWEN_PLACEMENT_FN(param_name)


__all__ = [
    "EXPERT_CLASSIFIER",
    "PLACEMENT_FN",
    "Qwen3MoEMoRWeightSpec",
    "install_mor_metadata",
    "load_hf_weights",
    "physical_model_config",
]
