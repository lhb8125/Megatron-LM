"""Pure-Python architecture and HF-folding metadata helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from mor_mlite.config import (
    DepthRouterConfig,
    MoRArchitectureConfig,
    MoRParallelConfig,
)

MEGATRON_LM_PINNED_SHA = "5c8315f12a64a7279eec58896af9e74ee3351b74"
HF_FOLDING_METADATA_FILENAME = "mor_config.json"
FOLDING_POLICIES = frozenset({"mean"})


def _exact_json_value_matches(actual: Any, expected: Any) -> bool:
    """Compare derived JSON fields without Python's bool/int/float coercions."""

    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            _exact_json_value_matches(actual[key], expected[key]) for key in expected
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _exact_json_value_matches(left, right)
            for left, right in zip(actual, expected, strict=True)
        )
    return actual == expected


def validate_folding_policy(policy: str) -> str:
    if not isinstance(policy, str):
        raise TypeError(f"folding_policy must be a string, got {policy!r}")
    normalized = policy.lower()
    if normalized not in FOLDING_POLICIES:
        raise ValueError(
            f"folding_policy must be one of {sorted(FOLDING_POLICIES)}, got {policy!r}"
        )
    return normalized


def physical_to_logical_layer_map(
    architecture: MoRArchitectureConfig,
) -> dict[int, tuple[int, ...]]:
    """Map each registered physical decoder layer to its HF logical source(s)."""

    result: dict[int, tuple[int, ...]] = {}
    for physical_index in range(architecture.n_start_layers):
        result[physical_index] = (physical_index,)

    recurrent_physical_start = architecture.n_start_layers
    recurrent_logical_start = architecture.n_start_layers
    for recurrent_index in range(architecture.n_recurrent_layers):
        physical_index = recurrent_physical_start + recurrent_index
        result[physical_index] = tuple(
            recurrent_logical_start + recursion * architecture.n_recurrent_layers + recurrent_index
            for recursion in range(architecture.num_recursions)
        )

    end_physical_start = architecture.n_start_layers + architecture.n_recurrent_layers
    end_logical_start = (
        architecture.n_start_layers + architecture.n_recurrent_layers * architecture.num_recursions
    )
    for end_index in range(architecture.n_end_layers):
        result[end_physical_start + end_index] = (end_logical_start + end_index,)

    if len(result) != architecture.physical_num_layers:
        raise AssertionError("physical-to-logical layer map is incomplete")
    return result


@dataclass(frozen=True, slots=True)
class MoRCheckpointMetadata:
    """Sidecar metadata needed to interpret a folded recurrent checkpoint."""

    architecture: MoRArchitectureConfig
    folding_policy: str = "mean"
    schema_version: int = 1
    model_type: str = "qwen3_moe_mor"
    base_model_type: str = "qwen3_moe"
    megatron_lm_sha: str = MEGATRON_LM_PINNED_SHA
    depth_router: DepthRouterConfig = field(default_factory=DepthRouterConfig)
    depth_router_seed: int = 1234
    hf_source: str = "unknown"
    magi_attention_version: str = "1.1.1"
    cp_transition: str = "magi_direct"
    parallel: MoRParallelConfig = field(default_factory=MoRParallelConfig)

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or not isinstance(self.schema_version, int):
            raise TypeError("schema_version must be an integer")
        if self.schema_version != 1:
            raise ValueError(f"unsupported MoR metadata schema {self.schema_version}")
        for name in (
            "model_type",
            "base_model_type",
            "megatron_lm_sha",
            "hf_source",
            "magi_attention_version",
            "cp_transition",
        ):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise TypeError(f"{name} must be a string")
            if not value:
                raise ValueError(f"{name} must not be empty")
        if not isinstance(self.architecture, MoRArchitectureConfig):
            raise TypeError("architecture must be MoRArchitectureConfig")
        if not isinstance(self.depth_router, DepthRouterConfig):
            raise TypeError("depth_router must be DepthRouterConfig")
        if not isinstance(self.parallel, MoRParallelConfig):
            raise TypeError("parallel must be MoRParallelConfig")
        self.parallel.validate_world_size(self.parallel.expected_world_size)
        object.__setattr__(self, "folding_policy", validate_folding_policy(self.folding_policy))
        if isinstance(self.depth_router_seed, bool) or not isinstance(self.depth_router_seed, int):
            raise TypeError("depth_router_seed must be an integer")
        if self.depth_router_seed < 0:
            raise ValueError("depth_router_seed must be a non-negative integer")
        if self.cp_transition not in {
            "magi_direct",
            "magi_canonical",
            "static_reference",
        }:
            raise ValueError(f"unsupported cp_transition {self.cp_transition!r}")
        if self.parallel.cp_transition != self.cp_transition:
            raise ValueError(
                "parallel.cp_transition must match the top-level checkpoint "
                f"cp_transition: {self.parallel.cp_transition!r} != {self.cp_transition!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["architecture"] = self.architecture.to_dict()
        values["depth_router"] = self.depth_router.to_dict()
        values["parallel"] = self.parallel.to_dict()
        values["physical_to_logical_layers"] = {
            str(physical): list(logical)
            for physical, logical in physical_to_logical_layer_map(self.architecture).items()
        }
        values["logical_num_layers"] = self.architecture.logical_num_layers
        values["physical_num_layers"] = self.architecture.physical_num_layers
        values["hf_export"] = "unsupported-import-only"
        return values

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> MoRCheckpointMetadata:
        required = {
            "architecture",
            "folding_policy",
            "schema_version",
            "model_type",
            "base_model_type",
            "megatron_lm_sha",
            "depth_router",
            "depth_router_seed",
            "hf_source",
            "magi_attention_version",
            "cp_transition",
            "parallel",
            "physical_to_logical_layers",
            "logical_num_layers",
            "physical_num_layers",
            "hf_export",
        }
        missing = sorted(required - values.keys())
        if missing:
            raise ValueError(f"MoR checkpoint sidecar is missing required fields: {missing}")
        unknown = sorted(values.keys() - required)
        if unknown:
            raise ValueError(f"MoR checkpoint sidecar has unknown fields: {unknown}")
        raw_architecture = values["architecture"]
        if not isinstance(raw_architecture, Mapping):
            raise TypeError("architecture checkpoint metadata must be a mapping")
        raw_router = values["depth_router"]
        if not isinstance(raw_router, Mapping):
            raise TypeError("depth_router checkpoint metadata must be a mapping")
        missing_router = sorted({"temperature", "alpha", "aux_loss_coef"} - raw_router.keys())
        if missing_router:
            raise ValueError(
                f"depth_router checkpoint metadata is missing required fields: {missing_router}"
            )
        raw_parallel = values["parallel"]
        if not isinstance(raw_parallel, Mapping):
            raise TypeError("parallel checkpoint metadata must be a mapping")
        missing_parallel = sorted(
            {"dp", "tp", "cp", "ep", "etp", "zero_stage", "cp_transition"} - raw_parallel.keys()
        )
        if missing_parallel:
            raise ValueError(
                f"parallel checkpoint metadata is missing required fields: {missing_parallel}"
            )
        schema_version = values["schema_version"]
        if isinstance(schema_version, bool) or not isinstance(schema_version, int):
            raise TypeError("schema_version checkpoint metadata must be an integer")
        depth_router_seed = values["depth_router_seed"]
        if isinstance(depth_router_seed, bool) or not isinstance(depth_router_seed, int):
            raise TypeError("depth_router_seed checkpoint metadata must be an integer")
        string_fields = (
            "folding_policy",
            "model_type",
            "base_model_type",
            "megatron_lm_sha",
            "hf_source",
            "magi_attention_version",
            "cp_transition",
        )
        invalid_strings = [field for field in string_fields if not isinstance(values[field], str)]
        if invalid_strings:
            raise TypeError(
                "MoR checkpoint sidecar fields must be strings: " + ", ".join(invalid_strings)
            )
        metadata = cls(
            architecture=MoRArchitectureConfig.from_dict(raw_architecture),
            folding_policy=values["folding_policy"],
            schema_version=schema_version,
            model_type=values["model_type"],
            base_model_type=values["base_model_type"],
            megatron_lm_sha=values["megatron_lm_sha"],
            depth_router=DepthRouterConfig.from_dict(raw_router),
            depth_router_seed=depth_router_seed,
            hf_source=values["hf_source"],
            magi_attention_version=values["magi_attention_version"],
            cp_transition=values["cp_transition"],
            parallel=MoRParallelConfig.from_dict(raw_parallel),
        )
        canonical = metadata.to_dict()
        for derived_field in (
            "physical_to_logical_layers",
            "logical_num_layers",
            "physical_num_layers",
            "hf_export",
        ):
            if not _exact_json_value_matches(values[derived_field], canonical[derived_field]):
                raise ValueError(
                    f"MoR checkpoint sidecar has inconsistent derived field {derived_field}"
                )
        return metadata


__all__ = [
    "FOLDING_POLICIES",
    "HF_FOLDING_METADATA_FILENAME",
    "MEGATRON_LM_PINNED_SHA",
    "MoRCheckpointMetadata",
    "physical_to_logical_layer_map",
    "validate_folding_policy",
]
