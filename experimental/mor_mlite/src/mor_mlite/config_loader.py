"""Strict loaders for the checked-in MoR preset JSON files.

The default files live in the repository's top-level ``configs`` directory.
Setuptools also installs the same files under ``share/mor_mlite/configs`` so a
non-editable wheel does not depend on the current working directory.
"""

from __future__ import annotations

import json
import math
import sysconfig
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from mor_mlite.config import (
    DepthRouterConfig,
    MoRArchitectureConfig,
    MoRParallelConfig,
)

PRESET_FILENAMES: Mapping[str, str] = {
    "tiny": "tiny.json",
    "qwen3-30b": "qwen3_30b.json",
}

_REQUIRED_PRESET_FIELDS = frozenset({"architecture", "depth_router", "parallel"})
_OPTIONAL_PRESET_FIELDS = frozenset({"hf_source", "model", "expected_hf"})
_TINY_MODEL_FIELDS = frozenset(
    {
        "vocab_size",
        "hidden_size",
        "num_attention_heads",
        "num_key_value_heads",
        "intermediate_size",
        "num_experts",
        "num_experts_per_tok",
        "max_position_embeddings",
        "rope_theta",
        "rms_norm_eps",
        "initializer_range",
    }
)
_QWEN_EXPECTED_HF_FIELDS = frozenset({"num_hidden_layers", "num_experts", "num_experts_per_tok"})
_TINY_TO_HF_FIELDS: Mapping[str, str] = {
    "vocab_size": "vocab_size",
    "hidden_size": "hidden_size",
    "num_attention_heads": "num_attention_heads",
    "num_key_value_heads": "num_key_value_heads",
    "intermediate_size": "moe_intermediate_size",
    "num_experts": "num_experts",
    "num_experts_per_tok": "num_experts_per_tok",
    "max_position_embeddings": "max_position_embeddings",
    "rope_theta": "rope_theta",
    "rms_norm_eps": "rms_norm_eps",
    "initializer_range": "initializer_range",
}


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonstandard_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON numeric constant {value!r} is not allowed")


def load_json(path: Path, *, expected_type: type[dict | list]) -> dict[str, Any] | list[Any]:
    """Load one JSON file with duplicate-key and top-level shape checks."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"MoR configuration file does not exist: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonstandard_json_constant,
        )
    except UnicodeDecodeError as exc:
        raise ValueError(f"MoR configuration must be UTF-8: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in MoR configuration {path}: {exc.msg}") from exc
    if not isinstance(value, expected_type):
        expected = "object" if expected_type is dict else "array"
        raise TypeError(f"MoR configuration {path} must contain a JSON {expected}")
    return value


def default_config_directory() -> Path:
    """Locate bundled configs without consulting the process working directory."""

    candidates = (
        # Editable/source checkout: ``src/mor_mlite/config_loader.py`` -> repo.
        Path(__file__).resolve().parents[2] / "configs",
        # Wheel install via ``tool.setuptools.data-files``.
        Path(sysconfig.get_path("data")) / "share" / "mor_mlite" / "configs",
        # ``pip install --target`` places data files beside site packages while
        # ``sysconfig`` continues to describe the interpreter's own prefix.
        Path(__file__).resolve().parents[1] / "share" / "mor_mlite" / "configs",
    )
    required = frozenset(PRESET_FILENAMES.values()) | {"topologies.json"}
    for candidate in candidates:
        if candidate.is_dir() and all((candidate / filename).is_file() for filename in required):
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"could not locate bundled MoR configs; searched: {searched}")


def preset_config_path(preset: str) -> Path:
    try:
        filename = PRESET_FILENAMES[preset]
    except KeyError as exc:
        choices = ", ".join(PRESET_FILENAMES)
        raise ValueError(f"unknown preset {preset!r}; choose one of: {choices}") from exc
    return default_config_directory() / filename


def topology_config_path() -> Path:
    return default_config_directory() / "topologies.json"


def parse_capacity_schedule(
    raw: str | Sequence[float], *, num_recursions: int
) -> str | tuple[float, ...]:
    """Parse ``linear`` or an exact comma-separated/per-round schedule."""

    if isinstance(raw, str):
        stripped = raw.strip()
        if stripped == "linear":
            return stripped
        try:
            values = tuple(float(item.strip()) for item in stripped.split(","))
        except ValueError as exc:
            raise ValueError(
                "capacity schedule must be 'linear' or comma-separated fractions"
            ) from exc
    else:
        values = tuple(float(item) for item in raw)
    if len(values) != num_recursions:
        raise ValueError(
            "capacity schedule must contain exactly num_recursions fractions: "
            f"expected {num_recursions}, got {len(values)}"
        )
    return values


def _mapping(value: Any, *, field: str, source: Path) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{field} in {source} must be a JSON object")
    return dict(value)


def _positive_int_field(values: Mapping[str, Any], name: str, *, source: Path) -> int:
    value = values.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} in {source} must be a positive integer, got {value!r}")
    return value


@dataclass(frozen=True, slots=True)
class MoRPresetConfig:
    """Validated contents of ``configs/tiny.json`` or ``qwen3_30b.json``."""

    name: str
    architecture: MoRArchitectureConfig
    depth_router: DepthRouterConfig
    parallel: MoRParallelConfig
    model: Mapping[str, Any]
    expected_hf: Mapping[str, Any]
    hf_source: str | None
    source: Path

    @property
    def num_experts(self) -> int:
        values = self.model if self.name == "tiny" else self.expected_hf
        return int(values["num_experts"])

    def with_overrides(
        self,
        *,
        n_start_layers: int | None = None,
        n_recurrent_layers: int | None = None,
        num_recursions: int | None = None,
        n_end_layers: int | None = None,
        capacity_schedule: str | Sequence[float] | None = None,
        router_temperature: float | None = None,
        router_alpha: float | None = None,
        router_aux_loss_coef: float | None = None,
    ) -> MoRPresetConfig:
        """Apply explicit CLI overrides after loading the JSON preset."""

        recursion_count = (
            self.architecture.num_recursions if num_recursions is None else num_recursions
        )
        schedule = self.architecture.capacity_schedule
        if capacity_schedule is not None:
            schedule = parse_capacity_schedule(
                capacity_schedule,
                num_recursions=recursion_count,
            )
        architecture = MoRArchitectureConfig(
            n_start_layers=(
                self.architecture.n_start_layers if n_start_layers is None else n_start_layers
            ),
            n_recurrent_layers=(
                self.architecture.n_recurrent_layers
                if n_recurrent_layers is None
                else n_recurrent_layers
            ),
            num_recursions=recursion_count,
            n_end_layers=(self.architecture.n_end_layers if n_end_layers is None else n_end_layers),
            capacity_schedule=schedule,
        )
        expected_depth = self.expected_hf.get("num_hidden_layers")
        if expected_depth is not None and int(expected_depth) != architecture.logical_num_layers:
            raise ValueError(
                "architecture override does not match expected_hf.num_hidden_layers: "
                f"logical depth {architecture.logical_num_layers} != {expected_depth}"
            )
        depth_router = DepthRouterConfig(
            temperature=(
                self.depth_router.temperature if router_temperature is None else router_temperature
            ),
            alpha=self.depth_router.alpha if router_alpha is None else router_alpha,
            aux_loss_coef=(
                self.depth_router.aux_loss_coef
                if router_aux_loss_coef is None
                else router_aux_loss_coef
            ),
        )
        return replace(self, architecture=architecture, depth_router=depth_router)

    def validate_base_hf_config(
        self,
        values: Mapping[str, Any],
        *,
        architecture: MoRArchitectureConfig | None = None,
    ) -> None:
        """Fail before model construction when a checkpoint belongs to another preset.

        A model-only DCP may change execution topology, but its base HF snapshot
        must still describe the selected model family and the sidecar's logical
        depth.  This also prevents a Qwen-30B checkpoint from accidentally
        entering the tiny runner's full-state reconstruction path.
        """

        architecture = architecture or self.architecture
        expected: dict[str, Any] = {"num_hidden_layers": architecture.logical_num_layers}
        if self.name == "tiny":
            expected.update(
                {
                    hf_field: self.model[preset_field]
                    for preset_field, hf_field in _TINY_TO_HF_FIELDS.items()
                }
            )
        else:
            expected.update(self.expected_hf)
            expected["num_hidden_layers"] = architecture.logical_num_layers

        missing = sorted(field for field in expected if field not in values)
        if missing:
            raise ValueError(
                f"base HF config for preset {self.name!r} is missing fields: {missing}"
            )
        mismatches = [
            f"{field}: checkpoint={values[field]!r}, expected={wanted!r}"
            for field, wanted in expected.items()
            if values[field] != wanted
        ]
        if mismatches:
            raise ValueError(
                f"base HF config is incompatible with preset {self.name!r}; "
                + "; ".join(mismatches)
            )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "architecture": self.architecture.to_dict(),
            "depth_router": self.depth_router.to_dict(),
            "parallel": self.parallel.to_dict(),
        }
        if self.model:
            result["model"] = dict(self.model)
        if self.expected_hf:
            result["expected_hf"] = dict(self.expected_hf)
        if self.hf_source is not None:
            result["hf_source"] = self.hf_source
        return result


def load_preset_config(preset: str, path: Path | None = None) -> MoRPresetConfig:
    """Load and validate a named preset or an explicit compatible JSON file."""

    if preset not in PRESET_FILENAMES:
        choices = ", ".join(PRESET_FILENAMES)
        raise ValueError(f"unknown preset {preset!r}; choose one of: {choices}")
    source = preset_config_path(preset) if path is None else Path(path)
    raw = load_json(source, expected_type=dict)
    assert isinstance(raw, dict)
    fields = frozenset(raw)
    missing = _REQUIRED_PRESET_FIELDS - fields
    unknown = fields - _REQUIRED_PRESET_FIELDS - _OPTIONAL_PRESET_FIELDS
    if missing:
        raise ValueError(f"preset {source} is missing required fields: {sorted(missing)}")
    if unknown:
        raise ValueError(f"preset {source} has unknown fields: {sorted(unknown)}")

    architecture = MoRArchitectureConfig.from_dict(
        _mapping(raw["architecture"], field="architecture", source=source)
    )
    depth_router = DepthRouterConfig.from_dict(
        _mapping(raw["depth_router"], field="depth_router", source=source)
    )
    parallel = MoRParallelConfig.from_dict(
        _mapping(raw["parallel"], field="parallel", source=source)
    )
    model = _mapping(raw.get("model", {}), field="model", source=source)
    expected_hf = _mapping(raw.get("expected_hf", {}), field="expected_hf", source=source)
    hf_source = raw.get("hf_source")
    if hf_source is not None and (not isinstance(hf_source, str) or not hf_source.strip()):
        raise ValueError(f"hf_source in {source} must be a non-empty string")

    if preset == "tiny":
        if expected_hf:
            raise ValueError(f"expected_hf is not valid for the tiny preset in {source}")
        unknown_model = frozenset(model) - _TINY_MODEL_FIELDS
        if unknown_model:
            raise ValueError(f"model in {source} has unknown fields: {sorted(unknown_model)}")
        for field in (
            "vocab_size",
            "hidden_size",
            "num_attention_heads",
            "num_key_value_heads",
            "intermediate_size",
            "num_experts",
            "num_experts_per_tok",
            "max_position_embeddings",
        ):
            _positive_int_field(model, field, source=source)
        for field in ("rope_theta", "rms_norm_eps", "initializer_range"):
            value = model.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{field} in {source} must be a positive number, got {value!r}")
        hidden_size = int(model["hidden_size"])
        attention_heads = int(model["num_attention_heads"])
        key_value_heads = int(model["num_key_value_heads"])
        if hidden_size % attention_heads:
            raise ValueError(f"hidden_size in {source} must be divisible by num_attention_heads")
        head_dim = hidden_size // attention_heads
        if head_dim % 16:
            raise ValueError(
                f"tiny preset head_dim={head_dim} in {source} must be a multiple of 16 "
                "for the MagiAttention sm90 acceptance path"
            )
        if attention_heads % key_value_heads:
            raise ValueError(
                f"num_attention_heads in {source} must be divisible by num_key_value_heads"
            )
        if int(model["num_experts_per_tok"]) > int(model["num_experts"]):
            raise ValueError(f"num_experts_per_tok in {source} cannot exceed num_experts")
    else:
        if model:
            raise ValueError(f"model is not valid for the qwen3-30b preset in {source}")
        unknown_expected_hf = frozenset(expected_hf) - _QWEN_EXPECTED_HF_FIELDS
        if unknown_expected_hf:
            raise ValueError(
                f"expected_hf in {source} has unknown fields: {sorted(unknown_expected_hf)}"
            )
        for field in _QWEN_EXPECTED_HF_FIELDS:
            _positive_int_field(expected_hf, field, source=source)
        if int(expected_hf["num_hidden_layers"]) != architecture.logical_num_layers:
            raise ValueError(
                f"expected_hf.num_hidden_layers in {source} does not match architecture "
                f"logical depth {architecture.logical_num_layers}"
            )

    num_experts = int((model if preset == "tiny" else expected_hf)["num_experts"])
    parallel.validate_world_size(parallel.expected_world_size, num_experts=num_experts)
    return MoRPresetConfig(
        name=preset,
        architecture=architecture,
        depth_router=depth_router,
        parallel=parallel,
        model=model,
        expected_hf=expected_hf,
        hf_source=hf_source,
        source=source.resolve(),
    )


__all__ = [
    "PRESET_FILENAMES",
    "MoRPresetConfig",
    "default_config_directory",
    "load_json",
    "load_preset_config",
    "parse_capacity_schedule",
    "preset_config_path",
    "topology_config_path",
]
