"""Configuration objects for the Mixture-of-Recursions model.

The objects in this module deliberately do not import Megatron or torch.  This
keeps configuration parsing usable in conversion and launch tooling before a
distributed process group is initialized.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from typing import Any


def _require_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def _require_exact_fields(
    name: str, values: Mapping[str, Any], required: frozenset[str]
) -> dict[str, Any]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping")
    keys = frozenset(values)
    missing = sorted(required - keys)
    unknown = sorted(keys - required)
    if missing:
        raise ValueError(f"{name} is missing required fields: {missing}")
    if unknown:
        raise ValueError(f"{name} has unknown fields: {unknown}")
    return dict(values)


@dataclass(frozen=True, slots=True)
class MoRArchitectureConfig:
    """Physical and logical depth of a MoR decoder.

    ``n_recurrent_layers`` physical layers are registered exactly once and are
    invoked ``num_recursions`` times.  A custom capacity schedule is expressed
    relative to each sample's *original* valid length, never relative to the
    previous round's already-reduced length.
    """

    n_start_layers: int
    n_recurrent_layers: int
    num_recursions: int
    n_end_layers: int
    capacity_schedule: str | Sequence[float] = "linear"

    def __post_init__(self) -> None:
        _require_positive_int("n_start_layers", self.n_start_layers)
        _require_positive_int("n_recurrent_layers", self.n_recurrent_layers)
        _require_positive_int("num_recursions", self.num_recursions)
        _require_positive_int("n_end_layers", self.n_end_layers)

        schedule = self.capacity_schedule
        if isinstance(schedule, str):
            if schedule != "linear":
                raise ValueError("capacity_schedule must be 'linear' or a sequence of fractions")
            return

        if any(
            isinstance(value, bool) or not isinstance(value, (int, float)) for value in schedule
        ):
            raise ValueError("capacity fractions must be numbers, not booleans or strings")
        values = tuple(float(value) for value in schedule)
        if len(values) != self.num_recursions:
            raise ValueError(
                "custom capacity_schedule must contain exactly num_recursions "
                f"entries, got {len(values)} and {self.num_recursions}"
            )
        if any(not math.isfinite(value) or value <= 0.0 or value > 1.0 for value in values):
            raise ValueError("capacity fractions must be finite and in (0, 1]")
        if values[0] != 1.0:
            raise ValueError("the first recurrent round must have capacity 1.0")
        if any(right > left for left, right in pairwise(values)):
            raise ValueError("capacity_schedule must be monotonically non-increasing")
        object.__setattr__(self, "capacity_schedule", values)

    @property
    def logical_num_layers(self) -> int:
        return (
            self.n_start_layers + self.n_recurrent_layers * self.num_recursions + self.n_end_layers
        )

    @property
    def physical_num_layers(self) -> int:
        return self.n_start_layers + self.n_recurrent_layers + self.n_end_layers

    @property
    def capacity_fractions(self) -> tuple[float, ...]:
        if self.capacity_schedule == "linear":
            k = self.num_recursions
            return tuple((k - round_index) / k for round_index in range(k))
        return tuple(self.capacity_schedule)

    def capacity_for_round(self, round_index: int) -> float:
        if (
            isinstance(round_index, bool)
            or not isinstance(round_index, int)
            or not 0 <= round_index < self.num_recursions
        ):
            raise IndexError(
                f"round_index must be in [0, {self.num_recursions}), got {round_index!r}"
            )
        return self.capacity_fractions[round_index]

    def top_k(self, original_valid_length: int, round_index: int) -> int:
        """Return the round budget for one sample.

        Empty samples stay empty.  A non-empty sample always retains at least
        one token, matching ``max(1, floor(capacity * original_length))``.
        """

        if (
            isinstance(original_valid_length, bool)
            or not isinstance(original_valid_length, int)
            or original_valid_length < 0
        ):
            raise ValueError("original_valid_length must be a non-negative integer")
        fraction = self.capacity_for_round(round_index)
        if original_valid_length == 0:
            return 0
        if self.capacity_schedule == "linear":
            return max(
                1,
                (self.num_recursions - round_index) * original_valid_length // self.num_recursions,
            )
        return max(
            1,
            math.floor(fraction * original_valid_length),
        )

    @classmethod
    def tiny(cls) -> MoRArchitectureConfig:
        return cls(1, 2, 3, 1)

    @classmethod
    def qwen3_30b(cls) -> MoRArchitectureConfig:
        return cls(3, 14, 3, 3)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        if not isinstance(self.capacity_schedule, str):
            result["capacity_schedule"] = list(self.capacity_schedule)
        return result

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> MoRArchitectureConfig:
        return cls(
            **_require_exact_fields(
                "MoRArchitectureConfig",
                values,
                frozenset(
                    {
                        "n_start_layers",
                        "n_recurrent_layers",
                        "num_recursions",
                        "n_end_layers",
                        "capacity_schedule",
                    }
                ),
            )
        )


@dataclass(frozen=True, slots=True)
class DepthRouterConfig:
    temperature: float = 1.0
    alpha: float = 0.1
    aux_loss_coef: float = 0.001

    def __post_init__(self) -> None:
        for name in ("temperature", "alpha", "aux_loss_coef"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a number, got {value!r}")
        if not math.isfinite(self.temperature) or self.temperature <= 0.0:
            raise ValueError("temperature must be finite and greater than zero")
        if not math.isfinite(self.alpha) or self.alpha <= 0.0:
            raise ValueError("alpha must be finite and greater than zero")
        if not math.isfinite(self.aux_loss_coef) or self.aux_loss_coef < 0.0:
            raise ValueError("aux_loss_coef must be finite and non-negative")
        object.__setattr__(self, "temperature", float(self.temperature))
        object.__setattr__(self, "alpha", float(self.alpha))
        object.__setattr__(self, "aux_loss_coef", float(self.aux_loss_coef))

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> DepthRouterConfig:
        return cls(
            **_require_exact_fields(
                "DepthRouterConfig",
                values,
                frozenset({"temperature", "alpha", "aux_loss_coef"}),
            )
        )


@dataclass(frozen=True, slots=True)
class MoRParallelConfig:
    """User-facing parallel shape.

    EP is a second view over the same world rather than an additional factor in
    ``WORLD_SIZE``.  The first release intentionally supports ZeRO-1 and ETP=1
    only.
    """

    dp: int = 1
    tp: int = 1
    cp: int = 1
    ep: int = 1
    etp: int = 1
    zero_stage: int = 1
    cp_transition: str = "magi_direct"

    def __post_init__(self) -> None:
        for name in ("dp", "tp", "cp", "ep", "etp"):
            _require_positive_int(name, getattr(self, name))
        if isinstance(self.zero_stage, bool) or not isinstance(self.zero_stage, int):
            raise TypeError(f"zero_stage must be an integer, got {self.zero_stage!r}")
        if self.etp != 1:
            raise ValueError("ETP>1 is outside the first-release scope")
        if self.zero_stage != 1:
            raise ValueError("only ZeRO-1 is supported in the first release")
        allowed = {"magi_canonical", "magi_direct", "static_reference"}
        if not isinstance(self.cp_transition, str):
            raise TypeError(f"cp_transition must be a string, got {self.cp_transition!r}")
        if self.cp_transition not in allowed:
            raise ValueError(
                f"cp_transition must be one of {sorted(allowed)}, got {self.cp_transition!r}"
            )

    @property
    def expected_world_size(self) -> int:
        return self.dp * self.tp * self.cp

    def validate_world_size(self, world_size: int, *, num_experts: int | None = None) -> None:
        _require_positive_int("world_size", world_size)
        if world_size != self.expected_world_size:
            raise ValueError(
                "WORLD_SIZE must equal dp * tp * cp: "
                f"{world_size} != {self.dp} * {self.tp} * {self.cp}"
            )
        if world_size % self.ep != 0:
            raise ValueError(f"WORLD_SIZE={world_size} must be divisible by ep={self.ep}")
        if num_experts is not None:
            _require_positive_int("num_experts", num_experts)
            if num_experts % self.ep != 0:
                raise ValueError(f"num_experts={num_experts} must be divisible by ep={self.ep}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> MoRParallelConfig:
        return cls(
            **_require_exact_fields(
                "MoRParallelConfig",
                values,
                frozenset({"dp", "tp", "cp", "ep", "etp", "zero_stage", "cp_transition"}),
            )
        )
