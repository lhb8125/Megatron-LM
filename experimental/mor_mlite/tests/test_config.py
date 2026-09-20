from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from mor_mlite.config import (
    DepthRouterConfig,
    MoRArchitectureConfig,
    MoRParallelConfig,
)


def test_tiny_linear_schedule_and_depths() -> None:
    config = MoRArchitectureConfig.tiny()
    assert config.capacity_fractions == (1.0, 2.0 / 3.0, 1.0 / 3.0)
    assert [config.top_k(10, round_index) for round_index in range(3)] == [10, 6, 3]
    assert config.logical_num_layers == 8
    assert config.physical_num_layers == 4


def test_qwen_preset_has_48_logical_but_20_physical_layers() -> None:
    config = MoRArchitectureConfig.qwen3_30b()
    assert (
        config.n_start_layers,
        config.n_recurrent_layers,
        config.num_recursions,
        config.n_end_layers,
    ) == (3, 14, 3, 3)
    assert config.logical_num_layers == 48
    assert config.physical_num_layers == 20


@pytest.mark.parametrize(
    "schedule",
    ([0.9, 0.6, 0.3], [1.0, 0.5], [1.0, 0.5, 0.6], [1.0, 0.0, 0.1]),
)
def test_invalid_custom_schedules_are_rejected(schedule: list[float]) -> None:
    with pytest.raises(ValueError):
        MoRArchitectureConfig(1, 2, 3, 1, schedule)


def test_custom_schedule_rejects_boolean_values() -> None:
    with pytest.raises(ValueError, match="not booleans"):
        MoRArchitectureConfig(1, 2, 3, 1, [True, 0.5, 0.25])


def test_empty_sample_has_no_artificial_token() -> None:
    assert MoRArchitectureConfig.tiny().top_k(0, 2) == 0


def test_config_round_trip() -> None:
    architecture = MoRArchitectureConfig(1, 2, 3, 1, [1.0, 0.7, 0.2])
    assert MoRArchitectureConfig.from_dict(architecture.to_dict()) == architecture
    router = DepthRouterConfig()
    assert DepthRouterConfig.from_dict(router.to_dict()) == router


def test_nested_config_parsers_require_exact_fields() -> None:
    architecture = MoRArchitectureConfig.tiny().to_dict()
    architecture.pop("capacity_schedule")
    with pytest.raises(ValueError, match="capacity_schedule"):
        MoRArchitectureConfig.from_dict(architecture)

    router = DepthRouterConfig().to_dict()
    router["typo"] = 1
    with pytest.raises(ValueError, match="unknown fields"):
        DepthRouterConfig.from_dict(router)

    parallel = MoRParallelConfig().to_dict()
    parallel.pop("zero_stage")
    with pytest.raises(ValueError, match="zero_stage"):
        MoRParallelConfig.from_dict(parallel)


def test_router_config_rejects_boolean_values() -> None:
    with pytest.raises(TypeError, match="must be a number"):
        DepthRouterConfig(temperature=True)


def test_parallel_shape_uses_ep_as_overlapping_view() -> None:
    config = MoRParallelConfig(dp=2, tp=2, cp=2, ep=4)
    assert config.expected_world_size == 8
    config.validate_world_size(8, num_experts=128)
    with pytest.raises(ValueError, match="WORLD_SIZE"):
        config.validate_world_size(16, num_experts=128)
    with pytest.raises(ValueError, match="num_experts"):
        config.validate_world_size(8, num_experts=126)


def test_first_release_scope_is_enforced() -> None:
    with pytest.raises(ValueError, match="ETP"):
        MoRParallelConfig(etp=2)
    with pytest.raises(ValueError, match="ZeRO-1"):
        MoRParallelConfig(zero_stage=2)
    with pytest.raises(TypeError, match="must be an integer"):
        MoRParallelConfig(zero_stage=True)
