from __future__ import annotations

import json
from pathlib import Path

import pytest

from mor_mlite import checkpoint_io
from mor_mlite.checkpoint_io import (
    build_checkpoint_metadata,
    load_mor_checkpoint,
    read_mor_sidecar,
    save_mor_checkpoint,
)
from mor_mlite.config import DepthRouterConfig, MoRArchitectureConfig, MoRParallelConfig
from mor_mlite.qwen3_moe_mor.metadata import MoRCheckpointMetadata


class _Runtime:
    def __init__(self) -> None:
        self.saved: list[tuple[object, str, dict[str, object]]] = []
        self.loaded: list[tuple[object, str, dict[str, object]]] = []

    def save_checkpoint(self, handle: object, path: str, **kwargs: object) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        self.saved.append((handle, path, kwargs))

    def load_checkpoint(self, handle: object, path: str, **kwargs: object) -> int:
        self.loaded.append((handle, path, kwargs))
        return 7


def _metadata(**overrides: object):
    values = {
        "architecture": MoRArchitectureConfig.tiny(),
        "depth_router": DepthRouterConfig(),
        "depth_router_seed": 1234,
        "hf_source": "org/model",
        "parallel": MoRParallelConfig(),
    }
    values.update(overrides)
    return build_checkpoint_metadata(**values)


def test_checkpoint_wrapper_writes_self_describing_sidecars(tmp_path: Path) -> None:
    hf = tmp_path / "hf"
    hf.mkdir()
    config = {"model_type": "qwen3_moe", "num_hidden_layers": 8}
    (hf / "config.json").write_text(json.dumps(config), encoding="utf-8")
    runtime = _Runtime()
    handle = object()
    checkpoint = tmp_path / "dcp"
    metadata = _metadata()

    save_mor_checkpoint(
        runtime,
        handle,
        checkpoint,
        step=7,
        metadata=metadata,
        base_hf_path=hf,
    )

    assert read_mor_sidecar(checkpoint) == metadata
    assert json.loads((checkpoint / "config.json").read_text(encoding="utf-8")) == config
    assert runtime.saved[0][2]["step"] == 7
    assert load_mor_checkpoint(runtime, handle, checkpoint, expected_metadata=metadata) == 7


def test_checkpoint_wrapper_propagates_rank_zero_sidecar_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hf = tmp_path / "hf"
    hf.mkdir()
    (hf / "config.json").write_text("{}", encoding="utf-8")

    def fail_write(*_args, **_kwargs) -> None:
        raise OSError("simulated Lustre failure")

    monkeypatch.setattr(checkpoint_io, "write_mor_sidecars", fail_write)
    with pytest.raises(RuntimeError, match="rank-zero.*simulated Lustre failure"):
        save_mor_checkpoint(
            _Runtime(),
            object(),
            tmp_path / "dcp",
            step=1,
            metadata=_metadata(),
            base_hf_path=hf,
        )


def test_checkpoint_wrapper_rejects_router_semantic_mismatch(tmp_path: Path) -> None:
    hf = tmp_path / "hf"
    hf.mkdir()
    (hf / "config.json").write_text("{}", encoding="utf-8")
    runtime = _Runtime()
    checkpoint = tmp_path / "dcp"
    save_mor_checkpoint(
        runtime,
        object(),
        checkpoint,
        step=1,
        metadata=_metadata(),
        base_hf_path=hf,
    )
    incompatible = build_checkpoint_metadata(
        architecture=MoRArchitectureConfig.tiny(),
        depth_router=DepthRouterConfig(temperature=0.5),
        depth_router_seed=1234,
        hf_source="org/model",
        parallel=MoRParallelConfig(),
    )
    with pytest.raises(ValueError, match="depth_router"):
        load_mor_checkpoint(
            runtime,
            object(),
            checkpoint,
            expected_metadata=incompatible,
        )


def test_checkpoint_sidecar_roundtrips_parallel_provenance(tmp_path: Path) -> None:
    checkpoint = tmp_path / "dcp"
    checkpoint.mkdir()
    parallel = MoRParallelConfig(
        dp=2,
        tp=2,
        cp=2,
        ep=4,
        cp_transition="magi_direct",
    )
    metadata = _metadata(
        depth_router=DepthRouterConfig(temperature=0.75, alpha=0.2, aux_loss_coef=0.01),
        depth_router_seed=9876,
        parallel=parallel,
    )
    (checkpoint / "mor_config.json").write_text(json.dumps(metadata.to_dict()), encoding="utf-8")

    restored = read_mor_sidecar(checkpoint)
    assert restored.architecture == MoRArchitectureConfig.tiny()
    assert restored.depth_router == DepthRouterConfig(
        temperature=0.75,
        alpha=0.2,
        aux_loss_coef=0.01,
    )
    assert restored.depth_router_seed == 9876
    assert restored.cp_transition == "magi_direct"
    assert restored.parallel == parallel


def test_checkpoint_metadata_rejects_impossible_expert_parallel_provenance() -> None:
    with pytest.raises(ValueError, match="divisible by ep"):
        _metadata(parallel=MoRParallelConfig(ep=4))


def test_model_only_load_allows_runtime_reshard_but_full_resume_is_strict(
    tmp_path: Path,
) -> None:
    hf = tmp_path / "hf"
    hf.mkdir()
    (hf / "config.json").write_text("{}", encoding="utf-8")
    checkpoint = tmp_path / "dcp"
    runtime = _Runtime()
    source = _metadata(parallel=MoRParallelConfig())
    save_mor_checkpoint(
        runtime,
        object(),
        checkpoint,
        step=0,
        metadata=source,
        base_hf_path=hf,
        save_rng=False,
        save_optimizer=False,
    )
    current = _metadata(
        parallel=MoRParallelConfig(dp=2, tp=2, cp=2, ep=1),
    )

    with pytest.raises(ValueError, match="parallel"):
        load_mor_checkpoint(runtime, object(), checkpoint, expected_metadata=current)

    assert (
        load_mor_checkpoint(
            runtime,
            object(),
            checkpoint,
            expected_metadata=current,
            strict_runtime=False,
            load_rng=False,
            load_optimizer=False,
        )
        == 7
    )
    assert runtime.loaded[-1][2]["load_model"] is True
    assert runtime.loaded[-1][2]["load_rng"] is False
    assert runtime.loaded[-1][2]["load_optimizer"] is False

    cross_ep = _metadata(parallel=MoRParallelConfig(dp=2, tp=2, cp=2, ep=4))
    with pytest.raises(ValueError, match="expert_parallel_size"):
        load_mor_checkpoint(
            runtime,
            object(),
            checkpoint,
            expected_metadata=cross_ep,
            strict_runtime=False,
            load_rng=False,
            load_optimizer=False,
        )


def test_checkpoint_sidecar_rejects_missing_or_forged_derived_metadata(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "dcp"
    checkpoint.mkdir()
    raw = _metadata().to_dict()
    raw.pop("megatron_lm_sha")
    (checkpoint / "mor_config.json").write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="missing required fields"):
        read_mor_sidecar(checkpoint)

    raw = _metadata().to_dict()
    raw["physical_num_layers"] = 999
    (checkpoint / "mor_config.json").write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="inconsistent derived field"):
        read_mor_sidecar(checkpoint)

    raw = _metadata().to_dict()
    raw["architecture"].pop("capacity_schedule")
    (checkpoint / "mor_config.json").write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="capacity_schedule"):
        read_mor_sidecar(checkpoint)


@pytest.mark.parametrize(
    ("field", "mutate"),
    [
        ("logical_num_layers", lambda raw: float(raw["logical_num_layers"])),
        ("physical_num_layers", lambda raw: float(raw["physical_num_layers"])),
        (
            "physical_to_logical_layers",
            lambda raw: {
                **raw["physical_to_logical_layers"],
                "0": [float(raw["physical_to_logical_layers"]["0"][0])],
            },
        ),
        ("hf_export", lambda _raw: True),
    ],
)
def test_checkpoint_sidecar_rejects_coerced_derived_field_types(
    tmp_path: Path, field: str, mutate
) -> None:
    checkpoint = tmp_path / "dcp"
    checkpoint.mkdir()
    raw = _metadata().to_dict()
    raw[field] = mutate(raw)
    (checkpoint / "mor_config.json").write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match=f"inconsistent derived field {field}"):
        read_mor_sidecar(checkpoint)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", "1", "schema_version"),
        ("depth_router_seed", 1.5, "depth_router_seed"),
        ("model_type", 123, "must be strings"),
    ],
)
def test_checkpoint_sidecar_rejects_type_coercion(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    checkpoint = tmp_path / "dcp"
    checkpoint.mkdir()
    raw = _metadata().to_dict()
    raw[field] = value
    (checkpoint / "mor_config.json").write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises((TypeError, ValueError), match=message):
        read_mor_sidecar(checkpoint)


def test_checkpoint_sidecar_rejects_unknown_and_duplicate_fields(tmp_path: Path) -> None:
    checkpoint = tmp_path / "dcp"
    checkpoint.mkdir()
    raw = _metadata().to_dict()
    raw["future_or_typo"] = True
    (checkpoint / "mor_config.json").write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown fields"):
        read_mor_sidecar(checkpoint)

    payload = json.dumps(_metadata().to_dict())
    duplicate = payload[:-1] + ', "schema_version": 1}'
    (checkpoint / "mor_config.json").write_text(duplicate, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        read_mor_sidecar(checkpoint)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"schema_version": True}, "schema_version"),
        ({"depth_router_seed": 1.5}, "depth_router_seed"),
        ({"hf_source": 123}, "hf_source"),
        ({"folding_policy": 1}, "folding_policy"),
    ],
)
def test_checkpoint_metadata_constructor_rejects_values_it_cannot_reload(
    overrides: dict[str, object], message: str
) -> None:
    values = {
        "architecture": MoRArchitectureConfig.tiny(),
        "depth_router": DepthRouterConfig(),
        "depth_router_seed": 1234,
        "hf_source": "org/model",
        "parallel": MoRParallelConfig(),
    }
    values.update(overrides)
    with pytest.raises((TypeError, ValueError), match=message):
        MoRCheckpointMetadata(**values)
