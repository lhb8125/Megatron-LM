from __future__ import annotations

import json
from pathlib import Path

import pytest

from mor_mlite import config_loader
from mor_mlite.config_loader import (
    default_config_directory,
    load_json,
    load_preset_config,
)
from mor_mlite.parity.topologies import (
    TOPOLOGY_MATRIX,
    get_topology,
    load_topology_matrix,
)


def test_checked_in_presets_are_the_runtime_source_of_truth() -> None:
    config_dir = default_config_directory()
    assert config_dir.name == "configs"

    tiny = load_preset_config("tiny")
    assert tiny.source == config_dir / "tiny.json"
    assert tiny.architecture.to_dict() == json.loads(tiny.source.read_text())["architecture"]
    assert tiny.depth_router.aux_loss_coef == pytest.approx(0.001)
    assert tiny.num_experts == 4
    assert tiny.model["hidden_size"] // tiny.model["num_attention_heads"] == 64
    assert tiny.parallel.cp_transition == "magi_direct"

    qwen = load_preset_config("qwen3-30b")
    assert (
        qwen.architecture.n_start_layers,
        qwen.architecture.n_recurrent_layers,
        qwen.architecture.num_recursions,
        qwen.architecture.n_end_layers,
    ) == (3, 14, 3, 3)
    assert qwen.architecture.logical_num_layers == 48
    assert qwen.architecture.physical_num_layers == 20
    assert qwen.hf_source == "Qwen/Qwen3-30B-A3B-Base"
    assert qwen.num_experts == 128


def test_default_config_directory_supports_pip_target_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package_file = tmp_path / "target" / "mor_mlite" / "config_loader.py"
    package_file.parent.mkdir(parents=True)
    installed_configs = tmp_path / "target" / "share" / "mor_mlite" / "configs"
    installed_configs.mkdir(parents=True)
    for filename in ("tiny.json", "qwen3_30b.json", "topologies.json"):
        (installed_configs / filename).write_text("{}", encoding="utf-8")

    monkeypatch.setattr(config_loader, "__file__", str(package_file))
    monkeypatch.setattr(config_loader.sysconfig, "get_path", lambda _: str(tmp_path / "prefix"))
    assert default_config_directory() == installed_configs


def test_checked_in_topology_json_defines_the_acceptance_matrix() -> None:
    loaded = load_topology_matrix()
    assert loaded == TOPOLOGY_MATRIX
    assert [item.name for item in loaded] == [
        "baseline",
        "zero1",
        "tp",
        "cp",
        "ep",
        "tp_cp_ep",
        "tp_dp_ep",
        "cp_dp_ep",
        "all",
    ]
    assert get_topology("all").to_dict() == {
        "name": "all",
        "world_size": 8,
        "tp": 2,
        "cp": 2,
        "dp": 2,
        "ep": 4,
        "etp": 1,
    }
    with pytest.raises(ValueError, match="unknown topology"):
        get_topology("unvalidated-custom-shape")


def test_explicit_architecture_capacity_and_router_overrides_are_validated() -> None:
    resolved = load_preset_config("tiny").with_overrides(
        n_start_layers=2,
        n_recurrent_layers=3,
        num_recursions=2,
        n_end_layers=2,
        capacity_schedule="1.0,0.25",
        router_temperature=0.5,
        router_alpha=0.2,
        router_aux_loss_coef=0.004,
    )
    assert resolved.architecture.to_dict() == {
        "n_start_layers": 2,
        "n_recurrent_layers": 3,
        "num_recursions": 2,
        "n_end_layers": 2,
        "capacity_schedule": [1.0, 0.25],
    }
    assert resolved.depth_router.to_dict() == {
        "temperature": 0.5,
        "alpha": 0.2,
        "aux_loss_coef": 0.004,
    }

    with pytest.raises(ValueError, match="exactly num_recursions"):
        load_preset_config("tiny").with_overrides(
            num_recursions=2,
            capacity_schedule="1.0,0.6,0.2",
        )

    with pytest.raises(ValueError, match="expected_hf.num_hidden_layers"):
        load_preset_config("qwen3-30b").with_overrides(num_recursions=2)


def test_loaders_fail_closed_on_duplicate_or_unknown_fields(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"name": "first", "name": "second"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_json(duplicate, expected_type=dict)

    nonstandard_number = tmp_path / "nonstandard-number.json"
    nonstandard_number.write_text('{"value": NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-standard JSON numeric constant"):
        load_json(nonstandard_number, expected_type=dict)

    bad_preset = tmp_path / "tiny.json"
    payload = load_preset_config("tiny").to_dict()
    payload["typoed_router"] = {}
    bad_preset.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown fields"):
        load_preset_config("tiny", bad_preset)

    incomplete_tiny = tmp_path / "incomplete-tiny.json"
    payload = load_preset_config("tiny").to_dict()
    del payload["model"]["initializer_range"]
    incomplete_tiny.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="initializer_range"):
        load_preset_config("tiny", incomplete_tiny)

    unsupported_head_dim = tmp_path / "unsupported-head-dim.json"
    payload = load_preset_config("tiny").to_dict()
    payload["model"]["hidden_size"] = 32
    unsupported_head_dim.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="multiple of 16"):
        load_preset_config("tiny", unsupported_head_dim)

    bad_qwen = tmp_path / "bad-qwen.json"
    payload = load_preset_config("qwen3-30b").to_dict()
    payload["expected_hf"]["unconsumed_shape"] = 7
    bad_qwen.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="expected_hf.*unknown fields"):
        load_preset_config("qwen3-30b", bad_qwen)

    bad_topologies = tmp_path / "topologies.json"
    bad_topologies.write_text(
        json.dumps(
            [
                {"name": "same", "world_size": 1, "tp": 1, "cp": 1, "dp": 1, "ep": 1},
                {"name": "same", "world_size": 1, "tp": 1, "cp": 1, "dp": 1, "ep": 1},
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate topology names"):
        load_topology_matrix(bad_topologies)


def test_train_dry_run_reports_resolved_json_and_explicit_overrides(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from mor_mlite.train import main

    assert (
        main(
            [
                "--dry-run",
                "--preset",
                "tiny",
                "--n-start-layers",
                "2",
                "--num-recursions",
                "2",
                "--capacity-schedule",
                "1.0,0.5",
                "--router-temperature",
                "0.75",
            ]
        )
        == 0
    )
    plan = json.loads(capsys.readouterr().out)
    assert plan["preset_config_source"].endswith("configs/tiny.json")
    assert plan["architecture"]["n_start_layers"] == 2
    assert plan["architecture"]["num_recursions"] == 2
    assert plan["architecture"]["capacity_schedule"] == [1.0, 0.5]
    assert plan["depth_router"]["temperature"] == pytest.approx(0.75)
    assert plan["topology"]["name"] == "baseline"


def test_training_cli_does_not_expose_unvalidated_parallel_degrees() -> None:
    from mor_mlite.train import _parser

    with pytest.raises(SystemExit):
        _parser().parse_args(["--dp", "2"])
