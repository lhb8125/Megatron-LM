from __future__ import annotations

import ast
import importlib
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest


def _loaded(prefix: str) -> set[str]:
    return {name for name in sys.modules if name == prefix or name.startswith(f"{prefix}.")}


def test_mlite_adapter_and_converter_import_without_mlite_or_torch() -> None:
    before_megatron = _loaded("megatron")
    before_torch = _loaded("torch")

    importlib.import_module("mor_mlite.parity.mlite")
    importlib.import_module("mor_mlite.convert_hf")

    assert _loaded("megatron") == before_megatron
    assert _loaded("torch") == before_torch


def test_hf_converter_dry_run_prints_exact_fold_map_without_runtime(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    before_megatron = _loaded("megatron")
    from mor_mlite.convert_hf import main

    hf_root = tmp_path / "hf"
    hf_root.mkdir()
    (hf_root / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_moe",
                "num_hidden_layers": 8,
                "num_experts": 4,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "checkpoint"
    result = main(
        [
            "--hf-path",
            str(hf_root),
            "--output",
            str(output),
            "--n-start-layers",
            "1",
            "--n-recurrent-layers",
            "2",
            "--num-recursions",
            "3",
            "--n-end-layers",
            "1",
            "--folding-policy",
            "mean",
            "--dry-run",
        ]
    )

    assert result == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["direction"] == "hf-to-mlite"
    assert plan["reverse_hf_export_supported"] is False
    assert plan["hf_source"] == str(hf_root)
    assert plan["magi_attention_version"] == "1.1.1"
    assert plan["depth_router"] == {
        "temperature": 1.0,
        "alpha": 0.1,
        "aux_loss_coef": 0.001,
    }
    assert plan["depth_router_seed"] == 1234
    assert plan["parallel"] == {
        "dp": 1,
        "tp": 1,
        "cp": 1,
        "ep": 1,
        "etp": 1,
        "zero_stage": 1,
        "cp_transition": "magi_direct",
    }
    assert plan["logical_num_layers"] == 8
    assert plan["physical_num_layers"] == 4
    assert plan["physical_to_logical_layers"] == {
        "0": [0],
        "1": [1, 3, 5],
        "2": [2, 4, 6],
        "3": [7],
    }
    assert not output.exists()
    assert _loaded("megatron") == before_megatron


def test_hf_converter_dry_run_validates_expert_topology_without_runtime(
    tmp_path: Path,
) -> None:
    before_megatron = _loaded("megatron")
    from mor_mlite.convert_hf import main

    hf_root = tmp_path / "hf"
    hf_root.mkdir()
    (hf_root / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_moe",
                "num_hidden_layers": 8,
                "num_experts": 4,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="num_experts=4 must be divisible by ep=3"):
        main(
            [
                "--hf-path",
                str(hf_root),
                "--output",
                str(tmp_path / "checkpoint"),
                "--n-start-layers",
                "1",
                "--n-recurrent-layers",
                "2",
                "--num-recursions",
                "3",
                "--n-end-layers",
                "1",
                "--dp",
                "3",
                "--ep",
                "3",
                "--dry-run",
            ]
        )
    assert _loaded("megatron") == before_megatron


def test_hf_converter_rejects_reverse_export_at_parse_time(tmp_path: Path) -> None:
    from mor_mlite.convert_hf import main

    with pytest.raises(SystemExit) as error:
        main(
            [
                "--direction",
                "export",
                "--hf-path",
                "unused",
                "--output",
                str(tmp_path / "out"),
            ]
        )
    assert error.value.code == 2


def test_hf_converter_rejects_stale_output_and_config_file_containment(
    tmp_path: Path,
) -> None:
    from mor_mlite.config import MoRArchitectureConfig
    from mor_mlite.convert_hf import build_import_plan

    hf_root = tmp_path / "hf"
    hf_root.mkdir()
    config_path = hf_root / "config.json"
    config_path.write_text(json.dumps({"num_hidden_layers": 8}), encoding="utf-8")
    topology = {"world_size": 1, "tp": 1, "cp": 1, "dp": 1, "ep": 1, "etp": 1}
    kwargs = {
        "hf_path": str(config_path),
        "architecture": MoRArchitectureConfig(1, 2, 3, 1),
        "folding_policy": "mean",
        "topology": topology,
        "cp_transition": "magi_direct",
        "seed": 1234,
    }

    with pytest.raises(ValueError, match="nested in the source"):
        build_import_plan(output=hf_root / "converted", **kwargs)

    outside = tmp_path / "converted"
    outside.mkdir()
    (outside / "step_99").mkdir()
    with pytest.raises(FileExistsError, match="stale step"):
        build_import_plan(output=outside, **kwargs)


def test_converter_source_contains_no_runtime_export_call() -> None:
    from mor_mlite import convert_hf

    tree = ast.parse(Path(convert_hf.__file__).read_text(encoding="utf-8"))
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "export_weights" not in called_attributes
    assert "save_hf_weights" not in called_attributes


def test_mcore_model_checkpoint_hook_is_independent_of_optimizer_choice() -> None:
    protocol_path = (
        Path(__file__).parents[1] / "src" / "mor_mlite" / "qwen3_moe_mor" / "protocol.py"
    )
    tree = ast.parse(protocol_path.read_text(encoding="utf-8"))
    build = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "build_model"
    )
    optimizer_branches = [
        node
        for node in build.body
        if isinstance(node, ast.If)
        and any(
            isinstance(part, ast.Attribute) and part.attr == "optimizer"
            for part in ast.walk(node.test)
        )
    ]
    assert optimizer_branches
    attach_indices = [
        index
        for index, statement in enumerate(build.body)
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == "attach_model_sharded_state_dict"
    ]
    assert len(attach_indices) == 1
    optimizer_branch_index = build.body.index(optimizer_branches[0])
    assert attach_indices[0] > optimizer_branch_index
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "attach_model_sharded_state_dict"
        for branch in optimizer_branches
        for node in ast.walk(branch)
    )


def test_eos_runtime_overlay_is_exact_and_fail_closed() -> None:
    root = Path(__file__).parents[1]
    common = (root / "scripts" / "eos" / "common.sh").read_text(encoding="utf-8")
    setup = (root / "scripts" / "eos" / "setup_env.sh").read_text(encoding="utf-8")
    probe = (root / "scripts" / "eos" / "base_image_probe.sh").read_text(encoding="utf-8")

    assert 'TORCH_WHEEL_VERSION="2.10.0+cu129"' in common
    assert 'TORCH_WHEEL_INDEX="https://download.pytorch.org/whl/cu129"' in common
    assert 'TRANSFORMER_ENGINE_VERSION="2.13.0"' in common
    assert 'MOR_VENV="${DEPS_ROOT}/venv-torch210-cu129-v3"' in common
    assert 'MOR_PIP_CONSTRAINT="${EOS_WORKDIR}/scripts/eos/constraints-cu129.txt"' in common
    assert 'NVRX_VERSION="0.6.0"' in common
    assert '"torch==${TORCH_WHEEL_VERSION}"' in setup
    assert '--index-url "https://pypi.org/simple"' in setup
    assert '--extra-index-url "${TORCH_WHEEL_INDEX}"' in setup
    assert '"nvidia-resiliency-ext==${NVRX_VERSION}"' in setup
    assert 'torch.version.cuda != "12.9"' in setup
    assert 'export PIP_CONSTRAINT="${MOR_PIP_CONSTRAINT}"' in setup
    assert 'export PIP_EXTRA_INDEX_URL="${TORCH_WHEEL_INDEX}"' in setup
    constraints = (root / "scripts" / "eos" / "constraints-cu129.txt").read_text(encoding="utf-8")
    assert "torch==2.10.0+cu129" in constraints
    assert "triton==3.6.0" in constraints
    assert "cuda-python==12.9.4" in constraints
    assert "cuda-bindings==12.9.4" in constraints
    assert "packaging==25.0" in constraints
    assert "MAGI_ATTENTION_SKIP_MAGI_ATTN_COMM_BUILD=1" in setup
    assert "export MAGI_ATTENTION_NATIVE_GRPCOLL=0" in common
    assert "export MAGI_ATTENTION_KERNEL_BACKEND=ffa" in common
    assert "export MAGI_ATTENTION_HIERARCHICAL_COMM=0" in common
    assert "export MAGI_ATTENTION_QO_COMM=0" in common
    assert "export MOR_VENV" in common
    assert 'distribution("transformer-engine-torch")' in setup
    assert 'distribution("transformer-engine-cu12")' in setup
    assert "moe_permute_and_pad_with_probs" in setup
    assert 'startswith("transformer-engine-cu13")' in setup
    assert "python -m mor_mlite.runtime_canary te" in setup
    canary = (root / "src" / "mor_mlite" / "runtime_canary.py").read_text(encoding="utf-8")
    assert "moe_permute_and_pad_with_probs" in canary
    assert "from megatron.lite.primitive.utils import moe as mlite_moe" in canary
    version_probe = (root / "scripts" / "eos" / "version_probe.sh").read_text(encoding="utf-8")
    assert "python -m mor_mlite.runtime_canary te --output" in version_probe
    tiny = (root / "scripts" / "eos" / "run_tiny_matrix.sh").read_text(encoding="utf-8")
    assert '"${SCRIPT_DIR}/magi_canary.sh"' in tiny
    assert "--checkpoint-save-only" in tiny
    assert (
        '--resume-checkpoint "${ARTIFACT_ROOT}/external_checkpoint_save/runtime-checkpoint"' in tiny
    )
    assert "-m mor_mlite.parity certify-checkpoint" in tiny
    magi_canary = (root / "scripts" / "eos" / "magi_canary.sh").read_text(encoding="utf-8")
    assert "--nproc-per-node=2" in magi_canary
    assert "test_magi_attention_operator.py" in magi_canary
    assert "MLITE_TEST_HARNESS=1" in magi_canary
    qwen_smoke = (root / "scripts" / "eos" / "run_qwen_smoke.sh").read_text(encoding="utf-8")
    assert "-m mor_mlite.convert_hf" in qwen_smoke
    assert '"${SCRIPT_DIR}/magi_canary.sh"' in qwen_smoke
    assert '--init-checkpoint "${ARTIFACT_ROOT}/folded_init_ep1"' in qwen_smoke
    assert '--init-checkpoint "${ARTIFACT_ROOT}/folded_init_ep4"' in qwen_smoke
    assert '--output "${ARTIFACT_ROOT}/folded_init_ep4"' in qwen_smoke
    assert '--output "${ARTIFACT_ROOT}/all_forward"' in qwen_smoke
    assert "--reference-dp-shards 2" in qwen_smoke
    assert "--checkpoint-save-only" in qwen_smoke
    assert '--resume-checkpoint "${ARTIFACT_ROOT}/all_train_save/runtime-checkpoint"' in qwen_smoke
    assert "-m mor_mlite.parity certify-checkpoint" in qwen_smoke
    assert 'PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"' in qwen_smoke
    assert 'startswith("13.1")' in probe


def test_mlite_source_has_no_stale_tiny_materializer_call() -> None:
    from mor_mlite.parity import mlite

    tree = ast.parse(Path(mlite.__file__).read_text(encoding="utf-8"))
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "_materialize_tiny_config" not in called_names
    assert "_materialize_tiny_hf" in called_names


def test_parity_cli_resolves_process_isolated_checkpoint_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from mor_mlite.parity import mlite
    from mor_mlite.parity.__main__ import main

    captured = []

    def fake_run(config):
        captured.append(config)
        return config.output

    monkeypatch.setattr(mlite, "run_mlite", fake_run)
    assert (
        main(
            [
                "run",
                "--backend",
                "mlite",
                "--precision",
                "bf16",
                "--checkpoint-save-only",
                "--output",
                str(tmp_path / "save"),
            ]
        )
        == 0
    )
    assert captured[-1].checkpoint_save_only is True
    assert captured[-1].checkpoint_roundtrip is False

    assert (
        main(
            [
                "run",
                "--backend",
                "mlite",
                "--precision",
                "bf16",
                "--resume-checkpoint",
                str(tmp_path / "checkpoint"),
                "--output",
                str(tmp_path / "resume"),
            ]
        )
        == 0
    )
    assert captured[-1].resume_checkpoint == tmp_path / "checkpoint"
    assert captured[-1].checkpoint_roundtrip is False
    capsys.readouterr()


def test_qwen_parity_rejects_same_process_checkpoint_roundtrip(tmp_path: Path) -> None:
    from mor_mlite.parity.mlite import MLiteRunConfig

    config = MLiteRunConfig(
        output=tmp_path,
        preset="qwen3-30b",
        checkpoint_roundtrip=True,
    )
    with pytest.raises(ValueError, match="separate.*torchrun processes"):
        config.validate()


def test_serial_reference_dp_shards_are_forward_only_baseline_contract(tmp_path: Path) -> None:
    from mor_mlite.parity.mlite import MLiteRunConfig

    valid = MLiteRunConfig(
        output=tmp_path / "valid",
        topology="baseline",
        forward_only=True,
        checkpoint_roundtrip=False,
        reference_dp_shards=2,
        seq_lens=(8, 8),
    )
    assert valid.validate().name == "baseline"

    invalid = (
        replace(valid, topology="zero1"),
        replace(valid, forward_only=False),
        replace(valid, route_mode="replay", replay_from=tmp_path / "replay"),
        replace(valid, reference_dp_shards=3),
        replace(valid, reference_dp_shards=1.5),
        replace(valid, reference_dp_shards=float("nan")),
    )
    for config in invalid:
        with pytest.raises(ValueError, match="reference_dp_shards"):
            config.validate()


def test_serial_reference_replay_contract_is_checked_before_runtime(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    from mor_mlite.parity.artifacts import save_artifact
    from mor_mlite.parity.mlite import MLiteRunConfig

    replay = save_artifact(
        tmp_path / "replay",
        metadata={
            "backend": "mlite",
            "preset": "tiny",
            "precision": "bf16",
            "reference_dp_shards": 2,
            "global_batch": {
                "sequence_lengths": [8, 8],
                "partition_policy": "deterministic-longest-first-whole-sequence",
                "sample_partitions": [[0], [1]],
            },
        },
        tensors={"forward/logits": torch.zeros(1)},
        routes=[],
    )
    valid = MLiteRunConfig(
        output=tmp_path / "candidate",
        topology="zero1",
        route_mode="replay",
        replay_from=replay,
        seq_lens=(8, 8),
    )
    assert valid.validate().dp == 2

    with pytest.raises(ValueError, match="must equal target dense-DP"):
        replace(valid, topology="baseline").validate()


def test_serial_reference_callback_identity_is_explicit_and_exact_once() -> None:
    from mor_mlite.parity.mlite import _claim_serial_reference_callback

    seen: set[tuple[int, int]] = set()
    batch = SimpleNamespace(extras={"mor_parity_step": 3, "mor_parity_microbatch": 1})
    assert (
        _claim_serial_reference_callback(
            batch,
            expected_step=3,
            num_microbatches=2,
            shard=0,
            seen=seen,
        )
        == 1
    )
    with pytest.raises(RuntimeError, match="duplicate"):
        _claim_serial_reference_callback(
            batch,
            expected_step=3,
            num_microbatches=2,
            shard=0,
            seen=seen,
        )
    with pytest.raises(RuntimeError, match="invalid step/microbatch"):
        _claim_serial_reference_callback(
            SimpleNamespace(extras={"mor_parity_step": 4, "mor_parity_microbatch": 0}),
            expected_step=3,
            num_microbatches=2,
            shard=1,
            seen=seen,
        )


def test_dense_dp_global_batch_partition_and_loss_scales_are_exact() -> None:
    from mor_mlite.config import MoRArchitectureConfig
    from mor_mlite.parity.mlite import (
        _balanced_sample_partitions,
        _dp_objective_scales,
    )

    architecture = MoRArchitectureConfig.tiny()
    partitions = _balanced_sample_partitions((9, 6, 3), 2)
    scales = _dp_objective_scales((9, 6, 3), partitions, architecture, training=True)

    assert partitions == ((0,), (1, 2))
    assert scales[0]["lm_scale"] == pytest.approx(16 / 15)
    assert scales[1]["lm_scale"] == pytest.approx(14 / 15)
    assert scales[0]["global_input_tokens"] == 18
    assert scales[1]["global_input_tokens"] == 18
    assert scales[0]["aux_scales"] == pytest.approx((1.0, 1.0, 1.0))
    assert scales[1]["aux_scales"] == pytest.approx((1.0, 1.0, 1.0))
    assert scales[0]["local_router_candidates_by_round"] == (9, 9, 6)
    assert scales[1]["local_router_candidates_by_round"] == (9, 9, 6)


def test_distributed_parameter_fingerprint_detects_payload_change() -> None:
    torch = pytest.importorskip("torch")
    from mor_mlite.parity.mlite import _distributed_parameter_fingerprint

    model = torch.nn.Linear(3, 2)
    handle = SimpleNamespace(_model=model, _extras={"model_chunks": [model]})
    original = _distributed_parameter_fingerprint(handle)
    repeated = _distributed_parameter_fingerprint(handle)
    assert original["sha256"] == repeated["sha256"]
    assert original["parameter_counts"] == [2]

    with torch.no_grad():
        model.weight[0, 0].add_(1.0)
    changed = _distributed_parameter_fingerprint(handle)
    assert changed["sha256"] != original["sha256"]


def test_dense_dp_uses_per_round_aux_scales_for_arbitrary_sequence_lengths() -> None:
    from mor_mlite.config import MoRArchitectureConfig
    from mor_mlite.parity.mlite import (
        _balanced_sample_partitions,
        _dp_objective_scales,
    )

    architecture = MoRArchitectureConfig.tiny()
    partitions = _balanced_sample_partitions((9, 5, 4), 2)
    scales = _dp_objective_scales((9, 5, 4), partitions, architecture, training=True)

    assert partitions == ((0,), (1, 2))
    assert scales[0]["local_router_candidates_by_round"] == (9, 9, 6)
    assert scales[1]["local_router_candidates_by_round"] == (9, 9, 5)
    assert scales[0]["aux_scales"] == pytest.approx((1.0, 1.0, 12 / 11))
    assert scales[1]["aux_scales"] == pytest.approx((1.0, 1.0, 10 / 11))


def test_runtime_checkpoint_sidecar_has_full_folding_provenance() -> None:
    from mor_mlite.config import MoRArchitectureConfig
    from mor_mlite.parity.mlite import _checkpoint_sidecar

    payload = _checkpoint_sidecar(
        MoRArchitectureConfig.tiny(),
        folding_policy="mean",
        hf_source="generated:test",
    )
    assert payload["hf_source"] == "generated:test"
    assert payload["magi_attention_version"] == "1.1.1"
    assert payload["megatron_lm_sha"] == ("5c8315f12a64a7279eec58896af9e74ee3351b74")
    assert payload["depth_router"] == {
        "temperature": 1.0,
        "alpha": 0.1,
        "aux_loss_coef": 0.001,
    }
    assert payload["depth_router_seed"] == 1234
    assert payload["cp_transition"] == "magi_direct"
    assert payload["parallel"] == {
        "dp": 1,
        "tp": 1,
        "cp": 1,
        "ep": 1,
        "etp": 1,
        "zero_stage": 1,
        "cp_transition": "magi_direct",
    }
    assert payload["physical_to_logical_layers"] == {
        "0": [0],
        "1": [1, 3, 5],
        "2": [2, 4, 6],
        "3": [7],
    }


def test_init_checkpoint_recovers_model_semantics_and_allows_dcp_reshard(
    tmp_path: Path,
) -> None:
    from mor_mlite.checkpoint_io import build_checkpoint_metadata
    from mor_mlite.config import (
        DepthRouterConfig,
        MoRArchitectureConfig,
        MoRParallelConfig,
    )
    from mor_mlite.parity.mlite import (
        MLiteRunConfig,
        _resolve_checkpoint_initialization,
    )

    checkpoint = tmp_path / "converted"
    checkpoint.mkdir()
    architecture = MoRArchitectureConfig(2, 11, 4, 2)
    router = DepthRouterConfig(temperature=0.75, alpha=0.2, aux_loss_coef=0.01)
    source_parallel = MoRParallelConfig()
    metadata = build_checkpoint_metadata(
        architecture=architecture,
        depth_router=router,
        depth_router_seed=9876,
        hf_source="Qwen/source",
        parallel=source_parallel,
    )
    (checkpoint / "mor_config.json").write_text(json.dumps(metadata.to_dict()), encoding="utf-8")
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_moe",
                "num_hidden_layers": architecture.logical_num_layers,
                "num_experts": 128,
                "num_experts_per_tok": 8,
            }
        ),
        encoding="utf-8",
    )
    config = MLiteRunConfig(
        output=tmp_path / "artifact",
        preset="qwen3-30b",
        topology="zero1",
        init_checkpoint=checkpoint,
        checkpoint_roundtrip=False,
    )

    topology = config.validate()
    initialization = _resolve_checkpoint_initialization(config, topology)

    assert initialization is not None
    assert initialization.metadata.architecture == architecture
    assert initialization.metadata.depth_router == router
    assert initialization.metadata.depth_router_seed == 9876
    assert initialization.cp_transition == "magi_direct"
    assert initialization.metadata.parallel == source_parallel
    assert initialization.runtime_metadata.parallel == topology.to_parallel_config()

    incompatible_ep = MLiteRunConfig(
        output=tmp_path / "incompatible",
        preset="qwen3-30b",
        topology="all",
        init_checkpoint=checkpoint,
        checkpoint_roundtrip=False,
    )
    with pytest.raises(ValueError, match="cannot reshard.*EP"):
        incompatible_ep.validate()

    conflicting_router = MLiteRunConfig(
        output=tmp_path / "conflicting-router",
        preset="qwen3-30b",
        topology="zero1",
        init_checkpoint=checkpoint,
        depth_router=DepthRouterConfig(aux_loss_coef=0.0),
        checkpoint_roundtrip=False,
    )
    with pytest.raises(ValueError, match="depth-router configuration conflicts"):
        conflicting_router.validate()


def test_init_checkpoint_rejects_a_checkpoint_from_another_preset(tmp_path: Path) -> None:
    from mor_mlite import train
    from mor_mlite.checkpoint_io import build_checkpoint_metadata
    from mor_mlite.config import DepthRouterConfig, MoRArchitectureConfig, MoRParallelConfig
    from mor_mlite.parity.mlite import MLiteRunConfig, _tiny_hf_dict

    checkpoint = tmp_path / "tiny-checkpoint"
    checkpoint.mkdir()
    architecture = MoRArchitectureConfig.tiny()
    metadata = build_checkpoint_metadata(
        architecture=architecture,
        depth_router=DepthRouterConfig(),
        depth_router_seed=1234,
        hf_source="generated:tiny",
        parallel=MoRParallelConfig(),
    )
    (checkpoint / "mor_config.json").write_text(json.dumps(metadata.to_dict()), encoding="utf-8")
    (checkpoint / "config.json").write_text(
        json.dumps(_tiny_hf_dict(architecture)), encoding="utf-8"
    )

    config = MLiteRunConfig(
        output=tmp_path / "artifact",
        preset="qwen3-30b",
        topology="baseline",
        init_checkpoint=checkpoint,
        checkpoint_roundtrip=False,
    )
    with pytest.raises(ValueError, match="incompatible with preset 'qwen3-30b'"):
        config.validate()

    with pytest.raises(ValueError, match="depth-router configuration conflicts"):
        train.main(
            [
                "--dry-run",
                "--backend",
                "mlite",
                "--preset",
                "tiny",
                "--init-checkpoint",
                str(checkpoint),
                "--router-aux-loss-coef",
                "0",
            ]
        )


def test_train_cli_passes_model_only_init_checkpoint_to_mlite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from mor_mlite import train
    from mor_mlite.parity import mlite

    captured = []

    def fake_run(config):
        captured.append(config)
        return config.output

    monkeypatch.setattr(mlite, "run_mlite", fake_run)
    checkpoint = tmp_path / "converted"
    result = train.main(
        [
            "--backend",
            "mlite",
            "--preset",
            "qwen3-30b",
            "--init-checkpoint",
            str(checkpoint),
            "--output",
            str(tmp_path / "artifact"),
            "--adam-eps",
            "2e-6",
        ]
    )

    assert result == 0
    assert captured[0].init_checkpoint == checkpoint
    assert captured[0].hf_path == ""
    assert captured[0].cp_transition is None
    assert captured[0].adam_eps == pytest.approx(2e-6)
    assert json.loads(capsys.readouterr().out)["artifact"] == str(tmp_path / "artifact")


def test_tiny_hf_source_is_topology_independent_and_folding_exact() -> None:
    torch = pytest.importorskip("torch")

    from mor_mlite.config import MoRArchitectureConfig
    from mor_mlite.parity.mlite import _tiny_hf_weights

    architecture = MoRArchitectureConfig.tiny()
    first = _tiny_hf_weights(architecture, seed=1234)
    second = _tiny_hf_weights(architecture, seed=1234)
    assert first.keys() == second.keys()
    assert all(torch.equal(first[name], second[name]) for name in first)
    for suffix in (
        "self_attn.q_proj.weight",
        "mlp.experts.3.down_proj.weight",
    ):
        assert torch.equal(
            first[f"model.layers.1.{suffix}"],
            first[f"model.layers.3.{suffix}"],
        )
        assert torch.equal(
            first[f"model.layers.3.{suffix}"],
            first[f"model.layers.5.{suffix}"],
        )
    assert "model.layers.7.mlp.experts.3.up_proj.weight" in first


class _ConfigBox:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakeRuntime:
    def __init__(self) -> None:
        self.build_calls = 0
        self.handle = object()

    def build_model(self):
        self.build_calls += 1
        return self.handle


@pytest.mark.parametrize(
    "cp, strict, backend", [(2, True, "magi"), (1, True, "local"), (1, False, "flash")]
)
def test_runtime_session_uses_public_runtime_config_and_build_model(
    monkeypatch: pytest.MonkeyPatch,
    cp: int,
    strict: bool,
    backend: str,
) -> None:
    import mor_mlite.register as registration
    from mor_mlite.config import MoRArchitectureConfig
    from mor_mlite.parity import mlite
    from mor_mlite.parity.topologies import Topology

    registered: list[bool] = []
    created: list[object] = []
    fake_runtime = _FakeRuntime()
    api = SimpleNamespace(
        RuntimeConfig=_ConfigBox,
        MegatronLiteConfig=_ConfigBox,
        OptimizerConfig=_ConfigBox,
        ParallelConfig=_ConfigBox,
        create_runtime=lambda config: (created.append(config), fake_runtime)[1],
    )
    monkeypatch.setattr(registration, "register_with_mlite", lambda: registered.append(True))
    monkeypatch.setattr(mlite, "_load_runtime_api", lambda: api)
    monkeypatch.setattr(
        mlite,
        "resolve_hf_checkpoint",
        lambda source, **_kwargs: SimpleNamespace(
            source=source, local_path=Path("/resolved/qwen"), downloaded=True
        ),
    )
    monkeypatch.setenv("WORLD_SIZE", "4")

    session = mlite.build_runtime_session(
        mlite.MLiteRuntimeBuildConfig(
            hf_path="Qwen/Qwen3-30B-A3B-Base",
            topology=Topology("test", 4, tp=2, cp=cp, dp=2 // cp, ep=2),
            architecture=MoRArchitectureConfig.qwen3_30b(),
            build_optimizer=False,
            cp_transition="magi_direct",
            route_mode="replay",
            strict=strict,
        )
    )

    assert registered == [True]
    assert created == [session.runtime_config]
    assert session.runtime is fake_runtime
    assert session.handle is fake_runtime.handle
    assert fake_runtime.build_calls == 1
    assert session.runtime_config.backend == "mlite"
    assert session.backend_config.model_name == "qwen3_moe_mor"
    assert session.backend_config.attention_backend_override == backend
    assert session.backend_config.impl_cfg["deterministic"] is strict
    assert session.backend_config.impl_cfg["local_attention_backend"] == (
        "magi_ffa" if cp == 1 and strict else "te"
    )
    assert session.backend_config.load_hf_weights is True
    assert session.backend_config.hf_path == "/resolved/qwen"
    assert session.backend_config.parallel.tp == 2
    assert session.backend_config.parallel.cp == cp
    assert session.backend_config.parallel.ep == 2
    assert session.backend_config.impl_cfg["optimizer"] is None
    assert session.backend_config.optimizer.adam_eps == pytest.approx(1e-6)
    assert session.backend_config.impl_cfg["route_mode"] == "replay"
    assert session.backend_config.impl_cfg["use_thd"] is True


def test_mlite_runtime_rejects_multinode_launch_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mor_mlite.parity.mlite import _validate_single_node_launch

    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    with pytest.raises(ValueError, match="single-node only"):
        _validate_single_node_launch(4)

    monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")
    monkeypatch.setenv("SLURM_NNODES", "2")
    with pytest.raises(ValueError, match="SLURM_NNODES=2"):
        _validate_single_node_launch(4)


def test_forward_only_is_a_first_class_mlite_run_config(tmp_path: Path) -> None:
    from mor_mlite.parity.mlite import MLiteRunConfig

    config = MLiteRunConfig(
        output=tmp_path,
        preset="qwen3-30b",
        topology="baseline",
        precision="bf16",
        hf_path="Qwen/Qwen3-30B-A3B-Base",
        forward_only=True,
        checkpoint_roundtrip=False,
    )
    assert config.validate().world_size == 1
    assert config.forward_only


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("field", ["lr", "adam_eps", "clip_grad"])
def test_parity_run_configs_reject_nonfinite_training_hyperparameters(
    tmp_path: Path, field: str, value: float
) -> None:
    from mor_mlite.parity.mlite import MLiteRunConfig
    from mor_mlite.parity.reference import ReferenceRunConfig

    for config in (ReferenceRunConfig(output=tmp_path), MLiteRunConfig(output=tmp_path)):
        setattr(config, field, value)
        with pytest.raises(ValueError, match="finite and positive"):
            config.validate()


@pytest.mark.parametrize("value", [0.0, -1e-6, float("nan")])
def test_train_dry_run_rejects_invalid_adam_epsilon(tmp_path: Path, value: float) -> None:
    from mor_mlite import train

    with pytest.raises(ValueError, match="finite and positive"):
        train.main(
            [
                "--backend",
                "reference",
                "--output",
                str(tmp_path / "artifact"),
                f"--adam-eps={value}",
                "--dry-run",
            ]
        )


def test_mlite_parity_requests_logits_and_uses_fresh_checkpoint_restore() -> None:
    from mor_mlite.parity import mlite

    source = Path(mlite.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    string_literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert "mor_return_full_logits" in string_literals
    assert "logits" in string_literals
    assert "hf_path=str(checkpoint)" in source
    assert "load_hf_weights=False" in source
    assert "save_mor_checkpoint(" in source
    assert "load_mor_checkpoint(" in source
    assert 'prefix="uninterrupted/"' in source
    assert "build_checkpoint_continuity_report(" in source
    assert 'config.preset == "tiny"' in source
    assert "dict(preset.model) == dict(default_tiny.model)" in source
    assert "architecture == preset.architecture" in source
    assert "verify_training_continuity = capture_full_state and not config.forward_only" in source
    assert "value.detach().cpu().clone()" in source

    # The forward-only batch must select the model's logits branch instead of
    # retaining labels and capturing training-only log-probabilities.
    assignments_to_none = {
        ast.unparse(node.targets[0])
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.value, ast.Constant)
        and node.value.value is None
    }
    assert "batch.labels" in assignments_to_none
    assert "batch.loss_mask" in assignments_to_none


class _FakeGradBucketGroup:
    def __init__(self) -> None:
        self.buckets = [object(), object()]
        self.is_first_batch = False
        self.grad_reduce_handle = None
        self.calls = 0

    def start_grad_sync(self) -> None:
        self.calls += 1


class _FakeDDPChunk:
    def __init__(self, group) -> None:
        self.ddp_config = SimpleNamespace(
            use_distributed_optimizer=True,
            overlap_grad_reduce=False,
        )
        self.force_all_reduce = False
        self.bucket_groups = [group]
        self.expert_parallel_bucket_groups = []


def test_gradient_sync_probe_wraps_runtime_finalizer_and_physical_buckets() -> None:
    from mor_mlite.parity.mlite import (
        _install_gradient_sync_probe,
        _merge_gradient_sync_step_reports,
        _summarize_gradient_sync_steps,
    )

    group = _FakeGradBucketGroup()
    finalize_calls = []

    def finalize() -> None:
        finalize_calls.append(True)
        group.start_grad_sync()

    handle = SimpleNamespace(
        _extras={
            "finalize_grads": finalize,
            "model_chunks": [_FakeDDPChunk(group)],
        }
    )
    probe = _install_gradient_sync_probe(handle, forward_only=False)
    probe.begin_step("train/step_000")
    handle._extras["finalize_grads"]()
    local = probe.finish_step("train/step_000")
    merged = _merge_gradient_sync_step_reports([local])
    summary = _summarize_gradient_sync_steps([merged], required=True)

    assert finalize_calls == [True]
    assert group.calls == 1
    assert summary["status"] == "available"
    assert summary["finalize_grads_calls_min"] == 1
    assert summary["finalize_grads_calls_max"] == 1
    assert summary["physical_bucket_sync_calls_min"] == 1
    assert summary["physical_bucket_sync_calls_max"] == 1
    assert len(local["physical_buckets"]["sync_calls"]) == 2


def test_forward_only_gradient_probe_is_explicitly_not_required() -> None:
    from mor_mlite.parity.mlite import (
        _install_gradient_sync_probe,
        _merge_gradient_sync_step_reports,
        _summarize_gradient_sync_steps,
    )

    probe = _install_gradient_sync_probe(SimpleNamespace(), forward_only=True)
    probe.begin_step("train/step_000")
    merged = _merge_gradient_sync_step_reports([probe.finish_step("train/step_000")])
    summary = _summarize_gradient_sync_steps([merged], required=False)
    assert summary["status"] == "not_required"
    assert summary["physical_bucket_sync_calls_max"] is None


class _ReadOnlyGradBucketGroup:
    __slots__ = ("buckets", "grad_reduce_handle", "is_first_batch")

    def __init__(self) -> None:
        self.buckets = [object()]
        self.is_first_batch = False
        self.grad_reduce_handle = None

    def start_grad_sync(self) -> None:
        return None


def test_unpatchable_bucket_probe_is_explicitly_unavailable() -> None:
    from mor_mlite.parity.mlite import _install_gradient_sync_probe

    group = _ReadOnlyGradBucketGroup()
    handle = SimpleNamespace(
        _extras={
            "finalize_grads": lambda: None,
            "model_chunks": [_FakeDDPChunk(group)],
        }
    )
    probe = _install_gradient_sync_probe(handle, forward_only=False)
    probe.begin_step("train/step_000")
    report = probe.finish_step("train/step_000")
    assert report["physical_buckets"]["status"] == "unavailable"
    assert "not patchable" in report["physical_buckets"]["reason"]


class _FakeParameterLayer:
    def __init__(self, parameter: object) -> None:
        self.parameter = parameter

    def named_parameters(self, *, recurse: bool, remove_duplicate: bool):
        assert recurse and not remove_duplicate
        return (("weight", self.parameter),)


class _FakeMoRModel:
    def __init__(self, architecture, *, duplicate_recurrent: bool = False) -> None:
        parameters = [object() for _ in range(architecture.physical_num_layers)]
        if duplicate_recurrent:
            parameters[architecture.n_start_layers + 1] = parameters[architecture.n_start_layers]
        self.layers = [_FakeParameterLayer(parameter) for parameter in parameters]
        self.mor_architecture = architecture

    def named_parameters(self, *, recurse: bool, remove_duplicate: bool):
        assert recurse and not remove_duplicate
        return tuple(
            (f"layers.{index}.weight", layer.parameter) for index, layer in enumerate(self.layers)
        )


def test_recurrent_parameter_registration_audit_detects_aliases() -> None:
    from mor_mlite.config import MoRArchitectureConfig
    from mor_mlite.parity.mlite import _assert_recurrent_parameters_registered_once

    architecture = MoRArchitectureConfig.tiny()
    model = _FakeMoRModel(architecture)
    handle = SimpleNamespace(_extras={"model_chunks": [model]})
    report = _assert_recurrent_parameters_registered_once(handle, architecture)
    assert report["status"] == "passed"
    assert report["max_registrations_per_recurrent_parameter"] == 1
    assert report["logical_reuses_per_recurrent_parameter"] == 3

    duplicated = _FakeMoRModel(architecture, duplicate_recurrent=True)
    duplicated_handle = SimpleNamespace(_extras={"model_chunks": [duplicated]})
    with pytest.raises(RuntimeError, match="registered exactly once"):
        _assert_recurrent_parameters_registered_once(duplicated_handle, architecture)


@pytest.mark.mlite
def test_pinned_mlite_runtime_contract_is_present_when_installed() -> None:
    pytest.importorskip("torch")
    runtime = pytest.importorskip("megatron.lite.runtime")
    contracts = pytest.importorskip("megatron.lite.runtime.contracts")

    assert callable(runtime.create_runtime)
    assert contracts.RuntimeConfig is runtime.RuntimeConfig
    for name in ("MegatronLiteConfig", "OptimizerConfig", "ParallelConfig"):
        assert getattr(contracts, name) is not None


@pytest.mark.mlite
@pytest.mark.cuda
def test_mlite_runtime_adapter_smoke_is_explicitly_opt_in(tmp_path: Path) -> None:
    if os.environ.get("MOR_RUN_MLITE_CLI_SMOKE") != "1":
        pytest.skip("set MOR_RUN_MLITE_CLI_SMOKE=1 under torchrun to run the CUDA smoke")
    torch = pytest.importorskip("torch")
    pytest.importorskip("megatron.lite.runtime")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    from mor_mlite.parity.mlite import MLiteRunConfig, run_mlite

    artifact = run_mlite(
        MLiteRunConfig(
            output=tmp_path / "artifact",
            preset="tiny",
            topology="baseline",
            precision="bf16",
            steps=1,
            num_microbatches=1,
            seq_lens=(4, 2),
            checkpoint_roundtrip=True,
            forward_only=False,
        )
    )
    assert artifact == tmp_path / "artifact"
    assert (artifact / "manifest.json").is_file()
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["parameter_capture"]["initial_weights"] is True
    assert manifest["parameter_capture"]["complete_gradients"] is True
    assert manifest["parameter_capture"]["post_step_weights"] is True
    assert manifest["checkpoint_restored_step"] == 1
    assert manifest["checkpoint_next_step"] is True
    assert manifest["checkpoint_uninterrupted_step"] is True
    assert manifest["checkpoint_continuity"]["status"] == "passed"
    assert manifest["model_structure"]["status"] == "passed"
    assert manifest["model_structure"]["max_registrations_per_recurrent_parameter"] == 1
    assert manifest["communication"]["grad_sync_probe"]["status"] == "available"
    assert manifest["communication"]["grad_sync_probe"]["finalize_grads_calls_max"] == 1
    assert manifest["communication"]["physical_bucket_sync_dispatch_max"] == 1
    assert manifest["expert_route_probe"]["enabled"] is True
    assert manifest["expert_route_probe"]["dummy_padding_excluded"] is True
    assert manifest["expert_route_probe"]["captured_contexts"] > 0
    assert any(
        name.startswith("expert_route/train/step_000/") and name.endswith("/topk_indices")
        for name in manifest["tensor_index"]
    )
    assert (artifact / "runtime-checkpoint" / "mor_config.json").is_file()
    assert any(
        name.startswith("forward/uninterrupted/step_001/") for name in manifest["tensor_index"]
    )
    assert any(name.startswith("forward/resume/step_001/") for name in manifest["tensor_index"])
