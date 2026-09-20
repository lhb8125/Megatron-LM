from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from mor_mlite.parity.artifacts import load_artifact, save_artifact
from mor_mlite.parity.compare import (
    _check_metadata_compatibility,
)
from mor_mlite.parity.compare import (
    compare_artifacts as _compare_artifacts,
)
from mor_mlite.parity.metrics import classify_cutoff, compare_tensor
from mor_mlite.parity.mlite import _expert_route_replay_index, _route_index
from mor_mlite.parity.topologies import TOPOLOGY_MATRIX, get_topology


def compare_artifacts(*args, **kwargs):
    # These unit fixtures isolate individual tensor/route gates and deliberately
    # contain partial evidence. Complete acceptance is tested separately.
    report = _compare_artifacts(*args, diagnostic=True, **kwargs)
    assert report["status"] == "partial"
    assert not report["acceptance_complete"]
    return report


def _gradient_sync_probe(
    *,
    status: str = "available",
    finalize_calls: int = 1,
    bucket_calls: int | None = 1,
) -> dict:
    step = {
        "global_step": "train/step_000",
        "required": True,
        "status": status,
        "reasons": [] if status == "available" else ["probe unavailable"],
        "finalize_grads_calls_min": finalize_calls,
        "finalize_grads_calls_max": finalize_calls,
        "physical_bucket_sync_calls_min": bucket_calls,
        "physical_bucket_sync_calls_max": bucket_calls,
        "ranks": [],
    }
    return {
        "required": True,
        "status": status,
        "steps": [step],
        "finalize_grads_calls_min": finalize_calls,
        "finalize_grads_calls_max": finalize_calls,
        "physical_bucket_sync_calls_min": bucket_calls,
        "physical_bucket_sync_calls_max": bucket_calls,
    }


def _communication() -> dict:
    return {
        "active_set_changes": 2,
        "hidden_rebalances": 2,
        "recurrent_inner_dispatches": 0,
        "early_exit_qkv_tokens": 0,
        "recurrent_qkv_checks": 6,
        "physical_bucket_sync_dispatch_max": 1,
        "grad_sync_probe": _gradient_sync_probe(),
    }


def test_serial_reference_dp_metadata_must_match_candidate_partition() -> None:
    global_batch = {
        "sequence_lengths": [8, 8],
        "partition_policy": "deterministic-longest-first-whole-sequence",
        "sample_partitions": [[0], [1]],
    }
    baseline = {
        "reference_dp_shards": 2,
        "global_batch": global_batch,
    }
    candidate = {
        "topology": {"dp": 2},
        "global_batch": dict(global_batch),
    }
    assert _check_metadata_compatibility(baseline, candidate)["passed"]

    wrong_dp = {**candidate, "topology": {"dp": 1}}
    assert not _check_metadata_compatibility(baseline, wrong_dp)["passed"]

    wrong_partition = {
        **candidate,
        "global_batch": {**global_batch, "sample_partitions": [[0, 1], []]},
    }
    assert not _check_metadata_compatibility(baseline, wrong_partition)["passed"]


def _route(*, selected: list[int] | None = None) -> dict:
    selected = [10, 11] if selected is None else selected
    positions = {10: 0, 11: 1, 12: 2}
    return {
        "schema_version": 1,
        "phase": "train",
        "step": 0,
        "microbatch": 0,
        "round": 1,
        "mode": "learned",
        "sample_ids": [0] * len(selected),
        "original_positions": [positions[token_id] for token_id in selected],
        "global_token_ids": selected,
        "source_tp_ranks": [0] * len(selected),
        "source_cp_ranks": [0] * len(selected),
        "source_local_rows": list(range(len(selected))),
        "target_tp_ranks": [0] * len(selected),
        "target_cp_ranks": [0] * len(selected),
        "target_local_rows": list(range(len(selected))),
        "selected_gates": [0.09] * len(selected),
        "active_cu_seqlens": [0, len(selected)],
        "padding_mask": [False] * len(selected),
        "cutoff_score_margins": {"0": 0.0},
        "candidate_global_token_ids": [10, 11, 12],
        "candidate_sample_ids": [0, 0, 0],
        "candidate_original_positions": [0, 1, 2],
        "candidate_scores": [0.9, 0.5, 0.5],
    }


def _save(
    path: Path,
    *,
    route_mode: str,
    routes: list[dict],
    tensors: dict[str, torch.Tensor] | None = None,
    communication: dict | None = None,
    forward_only: bool = False,
    backend: str | None = None,
    model_structure: dict | None = None,
    metadata_overrides: dict | None = None,
) -> Path:
    metadata = {
        "precision": "fp32",
        "route_mode": route_mode,
        "steps": 1,
        "checkpoint_next_step": False,
        "forward_only": forward_only,
        "communication": _communication() if communication is None else communication,
        "architecture": {"n_recurrent_layers": 2, "num_recursions": 3},
    }
    if backend is not None:
        metadata["backend"] = backend
    if model_structure is not None:
        metadata["model_structure"] = model_structure
    if metadata_overrides is not None:
        metadata.update(metadata_overrides)
    return save_artifact(
        path,
        metadata=metadata,
        tensors={"forward/logits": torch.tensor([1.0, 2.0])} if tensors is None else tensors,
        routes=routes,
    )


_EXPERT_CONTEXT = "expert_route/step_000/mb_000/logical_001/recurrent/round_000/physical_001"


def _expert_route_tensors(
    *,
    topk_dtype: torch.dtype = torch.int64,
    selected_score_scale: float = 1.0,
    live_score_scale: float | None = None,
    cutoff_margin: float = 0.75,
) -> dict[str, torch.Tensor]:
    if live_score_scale is None:
        live_score_scale = selected_score_scale
    return {
        f"{_EXPERT_CONTEXT}/global_token_ids": torch.tensor([10, 12], dtype=torch.int64),
        f"{_EXPERT_CONTEXT}/topk_indices": torch.tensor([[0, 2], [1, 3]], dtype=topk_dtype),
        f"{_EXPERT_CONTEXT}/selected_scores": torch.tensor(
            [[0.8, 0.2], [0.7, 0.3]], dtype=torch.float32
        )
        * selected_score_scale,
        f"{_EXPERT_CONTEXT}/live_selected_scores": torch.tensor(
            [[0.8, 0.2], [0.7, 0.3]], dtype=torch.float32
        )
        * live_score_scale,
        f"{_EXPERT_CONTEXT}/cutoff_logit_margins": torch.full(
            (2,), cutoff_margin, dtype=torch.float32
        ),
    }


def _expert_probe_metadata(*, enabled: bool = True, contexts: int = 1) -> dict:
    return {
        "enabled": enabled,
        "scope": "tiny-only-native-qwen-topk-before-token-dispatch",
        "identity_contract": "canonical-global-token-id-plus-logical-layer",
        "topk_indices": "dispatch-visible-exact",
        "cutoff_margin": "minimum-selected-minus-maximum-unselected-raw-logit",
        "dummy_padding_excluded": True,
        "captured_contexts": contexts,
    }


def _expert_acceptance_metadata(*, enabled: bool = True, contexts: int = 1) -> dict:
    return {
        "precision": "bf16",
        "preset": "tiny",
        "parameter_capture": {"enabled": True},
        "expert_route_probe": _expert_probe_metadata(enabled=enabled, contexts=contexts),
    }


def _expert_replay_metadata() -> dict:
    return {
        "precision": "bf16",
        "preset": "qwen3-30b",
        "parameter_capture": {"enabled": False},
        "expert_route_probe": _expert_probe_metadata(),
        "expert_route_replay": {
            "enabled": True,
            "scope": "forward-only-cross-topology-oracle",
            "identity_key": "logical-layer-plus-global-token-id",
            "expert_ids": "baseline-bitwise-exact",
            "selected_scores": "baseline-forward-values-with-live-gradient-ste",
            "live_selected_scores": "diagnostic-only-live-router-values",
            "training_native_router_unchanged": True,
        },
    }


_VALID_MLITE_STRUCTURE = {
    "status": "passed",
    "recurrent_physical_parameters": 4,
    "max_registrations_per_recurrent_parameter": 1,
}


def test_replay_index_uses_checkpoint_next_step_and_deduplicates_phases(
    tmp_path: Path,
) -> None:
    train = _route(selected=[10, 11])
    uninterrupted = _route(selected=[10, 12])
    uninterrupted.update({"phase": "uninterrupted", "step": 1})
    resumed = dict(uninterrupted)
    resumed["phase"] = "resume"
    artifact = _save(
        tmp_path / "artifact",
        route_mode="learned",
        routes=[train, uninterrupted, resumed],
    )

    index = _route_index(artifact)

    assert set(index) == {(0, 0), (1, 0)}
    assert index[(1, 0)][0].global_token_ids.tolist() == [10, 12]


def test_replay_index_rejects_conflicting_checkpoint_phases(tmp_path: Path) -> None:
    uninterrupted = _route(selected=[10, 11])
    uninterrupted.update({"phase": "uninterrupted", "step": 1})
    resumed = _route(selected=[10, 12])
    resumed.update({"phase": "resume", "step": 1})
    artifact = _save(
        tmp_path / "artifact",
        route_mode="learned",
        routes=[uninterrupted, resumed],
    )

    with pytest.raises(ValueError, match="conflicting RoutePlans"):
        _route_index(artifact)


def test_expert_route_replay_index_loads_logical_layer_plan(tmp_path: Path) -> None:
    tensors = {
        "expert_route/step_000/mb_000/logical_001/recurrent/round_000/"
        "physical_001/global_token_ids": torch.tensor([10, 12], dtype=torch.long),
        "expert_route/step_000/mb_000/logical_001/recurrent/round_000/"
        "physical_001/topk_indices": torch.tensor([[0, 2], [1, 3]], dtype=torch.long),
        "expert_route/step_000/mb_000/logical_001/recurrent/round_000/"
        "physical_001/selected_scores": torch.tensor([[0.6, 0.4], [0.7, 0.3]]),
    }
    artifact = _save(
        tmp_path / "artifact",
        route_mode="learned",
        routes=[_route()],
        tensors=tensors,
    )

    index = _expert_route_replay_index(artifact)

    assert set(index) == {(0, 0)}
    assert set(index[(0, 0)]) == {1}
    plan = index[(0, 0)][1]
    assert plan.global_token_ids.tolist() == [10, 12]
    assert plan.topk_indices.tolist() == [[0, 2], [1, 3]]
    torch.testing.assert_close(
        plan.selected_scores,
        torch.tensor([[0.6, 0.4], [0.7, 0.3]]),
        rtol=0.0,
        atol=0.0,
    )


def test_expert_route_replay_index_rejects_conflicting_checkpoint_phases(
    tmp_path: Path,
) -> None:
    base = "step_001/mb_000/logical_001/recurrent/round_000/physical_001"
    tensors = {
        f"expert_route/uninterrupted/{base}/global_token_ids": torch.tensor([10], dtype=torch.long),
        f"expert_route/uninterrupted/{base}/topk_indices": torch.tensor([[0, 2]], dtype=torch.long),
        f"expert_route/uninterrupted/{base}/selected_scores": torch.tensor([[0.6, 0.4]]),
        f"expert_route/resume/{base}/global_token_ids": torch.tensor([10], dtype=torch.long),
        f"expert_route/resume/{base}/topk_indices": torch.tensor([[1, 2]], dtype=torch.long),
        f"expert_route/resume/{base}/selected_scores": torch.tensor([[0.6, 0.4]]),
    }
    artifact = _save(
        tmp_path / "artifact",
        route_mode="learned",
        routes=[_route()],
        tensors=tensors,
    )

    with pytest.raises(ValueError, match="conflicting expert-route replay plans"):
        _expert_route_replay_index(artifact)


def test_fp32_and_bf16_tensor_tolerances() -> None:
    baseline = torch.tensor([1.0, -2.0, 3.0])
    assert compare_tensor(
        "forward", baseline, baseline + 1e-6, precision="fp32", kind="forward"
    ).passed
    assert not compare_tensor(
        "forward", baseline, baseline + 1e-3, precision="fp32", kind="forward"
    ).passed
    assert compare_tensor(
        "gradient", baseline, baseline + 4e-6, precision="fp32", kind="gradient"
    ).passed

    assert compare_tensor(
        "loss", torch.tensor(2.0), torch.tensor(2.009), precision="bf16", kind="loss"
    ).passed
    assert not compare_tensor(
        "loss", torch.tensor(2.0), torch.tensor(2.011), precision="bf16", kind="loss"
    ).passed
    assert compare_tensor(
        "forward", baseline, baseline * 1.019, precision="bf16", kind="forward"
    ).passed
    assert compare_tensor(
        "update", baseline, baseline * 1.029, precision="bf16", kind="update"
    ).passed
    assert not compare_tensor(
        "update", baseline, baseline * 1.04, precision="bf16", kind="update"
    ).passed


def test_bf16_gradient_uses_phase_step_full_vector_gate(tmp_path: Path) -> None:
    tensors = {
        "gradient/step_000/large": torch.ones(10_000),
        "gradient/step_000/small": torch.tensor([1.0e-4]),
    }
    candidate_tensors = {
        "gradient/step_000/large": tensors["gradient/step_000/large"].clone(),
        # A severe per-parameter relative error, but only one element in the
        # complete reconstructed gradient vector.
        "gradient/step_000/small": torch.tensor([1.0]),
    }
    baseline = _save(
        tmp_path / "baseline",
        route_mode="replay",
        routes=[_route()],
        tensors=tensors,
        metadata_overrides={"precision": "bf16"},
    )
    candidate = _save(
        tmp_path / "candidate",
        route_mode="replay",
        routes=[_route()],
        tensors=candidate_tensors,
        metadata_overrides={"precision": "bf16"},
    )

    report = compare_artifacts(baseline, candidate)

    assert report["passed"]
    outlier = next(item for item in report["tensor_results"] if item["name"].endswith("small"))
    assert not outlier["passed"]
    assert outlier["diagnostic_only"]
    assert not outlier["hard_gate"]
    aggregate = report["aggregate_results"]
    assert len(aggregate) == 1
    assert aggregate[0]["name"] == "gradient/train/step_000"
    assert aggregate[0]["passed"]
    assert aggregate[0]["tensor_count"] == 2
    assert aggregate[0]["numel"] == 10_001
    assert report["namespace_summary"]["gradient"]["diagnostic_outliers"] == [
        "gradient/step_000/small"
    ]


def test_bf16_gradient_full_vector_failure_is_a_hard_gate(tmp_path: Path) -> None:
    tensors = {"gradient/resume/step_007/weight": torch.ones(64)}
    baseline = _save(
        tmp_path / "baseline",
        route_mode="replay",
        routes=[_route()],
        tensors=tensors,
        metadata_overrides={"precision": "bf16"},
    )
    candidate = _save(
        tmp_path / "candidate",
        route_mode="replay",
        routes=[_route()],
        tensors={"gradient/resume/step_007/weight": torch.full((64,), 1.04)},
        metadata_overrides={"precision": "bf16"},
    )

    report = compare_artifacts(baseline, candidate)

    assert not report["passed"]
    assert report["aggregate_results"][0]["name"] == "gradient/resume/step_007"
    assert not report["aggregate_results"][0]["passed"]
    assert report["namespace_summary"]["gradient"]["aggregate_failures"] == [
        "gradient/resume/step_007"
    ]


def test_update_and_post_step_are_distinct_bf16_namespaces(tmp_path: Path) -> None:
    baseline_tensors = {
        "update/step_000/large": torch.ones(10_000),
        "update/step_000/small": torch.tensor([1.0e-4]),
        "post_step/step_000/weight": torch.ones(4),
    }
    candidate_tensors = {
        "update/step_000/large": torch.ones(10_000),
        "update/step_000/small": torch.tensor([1.0]),
        # Post-step weights remain their own hard-gated observation; they are
        # not folded into the optimizer-update aggregate.
        "post_step/step_000/weight": torch.full((4,), 1.04),
    }
    baseline = _save(
        tmp_path / "baseline",
        route_mode="replay",
        routes=[_route()],
        tensors=baseline_tensors,
        metadata_overrides={"precision": "bf16"},
    )
    candidate = _save(
        tmp_path / "candidate",
        route_mode="replay",
        routes=[_route()],
        tensors=candidate_tensors,
        metadata_overrides={"precision": "bf16"},
    )

    report = compare_artifacts(baseline, candidate)

    assert not report["passed"]
    assert [item["namespace"] for item in report["aggregate_results"]] == ["update"]
    assert report["aggregate_results"][0]["passed"]
    assert report["namespace_summary"]["update"]["passed"]
    assert not report["namespace_summary"]["post_step"]["passed"]
    post_step = next(item for item in report["tensor_results"] if item["namespace"] == "post_step")
    assert post_step["hard_gate"]
    assert not post_step["diagnostic_only"]


def test_initial_is_bitwise_exact_and_fp32_gradient_remains_per_tensor_strict(
    tmp_path: Path,
) -> None:
    baseline_tensors = {
        "initial/weight": torch.tensor([1.0], dtype=torch.float32),
        "gradient/step_000/large": torch.ones(10_000),
        "gradient/step_000/small": torch.tensor([1.0e-4]),
    }
    candidate_tensors = {
        "initial/weight": torch.nextafter(torch.tensor([1.0]), torch.tensor([2.0])),
        "gradient/step_000/large": torch.ones(10_000),
        "gradient/step_000/small": torch.tensor([1.0]),
    }
    baseline = _save(
        tmp_path / "baseline", route_mode="replay", routes=[_route()], tensors=baseline_tensors
    )
    candidate = _save(
        tmp_path / "candidate",
        route_mode="replay",
        routes=[_route()],
        tensors=candidate_tensors,
    )

    report = compare_artifacts(baseline, candidate)

    assert not report["passed"]
    assert report["aggregate_results"] == []
    initial = next(item for item in report["tensor_results"] if item["namespace"] == "initial")
    small_gradient = next(
        item for item in report["tensor_results"] if item["name"].endswith("small")
    )
    assert not initial["passed"]
    assert initial["detail"] == "initial tensor bytes differ"
    assert initial["hard_gate"]
    assert not small_gradient["passed"]
    assert small_gradient["hard_gate"]


def test_bf16_nonfinite_gradient_is_never_diagnostic_only(tmp_path: Path) -> None:
    name = "gradient/step_000/weight"
    baseline = _save(
        tmp_path / "baseline",
        route_mode="replay",
        routes=[_route()],
        tensors={name: torch.ones(2)},
        metadata_overrides={"precision": "bf16"},
    )
    candidate = _save(
        tmp_path / "candidate",
        route_mode="replay",
        routes=[_route()],
        tensors={name: torch.tensor([1.0, float("nan")])},
        metadata_overrides={"precision": "bf16"},
    )

    report = compare_artifacts(baseline, candidate)

    result = report["tensor_results"][0]
    assert not report["passed"]
    assert result["hard_gate"]
    assert not result["diagnostic_only"]
    assert "non-finite" in result["detail"]


def test_expert_routes_require_exact_identity_but_use_forward_score_tolerance(
    tmp_path: Path,
) -> None:
    baseline = _save(
        tmp_path / "baseline",
        route_mode="replay",
        routes=[_route()],
        tensors=_expert_route_tensors(),
        backend="mlite",
        model_structure=_VALID_MLITE_STRUCTURE,
        metadata_overrides=_expert_acceptance_metadata(),
    )
    candidate = _save(
        tmp_path / "candidate",
        route_mode="replay",
        routes=[_route()],
        tensors=_expert_route_tensors(selected_score_scale=1.019, cutoff_margin=0.76),
        backend="mlite",
        model_structure=_VALID_MLITE_STRUCTURE,
        metadata_overrides=_expert_acceptance_metadata(),
    )

    report = compare_artifacts(baseline, candidate, scope="forward")

    assert report["passed"]
    expert = report["expert_routes"]
    assert expert["required"]
    assert expert["enabled"]
    assert expert["metadata_compatible"]
    assert expert["context_count"] == 1
    assert expert["exact_identity_count"] == 1
    assert expert["exact_identity_tensor_count"] == 2
    assert expert["baseline_minimum_cutoff_margin"] == pytest.approx(0.75)
    assert report["namespace_summary"]["expert_route"]["tensor_count"] == 5
    identity = [
        item
        for item in report["tensor_results"]
        if item["name"].endswith(("/global_token_ids", "/topk_indices"))
    ]
    numerical = [
        item
        for item in report["tensor_results"]
        if item["name"].endswith(
            ("/selected_scores", "/live_selected_scores", "/cutoff_logit_margins")
        )
    ]
    assert all("bitwise exact" in item["detail"] for item in identity)
    assert all("relative_l2 <= 0.02" in item["detail"] for item in numerical)


def test_expert_live_scores_and_cutoff_are_diagnostic_but_replayed_scores_are_gated(
    tmp_path: Path,
) -> None:
    baseline = _save(
        tmp_path / "baseline",
        route_mode="replay",
        routes=[_route()],
        tensors=_expert_route_tensors(cutoff_margin=1.0e-6),
        backend="mlite",
        model_structure=_VALID_MLITE_STRUCTURE,
        metadata_overrides={
            "precision": "bf16",
            "preset": "qwen3-30b",
            "parameter_capture": {"enabled": False},
            "expert_route_probe": _expert_probe_metadata(),
        },
    )
    candidate = _save(
        tmp_path / "candidate",
        route_mode="replay",
        routes=[_route()],
        tensors=_expert_route_tensors(live_score_scale=1.1, cutoff_margin=0.25),
        backend="mlite",
        model_structure=_VALID_MLITE_STRUCTURE,
        metadata_overrides={
            "precision": "bf16",
            "preset": "qwen3-30b",
            "parameter_capture": {"enabled": False},
            "expert_route_probe": _expert_probe_metadata(),
        },
    )

    report = compare_artifacts(baseline, candidate, scope="forward")

    cutoff = next(
        item for item in report["tensor_results"] if item["name"].endswith("/cutoff_logit_margins")
    )
    live_scores = next(
        item for item in report["tensor_results"] if item["name"].endswith("/live_selected_scores")
    )
    scores = next(
        item for item in report["tensor_results"] if item["name"].endswith("/selected_scores")
    )
    assert report["passed"]
    assert not cutoff["passed"]
    assert cutoff["diagnostic_only"]
    assert not cutoff["hard_gate"]
    assert live_scores["diagnostic_only"]
    assert not live_scores["hard_gate"]
    assert not live_scores["passed"]
    assert scores["passed"]
    assert scores["hard_gate"]


def test_forward_expert_replay_requires_exact_consumed_scores_but_not_live_scores(
    tmp_path: Path,
) -> None:
    baseline = _save(
        tmp_path / "baseline",
        route_mode="learned",
        routes=[_route()],
        tensors=_expert_route_tensors(),
        forward_only=True,
        backend="mlite",
        model_structure=_VALID_MLITE_STRUCTURE,
        metadata_overrides={
            "precision": "bf16",
            "preset": "qwen3-30b",
            "parameter_capture": {"enabled": False},
            "expert_route_probe": _expert_probe_metadata(),
        },
    )
    candidate = _save(
        tmp_path / "candidate",
        route_mode="replay",
        routes=[_route()],
        tensors=_expert_route_tensors(live_score_scale=1.1),
        forward_only=True,
        backend="mlite",
        model_structure=_VALID_MLITE_STRUCTURE,
        metadata_overrides=_expert_replay_metadata(),
    )

    report = compare_artifacts(baseline, candidate, scope="forward")

    assert report["passed"]
    consumed = next(
        item for item in report["tensor_results"] if item["name"].endswith("/selected_scores")
    )
    live = next(
        item for item in report["tensor_results"] if item["name"].endswith("/live_selected_scores")
    )
    assert consumed["passed"] and consumed["hard_gate"]
    assert "bitwise exact" in consumed["detail"]
    assert not live["passed"] and live["diagnostic_only"]


def test_forward_expert_replay_rejects_nonexact_consumed_scores(tmp_path: Path) -> None:
    common = {
        "forward_only": True,
        "backend": "mlite",
        "model_structure": _VALID_MLITE_STRUCTURE,
    }
    baseline = _save(
        tmp_path / "baseline",
        route_mode="learned",
        routes=[_route()],
        tensors=_expert_route_tensors(),
        metadata_overrides={
            "precision": "bf16",
            "preset": "qwen3-30b",
            "parameter_capture": {"enabled": False},
            "expert_route_probe": _expert_probe_metadata(),
        },
        **common,
    )
    candidate = _save(
        tmp_path / "candidate",
        route_mode="replay",
        routes=[_route()],
        tensors=_expert_route_tensors(selected_score_scale=1.001),
        metadata_overrides=_expert_replay_metadata(),
        **common,
    )

    report = compare_artifacts(baseline, candidate, scope="forward")

    consumed = next(
        item for item in report["tensor_results"] if item["name"].endswith("/selected_scores")
    )
    assert not report["passed"]
    assert not consumed["passed"] and consumed["hard_gate"]
    assert "tensor bytes differ" in consumed["detail"]


@pytest.mark.parametrize("identity_field", ["global_token_ids", "topk_indices"])
def test_expert_route_identity_change_is_a_hard_failure(
    tmp_path: Path, identity_field: str
) -> None:
    baseline_tensors = _expert_route_tensors()
    candidate_tensors = _expert_route_tensors()
    candidate_tensors[f"{_EXPERT_CONTEXT}/{identity_field}"].reshape(-1)[0] += 1
    baseline = _save(
        tmp_path / "baseline",
        route_mode="replay",
        routes=[_route()],
        tensors=baseline_tensors,
        backend="mlite",
        model_structure=_VALID_MLITE_STRUCTURE,
        metadata_overrides=_expert_acceptance_metadata(),
    )
    candidate = _save(
        tmp_path / "candidate",
        route_mode="replay",
        routes=[_route()],
        tensors=candidate_tensors,
        backend="mlite",
        model_structure=_VALID_MLITE_STRUCTURE,
        metadata_overrides=_expert_acceptance_metadata(),
    )

    report = compare_artifacts(baseline, candidate)

    failed = next(
        item for item in report["tensor_results"] if item["name"].endswith(identity_field)
    )
    assert not report["passed"]
    assert failed["hard_gate"]
    assert not failed["passed"]
    assert "tensor bytes differ" in failed["detail"]
    assert report["expert_routes"]["exact_identity_count"] == 0
    assert report["expert_routes"]["exact_identity_tensor_count"] == 1


def test_expert_route_identity_dtype_must_match_exactly(tmp_path: Path) -> None:
    baseline = _save(
        tmp_path / "baseline",
        route_mode="replay",
        routes=[_route()],
        tensors=_expert_route_tensors(),
        backend="mlite",
        model_structure=_VALID_MLITE_STRUCTURE,
        metadata_overrides=_expert_acceptance_metadata(),
    )
    candidate = _save(
        tmp_path / "candidate",
        route_mode="replay",
        routes=[_route()],
        tensors=_expert_route_tensors(topk_dtype=torch.int32),
        backend="mlite",
        model_structure=_VALID_MLITE_STRUCTURE,
        metadata_overrides=_expert_acceptance_metadata(),
    )

    report = compare_artifacts(baseline, candidate)

    failed = next(
        item for item in report["tensor_results"] if item["name"].endswith("/topk_indices")
    )
    assert not report["passed"]
    assert failed["hard_gate"]
    assert "dtype mismatch" in failed["detail"]


def test_default_tiny_expert_route_margin_and_metadata_are_hard_gates(
    tmp_path: Path,
) -> None:
    tensors = _expert_route_tensors(cutoff_margin=0.5)
    baseline = _save(
        tmp_path / "baseline",
        route_mode="replay",
        routes=[_route()],
        tensors=tensors,
        backend="mlite",
        model_structure=_VALID_MLITE_STRUCTURE,
        metadata_overrides=_expert_acceptance_metadata(),
    )
    candidate = _save(
        tmp_path / "candidate",
        route_mode="replay",
        routes=[_route()],
        tensors={name: value.clone() for name, value in tensors.items()},
        backend="mlite",
        model_structure=_VALID_MLITE_STRUCTURE,
        metadata_overrides=_expert_acceptance_metadata(contexts=2),
    )

    report = compare_artifacts(baseline, candidate)

    expert = report["expert_routes"]
    assert not report["passed"]
    assert not expert["passed"]
    assert not expert["metadata_compatible"]
    assert expert["baseline_minimum_cutoff_margin"] == pytest.approx(0.5)
    assert any("captured_contexts" in detail for detail in expert["details"])
    assert any("must be > 0.5" in detail for detail in expert["details"])


def test_bitwise_identical_nonfinite_initial_tensor_still_fails(tmp_path: Path) -> None:
    name = "initial/weight"
    value = torch.tensor([float("nan")])
    baseline = _save(
        tmp_path / "baseline", route_mode="replay", routes=[_route()], tensors={name: value}
    )
    candidate = _save(
        tmp_path / "candidate", route_mode="replay", routes=[_route()], tensors={name: value}
    )

    report = compare_artifacts(baseline, candidate)

    assert not report["passed"]
    assert report["tensor_results"][0]["hard_gate"]
    assert report["tensor_results"][0]["detail"] == "initial tensor contains non-finite values"


def test_cutoff_classification_uses_margin_vs_twice_score_error() -> None:
    ids = torch.tensor([10, 11, 12])
    near = classify_cutoff(
        torch.tensor([0.9, 0.5, 0.5]),
        torch.tensor([0.9, 0.50001, 0.49999]),
        ids,
        2,
    )
    assert near.near_tie
    assert near.margin == 0.0
    assert set(near.ambiguity_ids) == {11, 12}

    separated = classify_cutoff(
        torch.tensor([0.9, 0.7, 0.2]),
        torch.tensor([0.9, 0.70001, 0.19999]),
        ids,
        2,
    )
    assert not separated.near_tie
    assert separated.margin > 2 * separated.score_error + 1e-7


def test_artifact_roundtrip_preserves_tensors_routes_and_hashes(tmp_path: Path) -> None:
    tensors = {
        "forward/logits": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "gradient/weight": torch.tensor([0.25, -0.5]),
    }
    path = _save(tmp_path / "artifact", route_mode="replay", routes=[_route()], tensors=tensors)
    manifest, restored, routes = load_artifact(path)

    assert manifest["route_count"] == 1
    assert set(manifest["tensor_index"]) == set(tensors)
    assert routes == [_route()]
    for name, tensor in tensors.items():
        torch.testing.assert_close(restored[name], tensor, rtol=0.0, atol=0.0)
        assert len(manifest["tensor_index"][name]["sha256"]) == 64


def test_artifact_hash_accepts_scalar_tensor() -> None:
    from mor_mlite.parity.artifacts import tensor_sha256

    digest = tensor_sha256(torch.tensor(1.25, dtype=torch.float32))
    assert len(digest) == 64


def test_artifact_load_rejects_tampered_tensor_index_and_duplicate_routes(
    tmp_path: Path,
) -> None:
    path = _save(tmp_path / "artifact", route_mode="replay", routes=[_route()])
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tensor_index"]["forward/logits"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity mismatch"):
        load_artifact(path)

    duplicate_path = _save(tmp_path / "duplicate", route_mode="replay", routes=[_route(), _route()])
    with pytest.raises(ValueError, match="duplicate RoutePlan key"):
        load_artifact(duplicate_path)


def test_artifact_compare_rejects_precision_mismatch(tmp_path: Path) -> None:
    baseline = _save(tmp_path / "baseline", route_mode="replay", routes=[_route()])
    candidate = _save(
        tmp_path / "candidate",
        route_mode="replay",
        routes=[_route()],
        metadata_overrides={"precision": "bf16"},
    )

    report = compare_artifacts(baseline, candidate)

    assert not report["passed"]
    assert not report["metadata_compatibility"]["passed"]


def test_artifact_compare_rejects_optimizer_contract_mismatch(tmp_path: Path) -> None:
    baseline = _save(
        tmp_path / "baseline",
        route_mode="replay",
        routes=[_route()],
        metadata_overrides={"optimizer": {"name": "adam", "adam_eps": 1e-6}},
    )
    candidate = _save(
        tmp_path / "candidate",
        route_mode="replay",
        routes=[_route()],
        metadata_overrides={"optimizer": {"name": "adam", "adam_eps": 1e-8}},
    )

    report = compare_artifacts(baseline, candidate)

    assert not report["passed"]
    assert not report["metadata_compatibility"]["passed"]
    assert (
        "optimizer differs between baseline and candidate"
        in report["metadata_compatibility"]["details"]
    )


def test_artifact_output_must_be_fresh(tmp_path: Path) -> None:
    from mor_mlite.parity.artifacts import require_fresh_artifact_directory

    output = tmp_path / "artifact"
    assert require_fresh_artifact_directory(output) == output
    output.mkdir()
    (output / "step_7").mkdir()
    with pytest.raises(FileExistsError, match="stale-step reuse"):
        require_fresh_artifact_directory(output)


def test_replay_artifact_compare_requires_bitwise_canonical_route_metadata(
    tmp_path: Path,
) -> None:
    baseline_route = _route()
    baseline_route["mode"] = "replay"
    candidate_route = dict(baseline_route)
    # Ownership can differ by topology; canonical token metadata cannot.
    candidate_route["source_tp_ranks"] = [1, 1]
    candidate_route["target_cp_ranks"] = [1, 0]
    baseline = _save(tmp_path / "baseline", route_mode="replay", routes=[baseline_route])
    candidate = _save(tmp_path / "candidate", route_mode="replay", routes=[candidate_route])
    assert compare_artifacts(baseline, candidate)["passed"]

    changed_route = dict(candidate_route)
    changed_route["original_positions"] = [0, 2]
    changed = _save(tmp_path / "changed", route_mode="replay", routes=[changed_route])
    report = compare_artifacts(baseline, changed)
    assert not report["passed"]
    assert not report["routes"]["passed"]

    changed_gate_route = dict(candidate_route)
    changed_gate_route["selected_gates"] = [0.09, 0.08]
    changed_gates = _save(
        tmp_path / "changed-gates", route_mode="replay", routes=[changed_gate_route]
    )
    report = compare_artifacts(baseline, changed_gates)
    assert not report["passed"]
    assert not report["routes"]["passed"]


def test_learned_artifact_compare_allows_only_cutoff_ambiguity_changes(
    tmp_path: Path,
) -> None:
    baseline_route = _route(selected=[10, 11])
    candidate_route = _route(selected=[10, 12])
    candidate_route["candidate_scores"] = [0.9, 0.49999, 0.50001]
    candidate_route["peer_selected_global_token_ids"] = [[10, 12], [10, 12]]
    baseline = _save(tmp_path / "baseline", route_mode="learned", routes=[baseline_route])
    candidate = _save(tmp_path / "candidate", route_mode="learned", routes=[candidate_route])
    report = compare_artifacts(baseline, candidate)

    assert report["passed"]
    assert report["routes"]["near_ties"] == 1
    assert report["routes"]["near_tie_rate"] == 1.0


def test_learned_route_gate_json_is_not_a_bitwise_identity_field(tmp_path: Path) -> None:
    baseline_route = _route()
    candidate_route = dict(baseline_route)
    candidate_route["selected_gates"] = [0.090006, 0.089997]
    baseline = _save(tmp_path / "baseline", route_mode="learned", routes=[baseline_route])
    candidate = _save(tmp_path / "candidate", route_mode="learned", routes=[candidate_route])

    report = compare_artifacts(baseline, candidate)

    assert report["passed"]
    assert report["routes"]["passed"]


def test_learned_near_tie_does_not_allow_selected_token_loss(tmp_path: Path) -> None:
    baseline_route = _route(selected=[10, 11])
    candidate_route = _route(selected=[10])
    candidate_route["candidate_scores"] = [0.9, 0.49999, 0.50001]
    baseline = _save(tmp_path / "baseline", route_mode="learned", routes=[baseline_route])
    candidate = _save(tmp_path / "candidate", route_mode="learned", routes=[candidate_route])

    report = compare_artifacts(baseline, candidate)

    assert not report["passed"]
    assert any(
        "cardinalit" in detail or "metadata sizes" in detail
        for detail in report["routes"]["details"]
    )


def test_learned_near_tie_still_requires_original_position_identity(tmp_path: Path) -> None:
    baseline_route = _route(selected=[10, 11])
    candidate_route = _route(selected=[10, 12])
    candidate_route["candidate_scores"] = [0.9, 0.49999, 0.50001]
    candidate_route["original_positions"] = [0, 1]
    baseline = _save(tmp_path / "baseline", route_mode="learned", routes=[baseline_route])
    candidate = _save(tmp_path / "candidate", route_mode="learned", routes=[candidate_route])

    report = compare_artifacts(baseline, candidate)

    assert not report["passed"]
    assert any("inconsistent sample/position" in detail for detail in report["routes"]["details"])


def test_learned_artifact_compare_rejects_tp_cp_peer_disagreement_even_when_baseline_matches(
    tmp_path: Path,
) -> None:
    baseline_route = _route()
    candidate_route = _route()
    candidate_route["peer_selected_global_token_ids"] = [[10, 11], [10, 12]]
    baseline = _save(tmp_path / "baseline", route_mode="learned", routes=[baseline_route])
    candidate = _save(tmp_path / "candidate", route_mode="learned", routes=[candidate_route])

    report = compare_artifacts(baseline, candidate)
    assert not report["passed"]
    assert any("peers disagree" in detail for detail in report["routes"]["details"])


def test_communication_assertions_are_part_of_artifact_comparison(tmp_path: Path) -> None:
    baseline = _save(tmp_path / "baseline", route_mode="replay", routes=[_route()])
    bad_communication = _communication()
    bad_communication["hidden_rebalances"] = 3
    bad_communication["recurrent_inner_dispatches"] = 1
    candidate = _save(
        tmp_path / "candidate",
        route_mode="replay",
        routes=[_route()],
        communication=bad_communication,
    )
    report = compare_artifacts(baseline, candidate)
    assert not report["passed"]
    assert len(report["communication"]["details"]) == 2


@pytest.mark.parametrize(
    ("communication", "detail"),
    [
        ({}, "no communication counters"),
        (
            {
                **_communication(),
                "grad_sync_probe": _gradient_sync_probe(status="unavailable"),
                "physical_bucket_sync_dispatch_max": None,
            },
            "instrumentation is unavailable",
        ),
        (
            {
                **_communication(),
                "physical_bucket_sync_dispatch_max": None,
            },
            "maximum is unavailable",
        ),
        (
            {
                **_communication(),
                "grad_sync_probe": _gradient_sync_probe(finalize_calls=2),
            },
            "finalize_grads calls must be exactly one",
        ),
        (
            {
                **_communication(),
                "grad_sync_probe": _gradient_sync_probe(bucket_calls=2),
                "physical_bucket_sync_dispatch_max": 2,
            },
            "physical parameter bucket synchronized 2 times",
        ),
    ],
)
def test_training_communication_probe_fails_closed(
    tmp_path: Path, communication: dict, detail: str
) -> None:
    baseline = _save(tmp_path / "baseline", route_mode="replay", routes=[_route()])
    candidate = _save(
        tmp_path / "candidate",
        route_mode="replay",
        routes=[_route()],
        communication=communication,
        backend="mlite",
        model_structure={
            "status": "passed",
            "recurrent_physical_parameters": 4,
            "max_registrations_per_recurrent_parameter": 1,
        },
    )

    report = compare_artifacts(baseline, candidate)
    assert not report["passed"]
    assert any(detail in item for item in report["communication"]["details"])


def test_forward_only_candidate_still_requires_recurrent_execution_probe(tmp_path: Path) -> None:
    baseline = _save(
        tmp_path / "baseline",
        route_mode="replay",
        routes=[_route()],
        forward_only=True,
        communication={},
    )
    candidate = _save(
        tmp_path / "candidate",
        route_mode="replay",
        routes=[_route()],
        forward_only=True,
        communication={},
    )

    report = compare_artifacts(baseline, candidate)
    assert not report["passed"]
    assert any(
        "missing recurrent communication instrumentation" in detail
        for detail in report["communication"]["details"]
    )


def test_reference_candidate_does_not_require_mlite_gradient_probe(tmp_path: Path) -> None:
    baseline = _save(tmp_path / "baseline", route_mode="replay", routes=[_route()])
    candidate = _save(
        tmp_path / "candidate",
        route_mode="replay",
        routes=[_route()],
        backend="reference",
        communication={
            "active_set_changes": 2,
            "hidden_rebalances": 2,
            "recurrent_inner_dispatches": 0,
            "early_exit_qkv_tokens": 0,
            "recurrent_qkv_checks": 6,
        },
    )

    report = compare_artifacts(baseline, candidate)
    assert report["communication"]["passed"]


def test_mlite_candidate_requires_single_registration_structure_evidence(
    tmp_path: Path,
) -> None:
    baseline = _save(tmp_path / "baseline", route_mode="replay", routes=[_route()])
    valid_structure = {
        "status": "passed",
        "recurrent_physical_parameters": 4,
        "max_registrations_per_recurrent_parameter": 1,
    }
    candidate = _save(
        tmp_path / "candidate",
        route_mode="replay",
        routes=[_route()],
        backend="mlite",
        model_structure=valid_structure,
    )
    assert compare_artifacts(baseline, candidate)["passed"]

    duplicated = dict(valid_structure)
    duplicated["max_registrations_per_recurrent_parameter"] = 2
    bad = _save(
        tmp_path / "bad",
        route_mode="replay",
        routes=[_route()],
        backend="mlite",
        model_structure=duplicated,
    )
    report = compare_artifacts(baseline, bad)
    assert not report["passed"]
    assert not report["model_structure"]["passed"]


def test_tiny_mlite_checkpoint_comparison_requires_continuity_evidence(
    tmp_path: Path,
) -> None:
    baseline = _save(tmp_path / "baseline", route_mode="replay", routes=[_route()])
    candidate = _save(
        tmp_path / "candidate",
        route_mode="replay",
        routes=[_route()],
        backend="mlite",
        model_structure={
            "status": "passed",
            "recurrent_physical_parameters": 4,
            "max_registrations_per_recurrent_parameter": 1,
        },
        metadata_overrides={
            "preset": "tiny",
            "checkpoint_next_step": True,
        },
    )

    report = compare_artifacts(baseline, candidate)
    assert not report["passed"]
    assert not report["checkpoint_continuity"]["candidate"]["passed"]
    assert any(
        "continuity report is missing" in detail
        for detail in report["checkpoint_continuity"]["candidate"]["details"]
    )


def test_qwen_checkpoint_smoke_requires_a_real_resumed_optimizer_update(
    tmp_path: Path,
) -> None:
    baseline = _save(tmp_path / "baseline", route_mode="replay", routes=[_route()])
    candidate = _save(
        tmp_path / "candidate",
        route_mode="replay",
        routes=[_route()],
        backend="mlite",
        model_structure={
            "status": "passed",
            "recurrent_physical_parameters": 4,
            "max_registrations_per_recurrent_parameter": 1,
        },
        metadata_overrides={
            "preset": "qwen3-30b",
            "checkpoint_next_step": True,
            "checkpoint_restored_step": 1,
            "checkpoint_parameter_fingerprint": {"status": "passed"},
            "optimizer_steps": [
                {"phase": "train", "updated": True, "grad_norm": 1.0},
                {"phase": "resume", "updated": False, "grad_norm": 1.0},
            ],
        },
    )

    report = compare_artifacts(baseline, candidate)
    assert not report["checkpoint_roundtrip"]["passed"]
    assert any(
        "resume optimizer step did not update" in detail
        for detail in report["checkpoint_roundtrip"]["details"]
    )


def test_forward_scope_ignores_training_only_and_resume_candidate_artifacts(
    tmp_path: Path,
) -> None:
    baseline_route = _route()
    baseline_route["mode"] = "replay"
    resumed_route = dict(baseline_route)
    resumed_route["step"] = 1
    resumed_route["phase"] = "resume"
    shared = {"forward/step_000/mb_000/log_probs": torch.tensor([-1.0, -2.0])}
    baseline = _save(
        tmp_path / "forward-baseline",
        route_mode="replay",
        routes=[baseline_route],
        tensors={**shared, "initial/weight": torch.tensor([1.0])},
    )
    candidate = _save(
        tmp_path / "training-candidate",
        route_mode="replay",
        routes=[baseline_route, resumed_route],
        tensors={
            **shared,
            "initial/weight": torch.tensor([9.0]),
            "gradient/step_000/weight": torch.tensor([2.0]),
            "post_step/step_000/weight": torch.tensor([3.0]),
            "forward/resume/step_001/mb_000/log_probs": torch.tensor([-3.0]),
        },
    )

    report = compare_artifacts(baseline, candidate, scope="forward")
    assert report["passed"]
    assert report["unexpected_tensors"] == []
    assert report["ignored_candidate_tensors"] == ["forward/resume/step_001/mb_000/log_probs"]


def test_acceptance_topology_matrix_exactly_matches_overlapping_ep_semantics() -> None:
    expected = [
        ("baseline", 1, 1, 1, 1, 1),
        ("zero1", 2, 1, 1, 2, 1),
        ("tp", 2, 2, 1, 1, 1),
        ("cp", 2, 1, 2, 1, 1),
        ("ep", 2, 1, 1, 2, 2),
        ("tp_cp_ep", 4, 2, 2, 1, 2),
        ("tp_dp_ep", 4, 2, 1, 2, 2),
        ("cp_dp_ep", 4, 1, 2, 2, 2),
        ("all", 8, 2, 2, 2, 4),
    ]
    actual = [
        (item.name, item.world_size, item.tp, item.cp, item.dp, item.ep) for item in TOPOLOGY_MATRIX
    ]
    assert actual == expected
    for topology in TOPOLOGY_MATRIX:
        topology.validate(num_experts=128)
        assert topology.world_size == topology.tp * topology.cp * topology.dp
    assert get_topology("all") is TOPOLOGY_MATRIX[-1]
