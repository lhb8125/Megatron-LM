from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from mor_mlite.config import MoRArchitectureConfig
from mor_mlite.data import make_synthetic_batch, original_sample_lengths, packed_lm_targets
from mor_mlite.parity.artifacts import load_artifact, save_artifact
from mor_mlite.parity.compare import _compare_learned_routes, compare_artifacts
from mor_mlite.parity.reference import ReferenceRunConfig, run_reference
from mor_mlite.tiny import TinyMoRModel


@pytest.mark.parametrize("recursions", [1, 3, 7, 10, 31])
def test_linear_capacity_matches_integer_budget(recursions):
    architecture = MoRArchitectureConfig(1, 1, recursions, 1)
    for length in range(201):
        for round_index in range(recursions):
            expected = max(1, (recursions - round_index) * length // recursions) if length else 0
            assert architecture.top_k(length, round_index) == expected


def test_custom_first_capacity_must_retain_every_token():
    with pytest.raises(ValueError, match="first recurrent round"):
        MoRArchitectureConfig(1, 1, 3, 1, (1 - 5e-13, 2 / 3, 1 / 3))


@pytest.mark.parametrize(
    "mutation", ["empty", "missing", "extra", "wrong_round", "positions", "samples"]
)
def test_reference_replay_rejects_corrupt_or_incomplete_oracle(mutation):
    model = TinyMoRModel()
    batch = make_synthetic_batch(seq_lens=(5, 3))
    with torch.no_grad():
        plans = {p.round_index: p for p in model(batch).route_plans}
        if mutation == "empty":
            plans = {}
        elif mutation == "missing":
            del plans[1]
        elif mutation == "extra":
            plans[3] = replace(plans[2], round_index=3)
        elif mutation == "wrong_round":
            plans[1] = replace(plans[1], round_index=2)
        elif mutation == "positions":
            plans[1] = replace(plans[1], original_positions=plans[1].original_positions + 100)
        else:
            plans[1] = replace(plans[1], sample_ids=plans[1].sample_ids + 10)
        with pytest.raises(ValueError, match="replay"):
            model(batch, route_mode="replay", replay_plans=plans)


def test_none_mask_keeps_all_real_targets_and_explicit_mask_shifts_once():
    batch = make_synthetic_batch(seq_lens=(5, 3), mask_last_token=False)
    labels, mask = packed_lm_targets(batch)
    assert mask.sum() == 8
    assert labels.tolist() == [*batch.labels[1:5].tolist(), 0, *batch.labels[6:8].tolist(), 0]
    batch.loss_mask = torch.tensor([0.0, 1.0, 0.0, 0.5, 1.0, 1.0, 0.0, 1.0])
    assert packed_lm_targets(batch)[1].tolist() == [1.0, 0.0, 0.5, 1.0, 0.0, 0.0, 1.0, 0.0]


def test_stable_noncontiguous_sample_ids_preserve_outputs_and_gradients():
    model = TinyMoRModel()
    batch = make_synthetic_batch(seq_lens=(5, 3))
    original = model(batch)
    original.total_loss.backward()
    gradients = {
        name: p.grad.clone() if p.grad is not None else None for name, p in model.named_parameters()
    }
    model.zero_grad(set_to_none=True)
    batch.extras["sample_ids"] = torch.where(batch.extras["sample_ids"] == 0, 10, 42)
    assert original_sample_lengths(batch.seq_lens, batch.extras["sample_ids"]) == {10: 5, 42: 3}
    changed = model(batch)
    changed.total_loss.backward()
    torch.testing.assert_close(original.logits, changed.logits, rtol=0, atol=0)
    for name, p in model.named_parameters():
        if gradients[name] is None:
            assert p.grad is None
        else:
            torch.testing.assert_close(gradients[name], p.grad, rtol=0, atol=0)


def test_fractional_mask_total_below_one_preserves_weighted_token_mean():
    model = TinyMoRModel()
    batch = make_synthetic_batch(seq_lens=(5, 3))
    batch.loss_mask.zero_()
    batch.loss_mask[2] = 1
    full = model(batch).lm_loss
    full_grad = torch.autograd.grad(full, model.lm_head.weight)[0]
    batch.loss_mask[2] = 0.25
    weighted = model(batch).lm_loss
    weighted_grad = torch.autograd.grad(weighted, model.lm_head.weight)[0]
    torch.testing.assert_close(weighted, full, rtol=0, atol=0)
    torch.testing.assert_close(weighted_grad, full_grad, rtol=0, atol=0)
    batch.loss_mask.zero_()
    empty = model(batch).lm_loss
    assert empty.item() == 0
    assert torch.count_nonzero(torch.autograd.grad(empty, model.lm_head.weight)[0]) == 0


@pytest.fixture(scope="module")
def complete_reference(tmp_path_factory):
    root = tmp_path_factory.mktemp("complete-reference")
    run_reference(
        ReferenceRunConfig(
            output=root / "run", device="cpu", steps=1, num_microbatches=1, seq_lens=(5, 3)
        )
    )
    return root / "run"


def test_complete_reference_acceptance_passes(complete_reference):
    report = compare_artifacts(complete_reference, complete_reference)
    assert report["passed"], {
        key: value
        for key, value in report.items()
        if key not in {"tensor_results", "artifact_identity"}
    }
    assert report["acceptance_complete"]


@pytest.mark.parametrize(
    "namespace",
    [
        "initial",
        "gradient",
        "update",
        "post_step",
        "forward",
        "loss",
        "routes",
        "one_parameter",
        "one_round",
    ],
)
def test_acceptance_rejects_same_missing_evidence_on_both_sides(
    complete_reference, tmp_path, namespace
):
    metadata, tensors, routes = load_artifact(complete_reference)
    if namespace == "routes":
        routes = []
    elif namespace == "one_round":
        routes = [r for r in routes if r["round"] != 1]
    elif namespace == "one_parameter":
        name = metadata["initialized_parameters"][0]
        tensors = {key: value for key, value in tensors.items() if not key.endswith("/" + name)}
    else:
        tensors = {
            key: value for key, value in tensors.items() if not key.startswith(namespace + "/")
        }
    save_artifact(tmp_path, metadata=metadata, tensors=tensors, routes=routes)
    report = compare_artifacts(tmp_path, tmp_path)
    assert not report["passed"]
    assert not report["acceptance_complete"]
    assert not report["evidence"]["passed"]


def test_identical_selection_still_counts_cutoff_ties():
    from test_parity import _route

    route = _route()
    report = _compare_learned_routes([route], [route])
    assert report["passed"]
    assert report["near_ties"] == 1
    assert report["changed_near_ties"] == 0
    assert report["near_tie_rate"] == 1


def test_artifact_rejects_source_changes_during_execution(tmp_path, monkeypatch):
    from mor_mlite.parity import artifacts

    monkeypatch.setattr(artifacts, "source_snapshot", lambda: {"sha256": "b" * 64})
    with pytest.raises(RuntimeError, match="source changed"):
        save_artifact(
            tmp_path, metadata={"source_snapshot": {"sha256": "a" * 64}}, tensors={}, routes=[]
        )


def test_forward_only_evidence_requires_logits_but_not_a_training_loss(complete_reference):
    from mor_mlite.parity.evidence import check_evidence

    metadata, tensors, routes = load_artifact(complete_reference)
    metadata["forward_only"] = True
    tensors = {key: value for key, value in tensors.items() if key.startswith("forward/")}
    assert check_evidence(metadata, tensors, routes, scope="forward")["passed"]
    metadata["operator_probe"] = {"acceptance_eligible": False}
    assert not check_evidence(metadata, tensors, routes, scope="forward")["passed"]
    del metadata["operator_probe"]
    assert not check_evidence(metadata, tensors, routes, scope="all")["passed"]
    del tensors["forward/step_000/mb_000/logits"]
    assert not check_evidence(metadata, tensors, routes, scope="forward")["passed"]
