from __future__ import annotations

import copy
import random

import numpy as np
import torch

from mor_mlite.parity.continuity import (
    build_checkpoint_continuity_report,
    capture_rng_state,
    check_checkpoint_continuity,
    state_fingerprint,
)


def _fingerprint(value: str) -> dict[str, str]:
    return {"sha256": value * 64}


def _route(phase: str) -> dict:
    return {
        "phase": phase,
        "step": 1,
        "microbatch": 0,
        "round": 1,
        "mode": "learned",
        "sample_ids": [0, 0],
        "original_positions": [0, 2],
        "global_token_ids": [10, 12],
        "active_cu_seqlens": [0, 2],
        "padding_mask": [False, False],
        # Floating routing diagnostics are compared through tensors, not as
        # RoutePlan identity fields.
        "selected_gates": [0.05, 0.04],
    }


def _continuity_inputs() -> dict:
    tensors = {}
    values = {
        "forward": ("mb_000/logits", torch.tensor([[1.0, -2.0]])),
        "loss": ("mb_000/total", torch.tensor(2.5)),
        "gradient": ("weight", torch.tensor([0.25, -0.5])),
        "update": ("weight", torch.tensor([-0.01, 0.01])),
        "post_step": ("weight", torch.tensor([0.99, 2.01])),
    }
    for namespace, (suffix, value) in values.items():
        for phase in ("uninterrupted", "resume"):
            tensors[f"{namespace}/{phase}/step_001/{suffix}"] = value.clone()
    fingerprints = {
        "parameters": _fingerprint("a"),
        "optimizer": _fingerprint("b"),
        "rng": _fingerprint("c"),
    }
    post_fingerprints = {
        "parameters": _fingerprint("d"),
        "optimizer": _fingerprint("e"),
        "rng": _fingerprint("f"),
    }
    return {
        "step": 1,
        "precision": "fp32",
        "tensors": tensors,
        "routes": [_route("uninterrupted"), _route("resume")],
        "optimizer_steps": [
            {
                "phase": phase,
                "step": 1,
                "updated": True,
                "grad_norm": 0.75,
                "num_zeros": 0,
            }
            for phase in ("uninterrupted", "resume")
        ],
        "saved_fingerprints": fingerprints,
        "restored_fingerprints": copy.deepcopy(fingerprints),
        "uninterrupted_fingerprints": post_fingerprints,
        "resumed_fingerprints": copy.deepcopy(post_fingerprints),
    }


def test_state_fingerprint_is_order_independent_and_covers_optimizer_payload() -> None:
    state = {
        "state": {
            0: {
                "step": torch.tensor(3.0),
                "exp_avg": torch.tensor([0.25, -0.5]),
                "exp_avg_sq": torch.tensor([0.1, 0.2]),
            }
        },
        "param_groups": [{"lr": 1e-3, "params": [0]}],
    }
    reordered = {"param_groups": copy.deepcopy(state["param_groups"]), "state": state["state"]}
    baseline = state_fingerprint(state)

    assert baseline == state_fingerprint(reordered)
    assert baseline["tensor_count"] == 3
    mutated = copy.deepcopy(state)
    mutated["state"][0]["exp_avg"][0] += 1.0
    assert state_fingerprint(mutated)["sha256"] != baseline["sha256"]


def test_rng_fingerprint_covers_python_numpy_and_torch_without_advancing() -> None:
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)

    first = state_fingerprint(capture_rng_state())
    second = state_fingerprint(capture_rng_state())
    assert first == second

    torch.rand(1)
    assert state_fingerprint(capture_rng_state())["sha256"] != first["sha256"]


def test_checkpoint_continuity_report_covers_state_and_next_step() -> None:
    report = build_checkpoint_continuity_report(**_continuity_inputs())

    assert report["status"] == "passed"
    assert report["next_step"]["tensor_coverage"] == {
        "forward": 1,
        "loss": 1,
        "gradient": 1,
        "update": 1,
        "post_step": 1,
    }
    assert report["next_step"]["routes_exact"]
    assert report["next_step"]["optimizer_step"]["passed"]
    assert check_checkpoint_continuity(report, required=True)["passed"]


def test_checkpoint_continuity_report_fails_on_optimizer_restore_or_gradient_drift() -> None:
    inputs = _continuity_inputs()
    inputs["restored_fingerprints"]["optimizer"] = _fingerprint("0")
    inputs["tensors"]["gradient/resume/step_001/weight"][0] += 0.1

    report = build_checkpoint_continuity_report(**inputs)

    assert report["status"] == "failed"
    assert not report["restored_state"]["optimizer"]["passed"]
    assert any(
        not result["passed"]
        for result in report["next_step"]["tensor_results"]
        if result["canonical_name"].startswith("gradient/")
    )
    assert not check_checkpoint_continuity(report, required=True)["passed"]


def test_checkpoint_continuity_contract_fails_closed_when_required() -> None:
    result = check_checkpoint_continuity(None, required=True)
    assert not result["passed"]
    assert "missing" in result["details"][0]
