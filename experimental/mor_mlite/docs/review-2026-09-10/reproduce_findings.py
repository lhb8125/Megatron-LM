"""Read-only-source review probes; artifacts are written only to a temporary directory.

Run from the package root:
  PYTHONPATH=src .venv/bin/python docs/review-2026-09-10/reproduce_findings.py
This is diagnostic evidence for the reviewed snapshot, not a regression test suite.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from mor_mlite.config import MoRArchitectureConfig
from mor_mlite.data import make_synthetic_batch
from mor_mlite.parity.artifacts import load_artifact, save_artifact
from mor_mlite.parity.compare import _compare_learned_routes, compare_artifacts
from mor_mlite.tiny.model import TinyMoRModel


def main() -> None:
    torch.manual_seed(1234)
    torch.set_num_threads(1)
    result = {}
    model = TinyMoRModel()
    batch = make_synthetic_batch(seq_lens=(5, 3))
    with torch.no_grad():
        empty_replay = model(batch, route_mode="replay", replay_plans={})
        result["empty_replay_actual_modes"] = [p.mode for p in empty_replay.route_plans]
        learned = model(batch)
        plans = {p.round_index: p for p in learned.route_plans}
        plans[1] = replace(plans[1], original_positions=plans[1].original_positions + 100)
        replay = model(batch, route_mode="replay", replay_plans=plans)
        result["corrupt_replay_positions"] = {
            "accepted": True,
            "supplied": plans[1].original_positions.tolist(),
            "returned": replay.route_plans[1].original_positions.tolist(),
        }

    unmasked = make_synthetic_batch(seq_lens=(5, 3), mask_last_token=False)
    _, mask = TinyMoRModel._lm_targets(unmasked)
    result["optional_mask"] = {
        "input_mask_is_none": unmasked.loss_mask is None,
        "real_tokens": unmasked.input_ids.numel(),
        "tiny_target_weight_sum": mask.sum().item(),
    }
    noncontiguous = make_synthetic_batch(seq_lens=(5, 3))
    noncontiguous.extras["sample_ids"] += 10
    try:
        model(noncontiguous)
    except ValueError as exc:
        result["stable_sample_ids_error"] = str(exc)

    almost_full = MoRArchitectureConfig(1, 1, 3, 1, [1 - 5e-13, 2 / 3, 1 / 3])
    linear = MoRArchitectureConfig(1, 1, 10, 1)
    result["capacity"] = {
        "accepted_first_round_k_for_length_8": almost_full.top_k(8, 0),
        "linear_round_3_k_for_length_90": linear.top_k(90, 3),
        "integer_formula": ((10 - 3) * 90) // 10,
    }

    route = {
        "schema_version": 1,
        "phase": "train",
        "step": 0,
        "microbatch": 0,
        "round": 1,
        "mode": "learned",
        "sample_ids": [0, 0],
        "original_positions": [0, 1],
        "global_token_ids": [10, 11],
        "source_tp_ranks": [0, 0],
        "source_cp_ranks": [0, 0],
        "source_local_rows": [0, 1],
        "target_tp_ranks": [0, 0],
        "target_cp_ranks": [0, 0],
        "target_local_rows": [0, 1],
        "selected_gates": [0.09, 0.05],
        "active_cu_seqlens": [0, 2],
        "padding_mask": [False, False],
        "cutoff_score_margins": {"0": 0.0},
        "candidate_global_token_ids": [10, 11, 12],
        "candidate_sample_ids": [0, 0, 0],
        "candidate_original_positions": [0, 1, 2],
        "candidate_scores": [0.9, 0.5, 0.5],
    }
    result["identical_exact_cutoff_tie"] = _compare_learned_routes([route], [route])

    source = Path(__file__).resolve().parents[2] / "artifacts/local_validation"
    with TemporaryDirectory(prefix="mor-review-") as directory:
        root = Path(directory)
        removed = {}
        for side, name in [
            ("baseline", "final_source_reference_train"),
            ("candidate", "final_source_reference_replay"),
        ]:
            metadata, tensors, routes = load_artifact(source / name)
            kept = {
                key: value
                for key, value in tensors.items()
                if key.split("/")[0] not in {"initial", "gradient", "update", "post_step"}
            }
            removed[side] = {"tensors": len(tensors) - len(kept), "routes": len(routes)}
            save_artifact(root / side, metadata=metadata, tensors=kept, routes=[])
        report = compare_artifacts(root / "baseline", root / "candidate")
        result["missing_training_evidence"] = {
            "removed": removed,
            "scope": report["scope"],
            "passed": report["passed"],
            "gradients": report["namespace_summary"]["gradient"]["tensor_count"],
            "routes": report["routes"]["total"],
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

