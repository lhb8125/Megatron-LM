"""Read-only tiny smoke/full-width initialization audit; no formal certificate."""

import argparse
import json
from pathlib import Path


def inspect_parameters(initial, world):
    if len(initial) != world:
        raise ValueError("missing initial parameter rank")
    dense = lambda row: {k: v for k, v in row.items() if ".moe.experts." not in k}
    if any(dense(row) != dense(initial[0]) for row in initial):
        raise ValueError("dense initialization differs across replicas")
    if any(initial[rank] != initial[rank % 16] for rank in range(world)):
        raise ValueError("expert-DP initialization differs across replicas")
    backbone = sum(v["numel"] for v in dense(initial[0]).values())
    backbone += sum(
        v["numel"] for row in initial[:16] for k, v in row.items() if ".moe.experts." in k
    )
    expert_hashes = [
        v["sha256"] for row in initial[:16] for k, v in row.items() if ".moe.experts." in k
    ]
    if not expert_hashes or len(set(expert_hashes)) != len(expert_hashes):
        raise ValueError("independent expert matrices unexpectedly have identical initial values")
    return backbone


def inspect_initialization(root, world):
    reports = [json.loads((root / f"rank-{rank}.json").read_text()) for rank in range(world)]
    if [r["rank"] for r in reports] != list(range(world)) or any(
        r["world_size"] != world for r in reports
    ):
        raise ValueError("initialization rank identity mismatch")
    if len({(r["arm"], r["source"]["sha256"], r["model_config_sha256"]) for r in reports}) != 1:
        raise ValueError("initialization ranks used different contracts")
    initial = [r["backbone"] for r in reports]
    backbone = inspect_parameters(initial, world)
    arm = reports[0]["arm"]
    expected = backbone + (3 * 2048 if arm == "D" else 0)
    if any(
        r["parameters"] != {"backbone_parameters": backbone, "independent_parameters": expected}
        for r in reports
    ):
        raise ValueError("full-width parameter counting disagrees with complete fingerprints")
    return initial, {
        "arm": arm,
        "ranks": world,
        "backbone_parameters": backbone,
        "independent_parameters": expected,
        "replicated_initialization_bitwise": True,
        "expert_initializations_distinct": True,
        "source_sha256": reports[0]["source"]["sha256"],
        "model_config_sha256": reports[0]["model_config_sha256"],
    }


def inspect(root, world):
    read = lambda name: json.loads((root / name).read_text())
    initial = [read(f"initial-rank-{rank}.json") for rank in range(world)]
    reports = [read(f"result-rank-{rank}.json") for rank in range(world)]
    if not all(
        r["optimizer_update"] and r["causality"]["passed"] and r["topology"]["passed"]
        for r in reports
    ):
        raise ValueError("not all ranks passed the smoke")
    if len({r["source"]["sha256"] for r in reports}) != 1:
        raise ValueError("rank sources differ")
    backbone = inspect_parameters(initial, world)
    arm = reports[0]["arm"]
    if any(r["arm"] != arm for r in reports):
        raise ValueError("mixed-arm evidence")
    expected = backbone + (3 * read("config.json")["hidden_size"] if arm == "D" else 0)
    optimizer = read("continuity-rank-0.json")["at_save"]["optimizer"]
    if sum(optimizer["master_parameter_bytes"]) != 4 * expected:
        raise ValueError(
            "optimizer master ownership does not cover unique model parameters exactly"
        )
    return initial, {
        "arm": arm,
        "ranks": world,
        "backbone_parameters": backbone,
        "independent_parameters": expected,
        "optimizer_unique_coverage": True,
        "replicated_initialization_bitwise": True,
        "expert_initializations_distinct": True,
        "source_sha256": reports[0]["source"]["sha256"],
        "max_causal_abs_error": max(r["causality"]["max_abs_error_local"] for r in reports),
        "zero_active_rank_rounds": reports[0]["causality"]["zero_active_rank_rounds"],
    }


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", type=Path, action="append")
    group.add_argument("--initialization-run", type=Path, action="append")
    parser.add_argument("--world-size", type=int, choices=[32, 64], default=32)
    args = parser.parse_args()
    initial, summaries = {}, {}
    for root in args.run or args.initialization_run:
        audit = inspect if args.run else inspect_initialization
        parameters, result = audit(root, args.world_size)
        arm = result["arm"]
        if arm in summaries:
            raise ValueError("duplicate arm")
        initial[arm], summaries[arm] = parameters, result
    if len({r["source_sha256"] for r in summaries.values()}) != 1:
        raise ValueError("arms used different source snapshots")
    pairs = {}
    for arm in ("C", "D"):
        if "B" in initial and arm in initial:
            if initial["B"] != initial[arm]:
                raise ValueError(f"B/{arm} backbone initialization is not bitwise identical")
            pairs[f"B/{arm}"] = True
    print(
        json.dumps(
            {
                "scope": "tiny native smoke evidence only"
                if args.run
                else "full-width initialization only; no training/Adam-state memory evidence",
                "arms": summaries,
                "backbone_initialization_bitwise": pairs,
                "missing_arms": sorted(set("ABCD") - set(summaries)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
