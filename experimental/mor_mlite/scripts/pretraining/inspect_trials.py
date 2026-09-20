"""Read-only MBS evidence audit; never freeze an unfinished sweep."""

import argparse
import json
import math
from pathlib import Path


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def single(path):
    values = rows(path)
    if len(values) != 1:
        raise ValueError(f"expected one complete record: {path}")
    return values[0]


def inspect(root):
    from mor_mlite.pretraining.tuning import Trial

    root = Path(root)
    result = single(root / "trial.jsonl")
    contract = result["contract"]
    manifest = json.loads((root / "manifest.json").read_text())
    if any(manifest.get(k) != v for k, v in contract.items()):
        raise ValueError("trial and manifest contracts differ")
    completion = single(root / "complete.jsonl")
    if completion["contract"] != contract or completion["step"] != 50 or contract["mode"] != "tune":
        raise ValueError("not a completed 50-update tuning run")
    if result["formal_training_tokens"] != 0:
        raise ValueError("performance trial must not consume formal tokens")
    checkpoint = json.loads((root / "checkpoint-000050/pretraining-state.json").read_text())
    experiment = contract["experiment"]
    tokens_per_step = experiment["seq_length"] * experiment["global_batch_size"]
    if (
        checkpoint["contract"] != contract
        or checkpoint["step"] != 50
        or checkpoint["cursor"] != 50 * tokens_per_step
    ):
        raise ValueError("checkpoint contract or data cursor differs")
    updates = rows(root / "loss.jsonl")
    if [r["step"] for r in updates] != list(range(1, 51)):
        raise ValueError("tuning updates are missing or duplicated")
    for record in updates:
        if record["tokens"] != record["step"] * tokens_per_step:
            raise ValueError("training token cursor differs")
        if (
            any(
                not math.isfinite(record[k])
                for k in ("lm_loss", "depth_aux", "grad_norm", "seconds")
            )
            or record["seconds"] <= 0
        ):
            raise ValueError("nonfinite or invalid update")
        if len(bytes.fromhex(record["input_sha256"])) != 32:
            raise ValueError("invalid delivered-input digest")
    evaluations = rows(root / "evaluation.jsonl")
    if [r["step"] for r in evaluations] != [0, 50] or any(
        not math.isfinite(r["nll"]) or r["targets"] <= 0 for r in evaluations
    ):
        raise ValueError("missing or invalid initial/final validation")
    memory = single(root / "memory.jsonl")["ranks"]
    if sorted(r["rank"] for r in memory) != list(range(experiment["world_size"])):
        raise ValueError("missing or duplicate memory rank")
    for record in memory:
        if record["samples"] < 1 or record["capacity_bytes"] <= 0:
            raise ValueError("missing device-memory samples")
        if not math.isclose(
            record["peak_device_fraction"],
            record["whole_device_peak_bytes"] / record["capacity_bytes"],
        ):
            raise ValueError("memory fraction disagrees with measured bytes")
        if any(record[k] < 0 for k in ("allocated_peak_bytes", "reserved_peak_bytes")):
            raise ValueError("invalid allocator peak")
    trial = Trial(**result["trial"])
    if (
        trial.mbs != experiment["micro_batch_size"]
        or trial.warmup_steps != 20
        or trial.measured_steps != 30
        or not all((trial.updates_ok, trial.evaluation_ok, trial.checkpoint_ok))
        or not math.isfinite(trial.tokens_per_second)
        or trial.tokens_per_second <= 0
        or not 0 < trial.peak_device_fraction <= 1
    ):
        raise ValueError("trial did not complete valid training/evaluation/checkpoint measurements")
    if not math.isclose(trial.peak_device_fraction, max(r["peak_device_fraction"] for r in memory)):
        raise ValueError("trial omitted a rank's memory peak")
    measured = sum(r["seconds"] for r in updates[20:])
    if not math.isclose(trial.tokens_per_second, 30 * tokens_per_step / measured, rel_tol=1e-6):
        raise ValueError("throughput does not match the final 30 updates")
    return trial, contract, [r["input_sha256"] for r in updates]


def main():
    from mor_mlite.pretraining.tuning import next_mbs, select_mbs

    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--run", type=Path, action="append", required=True)
    args = parser.parse_args()
    evidence = [inspect(root) for root in args.run]
    trials = [item[0] for item in evidence]
    identities = []
    for _, contract, hashes in evidence:
        comparable = {
            **contract,
            "experiment": {
                k: v for k, v in contract["experiment"].items() if k != "micro_batch_size"
            },
        }
        identities.append((json.dumps(comparable, sort_keys=True), hashes))
    if any(item != identities[0] for item in identities):
        raise ValueError("MBS trials changed model/data/schedule/source or delivered inputs")
    candidate = select_mbs(trials)
    following = next_mbs(trials[-1])
    print(
        json.dumps(
            {
                "scope": "measured tuning evidence only; no formal acceptance certificate",
                "next_mbs": following,
                "sweep_complete": following is None,
                "selected_mbs": candidate if following is None else None,
                "best_so_far": candidate,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
