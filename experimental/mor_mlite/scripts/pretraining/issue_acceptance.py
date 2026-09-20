"""Assemble formal gates from complete evidence, never from manually set booleans.

Tiny native probes establish algorithm semantics. Full-width initialization,
selected-MBS tuning and external-process indexed recovery bind the actual run.
The full corpus must be sealed and audited; performance-pilot data is rejected.
"""

import argparse
import json
import math
from pathlib import Path


def rank_reports(root, world, *, prefix="rank"):
    return [json.loads((root / f"{prefix}-{rank}.json").read_text()) for rank in range(world)]


def require_four(mapping, label):
    if set(mapping) != set("ABCD"):
        raise ValueError(f"{label} requires exactly A/B/C/D")


def checked_source(rows, source):
    if any(row["source"]["sha256"] != source for row in rows):
        raise ValueError("evidence belongs to another source snapshot")


def check_data_provenance(manifest, tokenizer_dir):
    from mor_mlite.pretraining.config import validate_data_purpose
    from mor_mlite.pretraining.data import sha256_file
    from mor_mlite.pretraining.prepare_raw import DATASET, REVISION, TOKENIZER, TOKENIZER_REVISION

    validate_data_purpose(manifest, "train")
    provenance = manifest["provenance"]
    contract = json.loads(provenance["dataset_source"])
    expected = {
        "dataset": DATASET,
        "revision": REVISION,
        "tokenizer": TOKENIZER,
        "tokenizer_revision": TOKENIZER_REVISION,
        "target_tokens": 50_000_000_000,
        "eos": 151643,
        "chat_template": False,
        "sampling": "sorted-train-shards-whole-document-prefix-v1",
    }
    if any(contract.get(key) != value for key, value in expected.items()):
        raise ValueError("formal corpus preparation differs from approved pinned provenance")
    if (
        provenance["tokenizer_source"] != f"{TOKENIZER}@{TOKENIZER_REVISION}"
        or manifest["stored_tokens"] < expected["target_tokens"]
        or manifest["seed"] != 1234
        or manifest["split"] != "sha256(token-content)-mod1000-zero-validation"
    ):
        raise ValueError("formal tokenizer, token budget or split differs")
    if not manifest["tokenizer_files"]:
        raise ValueError("missing tokenizer file fingerprints")
    for name, digest in manifest["tokenizer_files"].items():
        if Path(name).name != name or sha256_file(tokenizer_dir / name) != digest:
            raise ValueError("formal tokenizer files differ from the corpus tokenizer")


def main():
    from inspect_rejected_trial import inspect_attempt as inspect_trial
    from inspect_smokes import inspect, inspect_initialization
    from tuning_evidence import (
        check_screening_contract,
        combined_trials,
        load_screening,
        next_candidate,
    )

    from mor_mlite.pretraining.config import Experiment, validate_base_model, validate_data_purpose
    from mor_mlite.pretraining.data import sha256_file, verify_sources
    from mor_mlite.pretraining.probe import native_tree
    from mor_mlite.pretraining.tuning import select_mbs
    from mor_mlite.provenance import source_snapshot

    parser = argparse.ArgumentParser(allow_abbrev=False)
    for name in ("smoke-run", "initialization-run", "auxiliary-run", "resume-run", "trial-run"):
        parser.add_argument("--" + name, type=Path, action="append", required=True)
    for name in ("shared-run", "data", "hf-config", "environment", "native-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--world-size", type=int, choices=[32, 64], default=64)
    parser.add_argument("--screening-run", type=Path, action="append", default=[])
    args = parser.parse_args()
    source = source_snapshot()["sha256"]
    screening = load_screening(args.screening_run, world_size=args.world_size, source=source)
    manifest = json.loads((args.data / "manifest.json").read_text())
    model_config = json.loads((args.hf_config / "config.json").read_text())
    validate_data_purpose(manifest, "train")
    check_data_provenance(manifest, args.hf_config)
    validate_base_model(model_config, manifest)
    data_sha = sha256_file(args.data / "manifest.json")
    model_sha = sha256_file(args.hf_config / "config.json")
    environment_sha = sha256_file(args.environment)
    environment = json.loads(args.environment.read_text())
    if not environment.get("te_forward_backward") or environment["native"] != native_tree(
        args.native_root
    ):
        raise ValueError("candidate environment/native canary mismatch")

    # Every receipt is preserved as a content hash, including all ranks.
    roots = (
        args.smoke_run
        + args.initialization_run
        + args.auxiliary_run
        + args.resume_run
        + args.trial_run
        + args.screening_run
        + [args.shared_run]
    )
    evidence = {}
    for root in roots:
        for path in sorted(root.rglob("*.json*")):
            if path.is_file():
                evidence[str(path.resolve())] = sha256_file(path)

    smokes, initializations, auxiliaries, trials, resumes = {}, {}, {}, {}, {}
    for root in args.smoke_run:
        initial, summary = inspect(root, 32)
        if summary["source_sha256"] != source or summary["arm"] in smokes:
            raise ValueError("invalid or duplicate native smoke evidence")
        smokes[summary["arm"]] = initial
    require_four(smokes, "native smoke")
    if smokes["B"] != smokes["C"] or smokes["B"] != smokes["D"]:
        raise ValueError("tiny B/C/D backbone initialization differs")
    for root in args.initialization_run:
        initial, summary = inspect_initialization(root, args.world_size)
        if (
            summary["source_sha256"] != source
            or summary["model_config_sha256"] != model_sha
            or summary["arm"] in initializations
        ):
            raise ValueError("invalid or duplicate full-width initialization")
        initializations[summary["arm"]] = initial
    require_four(initializations, "full-width initialization")
    if initializations["B"] != initializations["C"] or initializations["B"] != initializations["D"]:
        raise ValueError("full-width B/C/D backbone initialization differs")
    for root in args.auxiliary_run:
        reports = rank_reports(root, 32)
        checked_source(reports, source)
        arm = reports[0]["arm"]
        if arm in auxiliaries or any(
            not r["passed"] or r["arm"] != arm or not r["optimizer_ownership"] for r in reports
        ):
            raise ValueError("invalid auxiliary normalization/ownership evidence")
        auxiliaries[arm] = True
    require_four(auxiliaries, "auxiliary normalization")
    shared = rank_reports(args.shared_run, 32)
    checked_source(shared, source)
    if any(
        not row["passed"]
        or not row["metrics"]
        or any(
            not all(math.isfinite(m[k]) and m[k] >= 0.999 for k in ("tensor_similarity", "cosine"))
            for m in row["metrics"]
        )
        for row in shared
    ):
        raise ValueError("shared-gradient unrolling check failed")

    for root in args.trial_run:
        trial, contract, hashes = inspect_trial(root)
        arm = contract["experiment"]["arm"]
        expected = Experiment(arm, world_size=args.world_size, micro_batch_size=trial.mbs).to_dict()
        if contract["experiment"] != expected or any(
            contract[k] != v
            for k, v in {
                "source_sha256": source,
                "data_sha256": data_sha,
                "model_config_sha256": model_sha,
                "environment_sha256": environment_sha,
            }.items()
        ):
            raise ValueError("MBS trial does not bind the approved formal data/config/environment")
        check_screening_contract(screening.get(arm, []), contract)
        topology = json.loads((root / "manifest.json").read_text())["topology"]
        if not topology["passed"] or topology["world_size"] != args.world_size:
            raise ValueError("MBS trial lacks matching multi-node EP locality evidence")
        trials.setdefault(arm, []).append((trial, hashes))
    require_four(trials, "MBS sweep")
    selected = {}
    common_hashes = None
    for arm, arm_trials in trials.items():
        arm_trials.sort(key=lambda value: value[0].mbs)
        screened = [value[0] for value in screening.get(arm, [])]
        final = [value[0] for value in arm_trials]
        selected[arm] = select_mbs(combined_trials(screened, final))
        if next_candidate(screened, final) is not None:
            raise ValueError("MBS sweep still requires another size; cannot freeze")
        for _, hashes in arm_trials:
            if hashes is None:
                continue  # Rejection has no fabricated completed input sequence.
            if common_hashes is None:
                common_hashes = hashes
            elif hashes != common_hashes:
                raise ValueError("different arms/MBS consumed different global inputs")

    for root in args.resume_run:
        reports = rank_reports(root, args.world_size, prefix="resume-rank")
        arm = reports[0]["contract"]["experiment"]["arm"]
        if arm in resumes or arm not in selected:
            raise ValueError("duplicate or unexpected indexed recovery arm")
        expected = Experiment(arm, world_size=args.world_size, micro_batch_size=selected[arm])
        for row in reports:
            contract = row["contract"]
            if (
                not row["passed"]
                or not contract["scope"].startswith("full-width")
                or contract["experiment"] != expected.to_dict()
                or contract["fixture"] != model_config
                or contract["data_sha256"] != data_sha
                or contract["source_sha256"] != source
                or row["cursor"] != 2 * expected.tokens_per_step
                or row["next_input_sha256"] != common_hashes[1]
            ):
                raise ValueError("full-width indexed resume does not match the frozen experiment")
        resumes[arm] = True
    require_four(resumes, "full-width indexed recovery")

    # Potentially large read-only corpus verification happens once, not on GPUs.
    verify_sources(args.data)
    args.output.mkdir(parents=True, exist_ok=False)
    checks = dict.fromkeys(
        (
            "causality",
            "initialization",
            "shared_gradients",
            "auxiliary_gradients",
            "data_integrity",
            "checkpoint_resume",
            "gb200_multinode",
        ),
        True,
    )
    for arm in "ABCD":
        receipt = {
            "source_sha256": source,
            "data_sha256": data_sha,
            "environment_sha256": environment_sha,
            "model_config_sha256": model_sha,
            "experiment": Experiment(
                arm, world_size=args.world_size, micro_batch_size=selected[arm]
            ).to_dict(),
            "checks": checks,
            "evidence_sha256": evidence,
            "scope": __doc__,
            "issuer_sha256": sha256_file(Path(__file__)),
            "startup_wrapper_sha256": sha256_file(Path(__file__).with_name("fused_python.sh")),
            "screening_evidence": [str(path.resolve()) for path in args.screening_run],
            "tuning_policy": "screening may seed search; selected MBS qualified on final data",
        }
        (args.output / f"{arm.lower()}.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"issued": str(args.output), "selected_mbs": selected}), flush=True)


if __name__ == "__main__":
    main()
