"""Capture a registered terminal CUDA-OOM attempt without changing GPU artifacts."""

import argparse
import json
import shlex
import subprocess
import tempfile
from pathlib import Path


def main():
    from inspect_rejected_trial import cuda_oom_terminal, inspect
    from mcore_devtoolkit.cluster_run.registry import default_registry

    from mor_mlite.pretraining.config import Experiment
    from mor_mlite.pretraining.data import sha256_file
    from mor_mlite.provenance import source_snapshot

    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--registry-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve existing evidence; use a new evidence directory")
    record = default_registry().get(args.registry_id)
    if record is None or not str(record.scheduler_id).isdigit():
        raise ValueError("not a registered SLURM job")
    command = record.command

    def argument(flag):
        if command.count(flag) != 1:
            raise ValueError(f"ambiguous registered option: {flag}")
        return command[command.index(flag) + 1]

    if argument("--mode") != "tune" or "mor_mlite.pretraining.train" not in command:
        raise ValueError("only actual tuning attempts can be rejected by this tool")
    config = Experiment(
        argument("--arm"),
        world_size=int(argument("--world-size")),
        micro_batch_size=int(argument("--mbs")),
    )
    metadata = record.metadata
    host = metadata["remote_host"]
    output = argument("--output")
    if not output.startswith(metadata["remote_project_path"] + "/runtime/mor-pretraining/"):
        raise ValueError("registered output is outside this experiment")
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host]
    accounting = subprocess.run(
        ssh + [f"sacct -n -P -j {record.scheduler_id} --format=JobID,State,ExitCode"],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    ).stdout
    log = subprocess.run(
        ssh + ["cat " + shlex.quote(metadata["remote_log_path"])],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    ).stdout
    cuda_oom_terminal(accounting, record.scheduler_id, log)
    manifest_text = subprocess.run(
        ssh + ["cat " + shlex.quote(output + "/manifest.json")],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    ).stdout
    manifest = json.loads(manifest_text)
    contract = {key: value for key, value in manifest.items() if key != "topology"}
    if (
        contract["experiment"] != config.to_dict()
        or contract["mode"] != "tune"
        or contract["source_sha256"] != source_snapshot()["sha256"]
    ):
        raise ValueError("failure manifest does not match registered configuration/current source")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".rejection-audit-", dir=args.output.parent) as tmp:
        temporary = Path(tmp) / "sealed"
        temporary.mkdir()
        contents = {
            "manifest.json": manifest_text,
            "scheduler.txt": accounting,
            "stderr.log": log,
            "submission.json": json.dumps(record.to_public_dict(), indent=2),
        }
        for name, text in contents.items():
            (temporary / name).write_text(text)
        failure = {
            "outcome": "rejected_cuda_oom",
            "formal_training_tokens": 0,
            "scheduler_id": str(record.scheduler_id),
            "registry_id": record.job_id,
            "contract": contract,
            "measured_trial": None,
            "files_sha256": {name: sha256_file(temporary / name) for name in contents},
        }
        (temporary / "rejection.json").write_text(json.dumps(failure, indent=2) + "\n")
        inspect(temporary)
        temporary.rename(args.output)
    print(
        json.dumps(
            {
                "recorded": str(args.output),
                "job": record.scheduler_id,
                "outcome": "rejected_cuda_oom",
                "arm": config.arm,
                "mbs": config.micro_batch_size,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
