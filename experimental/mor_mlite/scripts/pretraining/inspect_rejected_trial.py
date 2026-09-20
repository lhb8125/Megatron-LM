"""Audit explicit CUDA-OOM rejection evidence, never fabricate a successful trial.

The original scheduler record, model manifest and complete error log must be
retained. No peak, throughput, completed updates or checkpoint is inferred.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RejectedTrial:
    mbs: int

    @property
    def eligible(self):
        return False


def cuda_oom_terminal(accounting, job_id, log):
    records = [
        line.split("|") for line in accounting.splitlines() if line.split("|")[0] == str(job_id)
    ]
    if len(records) != 1 or len(records[0]) < 3:
        raise ValueError("missing/ambiguous failed scheduler record")
    state, code = records[0][1:3]
    if state not in {"FAILED", "OUT_OF_MEMORY"} or code == "0:0":
        raise ValueError("CUDA OOM rejection needs a terminal failed job")
    # Host OOM, allocation requests in normal logs, NCCL timeouts, and pending
    # jobs are not evidence of an infeasible CUDA microbatch size.
    if not re.search(
        r"(?:torch\.(?:cuda\.)?OutOfMemoryError|RuntimeError): CUDA out of memory\.", log
    ):
        raise ValueError("no actual CUDA out-of-memory exception")
    return state


def inspect(root):
    from mor_mlite.pretraining.config import Experiment
    from mor_mlite.pretraining.data import sha256_file

    root = Path(root)
    failure = json.loads((root / "rejection.json").read_text())
    if failure["outcome"] != "rejected_cuda_oom" or failure["formal_training_tokens"] != 0:
        raise ValueError("not an explicit tuning-only CUDA OOM rejection")
    expected = {"manifest.json", "scheduler.txt", "stderr.log", "submission.json"}
    if set(failure["files_sha256"]) != expected or any(
        sha256_file(root / name) != digest for name, digest in failure["files_sha256"].items()
    ):
        raise ValueError("rejection evidence changed or is incomplete")
    cuda_oom_terminal(
        (root / "scheduler.txt").read_text(),
        failure["scheduler_id"],
        (root / "stderr.log").read_text(),
    )
    manifest = json.loads((root / "manifest.json").read_text())
    contract = {k: v for k, v in manifest.items() if k != "topology"}
    config = Experiment(**contract["experiment"])
    submission = json.loads((root / "submission.json").read_text())
    if (
        str(submission["scheduler_id"]) != failure["scheduler_id"]
        or submission["job_id"] != failure["registry_id"]
    ):
        raise ValueError("rejection belongs to another registered job")
    command = json.loads(submission["command_json"])
    if "mor_mlite.pretraining.train" not in command:
        raise ValueError("rejected submission is not a production tuning entry")
    for flag, value in {
        "--mode": "tune",
        "--arm": config.arm,
        "--mbs": str(config.micro_batch_size),
        "--world-size": str(config.world_size),
    }.items():
        if command.count(flag) != 1 or command[command.index(flag) + 1] != value:
            raise ValueError("registered rejected command differs from the manifest")
    if contract["mode"] != "tune" or contract != failure["contract"]:
        raise ValueError("rejected job is not the recorded performance configuration")
    if (
        not manifest["topology"]["passed"]
        or manifest["topology"]["world_size"] != config.world_size
    ):
        raise ValueError("rejected job did not establish the intended EP topology")
    if failure["measured_trial"] is not None:
        raise ValueError("OOM rejection must not fabricate performance measurements")
    return RejectedTrial(config.micro_batch_size), contract, None


def inspect_attempt(root):
    from inspect_trials import inspect as inspect_success

    root = Path(root)
    if (root / "rejection.json").exists():
        if (root / "trial.jsonl").exists() or (root / "complete.jsonl").exists():
            raise ValueError("conflicting successful and rejected trial evidence")
        return inspect(root)
    return inspect_success(root)
