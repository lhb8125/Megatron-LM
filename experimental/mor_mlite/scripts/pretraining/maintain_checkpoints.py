"""Explicitly authorized retention and conservative next-allocation disk budget."""

import copy
import hashlib
import json
import pickle
import shlex
import subprocess
import time
from pathlib import Path


def prune_command(host, raw, *, apply=False):
    script = Path(__file__).with_name("prune_weight_files.py").read_text()
    digest = hashlib.sha256(raw).hexdigest()
    command = "python3 -c " + shlex.quote(script) + " --approved-sha256 " + digest
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host,
            command + (" --apply" if apply else "")]


def run_prune(host, raw, *, apply=False, stream=None):
    """Retry only proven pre-authentication failures, never ambiguous deletion."""
    from mcore_devtoolkit.cluster_run.remote import preauth_disconnect

    command = prune_command(host, raw, apply=apply)
    for attempt in range(3):
        result = subprocess.run(command, input=raw.decode(), capture_output=True,
                                text=True, timeout=180 if apply else 120)
        if stream is not None:
            stream.write((result.stdout + result.stderr).encode())
            stream.flush()
        if preauth_disconnect(result) and attempt < 2:
            time.sleep(5 * (attempt + 1))
            continue
        result.check_returncode()
        return result


def recover_prune_preauth(state, state_dir, manifest_name, total):
    """Reconcile unchanged targets read-only; main loop must make a fresh plan."""
    from mcore_devtoolkit.cluster_run.remote import preauth_disconnect
    from plan_checkpoint_retention import build_plan, remote_inventory

    if (state.get("status") != "failed" or state.get("schema_version") != 2
            or "intent" in state or not state.get("active")
            or state["binding"].get("prune_approved") != "True"):
        raise ValueError("not a recoverable authorized pruning failure")
    path = Path(state_dir) / manifest_name
    if (Path(manifest_name).name != manifest_name or not manifest_name.startswith("retention-")
            or path.suffix != ".json" or path.is_symlink() or path.with_suffix(".log").is_symlink()):
        raise ValueError("invalid local retention manifest")
    raw = path.read_bytes()
    command = prune_command(state["binding"]["host"], raw, apply=True)
    if state.get("error") != f"CalledProcessError: {subprocess.CalledProcessError(255, command)}":
        raise ValueError("failure was not this exact prune operation")
    lines = path.with_suffix(".log").read_text().splitlines()
    if not 1 <= len(lines) <= 3 or not all(preauth_disconnect(
            subprocess.CompletedProcess(command, 255, "", line)) for line in lines):
        raise ValueError("prune outcome is ambiguous; explicit reconciliation required")
    manifest = json.loads(raw)
    runs = {r["output"]: r for r in state["runs"] + list(state["active"].values())}
    if manifest["allowed_run_roots"] != sorted(runs) or not manifest["files"]:
        raise ValueError("retention manifest escaped current campaign")
    records = remote_inventory(state["binding"]["host"], list(runs.values()))
    plan = build_plan(state, records, total)
    candidates = {f["path"]: f for row in plan["candidates"] for f in row["files"]}
    if any(candidates.get(f["path"]) != f for f in manifest["files"]):
        raise ValueError("prune targets changed, missing or now protected")
    # Remote validator checks every inode/mtime/size, ownership and marker; no unlink.
    checked = json.loads(run_prune(state["binding"]["host"], raw).stdout)
    if checked != {"applied": False, "files": len(manifest["files"]),
                   "bytes": sum(f["size"] for f in manifest["files"])}:
        raise ValueError("unexpected read-only pruning validation result")
    result = copy.deepcopy(state)
    result.setdefault("prune_preauth_recoveries", []).append(
        {"error": result.pop("error"), "manifest": manifest_name,
         "sha256": hashlib.sha256(raw).hexdigest()})
    result["status"] = "watching"
    return result


def write_budget(records, allocations, milestone_steps=(), margin=512 * 2**30):
    """Reserve all remaining scheduled saves, not just the next checkpoint."""
    sizes = {}
    saved = {}
    for row in records:
        if row["marker"]:
            saved[str(Path(row["path"]).parent)] = max(
                saved.get(str(Path(row["path"]).parent), 0), row["step"]
            )
            if row["files"]:
                sizes[row["arm"]] = max(sizes.get(row["arm"], 0), sum(f["size"] for f in row["files"]))
    budget = margin
    for run in allocations:
        start = max(run["from_step"], saved.get(run["output"], 0))
        steps = {s for s in range(start + 1, run["to_step"] + 1) if s % 100 == 0}
        steps.update(s for s in milestone_steps if start < s <= run["to_step"])
        if start < run["to_step"]:
            steps.add(run["to_step"])
        if steps and run["arm"] not in sizes:
            raise ValueError("no measured checkpoint size for storage admission")
        budget += int(len(steps) * sizes.get(run["arm"], 0) * 1.10)
    return budget


def verify_retained(host, records, kept_paths, cache):
    from inspect_recovery import remote_files, validate_storage

    cache.mkdir(parents=True, exist_ok=True)
    for row in records:
        if row["path"] not in kept_paths or not row["marker"] or not row["files"]:
            continue
        native = row["path"] + f"/step_{row['step']}"
        files = remote_files(host, native)
        digest = files[".metadata"]["sha256"]
        metadata = cache / digest
        if not metadata.exists():
            subprocess.run(["rsync", "-a", f"{host}:{native}/.metadata", str(metadata)], check=True, timeout=120)
        if hashlib.sha256(metadata.read_bytes()).hexdigest() != digest:
            raise ValueError("retained metadata copy checksum mismatch")
        validate_storage(pickle.loads(metadata.read_bytes()), files)
        world = row["marker"]["contract"]["experiment"]["world_size"]
        if any(files.get(f"rng_state_rank_{r:05d}.pt", {}).get("size", 0) <= 0 for r in range(world)):
            raise ValueError("retained checkpoint missing RNG rank")


def maintain(state, state_dir, total, *, next_plan=None, retain_all=False, prune_on_pressure=False):
    """Caller must hold campaign lock and have explicit pruning authorization."""
    from plan_checkpoint_retention import build_plan, remote_inventory
    from readonly_probe import run_readonly

    from mor_mlite.pretraining.config import milestones

    if prune_on_pressure and (retain_all or state["binding"].get("prune_approved") != "True"):
        raise ValueError("pressure pruning requires explicit authorization and deletion enabled")
    if prune_on_pressure:
        # A fresh read-only budget must demonstrate pressure before any deletion.
        checked = maintain(state, state_dir, total, next_plan=next_plan, retain_all=True)
        if checked["admit"]:
            return checked
    host = state["binding"]["host"]
    runs = {r["output"]: r for r in state["runs"] + list(state["active"].values())}
    records = remote_inventory(host, list(runs.values()))
    plan = build_plan(state, records, total)
    removed = 0
    if plan["candidates"] and not retain_all:
        verify_retained(host, records, plan["kept_paths"], state_dir / "retention-metadata")
        manifest = {"allowed_run_roots": sorted(runs),
                    "files": [f for r in plan["candidates"] for f in r["files"]],
                    "retention_plan": plan}
        raw = (json.dumps(manifest, indent=2) + "\n").encode()
        prefix = state_dir / f"retention-{time.time_ns()}"
        with prefix.with_suffix(".json").open("xb") as stream:
            stream.write(raw)
        run_prune(host, raw)
        with prefix.with_suffix(".log").open("xb") as stream:
            applied = run_prune(host, raw, apply=True, stream=stream)
        final = json.loads(applied.stdout.splitlines()[-1])
        if not final["applied"] or final["files"] != len(manifest["files"]):
            raise ValueError("incomplete checkpoint pruning result; inspect log")
        removed = final["bytes"]
    root = state["binding"]["remote_root"] + "/runtime/mor-pretraining"
    code = "import os,sys; s=os.statvfs(sys.argv[1]); print(s.f_bavail*s.f_frsize)"
    free = int(run_readonly(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host,
         "python3 -c " + shlex.quote(code) + " " + shlex.quote(root)],
    ).stdout.strip())
    allocations = list(state["active"].values()) + ([next_plan] if next_plan else [])
    required = write_budget(records, allocations, milestones(total, 8_388_608))
    return {"removed_bytes": removed, "free_bytes": free, "required_bytes": required,
            "admit": free >= required}
