"""Read-only failed-run checkpoint preflight; never marks a run completed.

Only load metadata/RNG from our own trusted campaign outputs. Storage bounds
prove absence of truncation, not bitwise integrity of every tensor payload.
Actual native model/optimizer/RNG restore remains mandatory on continuation.
"""

import argparse
import hashlib
import json
import math
import pickle
import shlex
import zipfile
from pathlib import Path


def prefix_rows(path, first, last):
    """A quota failure may leave an incomplete trailing line, never a prefix hole."""
    result = []
    for line in path.read_text().splitlines():
        if len(result) == last - first + 1:
            break
        row = json.loads(line)
        if row["step"] != first + len(result):
            raise ValueError(f"discontinuous recovery prefix: {path}")
        result.append(row)
    if len(result) != last - first + 1:
        raise ValueError(f"missing recovery prefix: {path}")
    return result


def validate_storage(metadata, files):
    expected = {}
    for location in metadata.storage_data.values():
        name = location.relative_path
        if Path(name).name != name or location.offset < 0 or location.length <= 0:
            raise ValueError("unsafe or invalid checkpoint storage range")
        expected[name] = max(expected.get(name, 0), location.offset + location.length)
    if not expected:
        raise ValueError("empty checkpoint storage metadata")
    for name, end in expected.items():
        if files.get(name, {}).get("size", -1) < end:
            raise ValueError(f"missing/truncated checkpoint shard: {name}")
    keys = metadata.state_dict_metadata
    for prefix in ("optimizer.state.param.", "optimizer.state.exp_avg.", "optimizer.state.exp_avg_sq."):
        if not any(key.startswith(prefix) for key in keys):
            raise ValueError(f"missing optimizer tensors: {prefix}")
    return expected


def inspect(root, plan, receipt, step, files):
    import torch

    from mor_mlite.pretraining.config import Experiment, milestones

    if not plan["from_step"] < step < plan["to_step"]:
        raise ValueError("recovery must be an interior saved checkpoint")
    config = Experiment(**receipt["experiment"])
    contract = json.loads((root / "manifest.json").read_text())
    if (
        contract["mode"] != "train"
        or contract["experiment"] != config.to_dict()
        or contract["total_tokens"] != contract["total_steps"] * config.tokens_per_step
        or not contract["topology"]["passed"]
        or config.arm != plan["arm"]
        or any(contract[k] != receipt[k] for k in (
            "source_sha256", "data_sha256", "environment_sha256", "model_config_sha256"
        ))
    ):
        raise ValueError("changed accepted recovery contract")
    scientific = {k: v for k, v in contract.items() if k != "topology"}
    checkpoint = root / f"checkpoint-{step:06d}"
    marker = json.loads((checkpoint / "pretraining-state.json").read_text())
    if marker != {
        "contract": scientific, "step": step, "cursor": step * config.tokens_per_step,
        "scheduler": "pure-token-function-v1",
    }:
        raise ValueError("incomplete or incompatible recovery checkpoint")
    updates = prefix_rows(root / "loss.jsonl", plan["from_step"] + 1, step)
    for row in updates:
        if (
            row["tokens"] != row["step"] * config.tokens_per_step
            or not math.isclose(row["lr"], config.learning_rate(row["tokens"], contract["total_tokens"]), rel_tol=1e-12)
            or any(not math.isfinite(row[k]) for k in ("lm_loss", "depth_aux", "grad_norm", "seconds"))
            or row["seconds"] <= 0
            or len(bytes.fromhex(row["input_sha256"])) != 32
        ):
            raise ValueError("invalid recovery update")
    for rank in range(config.world_size):
        for row in prefix_rows(root / f"input-rank-{rank}.jsonl", plan["from_step"] + 1, step):
            if len(bytes.fromhex(row["sha256"])) != 32:
                raise ValueError("invalid rank input digest")
    required_eval = {s for s in milestones(contract["total_steps"], config.tokens_per_step)
                     if plan["from_step"] < s <= step}
    if plan["from_step"] == 0:
        required_eval.add(0)
    evaluations = []
    evaluation_path = root / "evaluation.jsonl"
    if evaluation_path.exists():
        evaluations = [json.loads(line) for line in evaluation_path.read_text().splitlines()]
    for s in required_eval:
        found = [r for r in evaluations if r["step"] == s]
        if len(found) != 1 or not math.isfinite(found[0]["nll"]) or found[0]["targets"] <= 0:
            raise ValueError("missing recovery milestone evaluation")
    native = checkpoint / f"step_{step}"
    metadata = pickle.loads((native / ".metadata").read_bytes())
    shards = validate_storage(metadata, files)
    for name in [".metadata", "metadata.json"] + [f"rng_state_rank_{r:05d}.pt" for r in range(config.world_size)]:
        path = native / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != files[name]["sha256"]:
            raise ValueError(f"local/remote metadata mismatch: {name}")
        if name.endswith(".pt"):
            with zipfile.ZipFile(path) as archive:
                if archive.testzip() is not None:
                    raise ValueError(f"corrupt RNG archive: {name}")
            rng = torch.load(path, map_location="cpu", weights_only=False)
            if not {"random_rng_state", "np_rng_state", "torch_rng_state", "cuda_rng_state"} <= rng.keys():
                raise ValueError("incomplete RNG state")
    return {
        "arm": plan["arm"], "scheduler_id": plan["scheduler_id"], "from_step": plan["from_step"],
        "checkpoint_step": step, "prefix_updates": len(updates), "native_shards": len(shards),
        "native_bytes": sum(files[n]["size"] for n in shards), "rng_ranks": config.world_size,
        "preflight_passed": True, "tensor_payload_checksums_verified": False,
        "native_restore_verified": False,
        "input_hashes": {str(r["step"]): r["input_sha256"] for r in updates},
    }


def remote_files_command(host, native):
    # Small files are hashed, large tensor shards are stat-only; no login-node
    # tensor loading or multi-hundred-GB checkpoint copy.
    code = """import hashlib,json,sys,zipfile
from pathlib import Path
p=Path(sys.argv[1]); result={}
for f in p.iterdir():
 if f.is_symlink() or not f.is_file(): raise ValueError('unexpected checkpoint entry')
 size=f.stat().st_size
 result[f.name]={'size':size}
 if not f.name.endswith('.distcp'):
  if size>64000000: raise ValueError('oversized checkpoint metadata')
  result[f.name]['sha256']=hashlib.sha256(f.read_bytes()).hexdigest()
  if f.name.startswith('rng_state_rank_'):
   with zipfile.ZipFile(f) as z:
    if z.testzip() is not None: raise ValueError('corrupt RNG archive')
print(json.dumps(result))
"""
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host,
            "python3 -c " + shlex.quote(code) + " " + shlex.quote(native)]


def remote_files(host, native):
    from readonly_probe import run_readonly

    result = run_readonly(remote_files_command(host, native))
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--arm", choices=list("ABCD"), required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    state = json.loads(args.state.read_text())
    plan = state["active"][args.arm]
    native = f"{plan['output']}/checkpoint-{args.step:06d}/step_{args.step}"
    files = remote_files(state["binding"]["host"], native)
    report = inspect(
        args.state.parent / "evidence" / plan["scheduler_id"], plan,
        json.loads(args.receipt.read_text()), args.step, files,
    )
    report["remote_files"] = files
    if args.report:
        with args.report.open("x") as stream:
            stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in {"input_hashes", "remote_files"}}, indent=2))


if __name__ == "__main__":
    main()
