"""Recipe -> cluster-run bridge. Default is sbatch rendering, not submission."""

from __future__ import annotations

import argparse
import json
import subprocess
import uuid
from pathlib import Path

from mor_mlite.pretraining.config import Experiment


def launch_spec(recipe, *, arm, world_size, mbs, command, env, container_sqsh):
    from mcore_devtoolkit.cluster_run.spec import ClusterRunSpec

    config = Experiment(arm, world_size=world_size, micro_batch_size=mbs)
    schedule = recipe["SLURM"]
    scientific = recipe["EXPERIMENT"]
    for field in (
        "global_batch_size",
        "seq_length",
        "seed",
        "ep",
        "lr",
        "min_lr",
        "warmup_fraction",
        "weight_decay",
        "clip_grad",
        "moe_aux",
        "depth_aux",
    ):
        if scientific[field] != getattr(config, field):
            raise ValueError(f"recipe differs from the approved experiment: {field}")
    if any(scientific[field] != 1 for field in ("tp", "cp", "pp", "etp")):
        raise ValueError("recipe parallelism differs from the approved experiment")
    if (
        scientific["betas"] != [config.adam_beta1, config.adam_beta2]
        or scientific["eps"] != config.adam_eps
        or scientific["precision"] != "bf16"
        or scientific["optimizer"] != "distributed_adamw"
    ):
        raise ValueError("recipe optimizer/precision differs from the approved experiment")
    if (
        schedule["segment"] != 4
        or schedule["gpus_per_node"] != 4
        or schedule["ntasks_per_node"] != 4
    ):
        raise ValueError("EP16 requires segment4 with four ranks/GPUs per node")
    return ClusterRunSpec(
        backend="slurm",
        command=command,
        cluster=schedule["cluster"],
        nnodes=world_size // 4,
        ntasks_per_node=4,
        gpus_per_node=4,
        segment=schedule["segment"],
        time_limit=schedule["time_limit"],
        job_name=f"mor-{arm.lower()}-mbs{mbs}",
        container_sqsh=container_sqsh,
        env=env,
    )


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    package = Path(__file__).resolve().parents[3]
    parser.add_argument("--recipe", type=Path, default=package / "configs/pretraining/oci-hsg.yaml")
    parser.add_argument("--arm", choices=list("ABCD"), required=True)
    parser.add_argument("--mbs", type=int, choices=[1, 2, 4, 8], required=True)
    parser.add_argument("--world-size", type=int, choices=[32, 64], default=64)
    parser.add_argument("--mode", choices=["tune", "train"], default="tune")
    parser.add_argument(
        "--smoke", action="store_true", help="Tiny-width real-model verification only"
    )
    for key in ("data", "hf-config", "environment", "output", "container-sqsh"):
        parser.add_argument("--" + key, required=key in ("output", "container-sqsh"))
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument(
        "--python", default="python", help="Interpreter inside the selected container"
    )
    parser.add_argument("--acceptance")
    parser.add_argument("--resume")
    parser.add_argument("--stop-tokens", type=int)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if not args.smoke and any(
        getattr(args, k) is None for k in ("data", "hf_config", "environment")
    ):
        parser.error("training/tuning needs --data, --hf-config, and --environment")
    if args.mode == "train" and not args.acceptance:
        parser.error("formal training needs --acceptance")
    import yaml
    from mcore_devtoolkit.cluster_run.config import load_cluster_config
    from mcore_devtoolkit.cluster_run.paths import get_project_root
    from mcore_devtoolkit.cluster_run.registry import default_registry
    from mcore_devtoolkit.cluster_run.remote import launch_remote_slurm
    from mcore_devtoolkit.cluster_run.render import render_slurm_preview
    from mcore_devtoolkit.job_launch.remote import resolve_login_info, ssh_run

    root = Path(get_project_root()).resolve()
    native = args.native_root.resolve()
    native_rel = native.relative_to(root)
    expected_sha = "5c8315f12a64a7279eec58896af9e74ee3351b74"
    if (
        subprocess.check_output(["git", "-C", str(native), "rev-parse", "HEAD"], text=True).strip()
        != expected_sha
    ):
        raise ValueError("wrong native source commit")
    if subprocess.check_output(
        ["git", "-C", str(native), "status", "--porcelain"], text=True
    ).strip():
        raise ValueError("native source snapshot is dirty")
    from mor_mlite.pretraining.snapshot import freeze

    snapshot = freeze(package, root / "runtime/mor-pretraining/sources", write=args.submit)
    package_rel = snapshot.relative_to(root)
    topology_rel = Path("runtime/mor-pretraining") / f"topology-{uuid.uuid4().hex}.txt"
    topology_remote = "${PROJECT_ROOT}/" + str(topology_rel)
    command = [
        "python",
        "-m",
        "mor_mlite.pretraining.train",
        "--arm",
        args.arm,
        "--mode",
        args.mode,
        "--world-size",
        str(args.world_size),
        "--mbs",
        str(args.mbs),
        "--native-root",
        "${PROJECT_ROOT}/" + str(native_rel),
        "--topology",
        topology_remote,
    ]
    for key in (
        "data",
        "hf_config",
        "environment",
        "output",
        "acceptance",
        "resume",
        "stop_tokens",
    ):
        value = getattr(args, key)
        if value is not None:
            command.extend(["--" + key.replace("_", "-"), str(value)])
    env = {
        "PYTHONPATH": ":".join(
            "${PROJECT_ROOT}/" + str(p)
            for p in (package_rel / "src", native_rel / "experimental/lite", native_rel)
        ),
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    recipe = yaml.safe_load(args.recipe.read_text())
    spec = launch_spec(
        recipe,
        arm=args.arm,
        world_size=args.world_size,
        mbs=args.mbs,
        command=command,
        env=env,
        container_sqsh=args.container_sqsh,
    )
    if args.smoke:
        spec.command = [
            "python",
            "-m",
            "mor_mlite.pretraining.smoke",
            "--arm",
            args.arm,
            "--topology",
            topology_remote,
            "--output",
            args.output,
        ]
        spec.time_limit = "0:30:00"
        spec.job_name += "-smoke"
        if args.resume:
            spec.command.extend(["--resume", args.resume])
    spec.command[0] = args.python
    if not args.submit:
        print(render_slurm_preview(spec))
        return
    login = resolve_login_info(load_cluster_config(spec.cluster))
    host = f"{login.user}@{login.hostname}"
    inventory = ssh_run(host, "scontrol show topology", ssh_options=login.ssh_options, timeout=60)
    if inventory.returncode or "BlockName=" not in inventory.stdout:
        raise ValueError("could not obtain real Slurm NVL topology inventory")
    (root / topology_rel).parent.mkdir(parents=True, exist_ok=True)
    (root / topology_rel).write_text(inventory.stdout)
    spec.sync_paths = [
        str(package_rel),
        str(native_rel / "megatron"),
        str(native_rel / "experimental/lite/megatron"),
        str(topology_rel),
    ]
    record = launch_remote_slurm(spec, default_registry())
    print(json.dumps(record.to_public_dict(), indent=2))


if __name__ == "__main__":
    main()
