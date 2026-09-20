"""Render or explicitly submit one source-bound sequential acceptance probe suite."""

import argparse
import json
import subprocess
import uuid
from pathlib import Path


def attention_startup_environment():
    from mor_mlite.pretraining.runtime import ATTENTION_BACKEND

    if ATTENTION_BACKEND != "fused":
        raise ValueError("probe startup environment must match the experiment attention backend")
    # Pinned native runtime otherwise inserts these keys after NCCL creates
    # its heartbeat thread. In this image, concurrent getenv/setenv can race
    # with a libc environ reallocation. Supply the same values before exec.
    return {"NVTE_FLASH_ATTN": "0", "NVTE_FUSED_ATTN": "1", "NVTE_UNFUSED_ATTN": "0"}


def suite_command(args, suite, topology):
    if len(set(args.arms)) != len(args.arms):
        raise ValueError("duplicate arms")
    expected_world = 64 if args.kind in ("initialization", "indexed") else 32
    if args.world_size != expected_world:
        raise ValueError(f"{args.kind} acceptance requires {expected_world} ranks")
    if args.kind == "shared" and args.arms != ["B"]:
        raise ValueError("shared-gradient probe requires B only")
    if args.kind != "indexed" and args.mbs != 1:
        raise ValueError("selected MBS applies only to indexed recovery")
    if args.kind != "smoke" and not args.fixture:
        raise ValueError("non-smoke probes require an explicit fixture")
    if args.kind in ("initialization", "indexed") and not args.data:
        raise ValueError("full-width probes require final indexed data")
    command = [args.python, suite, "--kind", args.kind, "--arms", *args.arms]
    command += ["--output", args.output]
    if args.kind == "smoke":
        command += ["--topology", topology]
    else:
        command += ["--fixture", args.fixture]
    if args.kind in ("initialization", "indexed"):
        command += ["--data", args.data]
    if args.kind == "indexed":
        command += ["--full-width", "--mbs", str(args.mbs)]
    if getattr(args, "debugger", None):
        command += ["--debugger", args.debugger]
    return command


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--kind",
        choices=["smoke", "auxiliary", "shared", "initialization", "indexed"],
        required=True,
    )
    parser.add_argument("--arms", nargs="+", choices=list("ABCD"), required=True)
    parser.add_argument("--world-size", type=int, choices=[32, 64], required=True)
    parser.add_argument("--mbs", type=int, choices=[1, 2, 4, 8], default=1)
    parser.add_argument("--fixture")
    parser.add_argument("--data")
    parser.add_argument("--output", required=True)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--container-sqsh", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--time-limit", default="0:30:00")
    parser.add_argument("--debugger", help="Optional gdb/cuda-gdb executable for native crash diagnosis")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    from mcore_devtoolkit.cluster_run.config import load_cluster_config
    from mcore_devtoolkit.cluster_run.paths import get_project_root
    from mcore_devtoolkit.cluster_run.registry import default_registry
    from mcore_devtoolkit.cluster_run.remote import launch_remote_slurm
    from mcore_devtoolkit.cluster_run.render import render_slurm_preview
    from mcore_devtoolkit.cluster_run.spec import ClusterRunSpec
    from mcore_devtoolkit.job_launch.remote import resolve_login_info, ssh_run

    from mor_mlite.pretraining.snapshot import freeze

    root = Path(get_project_root()).resolve()
    package = Path(__file__).resolve().parents[2]
    native = args.native_root.resolve()
    native_rel = native.relative_to(root)
    if (
        subprocess.check_output(["git", "-C", str(native), "rev-parse", "HEAD"], text=True).strip()
        != "5c8315f12a64a7279eec58896af9e74ee3351b74"
    ):
        raise ValueError("wrong native source commit")
    if subprocess.check_output(
        ["git", "-C", str(native), "status", "--porcelain"], text=True
    ).strip():
        raise ValueError("native source snapshot is dirty")
    snapshot = freeze(package, root / "runtime/mor-pretraining/sources", write=args.submit)
    source_rel = snapshot.relative_to(root)
    scripts_rel = Path(__file__).resolve().parent.relative_to(root)
    topology_rel = Path("runtime/mor-pretraining") / f"probe-topology-{uuid.uuid4().hex}.txt"

    def remote(path):
        return "${PROJECT_ROOT}/" + str(path)

    spec = ClusterRunSpec(
        backend="slurm",
        cluster="oci-hsg",
        nnodes=args.world_size // 4,
        segment=4,
        ntasks_per_node=4,
        gpus_per_node=4,
        time_limit=args.time_limit,
        job_name=f"mor-gate-{args.kind}",
        container_sqsh=args.container_sqsh,
        command=suite_command(args, remote(scripts_rel / "probe_suite.py"), remote(topology_rel)),
        env={
            **attention_startup_environment(),
            "PYTHONPATH": ":".join(
                map(
                    remote,
                    (
                        source_rel / "src",
                        native_rel / "experimental/lite",
                        native_rel,
                    ),
                )
            ),
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONFAULTHANDLER": "1",
            "TOKENIZERS_PARALLELISM": "false",
        },
        sync_paths=[
            str(source_rel),
            str(scripts_rel),
            str(native_rel / "megatron"),
            str(native_rel / "experimental/lite/megatron"),
        ],
    )
    if not args.submit:
        print(render_slurm_preview(spec))
        return
    if args.kind == "smoke":
        login = resolve_login_info(load_cluster_config(spec.cluster))
        inventory = ssh_run(
            f"{login.user}@{login.hostname}",
            "scontrol show topology",
            ssh_options=login.ssh_options,
            timeout=60,
        )
        if inventory.returncode or "BlockName=" not in inventory.stdout:
            raise ValueError("could not obtain real Slurm NVL topology inventory")
        (root / topology_rel).write_text(inventory.stdout)
        spec.sync_paths.append(str(topology_rel))
    record = launch_remote_slurm(spec, default_registry())
    print(json.dumps(record.to_public_dict(), indent=2))


if __name__ == "__main__":
    main()
