"""Sequential fresh-process probes on one SLURM allocation, fail closed."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def rendezvous_port(job_id, sequence, *, ephemeral_start=32768):
    # Successive Gloo/NCCL processes can consume ephemeral listener ports;
    # the next TCPStore must not select an adjacent port in that same pool.
    upper = min(30000, ephemeral_start)
    lower = 7000
    if upper - lower < 1000 or not 0 <= sequence < 16:
        raise ValueError("no safe fixed-port range for the sequential probe suite")
    return lower + (int(job_id) * 16 + sequence) % (upper - lower)


def diagnostic_command(command, debugger=None):
    if debugger is None:
        return command
    # Without --return-child-result, gdb can exit successfully after a crash.
    # Never let debugger success stand in for the probe's exit status.
    return [
        debugger,
        "--batch",
        "--return-child-result",
        "-ex",
        "set pagination off",
        "-ex",
        "set confirm off",
        "-ex",
        # Preserve the normal process address layout policy. GDB's default
        # disables ASLR and can mask address-sensitive startup failures.
        "set disable-randomization off",
        "-ex",
        "show disable-randomization",
        "-ex",
        "run",
        "-ex",
        "thread apply all bt 40",
        "--args",
        *command,
    ]


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--kind",
        choices=["auxiliary", "indexed", "shared", "initialization", "smoke"],
        required=True,
    )
    parser.add_argument("--arms", nargs="+", choices=list("ABCD"), required=True)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--topology", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full-width", action="store_true")
    parser.add_argument("--mbs", type=int, choices=[1, 2, 4, 8], default=1)
    parser.add_argument("--debugger", help="Optional native debugger; a crashed child remains a failure")
    args = parser.parse_args()
    if args.kind == "smoke" and args.topology is None:
        parser.error("smoke probes require --topology")
    if args.kind != "smoke" and args.fixture is None:
        parser.error("non-smoke probes require --fixture")
    if len(set(args.arms)) != len(args.arms):
        parser.error("duplicate arms")
    if args.kind in ("indexed", "initialization") and args.data is None:
        parser.error("indexed/initialization probes require --data")
    if args.kind == "shared" and args.arms != ["B"]:
        parser.error("shared-gradient reference applies to B only")
    if args.kind != "indexed" and (args.full_width or args.mbs != 1):
        parser.error("--full-width/--mbs apply only to indexed resume probes")
    ephemeral_start = int(Path("/proc/sys/net/ipv4/ip_local_port_range").read_text().split()[0])
    sequence = 0
    for arm in args.arms:
        base = args.output / arm.lower()
        script = {
            "auxiliary": "auxiliary_probe.py",
            "indexed": "indexed_resume_probe.py",
            "shared": "shared_gradient_probe.py",
            "initialization": "initialization_probe.py",
            "smoke": None,
        }[args.kind]
        common = (
            [sys.executable, "-m", "mor_mlite.pretraining.smoke"]
            if args.kind == "smoke"
            else [sys.executable, str(Path(__file__).with_name(script))]
        )
        if args.kind != "shared":
            common += ["--arm", arm]
        if args.kind == "smoke":
            common += ["--topology", str(args.topology)]
        elif args.kind in ("auxiliary", "initialization"):
            common += ["--hf-config", str(args.fixture)]
            if args.kind == "initialization":
                common += ["--data", str(args.data)]
        elif args.kind == "indexed":
            common += [
                "--fixture",
                str(args.fixture),
                "--data",
                str(args.data),
                "--mbs",
                str(args.mbs),
            ]
            if args.full_width:
                common += ["--full-width"]
        else:
            common += ["--fixture", str(args.fixture)]
        for resume in (False, True) if args.kind == "indexed" else (False,):
            command = common + [
                "--output",
                str(base.with_name(base.name + "-resume") if resume else base),
            ]
            if resume:
                command += ["--resume", str(base)]
            port = rendezvous_port(
                os.environ["SLURM_JOB_ID"], sequence, ephemeral_start=ephemeral_start
            )
            env = {**os.environ, "MASTER_PORT": str(port)}
            if args.debugger:
                env["DEBUGINFOD_URLS"] = ""
            subprocess.run(diagnostic_command(command, args.debugger), env=env, check=True)
            sequence += 1


if __name__ == "__main__":
    main()
