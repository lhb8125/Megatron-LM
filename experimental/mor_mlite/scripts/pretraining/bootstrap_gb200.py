"""Add the pinned missing dependency in an experiment-only container venv."""

import os
import subprocess
import sys
from pathlib import Path


def main():
    project = Path(os.environ["PROJECT_ROOT"])
    env_dir = project / "runtime/mor-pretraining/gb200-env-nvrx060"
    python = env_dir / "bin/python"
    if not python.exists():
        subprocess.run(
            [sys.executable, "-m", "venv", "--system-site-packages", str(env_dir)], check=True
        )
    clean = dict(os.environ)
    clean.pop("PYTHONPATH", None)
    subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "nvidia-resiliency-ext==0.6.0",
            "grpcio==1.78.0",
            "grpcio-tools==1.78.0",
            "protobuf==6.33.6",
            "nvidia-cutlass-dsl[cu13]==4.4.2",
        ],
        env=clean,
        check=True,
    )
    subprocess.run([str(python), "-m", "pip", "check"], env=clean, check=True)
    os.execv(str(python), [str(python), "-m", "mor_mlite.pretraining.probe", *sys.argv[1:]])


if __name__ == "__main__":
    main()
