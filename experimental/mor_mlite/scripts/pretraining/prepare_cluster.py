"""CPU-node dependency bootstrap; credentials never appear in shell argv/logs."""

import os
import platform
import subprocess
import sys
from pathlib import Path


def main():
    project = Path(os.environ["PROJECT_ROOT"])
    sys.path.insert(0, str(project / "src"))
    from mcore_devtoolkit.job_launch.config import load_dotenv

    load_dotenv()
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is not configured; no download attempted")
    # The bootstrap has already loaded toolkit configuration. Do not expose
    # the toolkit's source-tree dist-info to the independent data venv/pip.
    os.environ.pop("PYTHONPATH", None)
    env_dir = project / "runtime" / "mor-pretraining" / ("data-env-" + platform.machine())
    python = env_dir / "bin" / "python"
    if not python.exists():
        subprocess.run([sys.executable, "-m", "venv", str(env_dir)], check=True)
    subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "numpy==2.2.6",
            "transformers==4.51.3",
            "tokenizers==0.21.4",
            "huggingface-hub==0.34.4",
            "pyarrow==20.0.0",
        ],
        check=True,
    )
    package = Path(__file__).resolve().parents[2]
    os.environ["PYTHONPATH"] = str(package / "src")
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.execv(str(python), [str(python), "-m", "mor_mlite.pretraining.prepare_raw", *sys.argv[1:]])


if __name__ == "__main__":
    main()
