import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "script,template",
    [("submit_matrix.sh", "tiny_matrix.sbatch"), ("submit_qwen30b.sh", "qwen30b_smoke.sbatch")],
)
def test_submit_wrapper_uses_its_own_checkout(tmp_path, script, template):
    source = Path(__file__).resolve().parents[1]
    checkout = tmp_path / "isolated"
    scripts = checkout / "scripts" / "eos"
    scripts.mkdir(parents=True)
    for name in (script, "common.sh"):
        shutil.copyfile(source / "scripts" / "eos" / name, scripts / name)
    completed = subprocess.run(
        [
            "bash",
            "-c",
            'sbatch() { printf "%s\\n" "$@"; }; export -f sbatch; bash "$1"',
            "test",
            str(scripts / script),
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=True,
    )
    assert f"--chdir={checkout}" in completed.stdout
    assert f"--output={checkout}/logs/%x-%j.out" in completed.stdout
    assert str(checkout / "slurm" / template) in completed.stdout


def test_slurm_templates_do_not_fall_back_to_historical_source():
    root = Path(__file__).resolve().parents[1]
    for path in (root / "slurm").glob("*.sbatch"):
        subprocess.run(["bash", "-n", str(path)], check=True)
        assert "/lustre/fsw/coreai_devtech_all/hongbinl/mor_mlite" not in path.read_text()
