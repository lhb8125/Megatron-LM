# lawrence mcafee

# ~~~~~~~~ import ~~~~~~~~
import os
import subprocess

# from lutil import pax

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# account = "adlr_nlp_llmnext"
account = "llmservice_dev_mcore"
orig_root="/lustre/fs6/portfolios/adlr/users/lmcafee/retro/megatrons/core-converter/scripts/checkpoints/8t/orig"
conv_root="/lustre/fs6/portfolios/adlr/users/lmcafee/retro/megatrons/core-converter/scripts/checkpoints/8t"

groups = {
    "orig" : [
        ("15b", 8, 1, os.path.join(orig_root, "15b")),
        # ("340b", 8, 12, os.path.join(orig_root, "340b")),
    ],
    "conv" : [
        # *[ ("15b", 8, pp, os.path.join(conv_root, f"15b-pp{pp}"))
        #    for pp in (1, 2, 4) ],
        # *[ ("340b", 8, pp, os.path.join(conv_root, f"340b-pp{pp}"))
        #    for pp in (2, 4) ],
    ],
}

for group_key, tests in groups.items():
    for model_key, tp, pp, checkpoint_dir in tests:
        print(f"~~~~ launch g {group_key}, m {model_key}, pp {pp} ~~~~")
        subprocess.run([
            "sbatch",
            f"--export=GROUP_KEY={group_key},MODEL_KEY={model_key},PP={pp},CHECKPOINT_DIR={checkpoint_dir}",
            "-A",
            account,
            f"--job-name={account}-lmcafee:lmcafee_g{group_key}_m{model_key}-pp{pp}",
            f"--nodes={pp}",
            "./single.sh",
        ])

# eof
