# lawrence mcafee

# ~~~~~~~~ import ~~~~~~~~
import glob
import os

# from lutil import pax

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def softlink_data():

    src_dir = "/lustre/fsw/portfolios/adlr/projects/adlr_nlp_arch/adlr_nlp_sharing/nvllm-8t/data/tokens-shuffle"
    dst_dir = "/lustre/fs6/portfolios/adlr/users/lmcafee/retro/megatrons/core-converter/scripts/oci-8t/ppl/data"

    for ext in ("bin", "idx"):
        src_paths = glob.glob(src_dir + f"/**/*.{ext}", recursive=True)
        # pax("src_paths")
        for src_path in src_paths:
            dst_path = src_path.replace(src_dir, dst_dir)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            os.symlink(src_path, dst_path)
            # pax("src_path, dst_path")

    # pax("src_dir, dst_dir, src_paths")
    print("all done.")


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":

    softlink_data()
