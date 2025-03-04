#!/bin/bash

#SBATCH --nodes=8
#SBATCH --account coreai_dlalgo_llm
#SBATCH --partition batch
#SBATCH --ntasks-per-node=8
#SBATCH --time 0:05:00
#SBATCH --job-name=coreai_devtech_all:moe:1f1b_overlap
#SBATCH --dependency=singleton
#SBATCH --exclusive

# Prepare SLURM job

WORKSPACE=/lustre/fsw/coreai_devtech_all/pingtianl/a2a_olp
export $WORKSPACE
srun \
    --mpi=pmix -l \
    --ntasks-per-node=8 \
    --container-image=${WORKSPACE}/pytorch_2501.sqsh \
    --container-mounts=${WORKSPACE}:${WORKSPACE} \
    bash ${WORKSPACE}/github/tests/unit_tests/a2a_overlap/run.sh
EOF
set -e