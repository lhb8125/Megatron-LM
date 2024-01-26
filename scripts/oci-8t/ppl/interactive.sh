#!/bin/bash

set -u

GROUP_KEY="orig"
MODEL_KEY="15b"
PP=1
CHECKPOINT_DIR="/lustre/fs6/portfolios/adlr/users/lmcafee/retro/megatrons/core-converter/scripts/checkpoints/8t/orig/${MODEL_KEY}"

######## python command. ########

. ./build_cmd.sh

######## Command. ########

unset NCCL_DEBUG

NPROCS=8
CMD="\
    cd /lustre/fs6/portfolios/adlr/users/lmcafee/retro/megatrons/training-nov2023 && \
    export PYTHONPATH=${REPO_DIR}:/home/lmcafee/src && \
    python -m torch.distributed.run \
    --nproc_per_node ${NPROCS} \
    --nnodes 1 \
    --node_rank ${NODE_RANK} \
    --master_addr ${MASTER_ADDR} \
    --master_port 6000 \
    ${SCRIPT} ${ARGS} \
"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "CMD = '$CMD'."
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~"
eval $CMD

# eof.
