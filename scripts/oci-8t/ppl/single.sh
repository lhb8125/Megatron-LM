#!/bin/bash

#SBATCH -p batch_block1,batch_block2,batch_block3,batch_block4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH -A llmservice_nlp_fm
#SBATCH -t 4:00:00
#SBATCH --exclusive
#SBATCH --job-name=adlr-nlp:retro-mcore
#SBATCH --dependency=singleton

######## command. ########

. ./build_cmd.sh

# CMD=" \
#     cd ${REPO_DIR} && \
#     ${REPO_DIR}/bind.sh --cpu=${REPO_DIR}/dgxa100_ccx.sh --mem=${REPO_DIR}/dgxa100_ccx.sh python -u ${SCRIPT} ${ARGS} \
# CMD="cd ${REPO_DIR} && python -u ${SCRIPT} ${ARGS}"
CMD="cd /lustre/fs6/portfolios/adlr/users/lmcafee/retro/megatrons/training-nov2023 && python -u ${SCRIPT} ${ARGS}"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo $CMD
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

######## srun. ########

# IMAGE="gitlab-master.nvidia.com/adlr/megatron-lm/training-nov2023-inference"
# IMAGE="gitlab-master.nvidia.com/adlr/megatron-lm/training-nov2023"
IMAGE="gitlab-master.nvidia.com/adlr/megatron-lm/pytorch:23.09-py3-pretrain-draco_cw_ub_tot-te-apex"

MOUNTS="/home/lmcafee:/home/lmcafee"
# MOUNTS+=",/lustre/fsw/portfolios/adlr/users/lmcafee:/lustre/fsw/portfolios/adlr/users/lmcafee"
MOUNTS+=",/lustre/fsw/portfolios/adlr:/lustre/fsw/portfolios/adlr"
MOUNTS+=",/lustre/fs6/portfolios/adlr/users/lmcafee:/lustre/fs6/portfolios/adlr/users/lmcafee"
MOUNTS+=",/lustre/fs3/portfolios/adlr/projects/adlr_nlp_arch/adlr_nlp_sharing/nvllm-8t:/lustre/fs3/portfolios/adlr/projects/adlr_nlp_arch/adlr_nlp_sharing/nvllm-8t"

LOG_DIR="${REPO_DIR}/scripts/oci-8t/ppl/logs"
mkdir -p ${LOG_DIR}

srun -l --export=ALL,PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
     --container-image $IMAGE \
     --container-mounts $MOUNTS \
     --output=$LOG_DIR/"%j_${GROUP_KEY}_${MODEL_KEY}_pp${PP}.log" \
     sh -c "${CMD}"

# eof.
