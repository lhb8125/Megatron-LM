#!/bin/bash

#SBATCH -p large_runs_block1_APPROVL_RQUIRD
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH -A llmservice_nlp_fm
#SBATCH -t 8:00:00
#SBATCH --exclusive
#SBATCH --job-name=adlr-nlp:retro-mcore
#SBATCH --dependency=singleton

# ... SBATCH -p batch_block1,batch_block2,batch_block3,batch_block4

######## setup. ########

set -u

# export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_SOCKET_IFNAME=^vlan,lo
# unset NCCL_DEBUG
export NCCL_DEBUG=INFO

REPO_DIR="/lustre/fs6/portfolios/adlr/users/lmcafee/retro/megatrons/core-converter"
LOG_DIR="${REPO_DIR}/scripts/oci/logs"
mkdir -p ${LOG_DIR}

######## Command. ########

CMD="cd ${REPO_DIR} && bash scripts/oci/convert_340b.sh"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo $CMD
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

IMAGE="gitlab-master.nvidia.com/adlr/megatron-lm/pytorch:23.09-py3-pretrain-draco_cw_ub_tot-te-apex"

MOUNTS="/home/lmcafee:/home/lmcafee"
MOUNTS+=",/lustre/fsw/portfolios/adlr/users/lmcafee:/lustre/fsw/portfolios/adlr/users/lmcafee"
MOUNTS+=",/lustre/fs6/portfolios/adlr/users/lmcafee:/lustre/fs6/portfolios/adlr/users/lmcafee"
MOUNTS+=",/lustre/fs3/portfolios/adlr/projects/adlr_nlp_arch/adlr_nlp_sharing/nvllm-8t/checkpoints:/lustre/fs3/portfolios/adlr/projects/adlr_nlp_arch/adlr_nlp_sharing/nvllm-8t/checkpoints"

srun -l --export=ALL,PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
     --container-image $IMAGE \
     --container-mounts $MOUNTS \
     --output=$LOG_DIR/"%j_m${MODEL_SIZE}_pp${PP}.log" \
     sh -c "${CMD}"

# eof.
