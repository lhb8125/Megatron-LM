#!/bin/bash

#SBATCH -p luna
#SBATCH --nodes=1
#SBATCH -A llmservice_dev_mcore
#SBATCH -t 4:00:00
#SBATCH --exclusive
#SBATCH --job-name=llmservice_dev_mcore-lmcafee:lmcafee
#SBATCH --ntasks-per-node=1

# ... SBATCH --dependency=singleton

REPO_DIR="/lustre/fsw/adlr/adlr-nlp/lmcafee/data/retro/megatrons/core-converter"
LOG_DIR="${REPO_DIR}/scripts/selene/logs"
mkdir -p ${LOG_DIR}

######## setup. ########

set -u

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_SOCKET_IFNAME=^vlan,lo
unset NCCL_DEBUG

######## Command. ########

CMD="cd ${REPO_DIR} && bash scripts/selene/convert_340b.sh"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo $CMD
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

IMAGE="gitlab-master.nvidia.com/adlr/megatron-lm/pytorch:23.09-py3-pretrain-draco_cw_ub_tot-te-apex"
MOUNTS="/lustre/fsw/adlr:/lustre/fsw/adlr"
srun -l --export=ALL,PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
     --container-image $IMAGE \
     --container-mounts $MOUNTS \
     --output=$LOG_DIR/"%j_m${MODEL_SIZE}_p${PP}.log" \
     sh -c "${CMD}"

# eof.
