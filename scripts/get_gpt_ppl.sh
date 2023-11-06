#!/bin/bash

#SBATCH -p luna,interactive
#SBATCH --nodes=1
#SBATCH -A llmservice_nlp_retro
#SBATCH -t 0:30:00
#SBATCH --exclusive
#SBATCH --job-name=llmservice_nlp_retro-retro:gpt-nextlm-800m-test
#SBATCH --ntasks-per-node=8
#SBATCH --dependency=singleton

######## Arguments. ########

if [ "$#" != 1 ]; then
    echo "expected 1 arg, found ${#}."
    exit 1
fi
USE_CORE=$1
# ADD_RETRIEVER=$2
NPROCS=1 # 8
# NWORKERS=32






# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# customize / begin.
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

ADD_RETRIEVER=0
REPO_DIR="/lustre/fs6/portfolios/adlr/users/lmcafee/retro/megatrons/core-converter"
# CHECKPOINT_DIR="/lustre/fsw/adlr/adlr-nlp/boxinw/next-llm/pretrain-checkpoint"

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# customize / end.
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<







######## setup. ########

set -u

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_SOCKET_IFNAME=^vlan,lo
unset NCCL_DEBUG

# DIR=$(readlink -f `pwd`)
# DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
# LOG_DIR=$DIR/logs
# mkdir -p $LOG_DIR

# NAME="gpt3-800m-pretraining-gpt-fitting"

# CHECKPOINT_DIR="/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/${NAME}"


# if [ -f "$CHECKPOINT_DIR/latest_checkpointed_iteration.txt" ]
# then
#   LOAD_DIR=$CHECKPOINT_DIR
#   LOAD_OPTION=""
# else
#   LOAD_DIR="/lustre/fsw/adlr/adlr-nlp/adlr-nlp-sharing/nvllm-1.1t/checkpoints/gpt3-843m-multi-1.1t-gtc-llr"
#   LOAD_OPTION="--no-load-optim --finetune"
# fi

# echo $LOAD_DIR

ARGS=""

if [ "$USE_CORE" = "0" ]; then
    LOAD_DIR="/lustre/fs3/portfolios/adlr/projects/adlr_nlp_arch/adlr_nlp_sharing/nvllm-1.1t/checkpoints/gpt3-843m-multi-1.1t-gtc-llr"
else
    ARGS+=" --use-mcore-models"
    LOAD_DIR="/lustre/fs6/portfolios/adlr/users/lmcafee/retro/megatrons/core-converter/scripts/checkpoints/843m"
fi

# >>>
# LOAD_DIR="aaa"
# <<<
LOAD_OPTION="--no-load-optim --finetune --exit-interval 10"
LOAD_OPTION+=" --exit-on-missing-checkpoint"
LOAD_OPTION+=" --skip-train"

######## checkpoint. ########

# TENSORBOARD_DIR="$CHECKPOINT_DIR/tensorboard"
# mkdir -p ${TENSORBOARD_DIR}

######## data blend. ########

# . /lustre/fsw/adlr/adlr-nlp/boxinw/megatron-lm-pretrain/scripts/lawrence_blend_oci.sh
. /lustre/fs6/portfolios/adlr/users/lmcafee/retro/misc/next-llm-tokenizer/lawrence_blend_oci.sh

######## args. ########


#     --save-interval 2000 \
#     --save ${CHECKPOINT_DIR} \
#     --tensorboard-dir ${TENSORBOARD_DIR} \
#     --log-validation-ppl-to-tensorboard \
TP=1
ARGS+=" \
    --sequence-parallel \
    --recompute-activations \
    --use-flash-attn \
    --apply-layernorm-1p \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --no-position-embedding \
    --use-rotary-position-embeddings \
    --rotary-percent 0.5 \
    --swiglu \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --exit-duration-in-mins 220 \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size 1 \
    --load ${LOAD_DIR} ${LOAD_OPTION} \
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --micro-batch-size 1 \
    --global-batch-size 128 \
    --train-samples 25000000 \
    --lr-decay-samples 23750000 \
    --lr-warmup-samples 16667 \
    --lr 2.5e-5 \
    --min-lr 2.5e-6 \
    --lr-decay-style cosine \
    --log-interval 1 \
    --eval-iters 32 \
    --eval-interval 1260 \
    --tokenizer-type GPTSentencePieceTokenizer \
    --tokenizer-model /lustre/fsw/portfolios/adlr/projects/adlr_nlp_arch/adlr_nlp_sharing/nvllm-1.1t/utils/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model \
    --data-path ${DATA_BLEND} \
    --split 98,2,0 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.007 \
    --log-params-norm \
    --log-num-zeros-in-grad \
    --bf16 \
"

######## retro. ########

# if [ "$ADD_RETRIEVER" = "0" ]; then
SCRIPT=pretrain_gpt.py
# else
#     RETRO_WORKDIR=/lustre/fsw/adlr/adlr-nlp/boxinw/next-llm
#     ARGS="${ARGS} \
#     --retro-workdir ${RETRO_WORKDIR} \
#     --retro-add-retriever \
#     --num-workers ${NWORKERS} \
#     "
#     SCRIPT=pretrain_retro.py
# fi

######## Command. ########

# CMD=" \
#     cd ${REPO_DIR} && \
#     ${REPO_DIR}/bind.sh --cpu=${REPO_DIR}/dgxa100_ccx.sh --mem=${REPO_DIR}/dgxa100_ccx.sh python -u ${SCRIPT} ${ARGS} \
# "
# echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
# echo $CMD
# echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

# IMAGE="gitlab-master.nvidia.com/adlr/megatron-lm/lmcafee/retro-process-22.12"
# IMAGE="/lustre/fsw/adlr/adlr-nlp/boxinw/images/retrov2.sqsh"
# MOUNTS="/lustre/fsw/adlr:/lustre/fsw/adlr"
# srun -l \
#      --container-image $IMAGE \
#      --container-mounts $MOUNTS \
#      --output=$LOG_DIR/"%j_${NAME}_r${ADD_RETRIEVER}.log" \
#      sh -c "${CMD}"

# NPROCS=8
CMD="\
    cd ${REPO_DIR} && \
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
