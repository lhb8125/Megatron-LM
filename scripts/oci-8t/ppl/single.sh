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

set -u

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_SL=1
export NCCL_IB_TIMEOUT=19
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
export NCCL_P2P_NET_CHUNKSIZE=2097152

REPO_DIR="/lustre/fs6/portfolios/adlr/users/lmcafee/retro/megatrons/core-converter"
SCRIPT="pretrain_gpt.py"

# BLEND_NAME="gpt3-8t-shuffle"
# DATACACHE_DIR="/home/mpatwary/data/data-cache/${BLEND_NAME}"

# TENSORBOARD_DIR="$DIR/tensorboard/${NAME}"
# mkdir -p ${TENSORBOARD_DIR}

# Get the data blend
# . /home/mpatwary/data/adlr-nlp-sharing/nvllm-8t/utils/eng-unimax-e4-whole-cc-alpha1.5-non-cc-heuristic-mul-unimax-e1-code-a1.3-701515.sh
# . /lustre/fsw/portfolios/adlr/projects/adlr_nlp_arch/adlr_nlp_sharing/nvllm-8t/utils/eng-unimax-e4-whole-cc-alpha1.5-non-cc-heuristic-mul-unimax-e1-code-a1.3-701515.sh
. /lustre/fs6/portfolios/adlr/users/lmcafee/retro/megatrons/core-converter/scripts/oci-8t/ppl/blend.sh

#--recompute-activations \
# --load ${CHECKPOINT_DIR} \
# --save ${CHECKPOINT_DIR} \
# --save-interval 10000 \
# --tensorboard-dir ${TENSORBOARD_DIR}"
# --data-cache-path ${DATACACHE_DIR} \
# --eval-iters 32 \
# --eval-interval 2000 \
# --decoupled-lr 5.0e-4 \
# --decoupled-min-lr 4.5e-5 \
# --no-create-attention-mask-in-dataloader \
# TOKENIZER_MODEL="/home/mpatwary/data/adlr-nlp-sharing/nvllm-8t/utils/nemotron_2_256k.model"
# --tp-comm-overlap \
# --manual-gc \
TOKENIZER_MODEL="/lustre/fs3/portfolios/adlr/projects/adlr_nlp_arch/adlr_nlp_sharing/nvllm-8t/data/tokens-shuffle/utils/nemotron_2_256k.model"
EVAL_ITERS=1 # 10 # 32
EVAL_INTERVAL=2000
# MICRO_BATCH_SIZE=4
MICRO_BATCH_SIZE=1
if [ "${MODEL_KEY}" = "15b" ]; then
    ARGS=" \
        --distributed-timeout-minutes 60 \
        --use-mcore-models \
        --sequence-parallel \
        --apply-layernorm-1p \
        --untie-embeddings-and-output-weights \
        --disable-bias-linear \
        --no-position-embedding \
        --use-rotary-position-embeddings \
        --rotary-percent 0.5 \
        --squared-relu \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --exit-duration-in-mins 460 \
        --tensor-model-parallel-size 8 \
        --pipeline-model-parallel-size 1 \
        --num-layers 32 \
        --hidden-size 6144 \
        --num-attention-heads 48 \
        --group-query-attention \
        --num-query-groups 8 \
        --seq-length 4096 \
        --max-position-embeddings 4096 \
        --micro-batch-size ${MICRO_BATCH_SIZE} \
        --rampup-batch-size 384 384 97656250 \
        --global-batch-size 1152 \
        --train-samples 1953125000 \
        --lr-decay-samples 1949218748 \
        --lr-warmup-samples 3906252 \
        --lr 4.5e-4 \
        --min-lr 4.5e-5 \
        --lr-decay-style cosine \
        --log-interval 100 \
        --eval-iters ${EVAL_ITERS} \
        --eval-interval ${EVAL_INTERVAL} \
        --tokenizer-type GPTSentencePieceTokenizer \
        --tokenizer-model ${TOKENIZER_MODEL} \
        --data-path ${DATA_BLEND} \
        --split 99,1,0 \
        --clip-grad 1.0 \
        --weight-decay 0.1 \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --init-method-std 0.0134 \
        --log-params-norm \
        --log-num-zeros-in-grad \
        --log-throughput \
        --bf16 \
        --use-distributed-optimizer \
        --overlap-grad-reduce \
        --overlap-param-gather \
        --num-workers 1 \
    "
else
    # --data-cache-path ${DATACACHE_DIR} \
    # --load ${CHECKPOINT_DIR} \
    # --save ${CHECKPOINT_DIR} \
    # --save-interval 500 \
    # --tp-comm-overlap \
    # --decoupled-lr 5.0e-4 \
    # --decoupled-min-lr 1.0e-5 \
    # --no-create-attention-mask-in-dataloader \
    # --manual-gc \
    # --tensorboard-dir ${TENSORBOARD_DIR}"
    ARGS=" \
        --recompute-activations \
        --distributed-timeout-minutes 60 \
        --use-mcore-models \
        --sequence-parallel \
        --apply-layernorm-1p \
        --untie-embeddings-and-output-weights \
        --disable-bias-linear \
        --no-position-embedding \
        --use-rotary-position-embeddings \
        --rotary-percent 0.5 \
        --squared-relu \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --exit-duration-in-mins 5740 \
        --tensor-model-parallel-size 8 \
        --pipeline-model-parallel-size 12 \
        --num-layers 96 \
        --hidden-size 18432 \
        --num-attention-heads 96 \
        --group-query-attention \
        --num-query-groups 8 \
        --seq-length 4096 \
        --max-position-embeddings 4096 \
        --micro-batch-size 1 \
        --rampup-batch-size 768 768 97656250 \
        --global-batch-size 2304 \
        --train-samples 1953125000 \
        --lr-decay-samples 1949218748 \
        --lr-warmup-samples 3906252 \
        --lr 1.0e-4 \
        --min-lr 1.0e-5 \
        --lr-decay-style cosine \
        --log-interval 100 \
        --eval-iters ${EVAL_ITERS} \
        --eval-interval ${EVAL_INTERVAL} \
        --tokenizer-type GPTSentencePieceTokenizer \
        --tokenizer-model ${TOKENIZER_MODEL} \
        --data-path ${DATA_BLEND} \
        --split 99,1,0 \
        --clip-grad 1.0 \
        --weight-decay 0.1 \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --init-method-std 0.0063 \
        --log-num-zeros-in-grad \
        --log-throughput \
        --bf16 \
        --use-distributed-optimizer \
        --overlap-grad-reduce \
        --overlap-param-gather \
        --num-workers 6"

    if [ "${GROUP_KEY}" = "orig" ]; then
        ARGS+=" --num-layers-per-virtual-pipeline-stage 1"
    fi
fi

ARGS+=" --load ${CHECKPOINT_DIR}"
ARGS+=" --use-checkpoint-args"
ARGS+=" --no-load-optim --no-load-rng" # --finetune"
ARGS+=" --exit-interval 10"
ARGS+=" --exit-on-missing-checkpoint"
ARGS+=" --skip-train"
ARGS+=" --transformer-impl transformer_engine"

######## Command. ########

# CMD=" \
#     cd ${REPO_DIR} && \
#     ${REPO_DIR}/bind.sh --cpu=${REPO_DIR}/dgxa100_ccx.sh --mem=${REPO_DIR}/dgxa100_ccx.sh python -u ${SCRIPT} ${ARGS} \
# CMD="cd ${REPO_DIR} && python -u ${SCRIPT} ${ARGS}"
CMD="cd /lustre/fs6/portfolios/adlr/users/lmcafee/retro/megatrons/training-nov2023 && python -u ${SCRIPT} ${ARGS}"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo $CMD
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

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
