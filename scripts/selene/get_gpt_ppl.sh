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

if [ "$1" = "0" ]; then
    USE_CORE=0 TRANSFORMER_IMPL="local"
elif [ "$1" = "1" ]; then
    USE_CORE=1 TRANSFORMER_IMPL="transformer_engine"
else
    USE_CORE=1 TRANSFORMER_IMPL="local"
fi

# ADD_RETRIEVER=$2
NPROCS=8
# NWORKERS=32






# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# customize / begin.
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

ADD_RETRIEVER=0
# REPO_DIR="/lustre/fs6/portfolios/adlr/users/lmcafee/retro/megatrons/core-converter"
REPO_DIR="/lustre/fsw/adlr/adlr-nlp/lmcafee/data/retro/megatrons/core-converter"

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# customize / end.
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<







######## setup. ########

set -u

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_SOCKET_IFNAME=^vlan,lo
unset NCCL_DEBUG


# SIZE=843m SHARED_LOAD_DIR=gpt3-843m-multi-1.1t-gtc-llr TP=1
# SIZE=2b   SHARED_LOAD_DIR=gpt3-2b-multi-1.1t-gtc       TP=1
SIZE=8b   SHARED_LOAD_DIR=gpt3-8b-multi-1.1t-gtc       TP=4
# SIZE=22b  SHARED_LOAD_DIR=gpt3-22b-multi-1.1t-gtc      TP=8
# SIZE=43b SHARED_LOAD_DIR=gpt3-43b-multi-1.1t-gtc/tp8pp1 TP=8

ARGS=""
if [ "$USE_CORE" = "0" ]; then
    # LOAD_DIR="/lustre/fs3/portfolios/adlr/projects/adlr_nlp_arch/adlr_nlp_sharing/nvllm-1.1t/checkpoints/${SHARED_LOAD_DIR}"

    LOAD_DIR="/lustre/fsw/adlr/adlr-nlp/adlr-nlp-sharing/nvllm-3.5t/checkpoints/gpt3-8b-multi-3.5t/base/tp4pp1"
    # LOAD_DIR="/lustre/fsw/adlr/adlr-nlp/mpatwary/checkpoints/gpt3/share-checkpoints/gpt3-8b-multi-1.1t-gtc"
else
    ARGS+=" --transformer-impl ${TRANSFORMER_IMPL}"
    ARGS+=" --use-mcore-models"

    # LOAD_DIR="/lustre/fs6/portfolios/adlr/users/lmcafee/retro/megatrons/core-converter/scripts/checkpoints/${TRANSFORMER_IMPL}/${SIZE}"
    LOAD_DIR="/lustre/fsw/adlr/adlr-nlp/adlr-nlp-sharing/nvllm-3.5t-mcore/${TRANSFORMER_IMPL}/${SIZE}"
    if [ "$SIZE" = "43b" ]; then
	LOAD_DIR+="-tp8pp1"
    fi
fi

LOAD_OPTION="--no-load-optim --finetune --exit-interval 10"
LOAD_OPTION+=" --exit-on-missing-checkpoint"
LOAD_OPTION+=" --skip-train"
# >>>
# LOAD_OPTION+=" --use-checkpoint-args"
# <<<

######## checkpoint. ########

# TENSORBOARD_DIR="$CHECKPOINT_DIR/tensorboard"
# mkdir -p ${TENSORBOARD_DIR}

######## data blend. ########

# . /lustre/fsw/adlr/adlr-nlp/boxinw/megatron-lm-pretrain/scripts/lawrence_blend_oci.sh
# . /lustre/fs6/portfolios/adlr/users/lmcafee/retro/misc/next-llm-tokenizer/lawrence_blend_oci.sh
. /lustre/fsw/adlr/adlr-nlp/lmcafee/data/retro/megatrons/retro-mcore-test/scripts/843m/lawrence_blend_oci_soft.sh

######## args. ########

# EVAL_ITERS=32
EVAL_ITERS=4

#     --save-interval 2000 \
#     --save ${CHECKPOINT_DIR} \
#     --tensorboard-dir ${TENSORBOARD_DIR} \
#     --log-validation-ppl-to-tensorboard \

# TOKENIZER_MODEL="/lustre/fsw/portfolios/adlr/projects/adlr_nlp_arch/adlr_nlp_sharing/nvllm-1.1t/utils/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model"
TOKENIZER_MODEL="/lustre/fsw/adlr/adlr-nlp/adlr-nlp-sharing/nvllm-3.5t/utils/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model"
if [ "$SIZE" = "843m" ]; then
    ARGS+=" \
        --load ${LOAD_DIR} ${LOAD_OPTION} \
	\
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
        --eval-iters ${EVAL_ITERS} \
        --eval-interval 1260 \
        --tokenizer-type GPTSentencePieceTokenizer \
        --tokenizer-model ${TOKENIZER_MODEL} \
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

elif [ "$SIZE" = "2b" ]; then
        # --save-interval 20000 \
        # --save ${CHECKPOINT_DIR} \
        # --load ${CHECKPOINT_DIR} \
        # --tensorboard-dir ${TENSORBOARD_DIR} \
        # --DDP-impl local \
    ARGS+=" \
        --load ${LOAD_DIR} ${LOAD_OPTION} \
	\
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
        --exit-duration-in-mins 230 \
        --tensor-model-parallel-size 1 \
        --pipeline-model-parallel-size 1 \
        --num-layers 24 \
        --hidden-size 2048 \
        --num-attention-heads 16 \
        --seq-length 4096 \
        --max-position-embeddings 4096 \
        --micro-batch-size 1 \
        --global-batch-size 256 \
        --rampup-batch-size 64 64 65324160 \
        --train-samples 268554688 \
        --lr-decay-samples 255126953 \
        --lr-warmup-samples 122071 \
        --lr 2.0e-4\
        --min-lr 2.0e-5 \
        --lr-decay-style cosine \
        --log-interval 100 \
        --eval-iters ${EVAL_ITERS} \
        --eval-interval 2000 \
        --tokenizer-type GPTSentencePieceTokenizer \
        --tokenizer-model ${TOKENIZER_MODEL} \
        --data-path ${DATA_BLEND} \
        --split 99,1,0 \
        --clip-grad 1.0 \
        --weight-decay 0.1 \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --init-method-std 0.014 \
        --log-params-norm \
        --log-num-zeros-in-grad \
        --bf16 \
    "

elif [ "$SIZE" = "8b" ]; then
        # --save-interval 10000 \
        # --save ${CHECKPOINT_DIR} \
        # --load ${CHECKPOINT_DIR} \
        # --tensorboard-dir ${TENSORBOARD_DIR} \
        # --DDP-impl local \

    # LOAD_DIR="/lustre/fsw/portfolios/adlr/users/lmcafee/retro/megatrons/core-converter/scripts/checkpoints/nvllm-1.1t-8b/mlm"
    # LOAD_DIR="/lustre/fsw/portfolios/adlr/users/lmcafee/retro/megatrons/core-converter/scripts/checkpoints/nvllm-1.1t-8b/core-transformer_engine"
    # LOAD_DIR="/lustre/fsw/portfolios/adlr/users/lmcafee/retro/megatrons/core-converter/scripts/checkpoints/nvllm-1.1t-8b/core-local"
    # ARGS+=" \
    #     --load ${LOAD_DIR} ${LOAD_OPTION} \
    # 	\
    #     --sequence-parallel \
    #     --recompute-activations \
    #     --use-flash-attn \
    #     --apply-layernorm-1p \
    #     --untie-embeddings-and-output-weights \
    #     --disable-bias-linear \
    #     --no-position-embedding \
    #     --use-rotary-position-embeddings \
    #     --rotary-percent 0.5 \
    #     --swiglu \
    #     --attention-dropout 0.0 \
    #     --hidden-dropout 0.0 \
    #     --exit-duration-in-mins 230 \
    #     --tensor-model-parallel-size 4 \
    #     --pipeline-model-parallel-size 1 \
    #     --num-layers 32 \
    #     --hidden-size 4096 \
    #     --num-attention-heads 32 \
    #     --seq-length 4096 \
    #     --max-position-embeddings 4096 \
    #     --micro-batch-size 1 \
    #     --global-batch-size 256 \
    #     --rampup-batch-size 64 64 65324160 \
    #     --train-samples 268554688 \
    #     --lr-decay-samples 255126953 \
    #     --lr-warmup-samples 122071 \
    #     --lr 1.0e-4 \
    #     --min-lr 1.0e-5 \
    #     --lr-decay-style cosine \
    #     --log-interval 100 \
    #     --eval-iters ${EVAL_ITERS} \
    #     --eval-interval 2000 \
    #     --tokenizer-type GPTSentencePieceTokenizer \
    #     --tokenizer-model ${TOKENIZER_MODEL} \
    #     --data-path ${DATA_BLEND} \
    #     --split 99,1,0 \
    #     --clip-grad 1.0 \
    #     --weight-decay 0.1 \
    #     --adam-beta1 0.9 \
    #     --adam-beta2 0.95 \
    #     --init-method-std 0.010 \
    #     --log-params-norm \
    #     --log-num-zeros-in-grad \
    #     --bf16 \
    # "

    # --save-interval 5000 \
    # --save ${CHECKPOINT_DIR} \
    # --load ${CHECKPOINT_DIR} \
    # --tensorboard-dir ${TENSORBOARD_DIR} \
    # --DDP-impl local \
    # --blendable-index-path ${BLENDABLE_INDEX} \
    # --unsafe-flag-for-gpt-speedup \
    ARGS+=" \
        --load ${LOAD_DIR} ${LOAD_OPTION} \
	\
        --global-batch-size 1024 \
        --apply-layernorm-1p \
        --use-flash-attn \
        --untie-embeddings-and-output-weights \
        --disable-bias-linear \
        --no-position-embedding \
        --use-rotary-position-embeddings \
        --rotary-percent 0.5 \
        --squared-relu \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --exit-duration-in-mins 460 \
        --tensor-model-parallel-size 4 \
        --pipeline-model-parallel-size 1 \
        --sequence-parallel \
        --recompute-activations \
        --num-layers 32 \
        --hidden-size 4096 \
        --num-attention-heads 32 \
        --seq-length 4096 \
        --max-position-embeddings 4096 \
        --micro-batch-size 2 \
        --rampup-batch-size 256 256 207849600 \
        --train-samples 854492188 \
        --lr-decay-samples 811767576 \
        --lr-warmup-samples 122071 \
        --lr 3.0e-4 \
        --min-lr 3.0e-5 \
        --lr-decay-style cosine \
        --log-interval 100 \
        --eval-iters ${EVAL_ITERS} \
        --eval-interval 2000 \
        --tokenizer-type GPTSentencePieceTokenizer \
        --tokenizer-model ${TOKENIZER_MODEL} \
        --data-path ${DATA_BLEND} \
        --split 99,1,0 \
        --clip-grad 1.0 \
        --weight-decay 0.1 \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --init-method-std 0.010 \
        --log-params-norm \
        --log-num-zeros-in-grad \
        --bf16 \
    "

elif [ "$SIZE" = "22b" ]; then
        # --save-interval 10000 \
        # --save ${CHECKPOINT_DIR} \
        # --load ${CHECKPOINT_DIR} \
        # --DDP-impl local \
        # --tensorboard-dir ${TENSORBOARD_DIR} \
        # --unsafe-flag-for-gpt-speedup \
        # --use-container-fused-kernels \
        # --blendable-index-path ${BLENDABLE_INDEX} \
    ARGS+=" \
        --load ${LOAD_DIR} ${LOAD_OPTION} \
        \
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
        --exit-duration-in-mins 230 \
        --tensor-model-parallel-size 8 \
        --pipeline-model-parallel-size 1 \
        --num-layers 40 \
        --hidden-size 6144 \
        --num-attention-heads 48 \
        --seq-length 4096 \
        --max-position-embeddings 4096 \
        --micro-batch-size 1 \
        --global-batch-size 512 \
        --rampup-batch-size 128 128 65324032\
        --train-samples 268554688 \
        --lr-decay-samples 255126953 \
        --lr-warmup-samples 162761 \
        --lr 1.0e-4 \
        --min-lr 1.0e-5 \
        --lr-decay-style cosine \
        --log-interval 100 \
        --eval-iters ${EVAL_ITERS} \
        --eval-interval 2000 \
        --tokenizer-type GPTSentencePieceTokenizer \
        --tokenizer-model ${TOKENIZER_MODEL} \
        --data-path ${DATA_BLEND} \
        --split 99,1,0 \
        --clip-grad 1.0 \
        --weight-decay 0.1 \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --init-method-std 0.008 \
        --log-params-norm \
        --log-num-zeros-in-grad \
        --bf16 \
    "

elif [ "$SIZE" = "43b" ]; then
    ARGS+=" \
        --load ${LOAD_DIR} ${LOAD_OPTION} \
        \
        --use-flash-attn \
        --apply-layernorm-1p \
        --untie-embeddings-and-output-weights \
        --disable-bias-linear \
        --no-position-embedding \
        --use-rotary-position-embeddings \
        --rotary-percent 0.5 \
        --swiglu \
        --recompute-activations \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --exit-duration-in-mins 220 \
        --tensor-model-parallel-size ${TP} \
        --pipeline-model-parallel-size 1 \
        --num-layers 48 \
        --hidden-size 8192 \
        --num-attention-heads 64 \
        --seq-length 4096 \
        --max-position-embeddings 4096 \
        --micro-batch-size 1 \
        --global-batch-size 768 \
        --train-samples 25000000 \
        --lr-decay-samples 23750000 \
        --lr-warmup-samples 16667 \
        --lr 9.0e-6 \
        --min-lr 9e-7 \
        --lr-decay-style cosine \
        --log-interval 1 \
        --eval-iters ${EVAL_ITERS} \
        --eval-interval 1260 \
        --tokenizer-type GPTSentencePieceTokenizer \
        --tokenizer-model ${TOKENIZER_MODEL} \
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
        --use-distributed-optimizer \
    "

else
    echo "specialize for size '${SIZE}'."
    exit 1
fi

#     --attention-dropout 0.0 \
#     --hidden-dropout 0.0 \
#     --tensor-model-parallel-size ${TP} \
#     --pipeline-model-parallel-size 1 \
#     --num-layers 48 \
#     --hidden-size 8192 \
#     --num-attention-heads 64 \
#     --seq-length 4096 \
#     --max-position-embeddings 4096 \
#     --clip-grad 1.0 \
#     --weight-decay 0.1 \
#     --adam-beta1 0.9 \
#     --adam-beta2 0.95 \
#     --init-method-std 0.007 \
#     --use-distributed-optimizer \

#     --recompute-activations \
#     --no-position-embedding \
#     --use-rotary-position-embeddings \
# ARGS+=" \
#     --load ${LOAD_DIR} ${LOAD_OPTION} \
#     --use-checkpoint-args \
#     \
#     --exit-duration-in-mins 220 \
#     --micro-batch-size 1 \
#     --global-batch-size 768 \
#     --tokenizer-type GPTSentencePieceTokenizer \
#     --tokenizer-model ${TOKENIZER_MODEL} \
#     --data-path ${DATA_BLEND} \
#     --train-samples 25000000 \
#     --eval-iters ${EVAL_ITERS} \
#     --eval-interval 1260 \
#     --split 98,2,0 \
#     --lr-decay-samples 23750000 \
#     --lr-warmup-samples 16667 \
#     --lr 9.0e-6 \
#     --min-lr 9e-7 \
#     --lr-decay-style cosine \
#     --log-interval 1 \
#     --log-params-norm \
#     --log-num-zeros-in-grad \
#     --bf16 \
#     --use-flash-attn \
#     --apply-layernorm-1p \
#     --untie-embeddings-and-output-weights \
#     --disable-bias-linear \
#     --position-embedding-type rope \
#     --rotary-percent 0.5 \
#     --swiglu \
# "

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

NODE_RANK=0
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
