#!/bin/bash

# ... SBATCH -p luna -A adlr -t 4:00:00 --nodes=8 --exclusive --mem=0 --overcommit --ntasks-per-node=8 --dependency=singleton --job-name=adlr-nlp-largelm:gpt3-843m-multi-1.1t-gtc-llr

set -u

export NCCL_IB_TIMEOUT=19
export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
unset NCCL_DEBUG

REPO_DIR="/lustre/fs6/portfolios/adlr/users/lmcafee/retro/megatrons/core-converter"
NPROCS=1

# NAME="gpt3-843m-multi-1.1t-gtc-llr"

# DIR=`pwd`
# DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
# mkdir -p $DIR/logs

# CHECKPOINT_DIR="/lustre/fsw/adlr/adlr-nlp/adlr-nlp-sharing/nvllm-1.1t/checkpoints/${NAME}"
# BLENDABLE_INDEX="/lustre/fsw/adlr/adlr-gtc-43b/data/blendable-index/${NAME}"

# TENSORBOARD_DIR="$DIR/tensorboard/${NAME}"
# mkdir -p ${TENSORBOARD_DIR}

# Get the data blend
# . /lustre/fsw/adlr/adlr-nlp/adlr-nlp-sharing/nvllm-1.1t/data/tokens/multi-1.1t-gtc-blend-v0.1.sh
. /lustre/fs6/portfolios/adlr/users/lmcafee/retro/misc/next-llm-tokenizer/lawrence_blend_oci.sh
# DATA_BLEND="1.000 /lustre/fsw/portfolios/adlr/users/lmcafee/retro/data/MTNLG/NIHExporter_shuf_text_document"

#     --use-container-fused-kernels \
#     --blendable-index-path ${BLENDABLE_INDEX} \
# --train-samples 268554688 \
# --lr-decay-samples 255126953 \
# --lr-warmup-samples 81381 \
options=" \
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
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --micro-batch-size 2 \
    --global-batch-size 128 \
    --rampup-batch-size 32 32 65324160 \
    --train-samples 10000 \
    --lr-decay-samples 9900 \
    --lr-warmup-samples 100 \
    --lr 2.5e-4 \
    --min-lr 2.5e-5 \
    --lr-decay-style cosine \
    --log-interval 1 \
    --eval-iters 32 \
    --eval-interval 2000 \
    --tokenizer-type GPTSentencePieceTokenizer \
    --tokenizer-model /lustre/fsw/portfolios/adlr/projects/adlr_nlp_arch/adlr_nlp_sharing/nvllm-1.1t/utils/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model \
    --data-path ${DATA_BLEND} \
    --split 99,1,0 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.02 \
    --log-params-norm \
    --log-num-zeros-in-grad \
    --bf16"
    # --DDP-impl local"
    # --save-interval 20000 \
    # --save ${CHECKPOINT_DIR} \
    # --load ${CHECKPOINT_DIR} \
    # --tensorboard-dir ${TENSORBOARD_DIR}"

options+=" --use-mcore-models"
options+=" --transformer-impl transformer_engine"
# options+=" --allow-transformer-engine"

# run_cmd="${DIR}/bind.sh --cpu=${DIR}/dgxa100_ccx.sh --mem=${DIR}/dgxa100_ccx.sh python -u ${DIR}/pretrain_gpt.py ${options}"

# srun -l \
#      --container-image "/lustre/fsw/adlr/adlr-nlp/images/adlr+megatron-lm+pytorch+22.12-py3-eval_with_fused_kernels_pyspy.sqsh" \
#      --container-mounts "/lustre/fsw/adlr:/lustre/fsw/adlr" \
#      --output=$DIR/logs/%x_%j_$DATETIME.log sh -c "${run_cmd}"

# set +x

CMD="\
    cd ${REPO_DIR} && \
    export PYTHONPATH=${REPO_DIR}:/home/lmcafee/src && \
    python -m torch.distributed.run \
    --nproc_per_node ${NPROCS} \
    --nnodes 1 \
    --node_rank ${NODE_RANK} \
    --master_addr ${MASTER_ADDR} \
    --master_port 6000 \
    pretrain_gpt.py ${options} \
"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "CMD = '$CMD'."
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~"
eval $CMD

