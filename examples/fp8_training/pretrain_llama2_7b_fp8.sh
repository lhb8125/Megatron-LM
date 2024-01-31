#!/bin/bash
set -x
# Runs the "345M" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_TORCH_COMPILE=0

CHECKPOINT_PATH=???
DATA_PATH=???
MLM_PATH=???
TE_PATH=???

source ${DATA_PATH}/llama_blend.sh

export PYTHONPATH=${MLM_PATH}:${TE_PATH}:${PYTHONPATH}


WANDB="???"

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

LLAMA_ARGS="
    --num-layers 32 \
    --hidden-size 4096 \
    --ffn-hidden-size 11008 \
    --num-attention-heads 32 \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --micro-batch-size 2 \
    --global-batch-size 2048 \
    --lr 0.00015 \
    --train-iters 500000 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
"

DATA_ARGS="
    --data-path ${DATA_BLEND} \
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model ${DATA_PATH}/tokenizer.model \
    --split 990,8,2 \
    --data-cache-path ${DATA_PATH} \
"

OUTPUT_ARGS="
    --log-interval 10 \
    --save-interval 250 \
    --eval-interval 1000 \
    --eval-iters 20 \
    --timing-log-level 0 \
    --wandb-project ??? \
    --wandb-exp-name ??? \
    --wandb-save-dir ??? \
    --tensorboard-dir ??? \
    --tensorboard-log-interval 10 \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    --log-memory-to-tensorboard \
"

FP8_ARGS="
    --fp8-format hybrid \
    --fp8-margin 0 \
    --fp8-interval 1 \
    --fp8-amax-history-len 1024 \
    --fp8-amax-compute-algo max \
"
OTHER_ARGS="
    --tensor-model-parallel-size 2 \
    --pipeline-model-parallel-size 1 \
    --sequence-parallel \
    --bf16 \
    --use-distributed-optimizer \
    --use-flash-attn \
    --use-mcore-models \
    --transformer-impl transformer_engine \
"


#torchrun ${DISTRIBUTED_ARGS} ${MLM_PATH}/pretrain_gpt.py \
wandb login ${WANDB}
python ${MLM_PATH}/pretrain_gpt.py \
    ${LLAMA_ARGS} \
    ${DATA_ARGS} \
    ${OUTPUT_ARGS} \
    ${OTHER_ARGS} \
    ${FP8_ARGS} \
    --load ${CHECKPOINT_PATH} \
    --save ${CHECKPOINT_PATH} \
    --distributed-backend nccl
