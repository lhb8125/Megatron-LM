#!/bin/bash

set -u

export CUDA_DEVICE_MAX_CONNECTIONS=1

LOADER_TY="mcore"
if [ "${MODEL_SIZE}" = "340b" ]; then
    LOAD="/lustre/fsw/adlr/adlr-nlp/adlr-nlp-sharing/nvllm-8t/checkpoints/gpt3-340b-8t-shuffle-mup/base/"
    SAVE="340b-${PP}"
else

    LOAD="/lustre/fsw/adlr/adlr-nlp/adlr-nlp-sharing/nvllm-8t/checkpoints/gpt3-15b-8t-shuffle-mup/base"
    SAVE="15b"

    # export NVTE_APPLY_QK_LAYER_SCALING=1
    # LOAD="/lustre/fsw/adlr/adlr-nlp/adlr-nlp-sharing/nvllm-1.1t-mcore/transformer_engine/843m/"
    # SAVE="843m"

    # export NVTE_APPLY_QK_LAYER_SCALING=1
    # LOADER_TY="megatron"
    # LOAD="/lustre/fsw/adlr/adlr-nlp/adlr-nlp-sharing/nvllm-3.5t/checkpoints/gpt3-8b-multi-3.5t/base/tp4pp1"
    # SAVE="8b"
    # LOAD="/lustre/fsw/adlr/adlr-nlp/adlr-nlp-sharing/nvllm-1.1t/checkpoints/gpt3-843m-multi-1.1t-gtc-llr/"
    # SAVE="843m"
fi

ARGS=" \
    --model-type GPT \
    --loader ${LOADER_TY} \
    --saver mcore \
    --position-embedding-type rope \
    --transformer-impl transformer_engine \
    --target-tensor-parallel-size 8 \
    --target-pipeline-parallel-size ${PP} \
    --megatron-path /lustre/fsw/adlr/adlr-nlp/lmcafee/data/retro/megatrons/core-converter \
    --load-dir ${LOAD} \
    --save-dir /lustre/fsw/adlr/adlr-nlp/lmcafee/data/retro/megatrons/core-converter/scripts/checkpoints/${SAVE} \
"
# ARGS+=" --sequence-parallel"

python tools/checkpoint/convert.py ${ARGS}

# eof
