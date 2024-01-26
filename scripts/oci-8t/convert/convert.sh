#!/bin/bash

set -u

export CUDA_DEVICE_MAX_CONNECTIONS=1

# MEGATRON_PATH="/lustre/fsw/adlr/adlr-nlp/lmcafee/data/retro/megatrons/core-converter"
# LOAD_ROOT="/lustre/fsw/adlr/adlr-nlp/adlr-nlp-sharing/nvllm-8t/checkpoints"
# SAVE_ROOT="/lustre/fsw/adlr/adlr-nlp/lmcafee/data/retro/megatrons/core-converter/scripts/checkpoints"

MEGATRON_PATH="/lustre/fs6/portfolios/adlr/users/lmcafee/retro/megatrons/core-converter"
# LOAD_ROOT="/lustre/fs3/portfolios/adlr/projects/adlr_nlp_arch/adlr_nlp_sharing/nvllm-8t/checkpoints" # /gpt3-15b-8t-shuffle-mup/base"
LOAD_ROOT="/lustre/fs6/portfolios/adlr/users/lmcafee/retro/megatrons/core-converter/scripts/checkpoints/8t/orig"
SAVE_ROOT="/lustre/fs6/portfolios/adlr/users/lmcafee/retro/megatrons/core-converter/scripts/checkpoints/8t"

LOADER_TY="mcore"
if [ "${MODEL_SIZE}" = "15b" ]; then
    LOAD="gpt3-15b-8t-shuffle-mup/base"
    SAVE="15b"
else
    LOAD="gpt3-340b-8t-shuffle-mup/base/"
    SAVE="340b"
fi

# --load-dir ${LOAD_ROOT}/${LOAD} \
ARGS=" \
    --model-type GPT \
    --loader ${LOADER_TY} \
    --saver mcore \
    --position-embedding-type rope \
    --transformer-impl transformer_engine \
    --target-tensor-parallel-size 8 \
    --target-pipeline-parallel-size ${PP} \
    --megatron-path ${MEGATRON_PATH} \
    --load-dir ${LOAD_ROOT}/${SAVE} \
    --save-dir ${SAVE_ROOT}/${SAVE}-pp${PP} \
"
# ARGS+=" --sequence-parallel"

cd ${MEGATRON_PATH} && python tools/checkpoint/convert.py ${ARGS}

# eof
