#!/bin/bash

set -u

# export NVTE_APPLY_QK_LAYER_SCALING=1

# TRANSFORMER_IMPL=local
TRANSFORMER_IMPL=transformer_engine

MODEL_TYPE=text
# MODEL_TYPE=chat

MODEL_SIZE=7b TP=1
# MODEL_SIZE=13b TP=2
# MODEL_SIZE=70b TP=8

TOKENIZER_MODEL="/lustre/fs3/portfolios/adlr/projects/adlr_nlp_arch/adlr_nlp_sharing/llama-2/tokenizer.model"
LOAD_DIR="/lustre/fsw/portfolios/adlr/projects/adlr_nlp_arch/adlr_nlp_sharing/llama-2/checkpoints/huggingface/${MODEL_TYPE}/${MODEL_SIZE}"
SAVE_DIR="/lustre/fsw/portfolios/adlr/users/lmcafee/retro/megatrons/core-converter/scripts/checkpoints/hf-mcore/${TRANSFORMER_IMPL}/${MODEL_TYPE}/${MODEL_SIZE}"

# --tensor-model-parallel-size ${TP} \
# --position-embedding-type rope \

ARGS="\
    --model-type GPT \
    --loader llama2_hf \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --target-tensor-parallel-size ${TP} \
    --target-pipeline-parallel-size 1 \
    --megatron-path /lustre/fsw/portfolios/adlr/users/lmcafee/retro/megatrons/core-converter \
    --load-dir ${LOAD_DIR} \
    --save-dir ${SAVE_DIR} \
"
ARGS+=" --saver mcore --transformer-impl ${TRANSFORMER_IMPL}"
# ARGS+=" --saver megatron"

python tools/checkpoint/convert.py ${ARGS}

# eof
