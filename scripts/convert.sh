#!/bin/bash

set -u

# [oci] ROOT_DIR=/lustre/fsw/portfolios/adlr/users/lmcafee/llama/2; \
# [selene] ROOT_DIR=/lustre/fsw/adlr/adlr-nlp/adlr-nlp-sharing/llama-2; \
# TYPE=text; \ TYPE=chat; \
# SIZE=7b TP=1; \ SIZE=13b TP=2; \ SIZE=70b TP=8; \
# python util.py \
#     --model-type GPT \
#     --loader llama2_hf \
#     --saver megatron \
#     --target-tensor-parallel-size ${TP} \
#     --load-dir ${ROOT_DIR}/checkpoints/hf/${TYPE}/${SIZE} \
#     --save-dir ${ROOT_DIR}/checkpoints/megatron/${TYPE}/${SIZE} \
#     --tokenizer-model ${ROOT_DIR}/tokenizer.model

# --tokenizer-model /lustre/fsw/portfolios/adlr/projects/adlr_nlp_arch/adlr_nlp_sharing/nvllm-1.1t/utils/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# gpt3-843m-multi-1.1t-gtc-llr/run_gpt3_843m_gtc_llr.sh ... 1
# gpt3-2b-multi-1.1t-gtc/run_gpt3_2b_gtc.sh             ... 1
# gpt3-8b-multi-1.1t-gtc/run_gpt3_8b_gtc.sh             ... 4
# gpt3-22b-multi-1.1t-gtc/run_gpt3_22b_gtc.sh           ... 8
# gpt3-43b-multi-1.1t-gtc/training_gpt3_43b_gtc.sh      ... 8
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

KEY=843m DIR=gpt3-843m-multi-1.1t-gtc-llr TP=1
# KEY=2b   DIR=gpt3-2b-multi-1.1t-gtc       TP=1
# KEY=8b   DIR=gpt3-8b-multi-1.1t-gtc       TP=4
# KEY=22b  DIR=gpt3-22b-multi-1.1t-gtc      TP=8
# KEY=43b  DIR=gpt3-43b-multi-1.1t-gtc      TP=8

# python tools/checkpoint/convert.py \
#     --model-type GPT \
#     --loader megatron \
#     --saver mcore \
#     --target-tensor-parallel-size 1 \
#     --target-pipeline-parallel-size 1 \
#     --megatron-path /lustre/fs6/portfolios/adlr/users/lmcafee/retro/megatrons/core-converter \
#     --load-dir /lustre/fs3/portfolios/adlr/projects/adlr_nlp_arch/adlr_nlp_sharing/nvllm-1.1t/checkpoints/gpt3-843m-multi-1.1t-gtc-llr \
#     --save-dir /lustre/fs6/portfolios/adlr/users/lmcafee/retro/megatrons/core-converter/scripts/checkpoints/843m

export NVTE_APPLY_QK_LAYER_SCALING=1

TRANSFORMER_IMPL=local
# TRANSFORMER_IMPL=transformer_engine
python tools/checkpoint/convert.py \
    --model-type GPT \
    --loader megatron \
    --saver mcore \
    --transformer-impl ${TRANSFORMER_IMPL} \
    --target-tensor-parallel-size ${TP} \
    --target-pipeline-parallel-size 1 \
    --megatron-path /lustre/fs6/portfolios/adlr/users/lmcafee/retro/megatrons/core-converter \
    --load-dir /lustre/fs3/portfolios/adlr/projects/adlr_nlp_arch/adlr_nlp_sharing/nvllm-1.1t/checkpoints/${DIR} \
    --save-dir /lustre/fs6/portfolios/adlr/users/lmcafee/retro/megatrons/core-converter/scripts/checkpoints/${KEY}

# PP=4
# python tools/checkpoint/convert.py \
#     --model-type GPT \
#     --loader megatron \
#     --saver mcore \
#     --target-tensor-parallel-size 8 \
#     --target-pipeline-parallel-size ${PP} \
#     --megatron-path /lustre/fs6/portfolios/adlr/users/lmcafee/retro/megatrons/core-converter \
#     --load-dir /lustre/fsw/portfolios/adlr/projects/adlr_nlp_arch/adlr_nlp_sharing/nvllm-1.1t/checkpoints/gpt3-43b-multi-1.1t-gtc/tp8pp${PP}/ \
#     --save-dir /lustre/fs6/portfolios/adlr/users/lmcafee/retro/megatrons/core-converter/scripts/checkpoints/43b-tp8pp${PP}

# eof
