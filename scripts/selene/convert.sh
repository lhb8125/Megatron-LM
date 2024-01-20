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

export NVTE_APPLY_QK_LAYER_SCALING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1 # new

# TRANSFORMER_IMPL=local
TRANSFORMER_IMPL=transformer_engine

if [ "0" = "0" ]; then

    # KEY=843m DIR=gpt3-843m-multi-1.1t-gtc-llr TP=1
    # KEY=2b   DIR=gpt3-2b-multi-1.1t-gtc       TP=1
    KEY=8b   DIR=gpt3-8b-multi-1.1t-gtc       TP=4
    # KEY=22b  DIR=gpt3-22b-multi-1.1t-gtc      TP=8

    # >>>
    # ~~ oci ~~
    # MEGATRON_PATH="/lustre/fsw/portfolios/adlr/users/lmcafee/retro/megatrons/core-converter"
    # SHARING_ROOT="/lustre/fsw/portfolios/adlr/projects/adlr_nlp_arch/adlr_nlp_sharing"
    # LOAD_DIR="${SHARING_ROOT}/nvllm-1.1t/checkpoints/${DIR}"
    # LOAD_DIR="/lustre/fs6/portfolios/adlr/users/lmcafee/retro/megatrons/core-converter/scripts/checkpoints/nvllm-1.1t-8b/mlm"
    # SAVE_DIR="${SHARING_ROOT}/nvllm-1.1t-mcore/${TRANSFORMER_IMPL}/${KEY}"
    # +++
    # ~~ selene ~~
    MEGATRON_PATH="/lustre/fsw/adlr/adlr-nlp/lmcafee/data/retro/megatrons/core-converter"
    SHARING_ROOT="/lustre/fsw/adlr/adlr-nlp/adlr-nlp-sharing"
    # LOAD_DIR="/lustre/fsw/adlr/adlr-nlp/adlr-nlp-sharing/nvllm-3.5t/checkpoints/${DIR}"
    # SAVE_DIR="/lustre/fsw/adlr/adlr-nlp/adlr-nlp-sharing/nvllm-3.5t-mcore/${TRANSFORMER_IMPL}/${KEY}"
    # <<<

    # LOAD_DIR="${SHARING_ROOT}/nvllm-3.5t/checkpoints/${DIR}"
    # LOAD_DIR="/lustre/fsw/adlr/adlr-nlp/adlr-nlp-sharing/nvllm-3.5t/checkpoints/gpt3-8b-multi-3.5t"
    LOAD_DIR="/lustre/fsw/adlr/adlr-nlp/adlr-nlp-sharing/nvllm-3.5t/checkpoints/gpt3-8b-multi-3.5t/base/tp4pp1"

    # >>>
    SAVE_DIR="${SHARING_ROOT}/nvllm-3.5t-mcore/${TRANSFORMER_IMPL}/${KEY}"
    # SAVE_DIR="/lustre/fsw/adlr/adlr-nlp/lmcafee/data/retro/megatrons/core-converter/scripts/checkpoints/8b"
    # <<<

    # python -m tools.checkpoint.convert \
    python tools/checkpoint/convert.py \
        --model-type GPT \
        --loader megatron \
        --saver mcore \
	--position-embedding-type rope \
        --transformer-impl ${TRANSFORMER_IMPL} \
        --target-tensor-parallel-size ${TP} \
        --target-pipeline-parallel-size 1 \
        --megatron-path ${MEGATRON_PATH} \
        --load-dir ${LOAD_DIR} \
        --save-dir ${SAVE_DIR}

else

    KEY=43b  DIR=gpt3-43b-multi-1.1t-gtc      TP=8

    # 1, 2, 4, 4vp1
    PP=1 VP=""
    # PP=2 VP=""
    # PP=4 VP=""
    # PP=4 VP="vp1"
    python tools/checkpoint/convert.py \
        --model-type GPT \
        --loader megatron \
        --saver mcore \
	--position-embedding-type rope \
        --transformer-impl ${TRANSFORMER_IMPL} \
        --target-tensor-parallel-size ${TP} \
        --target-pipeline-parallel-size ${PP} \
        --megatron-path ${MEGATRON_PATH} \
        --load-dir /lustre/fsw/portfolios/adlr/projects/adlr_nlp_arch/adlr_nlp_sharing/nvllm-1.1t/checkpoints/gpt3-43b-multi-1.1t-gtc/tp8pp${PP}${VP} \
        --save-dir /lustre/fs6/portfolios/adlr/users/lmcafee/retro/megatrons/core-converter/scripts/checkpoints/${TRANSFORMER_IMPL}/${KEY}-tp8pp${PP}${VP}

fi

# eof
