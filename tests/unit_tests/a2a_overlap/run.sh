#!/bin/bash
set -x

#export CUDA_DEVICE_MAX_CONNECTIONS=1 
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
export NCCL_NVLS_ENABLE=0
export NVTE_FUSED_ATTN=1
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16

MLM_PATH=/lustre/fsw/coreai_devtech_all/pingtianl/a2a_olp/github
# TE_PATH=/lustre/fsw/coreai_devtech_all/pingtianl/a2a_olp/TransformerEngine
export PYTHONPATH=${MLM_PATH}:${PYTHONPATH}

DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`

EP=8 # change to 64 for 64 GPUs
TP=1
PP=1
EXPERTS=32 # change to 256 for 64 GPUs
CONFIG="--distributed-timeout-minutes 60 --tensor-model-parallel-size ${TP} --pipeline-model-parallel-size ${PP} --context-parallel-size 1 --expert-model-parallel-size $EP --use-distributed-optimizer --use-mcore-models --sequence-parallel --use-flash-attn --disable-bias-linear --micro-batch-size 1 --global-batch-size 512 --train-samples 585937500 --exit-duration-in-mins 230 --no-bias-swiglu-fusion --no-check-for-nan-in-loss-and-grad --no-rope-fusion --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 --transformer-impl transformer_engine --seq-length 4096 --data-cache-path ./mcore-benchmarking-v0.9/DeepSeek-V3-TP1PP16EP64VPP1-MBS1GBS512/cache --tokenizer-type HuggingFaceTokenizer --tokenizer-model deepseek-ai/DeepSeek-V3 --data-path /lustre/share/coreai_dlalgo_mcore/Dataset/Slimpajama/DeepSeek-V3/dsv3_text_document --split 99,1,0 --no-mmap-bin-files --no-create-attention-mask-in-dataloader --num-workers 6 --num-layers 1 --hidden-size 7168 --ffn-hidden-size 18432 --num-attention-heads 128 --kv-channels 128 --max-position-embeddings 4096 --position-embedding-type rope --rotary-base 10000 --make-vocab-size-divisible-by 3232 --normalization RMSNorm --norm-epsilon 1e-6 --swiglu --untie-embeddings-and-output-weights --multi-latent-attention --attention-dropout 0.0 --hidden-dropout 0.0 --clip-grad 1.0 --weight-decay 0.1 --qk-layernorm --lr-decay-samples 584765624 --lr-warmup-samples 162761 --lr-warmup-init 1.3e-7 --lr 1.3e-6 --min-lr 1.3e-7 --lr-decay-style cosine --adam-beta1 0.9 --adam-beta2 0.95 --num-experts ${EXPERTS} --moe-layer-freq '([0]*3+[1]*58)' --moe-ffn-hidden-size 2048 --moe-shared-expert-intermediate-size 2048 --moe-router-load-balancing-type seq_aux_loss --moe-router-topk 8 --moe-token-dispatcher-type alltoall --moe-router-pre-softmax --moe-grouped-gemm --moe-aux-loss-coeff 1e-3 --moe-router-group-topk 4 --moe-router-num-groups 8  --moe-router-topk-scaling-factor 2.5 --moe-router-score-function sigmoid  --moe-router-bias-update-rate 1e-3 --moe-expert-capacity-factor 1.0 --moe-pad-expert-input-to-capacity --q-lora-rank 1536 --kv-lora-rank 512 --qk-head-dim 128 --qk-pos-emb-head-dim 64 --v-head-dim 128 --rotary-scaling-factor 40 --eval-iters 32 --eval-interval 200 --finetune --auto-detect-ckpt-format --save ./mcore-benchmarking-v0.9/DeepSeek-V3-TP1PP16EP64VPP1-MBS1GBS512/checkpoints --save-interval 500 --dist-ckpt-strictness log_all --init-method-std 0.02 --log-timers-to-tensorboard --log-memory-to-tensorboard --log-num-zeros-in-grad --log-params-norm --log-validation-ppl-to-tensorboard --log-throughput --log-interval 1 --tensorboard-dir ./mcore-benchmarking-v0.9/DeepSeek-V3-TP1PP16EP64VPP1-MBS1GBS512/tensorboard --bf16 --no-async-tensor-model-parallel-allreduce"


# single node, 8 GPUs, run in interactive mode
run_cmd="torchrun --nproc_per_node=8 ${MLM_PATH}/tests/unit_tests/a2a_overlap/test_a2a_overlap.py ${CONFIG}"
# run_cmd="nsys profile -s none --stats=true -t nvtx,cuda,cudnn,cublas -o /lustre/fsw/coreai_devtech_all/pingtianl/a2a_olp/profile_a2a_overlap_${DATETIME}.nsys-rep  --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop torchrun --nproc_per_node=8 ${MLM_PATH}/tests/unit_tests/a2a_overlap/test_a2a_overlap.py ${CONFIG}"
${run_cmd}

