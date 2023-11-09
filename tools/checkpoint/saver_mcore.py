# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import os
import sys

import torch

# >>>
from lutil import pax, print_model
# <<<

def get_models(count, dtype, pre_process, post_process):
    # >>>
    # margs.apply_query_key_layer_scaling = False

    # margs.apply_query_key_layer_scaling = False
    # margs.async_tensor_model_parallel_allreduce = True
    # margs.bias_dropout_fusion = True
    # margs.consumed_train_samples = 0
    # margs.consumed_valid_samples = 0
    # margs.data_path = ['1.000', '/lustre/fsw/portfolios/adlr/users/lmcafee/retro/data/MTNLG/NIHExporter_shuf_text_document']
    # margs.log_interval = 1
    # margs.lr_decay_samples = 9900
    # margs.lr_warmup_samples = 100
    # margs.masked_softmax_fusion = True
    # margs.no_load_optim = None
    # margs.no_load_rng = None
    # margs.no_save_optim = None
    # margs.no_save_rng = None
    # margs.perform_initialization = True
    # margs.save = None
    # margs.save_interval = None
    # margs.tokenizer_model = '/lustre/fsw/portfolios/adlr/projects/adlr_nlp_arch/adlr_nlp_sharing/nvllm-1.1t/utils/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model'
    # margs.train_samples = 10000
    # margs.use_cpu_initialization = None
    # <<<
    models = [model_provider(pre_process, post_process).to(dtype) for _ in range(count)]
    # >>>
    # print_model("layer", models[0].decoder.layers[0])
    # exit()
    # pax("models")
    # <<<
    return models

def receive_local_modules(args, mpu, out_word_embed):

    # Make models for first pipeline stage and fill in embeddings
    mpu.set_pipeline_model_parallel_rank(0)
    post_process = args.target_pipeline_parallel_size == 1
    models = get_models(args.target_tensor_parallel_size, md.params_dtype, True, post_process)

    for tp_rank, model in enumerate(models):
        model.embedding.word_embeddings.weight.data.copy_(out_word_embed[tp_rank])
        if pos_embed is not None:
            model.embedding.position_embeddings.weight.data.copy_(pos_embed)
        else:
            assert not hasattr(model.embedding, "position_embeddings")

    # Transformer layers
    #-------------------
    total_layer_num = 0
    for pp_rank in range(args.target_pipeline_parallel_size):
        # For later pipeline parallel ranks, make the new models
        if pp_rank > 0:
            mpu.set_pipeline_model_parallel_rank(pp_rank)
            post_process = pp_rank == args.target_pipeline_parallel_size - 1
            models = get_models(args.target_tensor_parallel_size, md.params_dtype, False, post_process)

        for layer in range(len(models[0].decoder.layers)):
            msg = queue_get(f"transformer layer {total_layer_num}")

            # duplicated tensors
            input_norm_weight = msg.pop("input norm weight")
            if md.norm_has_bias:
                input_norm_bias = msg.pop("input norm bias")
            post_norm_weight = msg.pop("post norm weight")
            if md.norm_has_bias:
                post_norm_bias = msg.pop("post norm bias")
            if md.linear_bias:
                dense_bias = msg.pop("dense bias")
                mlp_l1_bias = msg.pop("mlp l1 bias")

            # Split up the parallel tensors
            qkv_weight = torch.chunk(msg.pop("qkv weight"), args.target_tensor_parallel_size, dim=0)
            dense_weight = torch.chunk(msg.pop("dense weight"), args.target_tensor_parallel_size, dim=1)
            mlp_l1_weight = torch.chunk(msg.pop("mlp l1 weight"), args.target_tensor_parallel_size, dim=1)

            # Special handling for swiglu
            if md.swiglu:
                mlp_l0_weight_W = torch.chunk(msg.pop("mlp l0 weight W"), args.target_tensor_parallel_size, dim=0)
                mlp_l0_weight_V = torch.chunk(msg.pop("mlp l0 weight V"), args.target_tensor_parallel_size, dim=0)
                mlp_l0_weight = [torch.cat(weights, dim=0) for weights in zip(mlp_l0_weight_W, mlp_l0_weight_V)]
            else:
                mlp_l0_weight = torch.chunk(msg.pop("mlp l0 weight"), args.target_tensor_parallel_size, dim=0)

            if md.linear_bias:
                qkv_bias = torch.chunk(msg.pop("qkv bias"), args.target_tensor_parallel_size, dim=0)
                if md.swiglu:
                    mlp_l0_bias_W = torch.chunk(msg.pop("mlp l0 bias W"), args.target_tensor_parallel_size, dim=0)
                    mlp_l0_bias_V = torch.chunk(msg.pop("mlp l0 bias V"), args.target_tensor_parallel_size, dim=0)
                    mlp_l0_bias = [torch.cat(bias, dim=0) for bias in zip(mlp_l0_bias_W, mlp_l0_bias_V)]
                else:
                    mlp_l0_bias = torch.chunk(msg.pop("mlp l0 bias"), args.target_tensor_parallel_size, dim=0)

            # Save them to the model
            for tp_rank in range(args.target_tensor_parallel_size):
                l = models[tp_rank].decoder.layers[layer]
                l.input_layernorm.weight.data.copy_(input_norm_weight)
                if md.norm_has_bias:
                    l.input_layernorm.bias.data.copy_(input_norm_bias)
                l.self_attention.linear_qkv.weight.data.copy_(qkv_weight[tp_rank])
                l.self_attention.linear_proj.weight.data.copy_(dense_weight[tp_rank])
                # >>>
                l.pre_mlp_layernorm.weight.data.copy_(post_norm_weight)
                if md.norm_has_bias:
                    l.pre_mlp_layernorm.bias.data.copy_(post_norm_bias)
                # +++
                # l.pre_cross_attn_layernorm.weight.data.copy_(post_norm_weight)
                # if md.norm_has_bias:
                #     l.pre_cross_attn_layernorm.bias.data.copy_(post_norm_bias)
                # <<<
                l.mlp.linear_fc1.weight.data.copy_(mlp_l0_weight[tp_rank])
                l.mlp.linear_fc2.weight.data.copy_(mlp_l1_weight[tp_rank])
                if md.linear_bias:
                    l.self_attention.linear_qkv.bias.data.copy_(qkv_bias[tp_rank])
                    l.self_attention.linear_proj.bias.data.copy_(dense_bias)
                    l.mlp.linear_fc1.bias.data.copy_(mlp_l0_bias[tp_rank])
                    l.mlp.linear_fc2.bias.data.copy_(mlp_l1_bias)

            total_layer_num = total_layer_num + 1
            check_message(msg)


        if post_process:
            msg = queue_get("final norm")
            final_norm_weight = msg.pop("weight")
            if md.norm_has_bias:
                final_norm_bias = msg.pop("bias")
            for tp_rank in range(args.target_tensor_parallel_size):
                models[tp_rank].decoder.final_layernorm.weight.data.copy_(final_norm_weight)
                if md.norm_has_bias:
                    models[tp_rank].decoder.final_layernorm.bias.data.copy_(final_norm_bias)
                if pp_rank != 0 and not md.output_layer:
                    # Copy word embeddings to final pipeline rank
                    models[tp_rank].word_embeddings.weight.data.copy_(out_word_embed[tp_rank])
            del final_norm_weight
            if md.norm_has_bias:
                del final_norm_bias
            check_message(msg)

            if md.output_layer:
                msg = queue_get("output layer")
                if not hasattr(models[0], 'output_layer'):
                    print("ERROR: got an output layer, but model does not have one")
                    exit(1)
                output_layer_weight = torch.chunk(msg.pop("weight"), args.target_tensor_parallel_size, dim=0)
                for tp_rank in range(args.target_tensor_parallel_size):
                    models[tp_rank].output_layer.weight.data.copy_(output_layer_weight[tp_rank])
                del output_layer_weight
                check_message(msg)

            msg = queue_get()
            if msg != "done" and msg["name"] == "pooler":
                if not hasattr(models[0], 'pooler'):
                    print("ERROR: got a pooler, but model does not have one")
                    exit(1)
                print("received pooler")
                pooler_weight = msg.pop("weight")
                pooler_bias = msg.pop("bias")
                for tp_rank in range(args.target_tensor_parallel_size):
                    models[tp_rank].pooler.dense.weight.data.copy_(pooler_weight)
                    models[tp_rank].pooler.dense.bias.data.copy_(pooler_bias)
                del pooler_weight
                del pooler_bias
                check_message(msg)
                msg = queue_get()

            if msg != "done" and msg["name"] == "lm head":
                if not hasattr(models[0], 'lm_head'):
                    print("ERROR: got an lm head, but model does not have one")
                    exit(1)
                print("received lm head")
                lm_head_dense_weight = msg.pop("dense weight")
                lm_head_dense_bias = msg.pop("dense bias")
                lm_head_norm_weight = msg.pop("norm weight")
                if md.norm_has_bias:
                    lm_head_norm_bias = msg.pop("norm bias")
                for tp_rank in range(args.target_tensor_parallel_size):
                    models[tp_rank].lm_head.dense.weight.data.copy_(lm_head_dense_weight)
                    models[tp_rank].lm_head.dense.bias.data.copy_(lm_head_dense_bias)
                    models[tp_rank].lm_head.norm.weight.data.copy_(lm_head_norm_weight)
                    if md.norm_has_bias:
                        models[tp_rank].lm_head.norm.bias.data.copy_(lm_head_norm_bias)
                check_message(msg)
                msg = queue_get()

            if msg != "done" and msg["name"] == "binary head":
                if not hasattr(models[0], 'binary_head'):
                    print("ERROR: got a binary head, but model does not have one")
                    exit(1)
                print("received binary head")
                binary_head_weight = msg.pop("weight")
                binary_head_bias = msg.pop("bias")
                for tp_rank in range(args.target_tensor_parallel_size):
                    models[tp_rank].binary_head.weight.data.copy_(binary_head_weight)
                    models[tp_rank].binary_head.bias.data.copy_(binary_head_bias)
                check_message(msg)
                msg = queue_get()

            if msg != "done":
                print("ERROR: got some more data but was expecting to be done")

        for tp_rank in range(args.target_tensor_parallel_size):
            mpu.set_tensor_model_parallel_rank(tp_rank)
            save_checkpoint(md.iteration, [models[tp_rank]], None, None)


def receive_te_modules(mpu, out_word_embed):

    raise Exception("hi.")

    # Make models for first pipeline stage and fill in embeddings
    mpu.set_pipeline_model_parallel_rank(0)
    post_process = args.target_pipeline_parallel_size == 1
    models = get_models(args.target_tensor_parallel_size, md.params_dtype, True, post_process)

    for tp_rank, model in enumerate(models):
        model.embedding.word_embeddings.weight.data.copy_(out_word_embed[tp_rank])
        if pos_embed is not None:
            model.embedding.position_embeddings.weight.data.copy_(pos_embed)
        else:
            assert not hasattr(model.embedding, "position_embeddings")

    # Transformer layers
    #-------------------
    total_layer_num = 0
    for pp_rank in range(args.target_pipeline_parallel_size):
        # For later pipeline parallel ranks, make the new models
        if pp_rank > 0:
            mpu.set_pipeline_model_parallel_rank(pp_rank)
            post_process = pp_rank == args.target_pipeline_parallel_size - 1
            models = get_models(args.target_tensor_parallel_size, md.params_dtype, False, post_process)

        for layer in range(len(models[0].decoder.layers)):
            msg = queue_get(f"transformer layer {total_layer_num}")

            # duplicated tensors
            input_norm_weight = msg.pop("input norm weight")
            if md.norm_has_bias:
                input_norm_bias = msg.pop("input norm bias")
            post_norm_weight = msg.pop("post norm weight")
            if md.norm_has_bias:
                post_norm_bias = msg.pop("post norm bias")
            if md.linear_bias:
                dense_bias = msg.pop("dense bias")
                mlp_l1_bias = msg.pop("mlp l1 bias")

            # Split up the parallel tensors
            qkv_weight = torch.chunk(msg.pop("qkv weight"), args.target_tensor_parallel_size, dim=0)
            dense_weight = torch.chunk(msg.pop("dense weight"), args.target_tensor_parallel_size, dim=1)
            mlp_l1_weight = torch.chunk(msg.pop("mlp l1 weight"), args.target_tensor_parallel_size, dim=1)

            # Special handling for swiglu
            if md.swiglu:
                mlp_l0_weight_W = torch.chunk(msg.pop("mlp l0 weight W"), args.target_tensor_parallel_size, dim=0)
                mlp_l0_weight_V = torch.chunk(msg.pop("mlp l0 weight V"), args.target_tensor_parallel_size, dim=0)
                mlp_l0_weight = [torch.cat(weights, dim=0) for weights in zip(mlp_l0_weight_W, mlp_l0_weight_V)]
            else:
                mlp_l0_weight = torch.chunk(msg.pop("mlp l0 weight"), args.target_tensor_parallel_size, dim=0)

            if md.linear_bias:
                qkv_bias = torch.chunk(msg.pop("qkv bias"), args.target_tensor_parallel_size, dim=0)
                if md.swiglu:
                    mlp_l0_bias_W = torch.chunk(msg.pop("mlp l0 bias W"), args.target_tensor_parallel_size, dim=0)
                    mlp_l0_bias_V = torch.chunk(msg.pop("mlp l0 bias V"), args.target_tensor_parallel_size, dim=0)
                    mlp_l0_bias = [torch.cat(bias, dim=0) for bias in zip(mlp_l0_bias_W, mlp_l0_bias_V)]
                else:
                    mlp_l0_bias = torch.chunk(msg.pop("mlp l0 bias"), args.target_tensor_parallel_size, dim=0)

            # Save them to the model
            for tp_rank in range(args.target_tensor_parallel_size):
                l = models[tp_rank].decoder.layers[layer]
                l.input_layernorm.weight.data.copy_(input_norm_weight)
                if md.norm_has_bias:
                    l.input_layernorm.bias.data.copy_(input_norm_bias)
                l.self_attention.linear_qkv.weight.data.copy_(qkv_weight[tp_rank])
                l.self_attention.linear_proj.weight.data.copy_(dense_weight[tp_rank])
                # >>>
                l.pre_mlp_layernorm.weight.data.copy_(post_norm_weight)
                if md.norm_has_bias:
                    l.pre_mlp_layernorm.bias.data.copy_(post_norm_bias)
                # +++
                # l.pre_cross_attn_layernorm.weight.data.copy_(post_norm_weight)
                # if md.norm_has_bias:
                #     l.pre_cross_attn_layernorm.bias.data.copy_(post_norm_bias)
                # <<<
                l.mlp.linear_fc1.weight.data.copy_(mlp_l0_weight[tp_rank])
                l.mlp.linear_fc2.weight.data.copy_(mlp_l1_weight[tp_rank])
                if md.linear_bias:
                    l.self_attention.linear_qkv.bias.data.copy_(qkv_bias[tp_rank])
                    l.self_attention.linear_proj.bias.data.copy_(dense_bias)
                    l.mlp.linear_fc1.bias.data.copy_(mlp_l0_bias[tp_rank])
                    l.mlp.linear_fc2.bias.data.copy_(mlp_l1_bias)

            total_layer_num = total_layer_num + 1
            check_message(msg)


        if post_process:
            msg = queue_get("final norm")
            final_norm_weight = msg.pop("weight")
            if md.norm_has_bias:
                final_norm_bias = msg.pop("bias")
            for tp_rank in range(args.target_tensor_parallel_size):
                models[tp_rank].decoder.final_layernorm.weight.data.copy_(final_norm_weight)
                if md.norm_has_bias:
                    models[tp_rank].decoder.final_layernorm.bias.data.copy_(final_norm_bias)
                if pp_rank != 0 and not md.output_layer:
                    # Copy word embeddings to final pipeline rank
                    models[tp_rank].word_embeddings.weight.data.copy_(out_word_embed[tp_rank])
            del final_norm_weight
            if md.norm_has_bias:
                del final_norm_bias
            check_message(msg)

            if md.output_layer:
                msg = queue_get("output layer")
                if not hasattr(models[0], 'output_layer'):
                    print("ERROR: got an output layer, but model does not have one")
                    exit(1)
                output_layer_weight = torch.chunk(msg.pop("weight"), args.target_tensor_parallel_size, dim=0)
                for tp_rank in range(args.target_tensor_parallel_size):
                    models[tp_rank].output_layer.weight.data.copy_(output_layer_weight[tp_rank])
                del output_layer_weight
                check_message(msg)

            msg = queue_get()
            if msg != "done" and msg["name"] == "pooler":
                if not hasattr(models[0], 'pooler'):
                    print("ERROR: got a pooler, but model does not have one")
                    exit(1)
                print("received pooler")
                pooler_weight = msg.pop("weight")
                pooler_bias = msg.pop("bias")
                for tp_rank in range(args.target_tensor_parallel_size):
                    models[tp_rank].pooler.dense.weight.data.copy_(pooler_weight)
                    models[tp_rank].pooler.dense.bias.data.copy_(pooler_bias)
                del pooler_weight
                del pooler_bias
                check_message(msg)
                msg = queue_get()

            if msg != "done" and msg["name"] == "lm head":
                if not hasattr(models[0], 'lm_head'):
                    print("ERROR: got an lm head, but model does not have one")
                    exit(1)
                print("received lm head")
                lm_head_dense_weight = msg.pop("dense weight")
                lm_head_dense_bias = msg.pop("dense bias")
                lm_head_norm_weight = msg.pop("norm weight")
                if md.norm_has_bias:
                    lm_head_norm_bias = msg.pop("norm bias")
                for tp_rank in range(args.target_tensor_parallel_size):
                    models[tp_rank].lm_head.dense.weight.data.copy_(lm_head_dense_weight)
                    models[tp_rank].lm_head.dense.bias.data.copy_(lm_head_dense_bias)
                    models[tp_rank].lm_head.norm.weight.data.copy_(lm_head_norm_weight)
                    if md.norm_has_bias:
                        models[tp_rank].lm_head.norm.bias.data.copy_(lm_head_norm_bias)
                check_message(msg)
                msg = queue_get()

            if msg != "done" and msg["name"] == "binary head":
                if not hasattr(models[0], 'binary_head'):
                    print("ERROR: got a binary head, but model does not have one")
                    exit(1)
                print("received binary head")
                binary_head_weight = msg.pop("weight")
                binary_head_bias = msg.pop("bias")
                for tp_rank in range(args.target_tensor_parallel_size):
                    models[tp_rank].binary_head.weight.data.copy_(binary_head_weight)
                    models[tp_rank].binary_head.bias.data.copy_(binary_head_bias)
                check_message(msg)
                msg = queue_get()

            if msg != "done":
                print("ERROR: got some more data but was expecting to be done")

        for tp_rank in range(args.target_tensor_parallel_size):
            mpu.set_tensor_model_parallel_rank(tp_rank)
            save_checkpoint(md.iteration, [models[tp_rank]], None, None)


def add_arguments(parser):
    group = parser.add_argument_group(title='M-Core saver')

    group.add_argument('--megatron-path', type=str, default=None,
                       help='Base directory of Megatron repository')

    group.add_argument('--target-tensor-parallel-size', type=int,
                       help='Target tensor model parallel size, defaults to the tensor parallel size '
                       'in the input checkpoint if provided by the loader, otherwise to 1')
    group.add_argument('--target-pipeline-parallel-size', type=int,
                       help='Target tensor model parallel size, default to the pipeline parall size '
                       'in the input checkpoint if provided by the loader, otherwise to 1')
    # >>>
    # group.add_argument('--transformer-impl', default='local',
    group.add_argument('--transformer-impl', required=True,
                       choices=['local', 'transformer_engine'],
                       help='Which Transformer implementation to use.')
    # <<<


def save_checkpoint(queue, args):

    # Search in directory above this
    sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     os.path.pardir,
                     os.path.pardir)))
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)

    try:
        from megatron.arguments import (parse_args, validate_args)
        from megatron.checkpointing import save_checkpoint
        from megatron.global_vars import set_global_variables, get_args
        from megatron.core.enums import ModelType
        from megatron.tokenizer.tokenizer import _vocab_size_with_padding
        from megatron import fused_kernels
        from megatron.core import mpu
    except ModuleNotFoundError:
        print("Unable to import Megatron, please specify the path to Megatron using --megatron-path. Exiting.")
        exit(1)

    def queue_get(name=None):
        val = queue.get()
        if val == "exit":
            print("Loader exited, exiting saver")
            exit(1)
        if name is not None and args.checking and val["name"] != name:
            val_name = val["name"]
            print(f'Unexpected message. Expecting "{name}" but got "{val_name}". Exiting saver.')
            exit(1)
        if name is not None:
            print(f"received {name}")
        return val

    def check_message(msg):
        if not args.checking:
            return
        msg_name = msg.pop("name")
        if len(msg.keys()) > 0:
            print(f"Unexpected values in {msg_name}:")
            for key in msg.keys():
                print(f"   {key}")
            print(f"Exiting. If you want to ignore this, use the argument --no-checking.")
            exit(1)


    md = queue_get()

    if args.target_tensor_parallel_size is None:
        if hasattr(md, 'previous_tensor_parallel_size'):
            args.target_tensor_parallel_size = md.previous_tensor_parallel_size
        else:
            print("loader did not provide a tensor parallel size and --target-tensor-parallel-size not provided on command line. "
                  "Default to 1.")
            args.target_tensor_parallel_size = 1

    if args.target_pipeline_parallel_size is None:
        if hasattr(md, 'previous_pipeline_parallel_size'):
            args.target_pipeline_parallel_size = md.previous_pipeline_parallel_size
        else:
            print("loader did not provide a pipeline parallel size and --target-pipeline-parallel-size not provided on command line. "
                  "Default to 1.")
            args.target_pipeline_parallel_size = 1


    # Arguments do sanity checks on the world size, but we don't care,
    # so trick it into thinking we are plenty of processes
    if args.target_tensor_parallel_size is not None and args.target_pipeline_parallel_size is not None:
        os.environ["WORLD_SIZE"] = f'{args.target_tensor_parallel_size * args.target_pipeline_parallel_size}'

    # We want all arguments to come from us
    sys.argv = ['script.py',
                '--num-layers', str(md.num_layers),
                '--hidden-size', str(md.hidden_size),
                '--seq-length', str(md.seq_length),
                '--num-attention-heads', str(md.num_attention_heads),
                '--max-position-embeddings', str(md.max_position_embeddings),
                '--position-embedding-type', str(md.position_embedding_type),
                '--tokenizer-type', str(md.tokenizer_type),
                '--tensor-model-parallel-size', str(args.target_tensor_parallel_size),
                '--pipeline-model-parallel-size', str(args.target_pipeline_parallel_size),
                '--no-masked-softmax-fusion',
                '--no-bias-gelu-fusion',
                '--no-bias-dropout-fusion',
                '--no-async-tensor-model-parallel-allreduce',
                '--use-cpu-initialization',
                '--micro-batch-size', '1',
                '--no-load-optim',
                '--no-load-rng',
                '--no-save-optim',
                '--no-save-rng',
                '--no-initialization',
                '--save-interval', '1',
                '--save', args.save_dir
                ]

    if md.make_vocab_size_divisible_by is not None:
        sys.argv.extend(['--make-vocab-size-divisible-by', str(md.make_vocab_size_divisible_by)])
    if md.params_dtype == torch.float16:
        sys.argv.append('--fp16')
    elif md.params_dtype == torch.bfloat16:
        sys.argv.append('--bf16')

    if md.output_layer:
        sys.argv.append('--untie-embeddings-and-output-weights')
    if not md.linear_bias:
        sys.argv.append('--disable-bias-linear')

    if md.model_type == 'BERT' and not md.bert_binary_head:
        sys.argv.append('--bert-no-binary-head')

    margs = parse_args()

    # >>>
    # pax({"perf init": margs.perform_initialization})
    # <<<

    if hasattr (md, 'checkpoint_args'):
        # These are arguments that we are either changing, or cause problems for validation if they are set
        # Note that some of these deal with T5 so will need to be changed if we support T5.
        args_to_keep = ['tensor_model_parallel_size', 'pipeline_model_parallel_size', 'world_size', 'params_dtype',
                        'num_layers_per_virtual_pipeline_stage', 'virtual_pipeline_model_parallel_size',
                        'masked_softmax_fusion', 'bias_gelu_fusion', 'bias_dropout_fusion',
                        'sequence_parallel', 'async_tensor_model_parallel_allreduce',
                        # >>>
                        'no_load_optim', 'no_load_rng', 'no_save_optim', 'no_save_rng',
                        # <<<
                        'vocab_file', 'tokenizer_model',
                        'save_interval', 'save',
                        'perform_initialization', 'use_cpu_initialization',
                        'recompute_granularity', 'recompute_num_layers', 'recompute_method',
                        'encoder_num_layers', 'encoder_seq_length',
                        'distribute_saved_activations',
                        'train_iters', 'lr_decay_iters', 'lr_warmup_iters', 'lr_warmup_fraction',
                        'start_weight_decay', 'end_weight_decay']

        for arg, value in vars(md.checkpoint_args).items():
            if arg in args_to_keep:
                continue
            if not hasattr(margs, arg):
                print(f"Checkpoint had argument {arg} but new arguments does not have this.")
                continue
            if getattr(margs, arg) != value:
                print(f"Overwriting default {arg} value {getattr(margs, arg)} with value from checkpoint {value}.")
                setattr(margs, arg, value)

    validate_args(margs)

    # Use M-core models & unset loaded paths.
    margs.use_mcore_models = True
    margs.blendable_index_path = None
    margs.data_path = []
    margs.load = None
    margs.save = args.save_dir
    margs.tensorboard_dir = None
    margs.tokenizer_model = None
    # >>>
    # margs.transformer_impl = "transformer_engine"
    margs.transformer_impl = args.transformer_impl
    # margs.no_load_optim = False
    # margs.no_load_rng = False
    # <<<

    set_global_variables(margs, build_tokenizer=False)

    # margs = megatron args
    margs = get_args()
    # >>>
    # pax({"transformer_impl": margs.transformer_impl})
    # pax({"perf init": margs.perform_initialization})
    # <<<

    if hasattr(md, 'consumed_train_samples'):
        margs.consumed_train_samples = md.consumed_train_samples
        margs.consumed_valid_samples = md.consumed_valid_samples
        print(f"Setting consumed_train_samples to {margs.consumed_train_samples}"
              f" and consumed_valid_samples to {margs.consumed_valid_samples}")
    else:
        print("consumed_train_samples not provided.")

    # Determine how to make our models
    if md.model_type == 'GPT':
        from pretrain_gpt import model_provider
        margs.model_type = ModelType.encoder_or_decoder
    elif md.model_type == 'BERT':
        from pretrain_bert import model_provider
        margs.model_type = ModelType.encoder_or_decoder
    else:
        raise Exception(f'unrecognized model type: {args.model_type}')

    # fake initializing distributed
    mpu.set_tensor_model_parallel_world_size(args.target_tensor_parallel_size)
    mpu.set_pipeline_model_parallel_world_size(args.target_pipeline_parallel_size)
    mpu.set_tensor_model_parallel_rank(0)
    mpu.set_pipeline_model_parallel_rank(0)
    fused_kernels.load(margs)

    # Embeddings
    #-----------
    embeddings_msg = queue_get("embeddings")

    pos_embed = None
    if md.position_embedding_type == 'learned_absolute':
        pos_embed = embeddings_msg.pop("position embeddings")
    orig_word_embed = embeddings_msg.pop("word embeddings")
    check_message(embeddings_msg)

    # Deal with padding
    if md.true_vocab_size is not None:
        # figure out what our padded vocab size is
        orig_vocab_size = orig_word_embed.shape[0]
        margs.padded_vocab_size = _vocab_size_with_padding(md.true_vocab_size, margs)

        # Cut out extra padding we don't need
        if orig_vocab_size > margs.padded_vocab_size:
            full_word_embed = orig_word_embed[0:margs.padded_vocab_size,:]

        # Expanding embedding to larger size by replicating final entry
        elif orig_vocab_size < margs.padded_vocab_size:
            padding_size = margs.padded_vocab_size - orig_vocab_size

            full_word_embed = torch.cat((
                orig_word_embed,
                orig_word_embed[-1].unsqueeze(0).expand(padding_size, -1)))

        # Same size!
        else:
            full_word_embed = orig_word_embed
    else:
        print("Original vocab size not specified, leaving embedding table as-is. "
              "If you've changed the tensor parallel size this could cause problems.")
        margs.padded_vocab_size = orig_word_embed.shape[0]
        full_word_embed = orig_word_embed

    # Split into new tensor model parallel sizes
    out_word_embed = torch.chunk(full_word_embed, args.target_tensor_parallel_size, dim=0)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    if margs.transformer_impl == "local":
        # receive_local_modules(mpu, out_word_embed)
        setter = MCoreLocalSetter
    elif margs.transformer_impl == "transformer_engine":
        # receive_te_modules(mpu, out_word_embed)
        setter = MCoreTESetter
    else:
        raise Exception(f"specialize for transformer_impl '{margs.transformer_impl}'.")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Make models for first pipeline stage and fill in embeddings
    mpu.set_pipeline_model_parallel_rank(0)
    post_process = args.target_pipeline_parallel_size == 1
    models = get_models(args.target_tensor_parallel_size, md.params_dtype, True, post_process)

    for tp_rank, model in enumerate(models):
        if pos_embed is None:
            assert not setter.has_position_embeddings()
        setter.set_embeddings(model, {
            "word" : out_word_embed[tp_rank],
            "position" : pos_embed,
        })

    # Transformer layers
    #-------------------
    total_layer_num = 0
    for pp_rank in range(args.target_pipeline_parallel_size):
        # For later pipeline parallel ranks, make the new models
        if pp_rank > 0:
            mpu.set_pipeline_model_parallel_rank(pp_rank)
            post_process = pp_rank == args.target_pipeline_parallel_size - 1
            models = get_models(args.target_tensor_parallel_size, md.params_dtype, False, post_process)

        for layer in range(len(models[0].decoder.layers)):
            msg = queue_get(f"transformer layer {total_layer_num}")

            # duplicated tensors
            input_norm_weight = msg.pop("input norm weight")
            if md.norm_has_bias:
                input_norm_bias = msg.pop("input norm bias")
            post_norm_weight = msg.pop("post norm weight")
            if md.norm_has_bias:
                post_norm_bias = msg.pop("post norm bias")
            if md.linear_bias:
                dense_bias = msg.pop("dense bias")
                mlp_l1_bias = msg.pop("mlp l1 bias")

            # Split up the parallel tensors
            qkv_weight = torch.chunk(msg.pop("qkv weight"), args.target_tensor_parallel_size, dim=0)
            dense_weight = torch.chunk(msg.pop("dense weight"), args.target_tensor_parallel_size, dim=1)
            mlp_l1_weight = torch.chunk(msg.pop("mlp l1 weight"), args.target_tensor_parallel_size, dim=1)

            # Special handling for swiglu
            if md.swiglu:
                mlp_l0_weight_W = torch.chunk(msg.pop("mlp l0 weight W"), args.target_tensor_parallel_size, dim=0)
                mlp_l0_weight_V = torch.chunk(msg.pop("mlp l0 weight V"), args.target_tensor_parallel_size, dim=0)
                mlp_l0_weight = [torch.cat(weights, dim=0) for weights in zip(mlp_l0_weight_W, mlp_l0_weight_V)]
            else:
                mlp_l0_weight = torch.chunk(msg.pop("mlp l0 weight"), args.target_tensor_parallel_size, dim=0)

            if md.linear_bias:
                qkv_bias = torch.chunk(msg.pop("qkv bias"), args.target_tensor_parallel_size, dim=0)
                if md.swiglu:
                    mlp_l0_bias_W = torch.chunk(msg.pop("mlp l0 bias W"), args.target_tensor_parallel_size, dim=0)
                    mlp_l0_bias_V = torch.chunk(msg.pop("mlp l0 bias V"), args.target_tensor_parallel_size, dim=0)
                    mlp_l0_bias = [torch.cat(bias, dim=0) for bias in zip(mlp_l0_bias_W, mlp_l0_bias_V)]
                else:
                    mlp_l0_bias = torch.chunk(msg.pop("mlp l0 bias"), args.target_tensor_parallel_size, dim=0)

            # Save them to the model
            for tp_rank in range(args.target_tensor_parallel_size):
                l = models[tp_rank].decoder.layers[layer]
                params_dict = {
                    "self_attn_norm_weight" : input_norm_weight,
                    "self_attn_qkv_weight" : qkv_weight[tp_rank],
                    "self_attn_proj_weight" : dense_weight[tp_rank],
                    "mlp_norm_weight" : post_norm_weight,
                    "mlp_fc1_weight" : mlp_l0_weight[tp_rank],
                    "mlp_fc2_weight" : mlp_l1_weight[tp_rank],
                }
                if md.norm_has_bias:
                    params_dict.update({
                        "self_attn_norm_bias" :
                        input_norm_bias if md.norm_has_bias else None,
                        "mlp_norm_bias" :
                        post_norm_bias if md.norm_has_bias else None,
                    })
                if md.linear_bias:
                    params_dict.update({
                        "self_attn_qkv_bias" : qkv_bias[tp_rank],
                        "self_attn_proj_bias" : dense_bias,
                        "mlp_fc1_bias" : mlp_l0_bias[tp_rank],
                        "mlp_fc2_bias" : mlp_l1_bias,
                    })
                setter.set_layer(l, params_dict)

            total_layer_num = total_layer_num + 1
            check_message(msg)


        if post_process:
            msg = queue_get("final norm")
            final_norm_weight = msg.pop("weight")
            if md.norm_has_bias:
                final_norm_bias = msg.pop("bias")
            # >>>
            # for tp_rank in range(args.target_tensor_parallel_size):
            for tp_rank, model in enumerate(models):
            # <<<
                # >>>
                # models[tp_rank].decoder.final_layernorm.weight.data.copy_(final_norm_weight)
                # if md.norm_has_bias:
                #     models[tp_rank].decoder.final_layernorm.bias.data.copy_(final_norm_bias)
                # +++
                setter.set_final_norm(model, {
                    "weight" : final_norm_weight,
                    "bias" : final_norm_bias if md.norm_has_bias else None,
                })
                # <<<
                if pp_rank != 0 and not md.output_layer:
                    # Copy word embeddings to final pipeline rank
                    # >>>
                    # models[tp_rank].word_embeddings.weight.data.copy_(out_word_embed[tp_rank])
                    setter.set_output_word_embeddings(model, out_word_embed[tp_rank])
                    # <<<
            del final_norm_weight
            if md.norm_has_bias:
                del final_norm_bias
            check_message(msg)

            if md.output_layer:
                msg = queue_get("output layer")
                if not hasattr(models[0], 'output_layer'):
                    print("ERROR: got an output layer, but model does not have one")
                    exit(1)
                output_layer_weight = torch.chunk(msg.pop("weight"), args.target_tensor_parallel_size, dim=0)
                # >>>
                # for tp_rank in range(args.target_tensor_parallel_size):
                #     models[tp_rank].output_layer.weight.data.copy_(output_layer_weight[tp_rank])
                # +++
                for tp_rank, model in enumerate(models):
                    setter.set_output_layer(model, output_layer_weight[tp_rank])
                # <<<
                del output_layer_weight
                check_message(msg)

            msg = queue_get()
            if msg != "done" and msg["name"] == "pooler":
                # >>>
                raise Exception("hi.")
                # <<<
                if not hasattr(models[0], 'pooler'):
                    print("ERROR: got a pooler, but model does not have one")
                    exit(1)
                print("received pooler")
                pooler_weight = msg.pop("weight")
                pooler_bias = msg.pop("bias")
                for tp_rank in range(args.target_tensor_parallel_size):
                    models[tp_rank].pooler.dense.weight.data.copy_(pooler_weight)
                    models[tp_rank].pooler.dense.bias.data.copy_(pooler_bias)
                del pooler_weight
                del pooler_bias
                check_message(msg)
                msg = queue_get()

            if msg != "done" and msg["name"] == "lm head":
                # >>>
                raise Exception("hi.")
                # <<<
                if not hasattr(models[0], 'lm_head'):
                    print("ERROR: got an lm head, but model does not have one")
                    exit(1)
                print("received lm head")
                lm_head_dense_weight = msg.pop("dense weight")
                lm_head_dense_bias = msg.pop("dense bias")
                lm_head_norm_weight = msg.pop("norm weight")
                if md.norm_has_bias:
                    lm_head_norm_bias = msg.pop("norm bias")
                for tp_rank in range(args.target_tensor_parallel_size):
                    models[tp_rank].lm_head.dense.weight.data.copy_(lm_head_dense_weight)
                    models[tp_rank].lm_head.dense.bias.data.copy_(lm_head_dense_bias)
                    models[tp_rank].lm_head.norm.weight.data.copy_(lm_head_norm_weight)
                    if md.norm_has_bias:
                        models[tp_rank].lm_head.norm.bias.data.copy_(lm_head_norm_bias)
                check_message(msg)
                msg = queue_get()

            if msg != "done" and msg["name"] == "binary head":
                # >>>
                raise Exception("hi.")
                # <<<
                if not hasattr(models[0], 'binary_head'):
                    print("ERROR: got a binary head, but model does not have one")
                    exit(1)
                print("received binary head")
                binary_head_weight = msg.pop("weight")
                binary_head_bias = msg.pop("bias")
                for tp_rank in range(args.target_tensor_parallel_size):
                    models[tp_rank].binary_head.weight.data.copy_(binary_head_weight)
                    models[tp_rank].binary_head.bias.data.copy_(binary_head_bias)
                check_message(msg)
                msg = queue_get()

            if msg != "done":
                print("ERROR: got some more data but was expecting to be done")

        for tp_rank in range(args.target_tensor_parallel_size):
            mpu.set_tensor_model_parallel_rank(tp_rank)
            save_checkpoint(md.iteration, [models[tp_rank]], None, None)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    print("Done!")
