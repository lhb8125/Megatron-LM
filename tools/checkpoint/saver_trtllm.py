# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import os
import sys
import torch
from importlib.metadata import version
from pkg_resources import packaging

from setter import ModelSetter
from utils import print_memory_usage


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
    group.add_argument('--architecture', required=True,
                       choices=['LlamaForCausalLM'],
                       help='Which model structure to convert.')

def quantize(weight, scale):
    return (weight * scale).to(torch.float8_e4m3fn).view(torch.int8)

def save_checkpoint(queue, args):

    # Transformer engine >= 0.12.0, for CPU initialization.
    te_version = packaging.version.Version(version("transformer-engine"))
    assert te_version >= packaging.version.Version("0.12.0"), \
        "transformer engine version: %s (>=0.12.0 required)." % te_version

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
    world_size = 1
    if args.target_tensor_parallel_size is not None and args.target_pipeline_parallel_size is not None:
        os.environ["WORLD_SIZE"] = f'{args.target_tensor_parallel_size * args.target_pipeline_parallel_size}'
        world_size = args.target_tensor_parallel_size * args.target_pipeline_parallel_size

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
                '--save', args.save_dir,
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

    if hasattr (md, 'checkpoint_args'):
        # These are arguments that we are either changing, or cause problems for validation if they are set
        # Note that some of these deal with T5 so will need to be changed if we support T5.
        args_to_keep = ['tensor_model_parallel_size', 'pipeline_model_parallel_size', 'world_size', 'params_dtype',
                        'num_layers_per_virtual_pipeline_stage', 'virtual_pipeline_model_parallel_size',
                        'masked_softmax_fusion', 'bias_gelu_fusion', 'bias_dropout_fusion',
                        'sequence_parallel', 'async_tensor_model_parallel_allreduce',
                        'no_load_optim', 'no_load_rng', 'no_save_optim', 'no_save_rng',
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

    # Explicitly copy sequence_parallel, apply_query_key_layer_scaling.
    margs.sequence_parallel = md.checkpoint_args.sequence_parallel
    margs.apply_query_key_layer_scaling = md.checkpoint_args.apply_query_key_layer_scaling


    ## Use M-core models & unset loaded paths.
    #margs.use_mcore_models = True
    #margs.blendable_index_path = None
    #margs.data_path = []
    #margs.load = None
    #margs.save = args.save_dir
    #margs.tensorboard_dir = None
    #margs.tokenizer_model = None
    #margs.transformer_impl = args.transformer_impl

    #set_global_variables(margs, build_tokenizer=False)

    ## Megatron args. (i.e., 'margs')
    #margs = get_args()

    #if hasattr(md, 'consumed_train_samples'):
    #    margs.consumed_train_samples = md.consumed_train_samples
    #    margs.consumed_valid_samples = md.consumed_valid_samples
    #    print(f"Setting consumed_train_samples to {margs.consumed_train_samples}"
    #          f" and consumed_valid_samples to {margs.consumed_valid_samples}")
    #else:
    #    print("consumed_train_samples not provided.")

    # Determine how to make our models
    #if md.model_type == 'GPT':
    #    from pretrain_gpt import model_provider
    #    margs.model_type = ModelType.encoder_or_decoder
    #else:
    #    raise Exception(f'unrecognized model type: {args.model_type}')

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
    #out_word_embed = torch.chunk(full_word_embed, args.target_tensor_parallel_size, dim=0)
    out_word_embed = full_word_embed

    if margs.fp16:
        dtype = 'float16'
    elif margs.bf16:
        dtype = 'bfloat16'
    else:
        dtype = 'float32'
    num_query_groups = margs.num_query_groups
    heads_per_group = margs.num_attention_heads // num_query_groups
    qkv_total_dim = margs.num_attention_heads + 2 * num_query_groups
    head_size = margs.hidden_size // margs.num_attention_heads

    config = {
        'architecture': args.architecture,
        'dtype': dtype,
        'logits_dtype': 'float32',
        'num_hidden_layers': margs.num_layers,
        'num_attention_heads': margs.num_attention_heads,
        'hidden_size': margs.hidden_size,
        'head_size': head_size,
        'intermediate_size': margs.ffn_hidden_size,
        'num_key_value_heads': margs.num_query_groups,
        'vocab_size': margs.padded_vocab_size,
        'position_embedding_type': 'rope_gpt_neox',
        'max_position_embeddings': margs.max_position_embeddings,
        'hidden_act': 'silu',
        'rotary_base': 10000.0,
        'rotary_scaling': None,
        'norm_epsilon': margs.norm_epsilon,
        'quantization': {
            'quant_algo': "FP8" if md.fp8 else None,
            'kv_cache_quant_algo': None,
            'exclude_modules': ['lm_head'],
        },
        'mapping': {
            'world_size': margs.world_size,
            'tp_size': margs.tensor_model_parallel_size,
            'pp_size': margs.pipeline_model_parallel_size,
        },
        'use_parallel_embedding': False,
        'embedding_sharding_dim': 0,
        'share_embedding_table': not margs.untie_embeddings_and_output_weights,
        'use_prompt_tuning': False,
        'moe_num_experts': 1,
        'moe_top_k': 0,
        'moe_tp_mode': None,
        'moe_normalization_mode': None,
        'enable_pos_shift': False,
        'dense_context_fmha': False,
        'max_lora_rank': 0,
        'lora_target_modules': None,
        'hf_modules_to_trtllm_modules': None,
        'trtllm_modules_to_hf_modules': None,
        'disable_weight_only_quant_plugin': None,
        'attn_bias': False,
        'mlp_bias': False,
    }
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
        import json
        json.dump(config, f, indent=4)

    # Make models for first pipeline stage and fill in embeddings
    #mpu.set_pipeline_model_parallel_rank(0)
    post_process = args.target_pipeline_parallel_size == 1
    #models = get_models(args.target_tensor_parallel_size, md.params_dtype, True, post_process)

    dst_weight = [{} for _ in range(args.target_tensor_parallel_size)]
    # Set embeddings.
    # --------------
    for tp_rank in range(args.target_tensor_parallel_size):
        dst_weight[tp_rank]['transformer.vocab_embedding.weight'] = out_word_embed

    # Transformer layers.
    # ------------------
    total_layer_num = 0
    if args.target_pipeline_parallel_size is not None:
        assert(md.num_layers % args.target_pipeline_parallel_size == 0)
        layers_per_pp_rank = md.num_layers // args.target_pipeline_parallel_size
    else:
        layers_per_pp_rank = md.num_layers
    for pp_rank in range(args.target_pipeline_parallel_size):
        # For later pipeline parallel ranks, make the new models
        if pp_rank > 0:
            dst_weight = [{} for _ in range(args.target_tensor_parallel_size)]
            #mpu.set_pipeline_model_parallel_rank(pp_rank)
            post_process = pp_rank == args.target_pipeline_parallel_size - 1
            #models = get_models(args.target_tensor_parallel_size, md.params_dtype, False, post_process)

        for layer in range(layers_per_pp_rank):
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
            qkv_weight = msg.pop("qkv weight")
            qkv_weight = qkv_weight.reshape([qkv_total_dim, head_size, margs.hidden_size])
            #qkv_weight = torch.chunk(msg.pop("qkv weight"), args.target_tensor_parallel_size, dim=0)
            q_slice = torch.cat(
                [
                    torch.arange((heads_per_group + 2) * i, (heads_per_group + 2) * i + heads_per_group)
                    for i in range(num_query_groups)
                ]
            )
            k_slice = torch.arange(heads_per_group, qkv_total_dim, (heads_per_group + 2))
            v_slice = torch.arange(heads_per_group + 1, qkv_total_dim, (heads_per_group + 2))
            q_weight = qkv_weight[q_slice].reshape(-1, margs.hidden_size).contiguous()
            k_weight = qkv_weight[k_slice].reshape(-1, margs.hidden_size).contiguous()
            v_weight = qkv_weight[v_slice].reshape(-1, margs.hidden_size).contiguous()
            q_weight = torch.chunk(q_weight, args.target_tensor_parallel_size, dim=0)
            k_weight = torch.chunk(k_weight, args.target_tensor_parallel_size, dim=0)
            v_weight = torch.chunk(v_weight, args.target_tensor_parallel_size, dim=0)
            qkv_weight = [torch.cat(weights, dim=0) for weights in zip(q_weight, k_weight, v_weight)]

            dense_weight = list(torch.chunk(msg.pop("dense weight"), args.target_tensor_parallel_size, dim=1))
            mlp_l1_weight = list(torch.chunk(msg.pop("mlp l1 weight"), args.target_tensor_parallel_size, dim=1))

            # Special handling for swiglu
            if md.swiglu:
                mlp_l0_weight_W = list(torch.chunk(msg.pop("mlp l0 weight W"), args.target_tensor_parallel_size, dim=0))
                mlp_l0_weight_V = list(torch.chunk(msg.pop("mlp l0 weight V"), args.target_tensor_parallel_size, dim=0))
                mlp_l0_weight = [torch.cat(weights, dim=0) for weights in zip(mlp_l0_weight_W, mlp_l0_weight_V)]
            else:
                mlp_l0_weight_W = list(torch.chunk(msg.pop("mlp l0 weight"), args.target_tensor_parallel_size, dim=0))

            if md.linear_bias:
                qkv_bias = torch.chunk(msg.pop("qkv bias"), args.target_tensor_parallel_size, dim=0)
                if md.swiglu:
                    mlp_l0_bias_W = torch.chunk(msg.pop("mlp l0 bias W"), args.target_tensor_parallel_size, dim=0)
                    mlp_l0_bias_V = torch.chunk(msg.pop("mlp l0 bias V"), args.target_tensor_parallel_size, dim=0)
                    mlp_l0_bias = [torch.cat(bias, dim=0) for bias in zip(mlp_l0_bias_W, mlp_l0_bias_V)]
                else:
                    mlp_l0_bias = torch.chunk(msg.pop("mlp l0 bias"), args.target_tensor_parallel_size, dim=0)

            # Save them to the model
            prefix = f'transformer.layers.{layer}.'
            # Check out fp8 scaling factors
            if md.fp8:
                qkv_fwd_scale = msg.pop("qkv fwd scale")
                qkv_bwd_scale = msg.pop("qkv bwd scale")
                dense_fwd_scale = msg.pop("dense fwd scale")
                dense_bwd_scale = msg.pop("dense bwd scale")
                mlp_l1_fwd_scale = msg.pop("mlp l1 fwd scale")
                mlp_l1_bwd_scale = msg.pop("mlp l1 bwd scale")
                mlp_l0_fwd_scale = msg.pop("mlp l0 fwd scale")
                mlp_l0_bwd_scale = msg.pop("mlp l0 bwd scale")

            for tp_rank in range(args.target_tensor_parallel_size):
                if md.fp8:
                    dst_weight[tp_rank][f"{prefix}attention.qkv.activation_scaling_factor"] = 1 / qkv_fwd_scale[0].view(1)
                    dst_weight[tp_rank][f"{prefix}attention.qkv.weights_scaling_factor"] = 1 / qkv_fwd_scale[1].view(1)
                    dst_weight[tp_rank][f"{prefix}attention.dense.activation_scaling_factor"] = 1 / dense_fwd_scale[0].view(1)
                    dst_weight[tp_rank][f"{prefix}attention.dense.weights_scaling_factor"] = 1 / dense_fwd_scale[1].view(1)
                    dst_weight[tp_rank][f"{prefix}mlp.fc.activation_scaling_factor"] = 1 / mlp_l0_fwd_scale[0].view(1)
                    dst_weight[tp_rank][f"{prefix}mlp.fc.weights_scaling_factor"] = 1 / mlp_l0_fwd_scale[1].view(1)
                    dst_weight[tp_rank][f"{prefix}mlp.gate.activation_scaling_factor"] = 1 / mlp_l0_fwd_scale[0].clone().view(1)
                    dst_weight[tp_rank][f"{prefix}mlp.gate.weights_scaling_factor"] = 1 / mlp_l0_fwd_scale[1].clone().view(1)
                    dst_weight[tp_rank][f"{prefix}mlp.proj.activation_scaling_factor"] = 1 / mlp_l1_fwd_scale[0].view(1)
                    dst_weight[tp_rank][f"{prefix}mlp.proj.weights_scaling_factor"] = 1 / mlp_l1_fwd_scale[1].view(1)
                    qkv_weight[tp_rank] = quantize(qkv_weight[tp_rank].contiguous(), qkv_fwd_scale[1].view(1))
                    dense_weight[tp_rank] = quantize(dense_weight[tp_rank].contiguous(), dense_fwd_scale[1].view(1))
                    mlp_l0_weight_W[tp_rank] = quantize(mlp_l0_weight_W[tp_rank].contiguous(), mlp_l0_fwd_scale[1].view(1))
                    mlp_l1_weight[tp_rank] = quantize(mlp_l1_weight[tp_rank].contiguous(), mlp_l1_fwd_scale[1].view(1))
                    if md.swiglu:
                        mlp_l0_weight_V[tp_rank] = quantize(mlp_l0_weight_V[tp_rank].contiguous(), mlp_l0_fwd_scale[1].clone().view(1))

                params_dict = {
                    f"{prefix}input_layernorm.weight" : input_norm_weight.contiguous(),
                    f"{prefix}attention.qkv.weight" : qkv_weight[tp_rank].contiguous(),
                    f"{prefix}attention.dense.weight" : dense_weight[tp_rank].contiguous(),
                    f"{prefix}post_layernorm.weight" : post_norm_weight.contiguous(),
                    f"{prefix}mlp.fc.weight" : mlp_l0_weight_W[tp_rank].contiguous(),
                    f"{prefix}mlp.proj.weight" : mlp_l1_weight[tp_rank].contiguous(),
                }
                if md.swiglu:
                    params_dict.update({
                        f"{prefix}mlp.gate.weight": mlp_l0_weight_V[tp_rank].contiguous(),
                    })
                if md.norm_has_bias:
                    params_dict.update({
                        f"{prefix}input_layernorm.bias" :
                        input_norm_bias if md.norm_has_bias else None,
                        f"{prefix}post_layernorm.bias" :
                        post_norm_bias if md.norm_has_bias else None,
                    })
                if md.linear_bias:
                    params_dict.update({
                        f"{prefix}attention.qkv.bias" : qkv_bias[tp_rank].contiguous(),
                        f"{prefix}attention.dense.bias" : dense_bias.contiguous(),
                        f"{prefix}mlp.fc.bias" : mlp_l0_bias[tp_rank].contiguous(),
                        f"{prefix}mlp.proj.bias" : mlp_l1_bias.contiguous(),
                    })
                dst_weight[tp_rank].update(params_dict)
            total_layer_num = total_layer_num + 1
            check_message(msg)


        if post_process:
            msg = queue_get("final norm")
            final_norm_weight = msg.pop("weight")
            if md.norm_has_bias:
                final_norm_bias = msg.pop("bias")
            for tp_rank in range(args.target_tensor_parallel_size):
                dst_weight[tp_rank]['transformer.ln_f.weight'] = final_norm_weight.contiguous()
                if md.norm_has_bias:
                    dst_weight[tp_rank]['transformer.ln_f.bias'] = final_norm_bias.contiguous()
                if pp_rank != 0 and not md.output_layer:
                    # Copy word embeddings to final pipeline rank
                    dst_weight[tp_rank]['transformer.lm_head.weight'] = out_word_embed[tp_rank].contiguous()
            del final_norm_weight
            if md.norm_has_bias:
                del final_norm_bias
            check_message(msg)

            if md.output_layer:
                msg = queue_get("output layer")
                output_layer_weight = torch.chunk(msg.pop("weight"), args.target_tensor_parallel_size, dim=0)
                for tp_rank in range(args.target_tensor_parallel_size):
                    dst_weight[tp_rank]['lm_head.weight'] = output_layer_weight[tp_rank].contiguous()
                del output_layer_weight
                check_message(msg)

        for tp_rank in range(args.target_tensor_parallel_size):
            rank = pp_rank * args.target_tensor_parallel_size + tp_rank
            import safetensors
            safetensors.torch.save_file(
                    dst_weight[tp_rank], os.path.join(args.save_dir, f'rank{rank}.safetensors'))

    print("Done!")
