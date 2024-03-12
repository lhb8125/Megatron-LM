# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import os
import psutil
import sys
import torch
from importlib.metadata import version
from pkg_resources import packaging

from setter import ModelSetter
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import OrderedDict


def add_arguments(parser):
    group = parser.add_argument_group(title='M-Core saver')

    group.add_argument('--megatron-path', type=str, default=None,
                       help='Base directory of Megatron repository')
    group.add_argument('--hf-in-path', type=str, default=None,
                       help="A HF model path, " "e.g. a folder containing https://huggingface.co/meta-llama/Llama-2-7b-hf/tree/main")
    group.add_argument('--hf-out-path', type=str, default=None,
                       help="Output HF model path, " "with the same format as above but user's own weights")


def save_checkpoint(queue, args):

    # Transformer engine >= 0.12.0, for CPU initialization.
    te_version = packaging.version.Version(version("transformer-engine"))
    assert te_version >= packaging.version.Version("0.12.0")

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

    # We want all arguments to come from us
    sys.argv = ['script.py',
                '--num-layers', str(md.num_layers),
                '--hidden-size', str(md.hidden_size),
                '--seq-length', str(md.seq_length),
                '--num-attention-heads', str(md.num_attention_heads),
                '--ffn-hidden-size', str(md.ffn_hidden_size),
                '--num-query-groups', str(md.num_query_groups),
                '--max-position-embeddings', str(md.max_position_embeddings),
                '--position-embedding-type', str(md.position_embedding_type),
                '--tokenizer-type', str(md.tokenizer_type),
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

    # >>>
    margs.sequence_parallel = md.checkpoint_args.sequence_parallel
    # margs.apply_query_key_layer_scaling = md.checkpoint_args.apply_query_key_layer_scaling
    # <<<

    validate_args(margs)

    # Use M-core models & unset loaded paths.
    margs.use_mcore_models = True
    margs.blendable_index_path = None
    margs.data_path = []
    margs.load = None
    margs.save = args.save_dir
    margs.tensorboard_dir = None
    margs.tokenizer_model = None
    margs.wandb_project = None

    set_global_variables(margs, build_tokenizer=False)

    # Megatron args. (i.e., 'margs')
    margs = get_args()

    if hasattr(md, 'consumed_train_samples'):
        margs.consumed_train_samples = md.consumed_train_samples
        margs.consumed_valid_samples = md.consumed_valid_samples
        print(f"Setting consumed_train_samples to {margs.consumed_train_samples}"
              f" and consumed_valid_samples to {margs.consumed_valid_samples}")
    else:
        print("consumed_train_samples not provided.")

    # fake initializing distributed
    #fused_kernels.load(margs)

    # Embeddings
    #-----------

    param_to_weights = lambda param: param.to(torch.bfloat16)

    num_layers = margs.num_layers
    hidden_size = margs.hidden_size
    head_num = margs.num_attention_heads
    ffn_hidden_size = margs.ffn_hidden_size
    #num_query_groups = margs.num_query_groups
    num_query_groups = head_num

    head_size = hidden_size // head_num
    heads_per_group = head_num // num_query_groups
    qkv_total_dim = head_num + 2 * num_query_groups

    checkpoint = OrderedDict()
    # Set embeddings.
    # --------------
    embeddings_msg=queue_get("embeddings")
    embed_weight=embeddings_msg.pop("word embeddings")
    embed_weights_base_name = f'model.embed_tokens.weight'
    checkpoint[embed_weights_base_name] = param_to_weights(embed_weight)
    check_message(embeddings_msg)

    # Transformer layers.
    # ------------------
    total_layer_num = 0
    for l in range(num_layers):
        msg = queue_get(f"transformer layer {l}")
        qkv_weights = msg.pop("qkv weight")
        qkv_weights = qkv_weights.reshape([qkv_total_dim, head_size, hidden_size])

        q_slice = torch.cat(
            [
                torch.arange((heads_per_group + 2) * i, (heads_per_group + 2) * i + heads_per_group)
                for i in range(num_query_groups)
            ]
        )
        k_slice = torch.arange(heads_per_group, qkv_total_dim, (heads_per_group + 2))
        v_slice = torch.arange(heads_per_group + 1, qkv_total_dim, (heads_per_group + 2))

        q_weights_base_name = f'model.layers.{l}.self_attn.q_proj.weight'
        k_weights_base_name = f'model.layers.{l}.self_attn.k_proj.weight'
        v_weights_base_name = f'model.layers.{l}.self_attn.v_proj.weight'

        checkpoint[q_weights_base_name] = param_to_weights(qkv_weights[q_slice].reshape(-1, hidden_size))
        checkpoint[k_weights_base_name] = param_to_weights(qkv_weights[k_slice].reshape(-1, hidden_size))
        checkpoint[v_weights_base_name] = param_to_weights(qkv_weights[v_slice].reshape(-1, hidden_size))

        o_weight = msg.pop("dense weight")
        o_weight_base_name = f'model.layers.{l}.self_attn.o_proj.weight'
        checkpoint[o_weight_base_name] = param_to_weights(o_weight)

        mlp_down_proj_weight = msg.pop("mlp l0 weight W")
        mlp_gate_proj_weight = msg.pop("mlp l0 weight V")

        mlp_down_proj_base_name = f'model.layers.{l}.mlp.gate_proj.weight'
        mlp_gate_proj_base_name = f'model.layers.{l}.mlp.up_proj.weight'

        checkpoint[mlp_down_proj_base_name] = param_to_weights(mlp_down_proj_weight)
        checkpoint[mlp_gate_proj_base_name] = param_to_weights(mlp_gate_proj_weight)

        mlp_up_proj_weight = msg.pop("mlp l1 weight")
        mlp_up_proj_base_name = f'model.layers.{l}.mlp.down_proj.weight'
        checkpoint[mlp_up_proj_base_name] = param_to_weights(mlp_up_proj_weight)

        input_ln_weight = msg.pop("input norm weight")
        input_ln_base_name = f'model.layers.{l}.input_layernorm.weight'
        checkpoint[input_ln_base_name] = param_to_weights(input_ln_weight)

        post_attn_ln_weight = msg.pop("post norm weight")
        post_attn_ln_base_name = f'model.layers.{l}.post_attention_layernorm.weight'
        checkpoint[post_attn_ln_base_name] = param_to_weights(post_attn_ln_weight)

        check_message(msg)

    final_ln_weight = queue_get("final norm").pop("weight")
    final_ln_base_name = f'model.norm.weight'
    checkpoint[final_ln_base_name] = param_to_weights(final_ln_weight)

    output_layer_weight = queue_get("output layer").pop("weight")
    output_layer_base_name = f'lm_head.weight'
    checkpoint[output_layer_base_name] = param_to_weights(output_layer_weight)

    #if msg != "done":
    #    print("ERROR: got some more data but was expecting to be done")

    #os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    #torch.save(checkpoint, args.out_file)
    #logging.info(f"Weights saved to {args.out_file}")
    model = AutoModelForCausalLM.from_pretrained(args.hf_in_path, local_files_only=True, ignore_mismatched_sizes=True)
    tokenizer = AutoTokenizer.from_pretrained(args.hf_in_path)
    model.load_state_dict(checkpoint)
    model.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)
    print(f"Full HF model saved to {args.hf_out_path}")
    print("Done!")
