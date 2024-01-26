# lawrence mcafee

# ~~~~~~~~ import ~~~~~~~~
import torch

from megatron import get_args # , initialize_megatron
from megatron.checkpointing import load_checkpoint
from megatron.core.enums import ModelType
from megatron.training import (
    # build_train_valid_test_datasets,
    # build_train_valid_test_data_loaders,
    build_train_valid_test_data_iterators,
    setup_model_and_optimizer, # get_model,
    update_train_iters,
)
from pretrain_gpt import (
    get_batch as _get_batch,
    model_provider,
    train_valid_test_datasets_provider,
)
# from pretrain_retro import core_model_provider, default_model_provider

from lutil import pax, tp

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
# def print_model_with_params(key, model, depth=0):
def print_model(key, model, depth=0):
    if depth == 0:
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("%s%s%s" % (
        "  " * depth,
        "" if key is None else f"({key}) ",
        type(model).__name__,
    ))
    for k, p in model.named_parameters(recurse=False):
        print("%s* %s : %s ... [%s]." % (
            "  " * (depth + 1),
            k,
            list(p.shape),
            # ",".join(map(str, p.view(-1)[None:None:p.numel()//4].tolist())),
            tp(p),
        ))
    for k, m in model.named_children():
        print_model(k, m, depth + 1)
    if depth == 0:
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("%s nparams : %d." % (key, sum(t.numel() for t in model.parameters())))
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

def compare_top_nparams(key, default_module, core_module):
    get_nparams = lambda m : "--" if m is None else sum(t.numel() for t in m.parameters())
    # >>>
    # get_param_shapes = lambda m : "--" if m is None else ", ".join(str(tuple(p.shape)) for p in m.parameters())
    get_param_shapes = lambda m : "--"
    # <<<
    # get_param_shapes = lambda m : "--" if m is None else "-some-"
    default_nparams = get_nparams(default_module)
    core_nparams = get_nparams(core_module)
    print("%10s : d %10s, c %10s ... %s ---- d %s, c %s." % (
        key,
        default_nparams,
        core_nparams,
        default_nparams - core_nparams if isinstance(default_nparams, int) and isinstance(core_nparams, int) else "--",
        get_param_shapes(default_module),
        get_param_shapes(core_module),
    ))

def compare_preprocess_nparams(default_model, core_model, batch):
    default_embedding = default_model.language_model.embedding
    core_embedding = core_model.embedding
    compare_top_nparams("emb", default_embedding, core_embedding)

    # args = get_args()
    # input_ids = torch.randint(
    #     low=0,
    #     high=10,
    #     size=(args.micro_batch_size, args.seq_length),
    #     dtype=torch.long,
    # )
    # position_ids = torch.randint(
    #     low=0,
    #     high=10,
    #     size=(args.micro_batch_size, args.hidden_size),
    #     dtype=torch.long,
    # )
    input_ids = batch["input_ids"]
    position_ids = batch["position_ids"]

    default_output = default_embedding(input_ids, position_ids)
    core_output = core_embedding(input_ids, position_ids)
    batch["hidden_states"] = default_output

    # >>>
    # print_model("default emb", default_embedding)
    # print_model("core emb", core_embedding)
    # pax("input_ids, position_ids, default_output, core_output, batch")
    # exit()
    # <<<

    # pax({
    #     "default_embedding" : type(default_embedding).__name__,
    #     "core_embedding" : type(core_embedding).__name__,
    # })

def compare_layer_nparams(key, layer_idx, default_layers, core_layers, batch):

    default_layer = default_layers[layer_idx]
    core_layer = core_layers[layer_idx]

    hidden_states = batch["hidden_states"]
    attention_mask = batch["attention_mask"]
    default_output = default_layer(hidden_states, attention_mask)
    core_output = core_layer(hidden_states, attention_mask)
    batch["hidden_states"] = default_output

    # >>>
    # print_model("default layer", default_layer)
    # print_model("core layer", core_layer)
    if layer_idx == 0:
        pax(
            {"layer_idx": f"{layer_idx} / {len(default_layers)}"},
            "default_output, core_output",
        )
    return
    # <<<

    compare_top_nparams(
        f"{key} {layer_idx} / pre sattn norm",
        default_layer.input_norm,
        core_layer.input_layernorm,
    )
    compare_top_nparams(
        f"{key} {layer_idx} /      self attn",
        default_layer.self_attention,
        core_layer.self_attention,
    )
    compare_top_nparams(
        f"{key} {layer_idx} / pre cattn norm",
        default_layer.post_attention_norm,
        core_layer.pre_cross_attn_layernorm,
    )
    compare_top_nparams(
        f"{key} {layer_idx} /     cross attn",
        default_layer.inter_attention,
        core_layer.cross_attention,
    )
    compare_top_nparams(
        f"{key} {layer_idx} /   pre mlp norm",
        default_layer.post_inter_attention_norm,
        core_layer.pre_mlp_layernorm,
    )
    compare_top_nparams(
        f"{key} {layer_idx} /            mlp",
        default_layer.mlp,
        core_layer.mlp,
    )
    compare_top_nparams(
        f"{key} {layer_idx} /      retriever",
        default_layer.retriever,
        None,
    )

    # pax({
    #     "default children" : list(dict(default_layer.named_children()).keys()),
    #     "core children" : list(dict(core_layer.named_children()).keys()),
    # })

    # compare_top_nparams(f"{key} {layer_idx}", default_layer, core_layer)

def compare_block_nparams(key, default_layers, core_layers, batch):
    assert len(default_layers) == len(core_layers)
    for i in range(len(default_layers)):
        compare_top_nparams(
            f"{key} block / {i}",
            default_layers[i],
            core_layers[i],
        )
        # compare_layer_nparams(
        #     f"{key} block / {i}",
        #     i,
        #     default_layers,
        #     core_layers,
        #     batch,
        # )

# def get_default_and_core_models():

#     # model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
#     #     model_provider, model_type)
#     return [
#         get_model(fn, ModelType.encoder_or_decoder)[0].module.module
#         for fn in (default_model_provider, core_model_provider)
#     ]
#     # unwrapped_model = unwrap_model(model)
def get_default_and_core_models():
    args = get_args()
    args.model_type = ModelType.encoder_or_decoder
    args.retro_add_retriever = False
    args.exit_on_missing_checkpoint = True
    args.no_load_optim = True
    args.no_load_rng = True

    args.use_mcore_models = False
    args.load="/lustre/fs3/portfolios/adlr/projects/adlr_nlp_arch/adlr_nlp_sharing/nvllm-1.1t/checkpoints/gpt3-843m-multi-1.1t-gtc-llr"
    # default_model = model_provider()
    # default_model = get_model(model_provider, args.model_type)
    default_model, _, _ = setup_model_and_optimizer(model_provider, args.model_type)
    load_checkpoint(default_model, None, None)

    args.use_mcore_models = True
    args.load="/lustre/fs6/portfolios/adlr/users/lmcafee/retro/megatrons/core-converter/scripts/checkpoints/843m"
    # core_model = model_provider()
    # core_model = get_model(model_provider, args.model_type)
    core_model, _, _ = setup_model_and_optimizer(model_provider, args.model_type)
    load_checkpoint(core_model, None, None)

    default_model = default_model[0].module.module
    core_model = core_model[0].module.module

    # pax({
    #     "default_model" : type(default_model).__name__,
    #     "core_model" : type(core_model).__name__,
    # })

    return default_model, core_model

def get_batch():
    args = get_args()
    update_train_iters(args)
    # train_ds, valid_ds, test_ds = build_train_valid_test_datasets(train_valid_test_datasets_provider)
    # train_loader, valid_loader, test_loader = build_train_valid_test_data_loaders(train_valid_test_datasets_provider)
    train_iter, valid_iter, test_iter = build_train_valid_test_data_iterators(train_valid_test_datasets_provider)

    # batch = list(train_loader)[0]
    # for batch in train_loader:
    #     break
    # batch = _get_batch(train_iter)
    tokens, labels, loss_mask, attention_mask, position_ids = \
        _get_batch(train_iter)
    batch = {
        "input_ids" : tokens,
        "position_ids" : position_ids,
        "attention_mask" : attention_mask,
        "labels" : labels,
        "loss_mask" : loss_mask,
    }

    # output_tensor = model(tokens, position_ids, attention_mask,
    #                       labels=labels)

    # pax("batch")

    # pax("train_loader, valid_loader, test_loader")

    # x = torch.rand((args.seq_length, args.micro_batch_size, args.hidden_size), dtype=torch.

    return batch

# def compare_models():
def compare_gpt_models():

    args = get_args()

    default_model, core_model = get_default_and_core_models()
    batch = get_batch()

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(default_model)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(core_model)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    default_layers = list(default_model.language_model.encoder.layers)
    core_layers = list(core_model.decoder.layers)

    compare_preprocess_nparams(default_model, core_model, batch)
    compare_block_nparams("decoder", default_layers, core_layers, batch)
    # pax("batch")
    # compare_layer_nparams("decoder layer", 5, default_layers, core_layers) # 5, 8
    compare_top_nparams("model", default_model, core_model)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    print_model("default final norm", default_model.language_model.encoder.final_norm)
    print_model("core final norm", core_model.decoder.final_layernorm)
    print_model("default final layer", default_model.language_model.output_layer)
    print_model("core final layer", core_model.output_layer)
    exit()

    pax(
        # "default_model, core_model",
        {
            "n default" : len(list(default_model.parameters())),
            "n core" : len(list(core_model.parameters())),
            "d children" : dict(default_model.named_children()),
            "c children" : dict(core_model.named_children()),
        },
    )

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# if __name__ == "__main__":

#     # Initalize and get arguments, timers, and Tensorboard writer.
#     # initialize_megatron(extra_args_provider=extra_args_provider,
#     #                     args_defaults=args_defaults)
#     initialize_megatron()
#     # Set pytorch JIT layer fusion options and warmup JIT functions.
#     set_jit_fusion_options()

#     # args = get_args()
#     # timers = get_timers()

#     mlm_model = get_mlm_model()
#     core_model = get_core_model()

#     # # Model, optimizer, and learning rate.
#     # timers('model-and-optimizer-setup', log_level=0).start(barrier=True)
#     # model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
#     #     model_provider, model_type)
#     # timers('model-and-optimizer-setup').stop()
#     # print_datetime('after model, optimizer, and learning rate '
#     #                'scheduler are built')
#     # config = get_model_config(model[0])

# eof
