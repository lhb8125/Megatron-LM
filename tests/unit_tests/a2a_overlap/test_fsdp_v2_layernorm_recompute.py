# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import gc
import traceback

import pytest
import torch

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallel
from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.dp_buffer import DataParallelBuffer
from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.fsdp_module import _FSDPRootContext
from megatron.core.enums import Fp8Recipe
from megatron.core.pipeline_parallel.utils import set_streams
from megatron.core.tensor_parallel.random import CheckpointWithoutOutput
from megatron.core.transformer import TransformerLayer
from megatron.core.utils import is_te_min_version
from tests.unit_tests.a2a_overlap.utils import (
    build_gpt_model,
    build_input_data,
    deterministic_mode,
    get_test_config,
    get_valid_flex_dispatcher_backend,
    get_valid_fp8_flags,
    overlap_train_step,
)
from tests.unit_tests.test_utilities import Utils

SEQ_LEN = 32
VOCAB_SIZE = 128
NUM_MICROBATCHES = 4
LR = 0.01


class TestFSDPV2LayerNormRecompute:
    """Production-shape regression for v2 LayerNorm recompute."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=4,
        )
        set_streams()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not is_te_min_version("2.3.0"), reason="Requires TE >= 2.3.0")
    def test_mxfp8_layernorm_recompute(self, monkeypatch):
        monkeypatch.setenv("MCORE_MOE_ROUTER_INPUT_LIFETIME_CHECK", "1")
        original_checkpoint = CheckpointWithoutOutput.checkpoint
        original_recompute = CheckpointWithoutOutput._recompute
        original_record_stream = DataParallelBuffer.record_unsharded_buffer_stream
        original_get_prefetch = _FSDPRootContext.get_prefetch_next_modules
        recompute_stream_pairs = []
        recorded_consumer_streams = []
        prefetch_directions = []

        def checkpoint_with_stream(checkpoint, *args, **kwargs):
            if checkpoint.debug_name is not None:
                checkpoint._test_forward_stream = torch.cuda.current_stream()
            return original_checkpoint(checkpoint, *args, **kwargs)

        def recompute_with_stream_check(checkpoint, grad):
            if checkpoint.debug_name is not None:
                recompute_stream_pairs.append(
                    (checkpoint._test_forward_stream, torch.cuda.current_stream())
                )
            return original_recompute(checkpoint, grad)

        def record_stream_with_check(buffer, stream):
            recorded_consumer_streams.append(stream)
            return original_record_stream(buffer, stream)

        def get_prefetch_with_check(ctx, module, bwd_pass=False):
            prefetch_directions.append((ctx.backward_phase, bwd_pass))
            return original_get_prefetch(ctx, module, bwd_pass=bwd_pass)

        monkeypatch.setattr(CheckpointWithoutOutput, "checkpoint", checkpoint_with_stream)
        monkeypatch.setattr(CheckpointWithoutOutput, "_recompute", recompute_with_stream_check)
        monkeypatch.setattr(
            DataParallelBuffer, "record_unsharded_buffer_stream", record_stream_with_check
        )
        monkeypatch.setattr(_FSDPRootContext, "get_prefetch_next_modules", get_prefetch_with_check)
        mxfp8_flags = [
            flag
            for flag in get_valid_fp8_flags()
            if flag is not None and flag[1] == Fp8Recipe.mxfp8
        ]
        if not mxfp8_flags:
            pytest.skip("Requires Blackwell with MXFP8 support")

        flex_backend = get_valid_flex_dispatcher_backend()
        if flex_backend != "hybridep":
            pytest.skip("Requires HybridEP support")

        recompute_kwargs = {
            "moe_token_dispatcher_type": "flex",
            "moe_flex_dispatcher_backend": flex_backend,
            "moe_router_topk": 8,
            "moe_router_padding_for_quantization": True,
            "moe_permute_fusion": True,
            "fp8": mxfp8_flags[0][0],
            "fp8_recipe": mxfp8_flags[0][1],
            "overlap_moe_expert_parallel_comm": True,
            "delay_wgrad_compute": True,
            "recompute_granularity": "selective",
            "recompute_modules": ["moe_act", "layernorm"],
        }

        def make_ddp_config():
            return DistributedDataParallelConfig(
                use_megatron_fsdp=True,
                use_megatron_fsdp_v2=True,
                data_parallel_sharding_strategy="optim_grads_params",
                overlap_grad_reduce=True,
                overlap_param_gather=True,
                fp8_param_gather=True,
                megatron_fsdp_main_params_dtype=None,
            )

        try:
            with deterministic_mode():
                data = build_input_data(seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE)
                recompute_config = get_test_config(
                    num_layers=2,
                    extra_kwargs=recompute_kwargs,
                    multi_latent_attention=False,
                    num_attention_heads=8,
                    kv_channels=64,
                )
                recompute_model = build_gpt_model(recompute_config, vocab_size=VOCAB_SIZE)
                recompute_model.bfloat16()
                assert all(
                    layer.recompute_pre_mlp_layernorm for layer in recompute_model.decoder.layers
                )
                recompute_fsdp = FullyShardedDataParallel(
                    config=recompute_config,
                    ddp_config=make_ddp_config(),
                    module=recompute_model,
                    fsdp_unit_modules=[TransformerLayer],
                )
                recompute_opt = torch.optim.SGD(recompute_fsdp.parameters(), lr=LR)

                rank = torch.distributed.get_rank()
                recompute_loss = overlap_train_step(
                    recompute_fsdp,
                    recompute_opt,
                    recompute_config,
                    data,
                    num_microbatches=NUM_MICROBATCHES,
                )
                assert torch.isfinite(
                    recompute_loss
                ), f"[rank {rank}] Non-finite loss: {recompute_loss.item()}"
                for name, param in recompute_fsdp.named_parameters():
                    if param.grad is not None:
                        assert torch.isfinite(
                            param.grad
                        ).all(), f"[rank {rank}] Non-finite gradient: {name}"
                assert recompute_stream_pairs
                assert all(
                    forward_stream == recompute_stream
                    for forward_stream, recompute_stream in recompute_stream_pairs
                ), "LayerNorm recompute must run on its original compute stream"
                assert recorded_consumer_streams
                assert all(
                    recompute_stream in recorded_consumer_streams
                    for _, recompute_stream in recompute_stream_pairs
                ), "LayerNorm recompute stream must own the unsharded weight-buffer lifetime"
                assert any(backward_phase for backward_phase, _ in prefetch_directions)
                assert all(
                    prefetch_bwd_pass
                    for backward_phase, prefetch_bwd_pass in prefetch_directions
                    if backward_phase
                ), "Every backward-phase prefetch must follow backward module order"

                del recompute_fsdp, recompute_opt
                gc.collect()
                torch.cuda.empty_cache()
        except Exception:
            traceback.print_exc()
            raise
