# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import gc
import traceback
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallel
from megatron.core.distributed.fsdp.src.megatron_fsdp.v2 import fsdp_module as fsdp_module_impl
from megatron.core.distributed.fsdp.src.megatron_fsdp.v2 import hooks as hooks_impl
from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.fsdp_module import (
    FSDPModule,
    _is_fp8_norm_param,
    _is_fp8_router_param,
    _should_defer_fp8_norm_sync,
)
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


def test_async_coalesced_unshard_waits_for_work(monkeypatch):
    order = []
    dp_group = object()

    class FakeCoalescingManager:
        def wait(self):
            order.append("wait")

    class FakeWeightBuffer:
        def __init__(self, name):
            self.name = name

        def unshard(self, bind_params=False):
            assert bind_params
            order.append(f"unshard:{self.name}")

    @contextmanager
    def fake_coalescing_manager(group, async_ops=False):
        assert group is dp_group
        assert async_ops
        yield FakeCoalescingManager()
        order.append("launch")

    monkeypatch.setattr(fsdp_module_impl, "_coalescing_manager", fake_coalescing_manager)
    fsdp_module_impl._unshard_weight_buffers(
        dp_group, [FakeWeightBuffer("a"), FakeWeightBuffer("b")], async_op=True
    )

    assert order == ["unshard:a", "unshard:b", "launch", "wait"]


def test_singleton_unshard_skips_coalescing_manager(monkeypatch):
    order = []

    class FakeWeightBuffer:
        _dp_world_size = 1

        def __init__(self, name):
            self.name = name

        def unshard(self, bind_params=False):
            assert bind_params
            order.append(self.name)

    def unexpected_coalescing_manager(*args, **kwargs):
        raise AssertionError("Singleton unshard must not enter a coalescing manager")

    monkeypatch.setattr(fsdp_module_impl, "_coalescing_manager", unexpected_coalescing_manager)
    fsdp_module_impl._unshard_weight_buffers(
        object(), [FakeWeightBuffer("a"), FakeWeightBuffer("b")], async_op=False
    )

    assert order == ["a", "b"]


@pytest.mark.parametrize(
    ("world_sizes", "expected"), [([1], True), ([1, 1], True), ([1, 2], False), ([], False)]
)
def test_singleton_unshard_detection(world_sizes, expected):
    class FakeWeightBuffer:
        def __init__(self, world_size):
            self._dp_world_size = world_size

    class FakePolicy:
        @staticmethod
        def weight_buffers_for_unshard(model_weight, transpose_weight, *, bwd_pass):
            assert bwd_pass
            return [model_weight, transpose_weight]

    class FakeParamGroup:
        mp_policy = FakePolicy()

        def __init__(self, buffers):
            self.model_weight_buffer = buffers[0] if buffers else None
            self.transpose_weight_buffer = buffers[1] if len(buffers) > 1 else None

    buffers = [FakeWeightBuffer(world_size) for world_size in world_sizes]
    module = type("FakeModule", (), {"_fsdp_param_groups": [FakeParamGroup(buffers)]})()

    assert fsdp_module_impl._uses_singleton_dp_for_unshard(module, bwd_pass=True) is expected


def test_backward_recompute_requests_backward_then_forward_buffers(monkeypatch):
    calls = []

    class FakeParamGroup:
        @staticmethod
        def _maybe_free_grad_data():
            calls.append("free_grad")

    class FakeTarget:
        _fsdp_state = SimpleNamespace(_is_root=False, enable_cuda_graph=False)
        _fsdp_root_context = SimpleNamespace(
            cuda_graph_active=False, backward_phase=True, enable_unshard_prefetch=True
        )
        _fsdp_param_groups = [FakeParamGroup()]

        @staticmethod
        def unshard(**kwargs):
            calls.append(kwargs)

    target = FakeTarget()
    monkeypatch.setattr(hooks_impl, "_find_fsdp_target", lambda _module: target)

    hooks_impl.mfsdp_forward_pre_hook(torch.nn.Identity(), (), {})

    assert calls == [
        {"async_op": True, "bwd_pass": True},
        {"async_op": True, "bwd_pass": False},
        "free_grad",
    ]


@pytest.mark.parametrize(("world_size", "expected"), [(1, True), (2, False), (None, False)])
def test_singleton_reduce_grad_detection(world_size, expected):
    grad_buffer = None
    if world_size is not None:
        grad_buffer = type("FakeGradBuffer", (), {"_dp_world_size": world_size})()
    param_group = type("FakeParamGroup", (), {"main_grad_buffer": grad_buffer})()

    assert fsdp_module_impl._uses_singleton_dp_for_reduce_grad(param_group) is expected


def test_fp8_norm_sync_selector_is_narrow():
    class FakePolicy:
        fp8 = SimpleNamespace(enabled=True)

        @staticmethod
        def is_fp8_param(param):
            return False

        @staticmethod
        def is_nvfp4_param(param):
            return False

    module = torch.nn.Module()
    module.config = SimpleNamespace(hidden_size=16, num_moe_experts=4)
    params = [torch.nn.Parameter(torch.ones(16)), torch.nn.Parameter(torch.ones(16))]
    router = torch.nn.Parameter(torch.ones(4, 16))

    assert _is_fp8_norm_param(
        module, "input_layernorm.weight", params[0], FakePolicy(), "optim_grads_params"
    )
    assert _is_fp8_router_param(
        module, "mlp.router.weight", router, FakePolicy(), "optim_grads_params"
    )

    assert _should_defer_fp8_norm_sync(
        module,
        ["input_layernorm.weight", "pre_mlp_layernorm.weight"],
        params,
        FakePolicy(),
        "optim_grads_params",
    )
    assert _should_defer_fp8_norm_sync(
        module,
        ["input_layernorm.weight", "router.weight"],
        [params[0], router],
        FakePolicy(),
        "optim_grads_params",
    )
    assert not _should_defer_fp8_norm_sync(
        module,
        ["input_layernorm.weight", "mlp.linear_fc1.weight"],
        [params[0], router],
        FakePolicy(),
        "optim_grads_params",
    )
    assert not _should_defer_fp8_norm_sync(
        module,
        ["input_layernorm.weight", "pre_mlp_layernorm.weight"],
        params,
        FakePolicy(),
        "optim_grads",
    )


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
        original_checkpoint = CheckpointWithoutOutput.checkpoint
        original_recompute = CheckpointWithoutOutput._recompute
        recompute_stream_pairs = []

        def checkpoint_with_stream(checkpoint, *args, **kwargs):
            checkpoint._test_forward_stream = torch.cuda.current_stream()
            return original_checkpoint(checkpoint, *args, **kwargs)

        def recompute_with_stream_check(checkpoint, grad):
            if hasattr(checkpoint, "_test_forward_stream"):
                recompute_stream_pairs.append(
                    (checkpoint._test_forward_stream, torch.cuda.current_stream())
                )
            return original_recompute(checkpoint, grad)

        monkeypatch.setattr(CheckpointWithoutOutput, "checkpoint", checkpoint_with_stream)
        monkeypatch.setattr(CheckpointWithoutOutput, "_recompute", recompute_with_stream_check)
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
            "fp8_param": True,
            "overlap_moe_expert_parallel_comm": True,
            "delay_wgrad_compute": True,
            "cuda_graph_impl": "full_iteration",
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
                keep_fp8_transpose_cache=True,
                megatron_fsdp_main_params_dtype=torch.float32,
                # This focused lifecycle test uses native SGD rather than the
                # production precision-aware optimizer, so params/grads must match.
                megatron_fsdp_main_grads_dtype=torch.float32,
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
                deferred_groups = []
                mxfp8_groups = []
                for fsdp_module in recompute_fsdp.modules():
                    if not isinstance(fsdp_module, FSDPModule):
                        continue
                    for param_names, param_group in fsdp_module._named_param_groups:
                        assert param_group.sharding_strategy == "optim_grads_params"
                        if param_group.transpose_weight_buffer is not None:
                            mxfp8_groups.append((param_names, param_group))
                            assert not param_group.transpose_weight_buffer.is_distributed
                        if param_group.defer_full_param_and_grad_sync:
                            deferred_groups.append((param_names, param_group))
                            assert param_group.replicate_model_weight_buffer
                            assert not param_group.model_weight_buffer.is_distributed
                            assert param_group.main_grad_buffer.is_distributed
                            normalized_names = [
                                name.lower().replace("_", "") for name in param_names
                            ]
                            assert any(
                                "layernorm" in name or "rmsnorm" in name
                                for name in normalized_names
                            )
                            assert all(
                                "layernorm" in name
                                or "rmsnorm" in name
                                or name.endswith("router.weight")
                                for name in normalized_names
                            )
                assert mxfp8_groups, "test model must contain MXFP8 parameter buffers"
                assert deferred_groups
                assert isinstance(recompute_model.embedding.word_embeddings, FSDPModule)
                assert isinstance(recompute_model.output_layer, FSDPModule)
                recompute_opt = torch.optim.SGD(recompute_fsdp.parameters(), lr=LR)

                rank = torch.distributed.get_rank()
                for _ in range(2):
                    recompute_loss = overlap_train_step(
                        recompute_fsdp,
                        recompute_opt,
                        recompute_config,
                        data,
                        num_microbatches=NUM_MICROBATCHES,
                        finalize_fsdp=True,
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
                assert all(
                    not param_group._deferred_grad_accumulated
                    and not param_group.model_weight_buffer.is_unsharded()
                    for _, param_group in deferred_groups
                )

                del recompute_fsdp, recompute_opt
                gc.collect()
                torch.cuda.empty_cache()
        except Exception:
            traceback.print_exc()
            raise
