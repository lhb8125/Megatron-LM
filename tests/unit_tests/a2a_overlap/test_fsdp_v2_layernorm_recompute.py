# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import traceback

import pytest

from megatron.core.enums import Fp8Recipe
from megatron.core.pipeline_parallel.utils import set_streams
from megatron.core.utils import is_te_min_version
from tests.unit_tests.a2a_overlap.test_fsdp_1f1b_overlap import (
    TestFSDP1F1BOverlap as _FSDPOverlapHarness,
)
from tests.unit_tests.a2a_overlap.utils import get_valid_fp8_flags
from tests.unit_tests.test_utilities import Utils


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
    def test_mxfp8_layernorm_recompute(self):
        mxfp8_flags = [
            flag
            for flag in get_valid_fp8_flags()
            if flag is not None and flag[1] == Fp8Recipe.mxfp8
        ]
        if not mxfp8_flags:
            pytest.skip("Requires Blackwell with MXFP8 support")

        try:
            _FSDPOverlapHarness._run_test_helper(
                self,
                dispatcher_type="alltoall",
                fp8_flag=mxfp8_flags[0],
                sharding_strategy="optim_grads_params",
                recompute_modules=["layernorm"],
                use_megatron_fsdp_v2=True,
                fp8_param_gather=True,
                test_only_kwargs={"delay_wgrad_compute": True},
            )
        except Exception:
            traceback.print_exc()
            raise
