import sys
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace

import pytest
from vllm.config.compilation import CUDAGraphMode

import vllm_moe_offload_ascend
from vllm_moe_offload_ascend.patches.patch_fused_moe import (
    _ascend_device_op_is_initializing,
    _ensure_moe_offload_splitting_op,
    _install_cann_compat_when_ready,
    _install_runtime_patches_when_ready,
    _moe_offload_kv_backstop_active,
    _moe_offload_kv_backstop_hint,
    _patch_kv_cache_capacity_backstop,
    _unpack_mlp_apply_result,
)


def test_register_compat_only_skips_moe_runtime_patches(monkeypatch):
    from vllm_moe_offload_ascend.patches import patch_fused_moe

    events = []
    monkeypatch.setattr(
        "vllm_moe_offload_ascend.env_registry.register_environment_variables",
        lambda: events.append("envs"),
    )
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_COMPAT_ONLY", "1")
    monkeypatch.setattr(
        patch_fused_moe,
        "apply_cann_compat_patches",
        lambda: events.append("compat"),
    )
    monkeypatch.setattr(
        patch_fused_moe,
        "apply_patches",
        lambda: pytest.fail("compat-only mode must not install MoE runtime patches"),
    )

    vllm_moe_offload_ascend.register()

    assert events == ["envs", "compat"]


def test_runtime_patch_install_waits_for_ascend_device_op(monkeypatch):
    from vllm_moe_offload_ascend.patches import patch_fused_moe

    device_op = ModuleType("vllm_ascend.device.device_op")
    monkeypatch.setitem(sys.modules, "vllm_ascend.device.device_op", device_op)
    monkeypatch.setattr(
        patch_fused_moe,
        "_register_plugin_ops",
        lambda: pytest.fail("must not import Ascend ops during device-op initialization"),
    )
    monkeypatch.setattr(
        patch_fused_moe,
        "_install_runtime_module_patches",
        lambda: pytest.fail("must not patch runtime modules during device-op initialization"),
    )

    assert _ascend_device_op_is_initializing() is True
    assert _install_runtime_patches_when_ready() is False


def test_runtime_patch_install_registers_custom_ops_before_moe_modules(monkeypatch):
    import importlib

    from vllm_moe_offload_ascend.patches import patch_fused_moe

    device_op = ModuleType("vllm_ascend.device.device_op")
    device_op.DeviceOperator = object
    monkeypatch.setitem(sys.modules, "vllm_ascend.device.device_op", device_op)
    events = []

    original_import_module = importlib.import_module

    def import_module(name, package=None):
        if name == "vllm_ascend.ops.register_custom_ops":
            events.append("register_custom_ops")
            return ModuleType(name)
        return original_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", import_module)
    monkeypatch.setattr(
        patch_fused_moe,
        "_register_plugin_ops",
        lambda: events.append("register_plugin_ops"),
    )
    monkeypatch.setattr(
        patch_fused_moe,
        "_install_runtime_module_patches",
        lambda: events.append("install_runtime_module_patches"),
    )
    monkeypatch.setattr(
        patch_fused_moe,
        "_patch_rms_norm_bias_cann_compat",
        lambda: events.append("patch_rms_norm_bias_cann_compat"),
    )

    assert _ascend_device_op_is_initializing() is False
    assert _install_runtime_patches_when_ready() is True
    assert events == [
        "register_custom_ops",
        "patch_rms_norm_bias_cann_compat",
        "register_plugin_ops",
        "install_runtime_module_patches",
    ]


def test_mix_placement_aiter_compat_suppresses_only_parent_constructor(monkeypatch):
    """Ascend's model-layout shim must not allocate ROCm CUDA AIter buffers."""
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm_moe_offload_ascend.patches import patch_fused_moe

    original_shared_enabled = rocm_aiter_ops.is_fusion_moe_shared_experts_enabled
    original_fused_enabled = rocm_aiter_ops.is_fused_moe_enabled
    monkeypatch.setattr(rocm_aiter_ops, "is_fusion_moe_shared_experts_enabled", lambda: True)
    monkeypatch.setattr(rocm_aiter_ops, "is_fused_moe_enabled", lambda: True)
    monkeypatch.setattr(patch_fused_moe, "_ascend_mix_placement_enabled", lambda: True)

    class FakeAscendFusedMoE:
        def __init__(self, **kwargs):
            self.shared_probe_in_parent = (
                rocm_aiter_ops.is_fusion_moe_shared_experts_enabled()
            )
            self.fused_probe_in_parent = rocm_aiter_ops.is_fused_moe_enabled()
            self.n_shared_experts = kwargs["n_shared_experts"]

    fake_module = SimpleNamespace(AscendFusedMoE=FakeAscendFusedMoE)
    patch_fused_moe._patch_ascend_mix_placement_aiter_compat(fake_module)
    layer = FakeAscendFusedMoE(n_shared_experts=2)

    assert layer.shared_probe_in_parent is False
    assert layer.fused_probe_in_parent is False
    assert layer._latchmoe_mix_placement_aiter_suppressed is True
    assert rocm_aiter_ops.is_fusion_moe_shared_experts_enabled() is True
    assert rocm_aiter_ops.is_fused_moe_enabled() is True
    assert fake_module.AscendFusedMoE._latchmoe_mix_placement_aiter_patch is True

    # Keep the originals reachable for clarity if this test is run without
    # pytest's monkeypatch teardown during an interactive investigation.
    assert callable(original_shared_enabled)
    assert callable(original_fused_enabled)


def test_mix_placement_aiter_compat_leaves_external_shared_path_unchanged(monkeypatch):
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm_moe_offload_ascend.patches import patch_fused_moe

    monkeypatch.setattr(rocm_aiter_ops, "is_fusion_moe_shared_experts_enabled", lambda: True)
    monkeypatch.setattr(rocm_aiter_ops, "is_fused_moe_enabled", lambda: True)
    monkeypatch.setattr(patch_fused_moe, "_ascend_mix_placement_enabled", lambda: False)

    class FakeAscendFusedMoE:
        def __init__(self, **kwargs):
            self.shared_probe_in_parent = (
                rocm_aiter_ops.is_fusion_moe_shared_experts_enabled()
            )
            self.fused_probe_in_parent = rocm_aiter_ops.is_fused_moe_enabled()
            self.n_shared_experts = kwargs["n_shared_experts"]

    fake_module = SimpleNamespace(AscendFusedMoE=FakeAscendFusedMoE)
    patch_fused_moe._patch_ascend_mix_placement_aiter_compat(fake_module)
    layer = FakeAscendFusedMoE(n_shared_experts=2)

    assert layer.shared_probe_in_parent is True
    assert layer.fused_probe_in_parent is True
    assert not hasattr(layer, "_latchmoe_mix_placement_aiter_suppressed")


def test_mix_placement_aiter_compat_expands_dispatcher_top_k(monkeypatch):
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm_moe_offload_ascend.patches import patch_fused_moe

    monkeypatch.setattr(rocm_aiter_ops, "is_fusion_moe_shared_experts_enabled", lambda: True)
    monkeypatch.setattr(rocm_aiter_ops, "is_fused_moe_enabled", lambda: True)
    monkeypatch.setattr(patch_fused_moe, "_ascend_mix_placement_enabled", lambda: True)
    setup_calls = []

    class FakeAscendFusedMoE:
        def __init__(self, **kwargs):
            self.n_shared_experts = kwargs["n_shared_experts"]
            self.moe_config = SimpleNamespace(experts_per_token=6)

    fake_module = SimpleNamespace(
        AscendFusedMoE=FakeAscendFusedMoE,
        setup_moe_comm_method=lambda config: setup_calls.append(config.experts_per_token),
    )
    patch_fused_moe._patch_ascend_mix_placement_aiter_compat(fake_module)
    layer = FakeAscendFusedMoE(n_shared_experts=2)

    assert layer.moe_config.experts_per_token == 8
    assert layer._latchmoe_mix_placement_dispatch_top_k == 8
    assert setup_calls == [8]


def test_mix_placement_router_compat_appends_shared_suffix_arguments(monkeypatch):
    """Pinned Ascend must tell its selector about materialized shared rows."""
    from vllm_moe_offload_ascend.patches import patch_fused_moe
    import vllm_moe_offload_ascend.moe_offload.runtime as runtime_impl

    calls = []

    class FakeBaseMethod:
        def create_weights(self, *args, **kwargs):
            del args, kwargs

    class FakeAscendMethod(FakeBaseMethod):
        def process_weights_after_loading(self, layer):
            del layer

        def apply(self, *, layer, **kwargs):
            return fake_module.select_experts(
                hidden_states=kwargs["hidden_states"],
                router_logits=kwargs["router_logits"],
                num_experts=kwargs["num_experts"],
            )

    def original_select_experts(**kwargs):
        calls.append(kwargs)
        return "weights", "ids"

    fake_module = SimpleNamespace(
        AscendUnquantizedFusedMoEMethod=FakeAscendMethod,
        UnquantizedFusedMoEMethod=FakeBaseMethod,
        select_experts=original_select_experts,
    )
    runtime = SimpleNamespace(
        config=SimpleNamespace(offload_stage_seam=False),
        should_use_fixed_slot_plan_for_layer=lambda _layer_id: False,
    )
    monkeypatch.setattr(runtime_impl, "get_moe_offload_runtime", lambda: runtime)

    patch_fused_moe._patch_unquantized_moe_method(fake_module)
    layer = SimpleNamespace(layer_id=7, mix_placement=True, n_shared_experts=2)
    hidden_states = object()
    router_logits = object()
    assert FakeAscendMethod().apply(
        layer=layer,
        hidden_states=hidden_states,
        router_logits=router_logits,
        num_experts=64,
    ) == ("weights", "ids")

    assert calls == [
        {
            "hidden_states": hidden_states,
            "router_logits": router_logits,
            "num_experts": 64,
            "mix_placement": True,
            "num_logical_experts": 64,
            "num_shared_experts": 2,
        }
    ]


def test_mix_placement_router_compat_leaves_regular_selector_arguments_unchanged(monkeypatch):
    from vllm_moe_offload_ascend.patches import patch_fused_moe
    import vllm_moe_offload_ascend.moe_offload.runtime as runtime_impl

    calls = []

    class FakeBaseMethod:
        def create_weights(self, *args, **kwargs):
            del args, kwargs

    class FakeAscendMethod(FakeBaseMethod):
        def process_weights_after_loading(self, layer):
            del layer

        def apply(self, *, layer, **kwargs):
            return fake_module.select_experts(
                hidden_states=kwargs["hidden_states"],
                router_logits=kwargs["router_logits"],
                num_experts=kwargs["num_experts"],
            )

    def original_select_experts(**kwargs):
        calls.append(kwargs)
        return "weights", "ids"

    fake_module = SimpleNamespace(
        AscendUnquantizedFusedMoEMethod=FakeAscendMethod,
        UnquantizedFusedMoEMethod=FakeBaseMethod,
        select_experts=original_select_experts,
    )
    runtime = SimpleNamespace(
        config=SimpleNamespace(offload_stage_seam=False),
        should_use_fixed_slot_plan_for_layer=lambda _layer_id: False,
    )
    monkeypatch.setattr(runtime_impl, "get_moe_offload_runtime", lambda: runtime)

    patch_fused_moe._patch_unquantized_moe_method(fake_module)
    layer = SimpleNamespace(layer_id=7, mix_placement=False, n_shared_experts=2)
    assert FakeAscendMethod().apply(
        layer=layer,
        hidden_states=object(),
        router_logits=object(),
        num_experts=64,
    ) == ("weights", "ids")

    assert list(calls[0]) == ["hidden_states", "router_logits", "num_experts"]


def test_mix_placement_injected_topk_preserves_shared_suffix_during_profile(monkeypatch):
    """Profile balancing must not rewrite a fused shared suffix as routed IDs."""
    from vllm_moe_offload_ascend.patches import patch_fused_moe
    from vllm_moe_offload_ascend.ops.fused_moe import moe_seam_inject
    import vllm_moe_offload_ascend.moe_offload.runtime as runtime_impl

    class FakeBaseMethod:
        def create_weights(self, *args, **kwargs):
            del args, kwargs

    class FakeAscendMethod(FakeBaseMethod):
        def process_weights_after_loading(self, layer):
            del layer

        def apply(self, *, layer, **kwargs):
            del layer
            return kwargs["enable_force_load_balance"]

    profile_events = []
    runtime = SimpleNamespace(
        config=SimpleNamespace(offload_stage_seam=False),
        should_use_fixed_slot_plan_for_layer=lambda _layer_id: False,
        fused_shared_lane_layout=lambda layer_id: (
            SimpleNamespace(
                routed_expert_count=64,
                shared_expert_count=2,
            )
            if layer_id == 7
            else None
        ),
        _record_profile_event=lambda *args, **kwargs: profile_events.append(
            (args, kwargs)
        ),
    )
    monkeypatch.setattr(runtime_impl, "get_moe_offload_runtime", lambda: runtime)
    fake_module = SimpleNamespace(
        AscendUnquantizedFusedMoEMethod=FakeAscendMethod,
        UnquantizedFusedMoEMethod=FakeBaseMethod,
        select_experts=lambda **_kwargs: ("weights", "ids"),
    )
    patch_fused_moe._patch_unquantized_moe_method(fake_module)

    topk_ids = SimpleNamespace(shape=(3, 8))
    moe_seam_inject.set_injected_topk(7, object(), topk_ids)
    try:
        assert FakeAscendMethod().apply(
            layer=SimpleNamespace(layer_id=7),
            enable_force_load_balance=True,
        ) is False
    finally:
        moe_seam_inject.clear_injected_topk(7)

    assert profile_events[0][0] == ("fused_shared_profile_router_preserved",)
    assert profile_events[0][1]["layer_id"] == 7
    assert profile_events[0][1]["payload"] == {
        "routed_expert_count": 64,
        "shared_expert_count": 2,
        "topk_width": 8,
    }


def test_mix_placement_router_parity_compares_appended_shared_suffix(monkeypatch):
    """The eager router oracle must use the same 6+routed shared ABI as seam."""
    import torch

    from vllm_moe_offload_ascend.patches import patch_fused_moe
    from vllm_moe_offload_ascend.ops.fused_moe import moe_seam_inject
    from vllm_moe_offload_ascend.moe_offload import router_parity
    import vllm_moe_offload_ascend.moe_offload.runtime as runtime_impl

    class FakeBaseMethod:
        def create_weights(self, *args, **kwargs):
            del args, kwargs

    class FakeAscendMethod(FakeBaseMethod):
        def process_weights_after_loading(self, layer):
            del layer

        def apply(self, *, layer, **kwargs):
            return fake_module.select_experts(
                hidden_states=kwargs["hidden_states"],
                router_logits=kwargs["router_logits"],
                num_experts=kwargs["num_experts"],
            )

    calls = []
    snapshots = []
    injected_weights = torch.tensor([[0.4, 0.3, 1.0, 1.0]])
    injected_ids = torch.tensor([[10, 11, 64, 65]], dtype=torch.int32)

    def native_selector(**kwargs):
        calls.append(kwargs)
        assert kwargs["mix_placement"] is True
        assert kwargs["num_logical_experts"] == 64
        assert kwargs["num_shared_experts"] == 2
        return injected_weights, injected_ids

    runtime = SimpleNamespace(
        config=SimpleNamespace(offload_stage_seam=False),
        should_use_fixed_slot_plan_for_layer=lambda _layer_id: False,
        fused_shared_lane_layout=lambda layer_id: (
            SimpleNamespace(routed_expert_count=64, shared_expert_count=2)
            if layer_id == 7
            else None
        ),
    )
    fake_module = SimpleNamespace(
        AscendUnquantizedFusedMoEMethod=FakeAscendMethod,
        UnquantizedFusedMoEMethod=FakeBaseMethod,
        select_experts=native_selector,
    )
    monkeypatch.setattr(runtime_impl, "get_moe_offload_runtime", lambda: runtime)
    monkeypatch.setattr(
        router_parity,
        "record_router_snapshot",
        lambda **kwargs: snapshots.append(kwargs),
    )
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_COMPARE_ROUTER", "1")
    patch_fused_moe._patch_unquantized_moe_method(fake_module)

    moe_seam_inject.set_injected_topk(7, injected_weights, injected_ids)
    try:
        actual = FakeAscendMethod().apply(
            layer=SimpleNamespace(layer_id=7),
            hidden_states=torch.zeros((1, 4)),
            router_logits=torch.zeros((1, 64)),
            num_experts=64,
        )
    finally:
        moe_seam_inject.clear_injected_topk(7)

    assert actual == (injected_weights, injected_ids)
    assert len(calls) == 1
    assert snapshots == [
        {
            "role": "native",
            "layer_id": 7,
            "router_logits": calls[0]["router_logits"],
            "topk_ids": injected_ids,
            "topk_weights": injected_weights,
        }
    ]


def test_summarize_fused_shared_ids_is_bounded_and_unique():
    import torch

    from vllm_moe_offload_ascend.patches import patch_fused_moe

    assert patch_fused_moe._summarize_fused_shared_ids(
        torch.tensor([[65, 64], [64, 65]], dtype=torch.int32)
    ) == [64, 65]


def test_cann_compat_install_does_not_install_moe_runtime(monkeypatch):
    import importlib

    from vllm_moe_offload_ascend.patches import patch_fused_moe

    device_op = ModuleType("vllm_ascend.device.device_op")
    device_op.DeviceOperator = object
    monkeypatch.setitem(sys.modules, "vllm_ascend.device.device_op", device_op)
    events = []

    original_import_module = importlib.import_module

    def import_module(name, package=None):
        if name == "vllm_ascend.ops.register_custom_ops":
            events.append("register_custom_ops")
            return ModuleType(name)
        return original_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", import_module)
    monkeypatch.setattr(
        patch_fused_moe,
        "_patch_rms_norm_bias_cann_compat",
        lambda: events.append("patch_rms_norm_bias_cann_compat"),
    )
    monkeypatch.setattr(
        patch_fused_moe,
        "_register_plugin_ops",
        lambda: pytest.fail("compat-only mode must not register MoE ops"),
    )
    monkeypatch.setattr(
        patch_fused_moe,
        "_install_runtime_module_patches",
        lambda: pytest.fail("compat-only mode must not install runtime hooks"),
    )

    assert _install_cann_compat_when_ready() is True
    assert events == [
        "register_custom_ops",
        "patch_rms_norm_bias_cann_compat",
    ]


def test_cann_moe_gating_falls_back_to_native_op(monkeypatch):
    import types

    import torch

    from vllm_moe_offload_ascend.patches import patch_fused_moe

    calls = []

    def native_moe_gating_top_k(x, k, **kwargs):
        calls.append((x, k, kwargs))
        return (
            torch.tensor([[1.0, 3.0]], dtype=torch.float32),
            torch.tensor([[7, 9]], dtype=torch.int64),
            torch.zeros((1, 4), dtype=torch.float32),
        )

    fake_torch_npu = types.ModuleType("torch_npu")
    fake_torch_npu.npu_moe_gating_top_k = native_moe_gating_top_k
    monkeypatch.setitem(sys.modules, "torch_npu", fake_torch_npu)

    class BaseDeviceAdaptor:
        @staticmethod
        def moe_gating_top_k(*args, **kwargs):
            raise AssertionError("the missing custom op path was not patched")

    device_op = types.ModuleType("vllm_ascend.device.device_op")
    device_op.BaseDeviceAdaptor = BaseDeviceAdaptor
    device_package = types.ModuleType("vllm_ascend.device")
    device_package.device_op = device_op
    monkeypatch.setitem(sys.modules, "vllm_ascend.device", device_package)
    monkeypatch.setitem(sys.modules, "vllm_ascend.device.device_op", device_op)

    assert patch_fused_moe._patch_moe_gating_top_k_cann_compat() is True
    weights, ids, out = BaseDeviceAdaptor.moe_gating_top_k(
        torch.zeros((1, 4)),
        k=2,
        k_group=1,
        group_count=1,
        group_select_mode=1,
        renorm=1,
        norm_type=0,
        out_flag=False,
    )

    assert calls[0][1:] == (2, {
        "bias": None,
        "k_group": 1,
        "group_count": 1,
        "group_select_mode": 1,
        "renorm": 0,
        "norm_type": 0,
        "out_flag": False,
        "routed_scaling_factor": 1.0,
        "eps": 1e-20,
    })
    torch.testing.assert_close(weights, torch.tensor([[0.25, 0.75]]))
    assert ids.dtype == torch.int32
    assert out.shape == (1, 4)
    assert patch_fused_moe._patch_moe_gating_top_k_cann_compat() is False


def test_cann_moe_init_routing_falls_back_to_native_op(monkeypatch):
    import types

    import torch

    from vllm_moe_offload_ascend.patches import patch_fused_moe

    calls = []

    def native_moe_init_routing_v2(hidden_states, topk_ids, **kwargs):
        calls.append((hidden_states, topk_ids, kwargs))
        return ("sorted", "row_idx", "expert_tokens", "scale")

    fake_torch_npu = types.ModuleType("torch_npu")
    fake_torch_npu.npu_moe_init_routing_v2 = native_moe_init_routing_v2
    monkeypatch.setitem(sys.modules, "torch_npu", fake_torch_npu)

    class BaseDeviceAdaptor:
        @staticmethod
        def npu_moe_init_routing(*args, **kwargs):
            raise AssertionError("the missing custom op path was not patched")

    device_op = types.ModuleType("vllm_ascend.device.device_op")
    device_op.BaseDeviceAdaptor = BaseDeviceAdaptor
    device_package = types.ModuleType("vllm_ascend.device")
    device_package.device_op = device_op
    monkeypatch.setitem(sys.modules, "vllm_ascend.device", device_package)
    monkeypatch.setitem(sys.modules, "vllm_ascend.device.device_op", device_op)

    assert patch_fused_moe._patch_moe_init_routing_cann_compat() is True
    hidden_states = torch.zeros((2, 4))
    topk_ids = torch.zeros((2, 2), dtype=torch.int32)
    result = BaseDeviceAdaptor.npu_moe_init_routing(
        hidden_states,
        topk_ids,
        active_num=-1,
        expert_num=4,
    )

    assert result == ("sorted", "row_idx", "expert_tokens", "scale")
    assert calls[0][2] == {
        "scale": None,
        "active_num": 4,
        "expert_capacity": -1,
        "expert_num": 4,
        "drop_pad_mode": 0,
        "expert_tokens_num_type": 1,
        "expert_tokens_num_flag": True,
        "quant_mode": -1,
        "active_expert_range": [0, 4],
        "row_idx_type": 0,
    }
    assert patch_fused_moe._patch_moe_init_routing_cann_compat() is False


def test_adapt_patch_retries_complete_ready_path(monkeypatch):
    import vllm_ascend.utils as ascend_utils

    from vllm_moe_offload_ascend.patches import patch_fused_moe

    events = []

    def original_adapt_patch(*args, **kwargs):
        events.append(("original", args, kwargs))
        return "adapted"

    monkeypatch.setattr(ascend_utils, "adapt_patch", original_adapt_patch)
    monkeypatch.setattr(
        patch_fused_moe,
        "_install_runtime_patches_when_ready",
        lambda: events.append("ready"),
    )
    monkeypatch.setattr(
        patch_fused_moe,
        "_patch_kv_cache_capacity_backstop",
        lambda: events.append("kv_backstop"),
    )

    patch_fused_moe._patch_adapt_patch_reinstall()

    assert ascend_utils.adapt_patch("worker", stage="late") == "adapted"
    assert events == [
        ("original", ("worker",), {"stage": "late"}),
        "ready",
        "kv_backstop",
    ]


def test_cann_rmsnorm_fallback_avoids_missing_custom_op(monkeypatch):
    import importlib

    from vllm_moe_offload_ascend.patches import patch_fused_moe

    class FakeRMSNorm:
        weight = "weight"
        variance_epsilon = 1e-6
        bias = None

        def forward_oot(self, x, residual=None):
            return ("original", x, residual)

    class FakeGemmaRMSNorm:
        weight = 3.0
        variance_epsilon = 1e-6

        def forward_oot(self, x, residual=None):
            return ("gemma-original", x, residual)

    fake_layernorm = ModuleType("vllm_ascend.ops.layernorm")
    fake_layernorm.AscendRMSNorm = FakeRMSNorm
    fake_layernorm.AscendGemmaRMSNorm = FakeGemmaRMSNorm
    original_import_module = importlib.import_module

    def import_module(name, package=None):
        if name == "vllm_ascend.ops.layernorm":
            return fake_layernorm
        return original_import_module(name, package)

    fake_torch_npu = ModuleType("torch_npu")
    fake_torch_npu.npu_add_rms_norm = lambda x, residual, weight, eps: (
        ("normalized", x, residual, weight, eps),
        None,
        "next_residual",
    )
    fake_torch_npu.npu_rms_norm = lambda x, weight, eps: (
        ("gemma-normalized", x, weight, eps),
        "gemma-residual",
    )

    monkeypatch.setattr(importlib, "import_module", import_module)
    monkeypatch.setattr(patch_fused_moe, "_opapi_supports_add_rms_norm_bias", lambda: False)
    monkeypatch.setitem(sys.modules, "torch_npu", fake_torch_npu)

    patch_fused_moe._patch_rms_norm_bias_cann_compat()

    assert FakeRMSNorm().forward_oot("x", "residual") == (
        ("normalized", "x", "residual", "weight", 1e-6),
        "next_residual",
    )
    assert FakeRMSNorm().forward_oot("x") == ("original", "x", None)
    assert FakeGemmaRMSNorm().forward_oot("x", "residual") == (
        ("normalized", "x", "residual", 4.0, 1e-6),
        "next_residual",
    )
    assert FakeGemmaRMSNorm().forward_oot("x") == (
        "gemma-normalized",
        "x",
        4.0,
        1e-6,
    )


def test_cann_rmsnorm_fallback_disables_unsupported_fusion_patterns(monkeypatch):
    import importlib

    from vllm_moe_offload_ascend.patches import patch_fused_moe

    fake_layernorm = ModuleType("vllm_ascend.ops.layernorm")
    fake_fusion_pass = ModuleType(
        "vllm_ascend.compilation.passes.norm_quant_fusion_pass"
    )
    fake_fusion_pass.enable_custom_op = lambda: True
    original_import_module = importlib.import_module

    def import_module(name, package=None):
        if name == "vllm_ascend.ops.layernorm":
            return fake_layernorm
        if name == "vllm_ascend.compilation.passes.norm_quant_fusion_pass":
            return fake_fusion_pass
        return original_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", import_module)
    monkeypatch.setattr(
        patch_fused_moe,
        "_opapi_supports_add_rms_norm_bias",
        lambda: False,
    )

    patch_fused_moe._patch_rms_norm_bias_cann_compat()

    assert fake_fusion_pass.enable_custom_op() is False
    assert getattr(
        fake_fusion_pass.enable_custom_op,
        "_latchmoe_cann_rmsnorm_compat",
        False,
    )


def test_stage_seam_registers_for_piecewise_cudagraph(monkeypatch):
    import vllm_ascend.platform as platform

    from vllm_moe_offload_ascend.patches import patch_fused_moe

    class FakeNPUPlatform:
        @classmethod
        def check_and_update_config(cls, _vllm_config):
            return None

    monkeypatch.setattr(platform, "NPUPlatform", FakeNPUPlatform)
    compat_calls = []
    monkeypatch.setattr(
        patch_fused_moe,
        "_patch_rms_norm_bias_cann_compat",
        lambda: compat_calls.append("rmsnorm_compat"),
    )
    monkeypatch.setattr(patch_fused_moe, "_install_runtime_module_patches", lambda: None)
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_STAGE_SEAM", "1")

    patch_fused_moe._patch_platform_splitting_ops()
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.PIECEWISE,
            splitting_ops=[],
        )
    )

    FakeNPUPlatform.check_and_update_config(config)

    assert compat_calls == ["rmsnorm_compat"]
    assert "vllm::moe_offload_stage" in config.compilation_config.splitting_ops


def test_engine_args_final_config_retries_stage_seam_patch(monkeypatch):
    import vllm.engine.arg_utils as arg_utils

    from vllm_moe_offload_ascend.patches import patch_fused_moe

    events = []

    class FakeEngineArgs:
        def create_engine_config(self):
            events.append("create")
            return SimpleNamespace(
                compilation_config=SimpleNamespace(
                    cudagraph_mode=CUDAGraphMode.PIECEWISE,
                    splitting_ops=[],
                )
            )

    monkeypatch.setattr(arg_utils, "EngineArgs", FakeEngineArgs)
    monkeypatch.setattr(
        patch_fused_moe,
        "_patch_platform_splitting_ops",
        lambda: events.append("retry_platform"),
    )
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_STAGE_SEAM", "1")

    # The function imports apply_moe_offload_defaults locally, so expose a fake
    # autoconfig module with the same API to keep this a lifecycle-only test.
    import vllm_moe_offload_ascend.moe_offload.autoconfig as autoconfig

    monkeypatch.setattr(
        autoconfig,
        "apply_moe_offload_defaults",
        lambda _args: events.append("defaults"),
    )
    patch_fused_moe._patch_engine_args_autoconfig()

    config = FakeEngineArgs().create_engine_config()

    assert events == ["defaults", "retry_platform", "create"]
    assert "vllm::moe_offload_stage" in config.compilation_config.splitting_ops


@pytest.mark.parametrize(
    "cudagraph_mode",
    [CUDAGraphMode.FULL, CUDAGraphMode.FULL_AND_PIECEWISE],
)
def test_stage_seam_fails_closed_for_any_full_graph(
    monkeypatch,
    cudagraph_mode,
):
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_STAGE_SEAM", "1")
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            cudagraph_mode=cudagraph_mode,
            splitting_ops=[],
        )
    )

    with pytest.raises(RuntimeError, match="requires PIECEWISE-only ACLGraph"):
        _ensure_moe_offload_splitting_op(config, fail_closed=True)


def test_stage_custom_op_is_a_top_level_splitting_graph():
    import torch
    from torch.fx.experimental.proxy_tensor import make_fx

    from vllm.compilation.backends import split_graph
    import vllm_moe_offload_ascend.ops.fused_moe.moe_offload_stage_op  # noqa: F401

    def model(x):
        routed = x + 1
        torch.ops.vllm.moe_offload_stage(routed, 0, 128, 0)
        return routed * 2

    graph = make_fx(model, tracing_mode="fake")(
        torch.ones(2, dtype=torch.int64)
    )
    _split_graph_module, split_items = split_graph(
        graph,
        ["vllm::moe_offload_stage"],
    )
    splitting_items = [item for item in split_items if item.is_splitting_graph]

    assert len(splitting_items) == 1
    call_targets = [
        node.target
        for node in splitting_items[0].graph.graph.nodes
        if node.op == "call_function"
    ]
    assert torch.ops.vllm.moe_offload_stage.default in call_targets


def test_comm_hook_resolves_contracts_from_new_runtime_args(monkeypatch):
    from vllm_moe_offload_ascend.patches import patch_fused_moe

    class FakeCommMethod:
        def fused_experts(self, fused_experts_input):
            return fused_experts_input

        def _maybe_apply_moe_offload_plan(self, fused_experts_input):
            return fused_experts_input

    runtime_args = ModuleType("vllm_ascend.ops.fused_moe.moe_runtime_args")
    runtime_args.MoEFusedExpertsInput = SimpleNamespace
    runtime_args.MoEOffloadParams = SimpleNamespace
    runtime_args.MoERoutingParams = SimpleNamespace
    runtime_args.MoEWeights = SimpleNamespace
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.ops.fused_moe.moe_runtime_args",
        runtime_args,
    )

    fake_comm = SimpleNamespace(
        MoECommMethod=FakeCommMethod,
        build_token_dispatch_input=lambda **kwargs: kwargs,
        build_mlp_compute_input=lambda **kwargs: kwargs,
        FusedExpertsResult=SimpleNamespace,
        setup_moe_comm_method=None,
    )

    patch_fused_moe._patch_moe_comm_method_runtime_hooks(fake_comm)

    assert (
        FakeCommMethod._maybe_apply_moe_offload_plan._ascend_moe_offload_patch_tag
        == "vllm_moe_offload_ascend.moe_comm_method_runtime"
    )


def test_graph_compatible_unregistered_slot_layer_fails_closed():
    from vllm_moe_offload_ascend.patches import patch_fused_moe

    runtime = SimpleNamespace(
        config=SimpleNamespace(graph_compatible_offload=True),
        should_use_fixed_slot_plan_for_layer=lambda layer_id: layer_id == 7,
        capture_safe_slot_weights=lambda **kwargs: None,
    )

    with pytest.raises(RuntimeError, match="refusing to capture or replay"):
        patch_fused_moe._graph_capture_slot_weights(runtime, layer_id=7)


def test_graph_capture_stage_rejects_unregistered_fixed_slot_layer(monkeypatch):
    import torch

    from vllm_moe_offload_ascend.moe_offload import runtime as runtime_mod
    from vllm_moe_offload_ascend.moe_offload.config import MoeOffloadConfig
    from vllm_moe_offload_ascend.moe_offload.runtime import MoeOffloadRuntime
    from vllm_moe_offload_ascend.ops.fused_moe import moe_offload_stage_op

    runtime = MoeOffloadRuntime(
        MoeOffloadConfig(
            enabled=True,
            num_slots=1,
            graph_compatible_offload=True,
        )
    )
    monkeypatch.setattr(runtime_mod, "_runtime", runtime)
    monkeypatch.setattr(runtime_mod, "_is_current_graph_capturing", lambda: True)

    with pytest.raises(RuntimeError, match="unregistered fixed-slot MoE layer"):
        moe_offload_stage_op._moe_offload_stage_impl(
            torch.tensor([[0]], dtype=torch.int32),
            layer_id=7,
            num_logical_experts=4,
            phase=0,
        )


def test_unpack_mlp_apply_result_supports_old_and_new_contracts():
    output = object()
    event = object()

    assert _unpack_mlp_apply_result(output) == (output, None)
    assert _unpack_mlp_apply_result((output, event)) == (output, event)
    with pytest.raises(RuntimeError, match="unsupported tuple"):
        _unpack_mlp_apply_result((output, event, object()))


def test_kv_backstop_is_scoped_to_moe_offload_env(monkeypatch):
    for name in (
        "VLLM_ASCEND_MOE_OFFLOAD_ENABLED",
        "VLLM_ASCEND_MOE_OFFLOAD_GB",
        "VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS",
    ):
        monkeypatch.delenv(name, raising=False)

    assert _moe_offload_kv_backstop_active() is False

    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_GB", "28")
    assert _moe_offload_kv_backstop_active() is True


def test_kv_backstop_hint_names_slot_reduction_action(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_GB", "28")
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS", "62")

    hint = _moe_offload_kv_backstop_hint(
        max_model_len=4096,
        max_concurrency=0.75,
    )

    assert "KV-capacity backstop" in hint
    assert "reduce `VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS`" in hint
    assert "offload_gb=28" in hint
    assert "num_slots=62" in hint
    assert "max_model_len=4096" in hint
    assert "max_concurrency=0.750x" in hint


def test_kv_backstop_wraps_vllm_capacity_error(monkeypatch):
    import vllm.v1.core.kv_cache_utils as kv_utils

    def original_check(*_args, **_kwargs):
        raise ValueError("base kv capacity error")

    monkeypatch.setattr(kv_utils, "_check_enough_kv_cache_memory", original_check)
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_GB", "28")
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS", "62")

    _patch_kv_cache_capacity_backstop()

    with pytest.raises(ValueError) as excinfo:
        kv_utils._check_enough_kv_cache_memory(
            0,
            lambda: 0,
            4096,
            lambda _available_memory: 0,
        )

    message = str(excinfo.value)
    assert "base kv capacity error" in message
    assert "KV-capacity backstop" in message
    assert "reduce `VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS`" in message


def test_kv_backstop_fails_after_resolved_blocks_if_one_request_cannot_fit(monkeypatch):
    import vllm.v1.core.kv_cache_utils as kv_utils

    called = False

    def original_report(_vllm_config, _kv_cache_config):
        nonlocal called
        called = True

    monkeypatch.setattr(
        kv_utils,
        "_report_kv_cache_config",
        original_report,
        raising=False,
    )
    monkeypatch.setattr(
        kv_utils,
        "get_max_concurrency_for_kv_cache_config",
        lambda _vllm_config, _kv_cache_config: 0.75,
    )
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_GB", "28")
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS", "62")
    config = SimpleNamespace(model_config=SimpleNamespace(max_model_len=4096))

    _patch_kv_cache_capacity_backstop()

    with pytest.raises(ValueError, match="max_concurrency=0.750x"):
        kv_utils._report_kv_cache_config(config, SimpleNamespace())

    assert called is False


def test_b2_wave_profile_summary_reports_overlap_and_stage_breakdown():
    from vllm_moe_offload_ascend.patches.patch_fused_moe import (
        _summarize_b2_wave_profiles,
    )

    summary = _summarize_b2_wave_profiles(
        [
            {
                "stage_mode": "main_slot_hit",
                "hits": 2,
                "misses": 0,
                "tokens": 17,
                "pairs": 17,
                "h2d_bytes": 0,
                "d2d_bytes": 0,
                "stage_issue_ms": 0.1,
                "stage_wait_ms": 0.0,
                "mlp_ms": 3.0,
                "gmm_ms": 2.0,
                "issued_before_compute": False,
                "issued_before_microbatch_materialize": False,
            },
            {
                "stage_mode": "async_double_buffer",
                "hits": 0,
                "misses": 2,
                "tokens": 23,
                "pairs": 23,
                "h2d_bytes": 1024,
                "d2d_bytes": 0,
                "stage_issue_ms": 0.2,
                "stage_wait_ms": 4.5,
                "mlp_ms": 5.0,
                "gmm_ms": 4.0,
                "prefetch_before_compute_count": 1,
                "prefetch_after_compute_count": 2,
                "issued_before_compute": True,
                "issued_before_microbatch_materialize": True,
                "issue_end_to_compute_ms": 6.0,
            },
            {
                "stage_mode": "async_double_buffer",
                "hits": 1,
                "misses": 1,
                "tokens": 11,
                "pairs": 11,
                "h2d_bytes": 512,
                "d2d_bytes": 256,
                "stage_issue_ms": 0.3,
                "stage_wait_ms": 1.5,
                "mlp_ms": 2.0,
                "gmm_ms": 1.0,
                "issued_before_compute": True,
            },
        ],
        layer_scatter_ms=12.75,
    )

    assert summary["wave_count"] == 3
    assert summary["hit_only_waves"] == 1
    assert summary["miss_only_waves"] == 1
    assert summary["mixed_waves"] == 1
    assert summary["main_slot_hit_waves"] == 1
    assert summary["staged_waves"] == 2
    assert summary["issued_before_compute_waves"] == 2
    assert summary["issued_before_microbatch_materialize_waves"] == 1
    assert summary["prefetch_before_compute_issues"] == 1
    assert summary["prefetch_after_compute_issues"] == 2
    assert summary["tokens"] == 51
    assert summary["pairs"] == 51
    assert summary["hits"] == 3
    assert summary["misses"] == 3
    assert summary["h2d_bytes"] == 1536
    assert summary["d2d_bytes"] == 256
    assert summary["stage_issue_ms"] == 0.6
    assert summary["stage_wait_ms"] == 6.0
    assert summary["mlp_ms"] == 10.0
    assert summary["gmm_ms"] == 7.0
    assert summary["layer_scatter_ms"] == 12.75
    assert summary["max_issue_end_to_compute_ms"] == 6.0
    assert summary["max_stage_wait_ms"] == 4.5


def test_b2_profile_details_omitted_by_default(monkeypatch):
    from vllm_moe_offload_ascend.patches.patch_fused_moe import (
        _attach_b2_profile_details,
    )

    class Jsonable:
        def to_jsonable(self):
            raise AssertionError("default details must not materialize json")

    monkeypatch.delenv("VLLM_ASCEND_MOE_B2_PROFILE_DETAILS", raising=False)
    payload = _attach_b2_profile_details(
        {"wave_summary": {"wave_count": 1}},
        wave_plan=Jsonable(),
        async_schedule=Jsonable(),
        async_stage=True,
        wave_profiles=[{"experts": [1, 2], "h2d_bytes": 128}],
    )

    assert payload == {
        "wave_summary": {"wave_count": 1},
        "profile_details": "omitted",
    }


def test_b2_profile_details_can_be_enabled(monkeypatch):
    from vllm_moe_offload_ascend.patches.patch_fused_moe import (
        _attach_b2_profile_details,
    )

    class Jsonable:
        def __init__(self, value):
            self.value = value

        def to_jsonable(self):
            return self.value

    monkeypatch.setenv("VLLM_ASCEND_MOE_B2_PROFILE_DETAILS", "1")
    payload = _attach_b2_profile_details(
        {"wave_summary": {"wave_count": 2}},
        wave_plan=Jsonable({"waves": [[3, 4]]}),
        async_schedule=Jsonable({"enabled": True}),
        async_stage=True,
        wave_profiles=[{"experts": [3, 4], "h2d_bytes": 256}],
    )

    assert payload["wave_summary"] == {"wave_count": 2}
    assert payload["wave_plan"] == {"waves": [[3, 4]]}
    assert payload["async_schedule"] == {"enabled": True}
    assert payload["waves"] == [{"experts": [3, 4], "h2d_bytes": 256}]
    assert "profile_details" not in payload


def test_b2_reference_full_tokens_is_disabled_by_default(monkeypatch):
    from vllm_moe_offload_ascend.patches.patch_fused_moe import (
        _b2_reference_full_tokens_enabled,
    )

    monkeypatch.delenv(
        "VLLM_ASCEND_MOE_OFFLOAD_B2_REFERENCE_FULL_TOKENS",
        raising=False,
    )

    assert _b2_reference_full_tokens_enabled(async_stage=False) is False


def test_b2_reference_full_tokens_accepts_sync_staging(monkeypatch):
    from vllm_moe_offload_ascend.patches.patch_fused_moe import (
        _b2_reference_full_tokens_enabled,
    )

    monkeypatch.setenv(
        "VLLM_ASCEND_MOE_OFFLOAD_B2_REFERENCE_FULL_TOKENS",
        "1",
    )

    assert _b2_reference_full_tokens_enabled(async_stage=False) is True


def test_b2_reference_full_tokens_rejects_async_staging(monkeypatch):
    from vllm_moe_offload_ascend.patches.patch_fused_moe import (
        _b2_reference_full_tokens_enabled,
    )

    monkeypatch.setenv(
        "VLLM_ASCEND_MOE_OFFLOAD_B2_REFERENCE_FULL_TOKENS",
        "1",
    )

    with pytest.raises(RuntimeError, match="only supports synchronous B2 staging"):
        _b2_reference_full_tokens_enabled(async_stage=True)


def test_b2_reference_full_layer_is_disabled_by_default(monkeypatch):
    from vllm_moe_offload_ascend.patches.patch_fused_moe import (
        _b2_reference_full_layer_enabled,
    )

    monkeypatch.delenv(
        "VLLM_ASCEND_MOE_OFFLOAD_B2_REFERENCE_FULL_LAYER",
        raising=False,
    )

    assert _b2_reference_full_layer_enabled(async_stage=False) is False


def test_b2_reference_full_layer_accepts_sync_staging(monkeypatch):
    from vllm_moe_offload_ascend.patches.patch_fused_moe import (
        _b2_reference_full_layer_enabled,
    )

    monkeypatch.setenv(
        "VLLM_ASCEND_MOE_OFFLOAD_B2_REFERENCE_FULL_LAYER",
        "1",
    )

    assert _b2_reference_full_layer_enabled(async_stage=False) is True


def test_b2_reference_full_layer_uses_blocking_copy_with_async_decode(monkeypatch):
    from vllm_moe_offload_ascend.patches.patch_fused_moe import (
        _b2_reference_full_layer_enabled,
    )

    monkeypatch.setenv(
        "VLLM_ASCEND_MOE_OFFLOAD_B2_REFERENCE_FULL_LAYER",
        "1",
    )

    assert _b2_reference_full_layer_enabled(async_stage=True) is True


def test_b2_overflow_mode_defaults_to_multi_wave(monkeypatch):
    from vllm_moe_offload_ascend.patches.patch_fused_moe import (
        _b2_overflow_mode,
    )

    monkeypatch.delenv(
        "VLLM_ASCEND_MOE_OFFLOAD_B2_OVERFLOW_MODE",
        raising=False,
    )

    assert _b2_overflow_mode() == "multi_wave"


def test_b2_overflow_mode_accepts_legacy_experimental_wave_alias(monkeypatch):
    from vllm_moe_offload_ascend.patches.patch_fused_moe import (
        _b2_overflow_mode,
    )

    monkeypatch.setenv(
        "VLLM_ASCEND_MOE_OFFLOAD_B2_OVERFLOW_MODE",
        "experimental_wave",
    )

    assert _b2_overflow_mode() == "multi_wave"


def test_b2_overflow_mode_allows_explicit_full_layer(monkeypatch):
    from vllm_moe_offload_ascend.patches.patch_fused_moe import (
        _b2_overflow_mode,
    )

    monkeypatch.setenv(
        "VLLM_ASCEND_MOE_OFFLOAD_B2_OVERFLOW_MODE",
        "full_layer",
    )

    assert _b2_overflow_mode() == "full_layer"


def test_b2_overflow_mode_rejects_unknown_value(monkeypatch):
    from vllm_moe_offload_ascend.patches.patch_fused_moe import (
        _b2_overflow_mode,
    )

    monkeypatch.setenv(
        "VLLM_ASCEND_MOE_OFFLOAD_B2_OVERFLOW_MODE",
        "fast",
    )

    with pytest.raises(RuntimeError, match="must be 'multi_wave'"):
        _b2_overflow_mode()


def test_b2_multi_wave_preflight_falls_back_without_native_recombine(monkeypatch):
    from vllm_moe_offload_ascend.patches.patch_fused_moe import (
        _b2_multi_wave_preflight_failure,
    )

    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_B2_DIRECT_SCATTER", "0")
    runtime = SimpleNamespace(
        config=SimpleNamespace(
            num_slots=32,
            async_load=True,
            effective_prefill_prefetch_depth=1,
            effective_prefill_buffer_count=2,
        )
    )

    assert (
        _b2_multi_wave_preflight_failure(runtime)
        == "native_recombine_disabled"
    )


def test_b2_multi_wave_preflight_requires_two_buffers_for_overlap(monkeypatch):
    from vllm_moe_offload_ascend.patches.patch_fused_moe import (
        _b2_multi_wave_preflight_failure,
    )

    monkeypatch.delenv(
        "VLLM_ASCEND_MOE_OFFLOAD_B2_DIRECT_SCATTER",
        raising=False,
    )
    runtime = SimpleNamespace(
        config=SimpleNamespace(
            num_slots=32,
            async_load=True,
            effective_prefill_prefetch_depth=1,
            effective_prefill_buffer_count=1,
        )
    )

    assert (
        _b2_multi_wave_preflight_failure(runtime)
        == "overlap_requires_two_buffers"
    )


def test_b2_profile_details_accepts_sync_execution_without_schedule(monkeypatch):
    from vllm_moe_offload_ascend.patches.patch_fused_moe import (
        _attach_b2_profile_details,
    )

    class Jsonable:
        def to_jsonable(self):
            return {"waves": [[1, 2]]}

    monkeypatch.setenv("VLLM_ASCEND_MOE_B2_PROFILE_DETAILS", "1")
    payload = _attach_b2_profile_details(
        {},
        wave_plan=Jsonable(),
        async_schedule=None,
        async_stage=False,
        wave_profiles=[{"experts": [1, 2], "stage_mode": "sync_slot_cache"}],
    )

    assert payload["wave_plan"] == {"waves": [[1, 2]]}
    assert payload["async_schedule"] is None
    assert payload["waves"] == [
        {"experts": [1, 2], "stage_mode": "sync_slot_cache"}
    ]


def test_estimate_b2_wave_h2d_bytes_uses_cached_layer_bytes():
    from vllm_moe_offload_ascend.patches.patch_fused_moe import (
        _estimate_b2_wave_h2d_bytes,
    )

    class FakeRuntime:
        def cached_layer_expert_weight_bytes(self, *, layer_id):
            assert layer_id == 7
            return 128

        def estimate_expert_weight_bytes(self, **kwargs):
            raise AssertionError("cached layer bytes should avoid per-expert estimates")

    assert _estimate_b2_wave_h2d_bytes(
        FakeRuntime(),
        layer_id=7,
        wave=(1, 2, 3, 4),
        readiness={2: True},
    ) == 128 * 3


def test_estimate_b2_wave_h2d_bytes_falls_back_without_cache():
    from vllm_moe_offload_ascend.patches.patch_fused_moe import (
        _estimate_b2_wave_h2d_bytes,
    )

    class FakeRuntime:
        def __init__(self):
            self.calls = []

        def estimate_expert_weight_bytes(self, *, layer_id, expert_id):
            self.calls.append((layer_id, expert_id))
            return 10 + int(expert_id)

    runtime = FakeRuntime()

    assert _estimate_b2_wave_h2d_bytes(
        runtime,
        layer_id=7,
        wave=(1, 2, 3),
        readiness={2: True},
    ) == 24
    assert runtime.calls == [(7, 1), (7, 3)]


def test_register_does_not_alias_plugin_modules_into_vllm_ascend(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_GB", "14")

    vllm_moe_offload_ascend.register()

    assert "vllm_ascend.moe_offload" not in sys.modules
    assert all(
        not name.startswith("vllm_ascend.moe_offload.") for name in sys.modules
    )


def test_register_imports_plugin_owned_sew_custom_ops(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_GB", "14")

    vllm_moe_offload_ascend.register()

    assert "vllm_moe_offload_ascend.ops.fused_moe.moe_offload_stage_op" in sys.modules
    assert "vllm_moe_offload_ascend.ops.fused_moe.moe_router_op" in sys.modules
    assert "vllm_ascend.ops.fused_moe.moe_offload_stage_op" not in sys.modules


def test_register_rebinds_already_imported_hook_globals(monkeypatch):
    import vllm_ascend.ops.fused_moe.fused_moe as fused_moe
    import vllm_ascend.ops.fused_moe.moe_comm_method as moe_comm_method
    import vllm_ascend.ops.fused_moe.token_dispatcher as token_dispatcher

    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_GB", "14")

    vllm_moe_offload_ascend.register()

    assert fused_moe.get_moe_offload_runtime.__module__ == (
        "vllm_moe_offload_ascend.moe_offload.runtime"
    )
    assert moe_comm_method.get_moe_offload_runtime.__module__ == (
        "vllm_moe_offload_ascend.moe_offload.runtime"
    )
    assert moe_comm_method.MoeOffloadDecisionPath.__module__ == (
        "vllm_moe_offload_ascend.moe_offload.runtime"
    )
    assert token_dispatcher.get_moe_pipeline_profiler.__module__ == (
        "vllm_moe_offload_ascend.moe_offload.pipeline"
    )


def test_external_shared_multistream_starts_at_routed_dispatch(monkeypatch):
    """The external shared first projection must be issued before B2 work."""
    import torch

    from vllm_moe_offload_ascend.patches import patch_fused_moe
    import vllm_ascend.utils as ascend_utils

    events = []

    class FakeStream:
        def __init__(self, name):
            self.name = name

        def record_event(self):
            events.append(("record_event", self.name))
            return "hidden_states_ready"

        def wait_event(self, event):
            events.append(("wait_event", self.name, event))

        def wait_stream(self, stream):
            events.append(("wait_stream", self.name, stream.name))

    default_stream = FakeStream("default")
    shared_stream = FakeStream("shared")

    class FakeNpu:
        active_stream = default_stream

        @classmethod
        def current_stream(cls):
            return cls.active_stream

        @classmethod
        def stream(cls, target):
            class StreamContext:
                def __enter__(self):
                    self.previous = FakeNpu.active_stream
                    FakeNpu.active_stream = target

                def __exit__(self, exc_type, exc, traceback):
                    FakeNpu.active_stream = self.previous

            return StreamContext()

    monkeypatch.setattr(torch, "npu", FakeNpu)
    monkeypatch.setattr(
        ascend_utils,
        "shared_experts_calculation_stream",
        lambda: shared_stream,
    )

    class FakeAscendFusedMoE:
        _latchmoe_external_shared_overlap_patch = False

        def __init__(self):
            self.multistream_overlap_shared_expert = True
            self.shared_multistream_overlap_gate = False
            self.is_internal_router = False
            self._shared_experts = object()
            self.moe_config = SimpleNamespace(
                dp_size=1,
                ep_size=1,
                tp_size=1,
                pcp_size=1,
            )

        def shared_forward_impl(self, hidden_states, router_logits):
            events.append(("upstream", hidden_states, router_logits))
            return "upstream"

        def _shared_experts_part1(self, hidden_states):
            events.append(("shared_part1", hidden_states))
            return "part1"

        def forward_impl(self, *, hidden_states, router_logits, return_with_event):
            dispatch_ready = default_stream.record_event()
            assert patch_fused_moe._launch_external_shared_overlap_if_active(
                dispatch_ready
            )
            events.append(("routed", hidden_states, router_logits, return_with_event))
            return SimpleNamespace(
                routed_out="routed_out",
                before_combine_evt="routed_combine_ready",
            )

        def _shared_experts_part2(self, hidden_states, part1):
            events.append(("shared_part2", hidden_states, part1))
            return "shared_out"

    fake_fused_moe = SimpleNamespace(AscendFusedMoE=FakeAscendFusedMoE)
    patch_fused_moe._patch_external_shared_multistream_overlap(fake_fused_moe)

    layer = FakeAscendFusedMoE()
    assert layer.shared_forward_impl("hidden", "router") == ("shared_out", "routed_out")
    assert events == [
        ("record_event", "default"),
        ("wait_event", "shared", "hidden_states_ready"),
        ("shared_part1", "hidden"),
        ("routed", "hidden", "router", True),
        ("wait_event", "shared", "routed_combine_ready"),
        ("shared_part2", "hidden", "part1"),
        ("wait_stream", "default", "shared"),
    ]


def test_external_shared_multistream_leaves_internal_router_upstream(monkeypatch):
    from vllm_moe_offload_ascend.patches import patch_fused_moe

    class FakeAscendFusedMoE:
        _latchmoe_external_shared_overlap_patch = False

        def __init__(self):
            self.multistream_overlap_shared_expert = True
            self.shared_multistream_overlap_gate = False
            self.is_internal_router = True
            self._shared_experts = object()
            self.moe_config = SimpleNamespace(
                dp_size=1,
                ep_size=1,
                tp_size=1,
                pcp_size=1,
            )

        def shared_forward_impl(self, hidden_states, router_logits):
            return ("upstream", hidden_states, router_logits)

    fake_fused_moe = SimpleNamespace(AscendFusedMoE=FakeAscendFusedMoE)
    patch_fused_moe._patch_external_shared_multistream_overlap(fake_fused_moe)

    layer = FakeAscendFusedMoE()
    assert layer.shared_forward_impl("hidden", "router") == (
        "upstream",
        "hidden",
        "router",
    )


def test_seam_forward_prefill_resident_uses_native_fused_moe(monkeypatch):
    import torch

    from vllm_moe_offload_ascend.patches import patch_fused_moe

    class FakeRunner:
        _ascend_moe_offload_seam_patch = False

        def _select_forward(self):
            raise AssertionError("original select_forward should not be used")

    fake_fused_moe = SimpleNamespace(AscendMoERunner=FakeRunner)
    patch_fused_moe._patch_ascend_moe_runner(fake_fused_moe)

    events = []

    class FakeRuntime:
        config = SimpleNamespace(
            offload_stage_seam=True,
            gmm_profile_path="/tmp/test-prefill-resident-native.jsonl",
        )

        def __init__(self):
            self.resident_checks = []

        def is_resident_layer(self, layer_id):
            self.resident_checks.append(int(layer_id))
            return int(layer_id) == 3

        def _record_profile_event(self, name, *, layer_id, start, payload):
            events.append(
                {
                    "name": name,
                    "layer_id": layer_id,
                    "payload": payload,
                    "start_type": type(start).__name__,
                }
            )

    runtime = FakeRuntime()
    vllm_moe_offload_ascend.register()
    import vllm_moe_offload_ascend.moe_offload.runtime as runtime_mod

    monkeypatch.setattr(patch_fused_moe, "_current_forward_is_prefill", lambda: True)
    monkeypatch.setattr(
        runtime_mod,
        "get_moe_offload_runtime",
        lambda: runtime,
    )

    runner = FakeRunner()
    runner._seam_active = True
    runner._seam_layer_id = 3
    runner._seam_num_logical_experts = 128
    runner.layer_name = "model.layers.3.mlp.experts"

    hidden_states = torch.empty((4, 16), dtype=torch.float32)
    router_logits = torch.empty((4, 128), dtype=torch.float32)
    expected = torch.empty_like(hidden_states)
    calls = []

    def fake_moe_forward(
        hidden,
        logits,
        shared_experts_input,
        input_ids,
        layer_name,
        hidden_dim_unpadded,
    ):
        calls.append(
            (
                "moe_forward",
                layer_name,
                tuple(hidden.shape),
                hidden_dim_unpadded,
            )
        )
        return expected

    def forbidden_op(*args, **kwargs):
        raise AssertionError("resident Prefill must bypass router/stage/mlp seam")

    monkeypatch.setattr(
        torch.ops.vllm,
        "moe_forward",
        fake_moe_forward,
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops.vllm,
        "moe_router_indirect",
        forbidden_op,
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops.vllm,
        "moe_offload_stage",
        forbidden_op,
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops.vllm,
        "moe_mlp",
        forbidden_op,
        raising=False,
    )

    out = runner._seam_forward_entry(
        hidden_states,
        router_logits,
        shared_experts_input=None,
        input_ids=None,
        layer_name="ignored.layer.name",
        hidden_dim_unpadded=0,
    )

    assert out is expected
    assert runtime.resident_checks == [3]
    assert calls == [("moe_forward", "ignored.layer.name", (4, 16), 0)]
    assert events == [
        {
            "name": "prefill_resident_native",
            "layer_id": 3,
            "payload": {"n_tokens": 4, "path": "native_fused_moe"},
            "start_type": "float",
        }
    ]


@pytest.mark.parametrize("compile_tracing", [False, True])
def test_seam_forward_compares_native_layer_boundary_without_changing_result(
    monkeypatch,
    capsys,
    compile_tracing,
):
    import torch

    from vllm_moe_offload_ascend.moe_offload.config import MoeOffloadConfig
    from vllm_moe_offload_ascend.moe_offload.runtime import MoeOffloadRuntime
    from vllm_moe_offload_ascend.patches import patch_fused_moe
    from vllm.model_executor.layers.fused_moe.runner import moe_runner

    class FakeRunner:
        _ascend_moe_offload_seam_patch = False

        def _select_forward(self):
            raise AssertionError("original select_forward should not be used")

    fake_fused_moe = SimpleNamespace(AscendMoERunner=FakeRunner)
    patch_fused_moe._patch_ascend_moe_runner(fake_fused_moe)

    runtime = MoeOffloadRuntime(
        MoeOffloadConfig(
            enabled=True,
            num_slots=8,
            offload_stage_seam=True,
        )
    )
    vllm_moe_offload_ascend.register()
    import vllm_moe_offload_ascend.moe_offload.runtime as runtime_mod

    monkeypatch.setattr(
        runtime_mod,
        "get_moe_offload_runtime",
        lambda: runtime,
    )
    monkeypatch.setenv(
        "VLLM_ASCEND_MOE_OFFLOAD_COMPARE_LAYER_BOUNDARY",
        "1",
    )
    monkeypatch.setattr(
        patch_fused_moe,
        "_is_torch_compile_tracing",
        lambda: compile_tracing,
    )

    runner = FakeRunner()
    runner._seam_active = True
    runner._seam_layer_id = 3
    runner._seam_num_logical_experts = 128
    runner.layer_name = "model.layers.3.mlp.experts"

    expected_layer = object()
    lookup_names = []

    def fake_get_layer_from_name(name):
        lookup_names.append(name)
        return expected_layer

    monkeypatch.setattr(
        moe_runner,
        "get_layer_from_name",
        fake_get_layer_from_name,
    )

    hidden_states = torch.zeros((4, 16), dtype=torch.float32)
    router_logits = torch.zeros((4, 128), dtype=torch.float32)
    native_out = (
        torch.zeros_like(hidden_states),
        torch.ones_like(hidden_states),
    )
    offload_out = torch.full_like(hidden_states, 3.0)
    calls = []
    native_inputs = []

    def fake_moe_forward(*args):
        native_inputs.append((args[0], args[1], args[4]))
        calls.append(
            (
                "native",
                runtime.should_use_fixed_slot_plan_for_layer(3),
            )
        )
        return native_out

    def fake_router(*args):
        calls.append(
            (
                "router",
                runtime.should_use_fixed_slot_plan_for_layer(3),
            )
        )
        return (
            torch.ones((4, 2), dtype=torch.float32),
            torch.zeros((4, 2), dtype=torch.int32),
        )

    def fake_stage(*args):
        calls.append(
            (
                "stage",
                runtime.should_use_fixed_slot_plan_for_layer(3),
            )
        )

    def fake_mlp(*args):
        calls.append(
            (
                "mlp",
                runtime.should_use_fixed_slot_plan_for_layer(3),
            )
        )
        return offload_out

    monkeypatch.setattr(
        torch.ops.vllm,
        "moe_forward",
        fake_moe_forward,
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops.vllm,
        "moe_router_indirect",
        fake_router,
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops.vllm,
        "moe_offload_stage",
        fake_stage,
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops.vllm,
        "moe_mlp",
        fake_mlp,
        raising=False,
    )

    result = runner._seam_forward_entry(
        hidden_states,
        router_logits,
        shared_experts_input=None,
        input_ids=None,
        layer_name="from_forward_context",
    )

    assert result is offload_out
    assert lookup_names == [
        "from_forward_context",
        "model.layers.3.mlp.experts",
    ]
    native_hidden_states, native_router_logits, native_layer_name = native_inputs[0]
    assert native_hidden_states is not hidden_states
    assert native_router_logits is not router_logits
    assert torch.equal(native_hidden_states, hidden_states)
    assert torch.equal(native_router_logits, router_logits)
    assert native_layer_name == "model.layers.3.mlp.experts"
    assert calls == [
        ("native", False),
        ("router", True),
        ("stage", True),
        ("mlp", True),
    ]
    output = capsys.readouterr().out
    if compile_tracing:
        assert "SEW_LAYER_BOUNDARY_COMPARE" not in output
    else:
        assert "SEW_LAYER_BOUNDARY_COMPARE layer=3 tokens=4" in output
        assert "output_equal=False" in output
        assert "output_max_abs=2.0" in output
    assert runtime.should_use_fixed_slot_plan_for_layer(3) is True


def test_b2_prefill_skips_resident_layer_before_route_stats(monkeypatch):
    import torch

    import vllm_moe_offload_ascend.moe_offload.runtime as runtime_impl
    from vllm_moe_offload_ascend.patches import patch_fused_moe

    class FakeCommMethod:
        _ascend_moe_offload_runtime_patch = False

        def fused_experts(self, fused_experts_input):
            return "native"

        def _maybe_apply_moe_offload_plan(self, fused_experts_input):
            raise AssertionError("original offload plan should not be used")

    class FakeRuntime:
        config = SimpleNamespace(
            graph_compatible_offload=False,
            b2_wave_prefill=True,
            gmm_profile_path="/tmp/test-prefill-resident-native-comm.jsonl",
        )

        def __init__(self):
            self.route_stats_calls = 0
            self.fixed_slot_checks = 0
            self.events = []

        def is_resident_layer(self, layer_id):
            return int(layer_id) == 7

        def should_use_fixed_slot_plan_for_layer(self, layer_id):
            self.fixed_slot_checks += 1
            return True

        def consume_prefill_route_stats_record(self, **kwargs):
            self.route_stats_calls += 1
            raise AssertionError("resident layer must skip route stats")

        def _record_profile_event(self, name, *, layer_id, start, payload):
            self.events.append(
                {
                    "name": name,
                    "layer_id": int(layer_id),
                    "payload": payload,
                    "start_type": type(start).__name__,
                }
            )

    runtime = FakeRuntime()
    monkeypatch.setattr(patch_fused_moe, "_current_forward_is_prefill", lambda: True)
    monkeypatch.setattr(
        runtime_impl,
        "get_moe_offload_runtime",
        lambda: runtime,
    )

    fake_comm = SimpleNamespace(
        MoECommMethod=FakeCommMethod,
        build_token_dispatch_input=lambda **kwargs: kwargs,
        build_mlp_compute_input=lambda **kwargs: kwargs,
        FusedExpertsResult=SimpleNamespace,
        MoEFusedExpertsInput=SimpleNamespace,
        MoEWeights=SimpleNamespace,
        MoEOffloadParams=SimpleNamespace,
        MoERoutingParams=SimpleNamespace,
        setup_moe_comm_method=None,
    )
    patch_fused_moe._patch_moe_comm_method_runtime_hooks(fake_comm)

    comm = FakeCommMethod()
    TokenDispatcherWithAllGather = type("TokenDispatcherWithAllGather", (), {})
    comm.token_dispatcher = TokenDispatcherWithAllGather()
    comm._run_b2_wave_prefill = lambda **kwargs: (_ for _ in ()).throw(
        AssertionError("resident layer must not run B2")
    )
    fused_experts_input = SimpleNamespace(
        hidden_states=torch.empty((4, 16)),
        topk_ids=torch.tensor([[1, 2], [2, 3], [3, 4], [4, 5]]),
        offload=SimpleNamespace(enabled=True, layer_id=7),
    )

    assert comm._maybe_run_b2_wave_prefill(
        fused_experts_input,
        before_dispatch_evt=None,
    ) is None
    assert comm.fused_experts(fused_experts_input) == "native"
    assert runtime.route_stats_calls == 0
    assert runtime.fixed_slot_checks == 0
    assert runtime.events == [
        {
            "name": "prefill_resident_native",
            "layer_id": 7,
            "payload": {
                "n_tokens": 4,
                "path": "native_fused_moe",
                "entry": "comm_method",
            },
            "start_type": "float",
        }
    ]


def test_b2_runs_exact_pair_waves_for_multi_request_decode_overflow(monkeypatch):
    import torch

    import vllm_moe_offload_ascend.moe_offload.runtime as runtime_impl
    from vllm_moe_offload_ascend.patches import patch_fused_moe

    class FakeCommMethod:
        _ascend_moe_offload_runtime_patch = False

        def fused_experts(self, fused_experts_input):
            return "native"

        def _maybe_apply_moe_offload_plan(self, fused_experts_input):
            return fused_experts_input

    class FakeRuntime:
        config = SimpleNamespace(
            graph_compatible_offload=False,
            b2_wave_prefill=True,
            gmm_profile_path="",
        )

        def is_resident_layer(self, layer_id):
            return False

        def should_use_fixed_slot_plan_for_layer(self, layer_id):
            return True

        def consume_prefill_route_stats_record(self, **kwargs):
            return None

        def should_use_b2_pair_waves(self, *, layer_id, active_expert_count):
            return active_expert_count > 2

    monkeypatch.setattr(patch_fused_moe, "_current_forward_is_prefill", lambda: False)
    monkeypatch.setattr(
        runtime_impl,
        "get_moe_offload_runtime",
        lambda: FakeRuntime(),
    )

    fake_comm = SimpleNamespace(
        MoECommMethod=FakeCommMethod,
        build_token_dispatch_input=lambda **kwargs: kwargs,
        build_mlp_compute_input=lambda **kwargs: kwargs,
        FusedExpertsResult=SimpleNamespace,
        MoEFusedExpertsInput=SimpleNamespace,
        MoEWeights=SimpleNamespace,
        MoEOffloadParams=SimpleNamespace,
        MoERoutingParams=SimpleNamespace,
        setup_moe_comm_method=None,
    )
    patch_fused_moe._patch_moe_comm_method_runtime_hooks(fake_comm)

    comm = FakeCommMethod()
    TokenDispatcherWithAllGather = type("TokenDispatcherWithAllGather", (), {})
    comm.token_dispatcher = TokenDispatcherWithAllGather()
    calls = []
    comm._run_b2_wave_prefill = lambda **kwargs: calls.append(kwargs) or "b2"
    fused_experts_input = SimpleNamespace(
        hidden_states=torch.empty((4, 16)),
        topk_ids=torch.tensor([[1, 2], [2, 3], [3, 4], [4, 5]]),
        offload=SimpleNamespace(enabled=True, layer_id=7),
    )

    assert comm._maybe_run_b2_wave_prefill(
        fused_experts_input,
        before_dispatch_evt=None,
    ) == "b2"
    assert calls[0]["control_profile"]["forward_phase"] == "decode"


def test_b2_fused_shared_counts_only_routed_pairs_for_wave_capacity(monkeypatch):
    import torch

    import vllm_moe_offload_ascend.moe_offload.runtime as runtime_impl
    from vllm_moe_offload_ascend.patches import patch_fused_moe

    class FakeCommMethod:
        _ascend_moe_offload_runtime_patch = False

        def fused_experts(self, fused_experts_input):
            return "native"

        def _maybe_apply_moe_offload_plan(self, fused_experts_input):
            return fused_experts_input

    fused_layout = SimpleNamespace(
        routed_expert_count=2,
        shared_expert_count=1,
        total_logical_expert_count=3,
    )

    class FakeRuntime:
        config = SimpleNamespace(
            graph_compatible_offload=False,
            b2_wave_prefill=True,
            gmm_profile_path="",
            max_num_seqs_hint=0,
            num_slots=1,
        )

        def fused_shared_lane_layout(self, layer_id):
            assert int(layer_id) == 7
            return fused_layout

        def is_resident_layer(self, layer_id):
            return False

        def should_use_fixed_slot_plan_for_layer(self, layer_id):
            return True

        def consume_prefill_route_stats_record(self, **kwargs):
            return None

        def should_use_b2_pair_waves(self, *, layer_id, active_expert_count):
            return active_expert_count > self.config.num_slots

    monkeypatch.setattr(patch_fused_moe, "_current_forward_is_prefill", lambda: False)
    monkeypatch.setattr(runtime_impl, "get_moe_offload_runtime", lambda: FakeRuntime())
    fake_comm = SimpleNamespace(
        MoECommMethod=FakeCommMethod,
        build_token_dispatch_input=lambda **kwargs: kwargs,
        build_mlp_compute_input=lambda **kwargs: kwargs,
        FusedExpertsResult=SimpleNamespace,
        MoEFusedExpertsInput=SimpleNamespace,
        MoEWeights=SimpleNamespace,
        MoEOffloadParams=SimpleNamespace,
        MoERoutingParams=SimpleNamespace,
        setup_moe_comm_method=None,
    )
    patch_fused_moe._patch_moe_comm_method_runtime_hooks(fake_comm)

    comm = FakeCommMethod()
    TokenDispatcherWithAllGather = type("TokenDispatcherWithAllGather", (), {})
    comm.token_dispatcher = TokenDispatcherWithAllGather()
    calls = []
    comm._run_b2_fused_shared_once = lambda **kwargs: calls.append(kwargs) or "b2"
    fused_experts_input = SimpleNamespace(
        hidden_states=torch.empty((2, 16)),
        topk_ids=torch.tensor([[0, 1, 2], [1, 0, 2]]),
        topk_weights=torch.ones((2, 3)),
        offload=SimpleNamespace(enabled=True, layer_id=7),
    )

    assert comm._maybe_run_b2_wave_prefill(
        fused_experts_input,
        before_dispatch_evt=None,
    ) == "b2"
    assert calls[0]["token_counts"] == {0: 2, 1: 2}
    assert calls[0]["control_profile"]["pair_offsets_by_expert"] is None


def test_b2_fused_shared_executes_pinned_suffix_once_and_combines_outputs(monkeypatch):
    import torch

    import vllm_moe_offload_ascend.moe_offload.runtime as runtime_impl
    from vllm_moe_offload_ascend.patches import patch_fused_moe

    @dataclass(frozen=True)
    class FakeFusedExpertsInput:
        hidden_states: object
        topk_ids: object
        topk_weights: object
        offload: object

    @dataclass(frozen=True)
    class FakeFusedExpertsResult:
        routed_out: object
        dispatch_marker: str

    shared_calls = []

    class FakeCommMethod:
        _ascend_moe_offload_runtime_patch = False

        def fused_experts(self, fused_experts_input):
            shared_calls.append((fused_experts_input, self.token_dispatcher.top_k))
            return FakeFusedExpertsResult(
                routed_out=torch.full_like(fused_experts_input.hidden_states, 3.0),
                dispatch_marker="shared",
            )

        def _maybe_apply_moe_offload_plan(self, fused_experts_input):
            return fused_experts_input

    class PinnedWeights:
        def __init__(self):
            self.expected_device_types = []

        def validate_backend_ready(self, *, expected_device_type):
            self.expected_device_types.append(expected_device_type)

    pinned_weights = PinnedWeights()
    events = []

    class FakeRuntime:
        def capture_safe_slot_weights(self, *, layer_id):
            assert int(layer_id) == 7
            return pinned_weights

        def _record_profile_event(self, name, *, layer_id, start, payload):
            events.append((name, int(layer_id), payload))

    runtime = FakeRuntime()
    monkeypatch.setattr(runtime_impl, "get_moe_offload_runtime", lambda: runtime)
    fake_comm = SimpleNamespace(
        MoECommMethod=FakeCommMethod,
        build_token_dispatch_input=lambda **kwargs: kwargs,
        build_mlp_compute_input=lambda **kwargs: kwargs,
        FusedExpertsResult=FakeFusedExpertsResult,
        MoEFusedExpertsInput=FakeFusedExpertsInput,
        MoEWeights=SimpleNamespace,
        MoEOffloadParams=SimpleNamespace,
        MoERoutingParams=SimpleNamespace,
        setup_moe_comm_method=None,
    )
    patch_fused_moe._patch_moe_comm_method_runtime_hooks(fake_comm)

    comm = FakeCommMethod()
    comm.token_dispatcher = SimpleNamespace(top_k=2)
    routed_calls = []

    def run_routed_b2(**kwargs):
        routed_calls.append(kwargs)
        return FakeFusedExpertsResult(
            routed_out=torch.full_like(
                kwargs["fused_experts_input"].hidden_states,
                2.0,
            ),
            dispatch_marker="routed",
        )

    comm._run_b2_wave_prefill = run_routed_b2
    comm._with_prepared_slot_weights = lambda input_value, weights: (
        input_value if weights is pinned_weights else None
    )
    fused_experts_input = FakeFusedExpertsInput(
        hidden_states=torch.zeros((2, 4)),
        topk_ids=torch.tensor([[0, 1, 2], [1, 0, 2]], dtype=torch.int32),
        topk_weights=torch.tensor([[0.6, 0.4, 1.0], [0.7, 0.3, 1.0]]),
        offload=SimpleNamespace(
            enabled=True,
            layer_id=7,
            expected_device_type="cpu",
        ),
    )
    layout = SimpleNamespace(
        routed_expert_count=2,
        shared_expert_count=1,
        total_logical_expert_count=3,
    )

    result = comm._run_b2_fused_shared_once(
        fused_experts_input=fused_experts_input,
        before_dispatch_evt=None,
        token_counts={0: 2, 1: 2},
        shared_layout=layout,
        control_profile={"pair_offsets_by_expert": (0, 1)},
    )

    assert len(routed_calls) == 1
    assert routed_calls[0]["fused_experts_input"].topk_ids.tolist() == [[0, 1], [1, 0]]
    assert routed_calls[0]["control_profile"]["pair_offsets_by_expert"] is None
    assert len(shared_calls) == 1
    shared_input, dispatch_top_k = shared_calls[0]
    assert dispatch_top_k == 1
    assert shared_input.topk_ids.tolist() == [[2], [2]]
    assert shared_input.topk_weights.tolist() == [[1.0], [1.0]]
    assert comm.token_dispatcher.top_k == 2
    assert pinned_weights.expected_device_types == ["cpu"]
    assert torch.equal(result.routed_out, torch.full((2, 4), 5.0))
    assert result.dispatch_marker == "routed"
    assert events == [
        (
            "fused_shared_b2_once",
            7,
            {
                "routed_expert_count": 2,
                "pinned_shared_expert_count": 1,
                "shared_pairs": 2,
                "shared_pair_execution_count": 1,
                "routed_pairs": 4,
            },
        )
    ]


def test_profile_b2_runs_without_explicit_feature_flag(monkeypatch):
    import torch

    import vllm_moe_offload_ascend.moe_offload.runtime as runtime_impl
    from vllm_moe_offload_ascend.ops.fused_moe import moe_offload_stage_op
    from vllm_moe_offload_ascend.patches import patch_fused_moe

    class FakeCommMethod:
        _ascend_moe_offload_runtime_patch = False

        def fused_experts(self, fused_experts_input):
            return "native"

        def _maybe_apply_moe_offload_plan(self, fused_experts_input):
            return fused_experts_input

    cached_stats = SimpleNamespace(
        token_counts_by_expert={0: 1, 1: 1, 2: 1, 3: 1},
        pair_offsets_by_expert=None,
    )

    class FakeRuntime:
        config = SimpleNamespace(
            graph_compatible_offload=False,
            b2_wave_prefill=False,
            gmm_profile_path="",
            max_num_seqs_hint=0,
            num_slots=2,
        )

        def is_resident_layer(self, layer_id):
            return False

        def should_use_fixed_slot_plan_for_layer(self, layer_id):
            return True

        def consume_prefill_route_stats_record(self, **kwargs):
            return cached_stats

        def should_use_b2_pair_waves(self, *, layer_id, active_expert_count):
            return False

    monkeypatch.setattr(patch_fused_moe, "_current_forward_is_prefill", lambda: None)
    monkeypatch.setattr(
        runtime_impl,
        "get_moe_offload_runtime",
        lambda: FakeRuntime(),
    )

    fake_comm = SimpleNamespace(
        MoECommMethod=FakeCommMethod,
        build_token_dispatch_input=lambda **kwargs: kwargs,
        build_mlp_compute_input=lambda **kwargs: kwargs,
        FusedExpertsResult=SimpleNamespace,
        MoEFusedExpertsInput=SimpleNamespace,
        MoEWeights=SimpleNamespace,
        MoEOffloadParams=SimpleNamespace,
        MoERoutingParams=SimpleNamespace,
        setup_moe_comm_method=None,
    )
    patch_fused_moe._patch_moe_comm_method_runtime_hooks(fake_comm)

    comm = FakeCommMethod()
    TokenDispatcherWithAllGather = type("TokenDispatcherWithAllGather", (), {})
    comm.token_dispatcher = TokenDispatcherWithAllGather()
    calls = []
    comm._run_b2_wave_prefill = lambda **kwargs: calls.append(kwargs) or "b2"
    fused_experts_input = SimpleNamespace(
        hidden_states=torch.empty((1, 16)),
        topk_ids=torch.tensor([[0, 1, 2, 3]], dtype=torch.int64),
        offload=SimpleNamespace(enabled=True, layer_id=7),
    )

    token = moe_offload_stage_op.set_moe_offload_profile_run_active(True)
    try:
        assert comm._maybe_run_b2_wave_prefill(
            fused_experts_input,
            before_dispatch_evt=None,
        ) == "b2"
    finally:
        moe_offload_stage_op.reset_moe_offload_profile_run_active(token)

    assert calls[0]["control_profile"]["route_stats_cache_hit"] is True
    assert calls[0]["control_profile"]["profile_adaptive_wave"] is True

    token = moe_offload_stage_op.set_moe_offload_graph_warmup_active(True)
    try:
        assert comm._maybe_run_b2_wave_prefill(
            fused_experts_input,
            before_dispatch_evt=None,
        ) == "b2"
    finally:
        moe_offload_stage_op.reset_moe_offload_graph_warmup_active(token)

    assert calls[1]["control_profile"]["route_stats_cache_hit"] is True
    assert calls[1]["control_profile"]["profile_adaptive_wave"] is False
    assert calls[1]["control_profile"]["graph_warmup_adaptive_wave"] is True


def test_b2_recoverable_wave_failure_runs_full_layer_fallback(monkeypatch):
    import torch

    import vllm_moe_offload_ascend.moe_offload.runtime as runtime_impl
    from vllm_moe_offload_ascend.moe_offload.phase_split import B2WaveFallback
    from vllm_moe_offload_ascend.patches import patch_fused_moe

    class FakeCommMethod:
        _ascend_moe_offload_runtime_patch = False

        def fused_experts(self, fused_experts_input):
            return "native"

        def _maybe_apply_moe_offload_plan(self, fused_experts_input):
            return fused_experts_input

    class FakeRuntime:
        config = SimpleNamespace(
            graph_compatible_offload=False,
            b2_wave_prefill=True,
            gmm_profile_path="",
            num_slots=2,
        )

        def is_resident_layer(self, layer_id):
            return False

        def should_use_fixed_slot_plan_for_layer(self, layer_id):
            return True

        def consume_prefill_route_stats_record(self, **kwargs):
            return None

        def should_use_b2_pair_waves(self, *, layer_id, active_expert_count):
            return active_expert_count > 2

    monkeypatch.setattr(patch_fused_moe, "_current_forward_is_prefill", lambda: False)
    monkeypatch.setattr(
        runtime_impl,
        "get_moe_offload_runtime",
        lambda: FakeRuntime(),
    )

    fake_comm = SimpleNamespace(
        MoECommMethod=FakeCommMethod,
        build_token_dispatch_input=lambda **kwargs: kwargs,
        build_mlp_compute_input=lambda **kwargs: kwargs,
        FusedExpertsResult=SimpleNamespace,
        MoEFusedExpertsInput=SimpleNamespace,
        MoEWeights=SimpleNamespace,
        MoEOffloadParams=SimpleNamespace,
        MoERoutingParams=SimpleNamespace,
        setup_moe_comm_method=None,
    )
    patch_fused_moe._patch_moe_comm_method_runtime_hooks(fake_comm)

    comm = FakeCommMethod()
    TokenDispatcherWithAllGather = type("TokenDispatcherWithAllGather", (), {})
    comm.token_dispatcher = TokenDispatcherWithAllGather()
    comm._run_b2_wave_prefill = lambda **kwargs: (_ for _ in ()).throw(
        B2WaveFallback("incomplete native pair coverage")
    )
    fallback_calls = []
    comm._run_b2_full_layer_reference = (
        lambda **kwargs: fallback_calls.append(kwargs) or "full_layer"
    )
    fused_experts_input = SimpleNamespace(
        hidden_states=torch.empty((4, 16)),
        topk_ids=torch.tensor([[1, 2], [2, 3], [3, 4], [4, 5]]),
        offload=SimpleNamespace(enabled=True, layer_id=7),
    )

    assert comm._maybe_run_b2_wave_prefill(
        fused_experts_input,
        before_dispatch_evt=None,
    ) == "full_layer"
    assert len(fallback_calls) == 1
    assert fallback_calls[0]["fallback_reason"].startswith(
        "B2WaveFallback: incomplete native pair coverage"
    )


def test_stage_op_defers_capacity_overflow_to_b2_without_phase_hint(monkeypatch):
    import torch

    import vllm_moe_offload_ascend.moe_offload.runtime as runtime_impl
    from vllm_moe_offload_ascend.ops.fused_moe import moe_offload_stage_op

    class FakeRuntime:
        config = SimpleNamespace(
            b2_wave_prefill=True,
            num_slots=2,
            max_num_seqs_hint=1,
        )

        def __init__(self):
            self.cached = None
            self.stage_calls = 0

        def is_static_residency_regime(self, num_logical_experts):
            return False

        def should_use_fixed_slot_plan_for_layer(self, layer_id):
            return True

        def is_layer_registered(self, layer_id):
            return True

        def should_use_b2_wave_prefill(
            self,
            *,
            layer_id,
            active_expert_count,
            is_prefill,
        ):
            return bool(is_prefill) and active_expert_count > self.config.num_slots

        def cache_prefill_route_stats(self, **kwargs):
            self.cached = kwargs

        def stage_fixed_slot_plan(self, **kwargs):
            self.stage_calls += 1
            raise AssertionError("overflow must be handed to B2, not staged once")

    runtime = FakeRuntime()
    monkeypatch.setattr(runtime_impl, "_is_current_graph_capturing", lambda: False)
    monkeypatch.setattr(runtime_impl, "get_moe_offload_runtime", lambda: runtime)

    topk_ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64)
    moe_offload_stage_op._moe_offload_stage_impl(
        topk_ids,
        layer_id=7,
        num_logical_experts=64,
        phase=moe_offload_stage_op.PHASE_UNKNOWN,
    )

    assert runtime.stage_calls == 0
    assert runtime.cached is None


def test_profile_dummy_run_scopes_adaptive_wave_context(monkeypatch):
    from vllm_moe_offload_ascend.ops.fused_moe import moe_offload_stage_op
    from vllm_moe_offload_ascend.patches import patch_fused_moe

    calls = []

    class FakeRunner:
        def _dummy_run(self, num_tokens, is_profile=False):
            calls.append(moe_offload_stage_op.is_moe_offload_profile_run_active())
            return num_tokens

    fake_module = ModuleType("fake_vllm_model_runner")
    fake_module.GPUModelRunner = FakeRunner
    fake_module.NPUModelRunner = FakeRunner

    def fake_import(name):
        if name == "vllm_ascend.worker.model_runner_v1":
            return fake_module
        raise ImportError(name)

    monkeypatch.setattr(patch_fused_moe.importlib, "import_module", fake_import)

    patch_fused_moe._patch_model_runner_profile_context()
    runner = FakeRunner()
    assert moe_offload_stage_op.is_moe_offload_profile_run_active() is False
    assert runner._dummy_run(1, True) == 1
    assert moe_offload_stage_op.is_moe_offload_profile_run_active() is False
    assert runner._dummy_run(1, False) == 1
    assert calls == [True, False]


def test_graph_warmup_scopes_adaptive_wave_context(monkeypatch):
    from vllm_moe_offload_ascend.ops.fused_moe import moe_offload_stage_op
    from vllm_moe_offload_ascend.patches import patch_fused_moe

    calls = []

    class FakeRunner:
        def capture_model(self):
            calls.append(
                (
                    moe_offload_stage_op.is_moe_offload_graph_warmup_active(),
                    moe_offload_stage_op.is_moe_offload_profile_run_active(),
                )
            )
            return 1

    fake_module = ModuleType("fake_vllm_model_runner")
    fake_module.GPUModelRunner = FakeRunner
    fake_module.NPUModelRunner = FakeRunner

    def fake_import(name):
        if name == "vllm_ascend.worker.model_runner_v1":
            return fake_module
        raise ImportError(name)

    monkeypatch.setattr(patch_fused_moe.importlib, "import_module", fake_import)

    patch_fused_moe._patch_model_runner_graph_warmup_context()
    runner = FakeRunner()
    assert moe_offload_stage_op.is_moe_offload_graph_warmup_active() is False
    assert runner.capture_model() == 1
    assert moe_offload_stage_op.is_moe_offload_graph_warmup_active() is False
    assert calls == [(True, False)]


def test_stage_op_profile_overflow_adapts_to_b2_without_feature_flag(monkeypatch):
    import torch

    import vllm_moe_offload_ascend.moe_offload.runtime as runtime_impl
    from vllm_moe_offload_ascend.ops.fused_moe import moe_offload_stage_op

    class FakeRuntime:
        config = SimpleNamespace(
            b2_wave_prefill=False,
            num_slots=2,
            max_num_seqs_hint=0,
        )

        def __init__(self):
            self.cached = None
            self.stage_calls = 0
            self.events = []

        def is_static_residency_regime(self, num_logical_experts):
            return False

        def should_use_fixed_slot_plan_for_layer(self, layer_id):
            return True

        def is_layer_registered(self, layer_id):
            return True

        def cache_prefill_route_stats(self, **kwargs):
            self.cached = kwargs

        def stage_fixed_slot_plan(self, **kwargs):
            self.stage_calls += 1
            raise AssertionError("profile overflow must be handed to B2")

        def _record_profile_event(self, name, *, layer_id, start, payload):
            self.events.append((name, layer_id, payload))

    runtime = FakeRuntime()
    monkeypatch.setattr(runtime_impl, "_is_current_graph_capturing", lambda: False)
    monkeypatch.setattr(runtime_impl, "get_moe_offload_runtime", lambda: runtime)

    token = moe_offload_stage_op.set_moe_offload_profile_run_active(True)
    try:
        moe_offload_stage_op._moe_offload_stage_impl(
            torch.tensor([[0, 1, 2, 3]], dtype=torch.int64),
            layer_id=7,
            num_logical_experts=64,
            phase=moe_offload_stage_op.PHASE_UNKNOWN,
        )
    finally:
        moe_offload_stage_op.reset_moe_offload_profile_run_active(token)

    assert runtime.stage_calls == 0
    assert runtime.cached is not None
    assert runtime.cached["token_counts_by_expert"] == {0: 1, 1: 1, 2: 1, 3: 1}
    assert runtime.events == [
        (
            "profile_adaptive_wave_decision",
            7,
            {
                "active_expert_count": 4,
                "num_slots": 2,
                "decision": "multi_wave",
                "forward_phase": moe_offload_stage_op.PHASE_UNKNOWN,
                "decision_source": "profile",
            },
        )
    ]


def test_stage_op_graph_warmup_overflow_adapts_to_b2(monkeypatch):
    import torch

    import vllm_moe_offload_ascend.moe_offload.runtime as runtime_impl
    from vllm_moe_offload_ascend.ops.fused_moe import moe_offload_stage_op

    class FakeRuntime:
        config = SimpleNamespace(
            b2_wave_prefill=True,
            num_slots=2,
            max_num_seqs_hint=1,
        )

        def __init__(self):
            self.cached = None
            self.stage_calls = 0
            self.events = []

        def is_static_residency_regime(self, num_logical_experts):
            return False

        def should_use_fixed_slot_plan_for_layer(self, layer_id):
            return True

        def is_layer_registered(self, layer_id):
            return True

        def cache_prefill_route_stats(self, **kwargs):
            self.cached = kwargs

        def stage_fixed_slot_plan(self, **kwargs):
            self.stage_calls += 1
            raise AssertionError("graph warm-up overflow must be handed to B2")

        def _record_profile_event(self, name, *, layer_id, start, payload):
            self.events.append((name, layer_id, payload))

    runtime = FakeRuntime()
    monkeypatch.setattr(runtime_impl, "_is_current_graph_capturing", lambda: False)
    monkeypatch.setattr(runtime_impl, "get_moe_offload_runtime", lambda: runtime)

    token = moe_offload_stage_op.set_moe_offload_graph_warmup_active(True)
    try:
        moe_offload_stage_op._moe_offload_stage_impl(
            torch.tensor([[0, 1, 2, 3]], dtype=torch.int64),
            layer_id=7,
            num_logical_experts=64,
            phase=moe_offload_stage_op.PHASE_UNKNOWN,
        )
    finally:
        moe_offload_stage_op.reset_moe_offload_graph_warmup_active(token)

    assert runtime.stage_calls == 0
    assert runtime.cached is not None
    assert runtime.events[-1][2]["decision"] == "multi_wave"
    assert runtime.events[-1][2]["decision_source"] == "graph_warmup"


def test_stage_op_profile_fit_keeps_single_slot_staging(monkeypatch):
    import torch

    import vllm_moe_offload_ascend.moe_offload.runtime as runtime_impl
    from vllm_moe_offload_ascend.ops.fused_moe import moe_offload_stage_op

    class FakeRuntime:
        config = SimpleNamespace(
            b2_wave_prefill=False,
            num_slots=2,
            max_num_seqs_hint=0,
        )

        def __init__(self):
            self.cached = None
            self.stage_calls = []

        def is_static_residency_regime(self, num_logical_experts):
            return False

        def should_use_fixed_slot_plan_for_layer(self, layer_id):
            return True

        def is_layer_registered(self, layer_id):
            return True

        def stage_fixed_slot_plan(self, **kwargs):
            self.stage_calls.append(kwargs)

    runtime = FakeRuntime()
    monkeypatch.setattr(runtime_impl, "_is_current_graph_capturing", lambda: False)
    monkeypatch.setattr(runtime_impl, "get_moe_offload_runtime", lambda: runtime)

    token = moe_offload_stage_op.set_moe_offload_profile_run_active(True)
    try:
        moe_offload_stage_op._moe_offload_stage_impl(
            torch.tensor([[0, 1, 1, 0]], dtype=torch.int64),
            layer_id=7,
            num_logical_experts=64,
            phase=moe_offload_stage_op.PHASE_UNKNOWN,
        )
    finally:
        moe_offload_stage_op.reset_moe_offload_profile_run_active(token)

    assert runtime.cached is None
    assert runtime.stage_calls == [
        {
            "layer_id": 7,
            "active_experts": (0, 1),
            "num_logical_experts": 64,
            "phase": "unknown",
            "num_tokens": 1,
            "top_k": 4,
            "expert_token_counts": {0: 2, 1: 2},
        }
    ]


def test_stage_op_does_not_defer_decode_overflow_without_shape_hint(monkeypatch):
    import torch

    import vllm_moe_offload_ascend.moe_offload.runtime as runtime_impl
    from vllm_moe_offload_ascend.ops.fused_moe import moe_offload_stage_op

    class FakeRuntime:
        config = SimpleNamespace(
            b2_wave_prefill=True,
            num_slots=2,
            max_num_seqs_hint=0,
        )

        def __init__(self):
            self.cached = None
            self.stage_calls = 0

        def is_static_residency_regime(self, num_logical_experts):
            return False

        def should_use_fixed_slot_plan_for_layer(self, layer_id):
            return True

        def is_layer_registered(self, layer_id):
            return True

        def should_use_b2_wave_prefill(
            self,
            *,
            layer_id,
            active_expert_count,
            is_prefill,
        ):
            return bool(is_prefill) and active_expert_count > self.config.num_slots

        def cache_prefill_route_stats(self, **kwargs):
            self.cached = kwargs

        def stage_fixed_slot_plan(self, **kwargs):
            self.stage_calls += 1
            raise RuntimeError("working set overflow")

    runtime = FakeRuntime()
    monkeypatch.setattr(runtime_impl, "_is_current_graph_capturing", lambda: False)
    monkeypatch.setattr(runtime_impl, "get_moe_offload_runtime", lambda: runtime)

    topk_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.int64)
    with pytest.raises(RuntimeError, match="working set overflow"):
        moe_offload_stage_op._moe_offload_stage_impl(
            topk_ids,
            layer_id=7,
            num_logical_experts=64,
            phase=moe_offload_stage_op.PHASE_UNKNOWN,
        )

    assert runtime.stage_calls == 1
    assert runtime.cached is None


def test_stage_op_confirmed_decode_overflow_defers_to_exact_pair_waves(monkeypatch):
    import torch

    import vllm_moe_offload_ascend.moe_offload.runtime as runtime_impl
    from vllm_moe_offload_ascend.ops.fused_moe import moe_offload_stage_op

    class FakeRuntime:
        config = SimpleNamespace(
            b2_wave_prefill=True,
            num_slots=2,
            # Large enough that the token heuristic alone would say "prefill".
            max_num_seqs_hint=1,
        )

        def __init__(self):
            self.cached = None
            self.stage_calls = 0

        def is_static_residency_regime(self, num_logical_experts):
            return False

        def should_use_fixed_slot_plan_for_layer(self, layer_id):
            return True

        def is_layer_registered(self, layer_id):
            return True

        def should_use_b2_pair_waves(self, *, layer_id, active_expert_count):
            return active_expert_count > self.config.num_slots

        def cache_prefill_route_stats(self, **kwargs):
            self.cached = kwargs

        def stage_fixed_slot_plan(self, **kwargs):
            self.stage_calls += 1
            raise AssertionError("decode overflow must hand off to exact pair waves")

    runtime = FakeRuntime()
    monkeypatch.setattr(runtime_impl, "_is_current_graph_capturing", lambda: False)
    monkeypatch.setattr(runtime_impl, "get_moe_offload_runtime", lambda: runtime)

    # Many tokens (prefill-shaped) but phase is CONFIRMED decode.
    topk_ids = torch.tensor([[0, 1, 2, 3]] * 8, dtype=torch.int64)
    moe_offload_stage_op._moe_offload_stage_impl(
        topk_ids,
        layer_id=7,
        num_logical_experts=64,
        phase=moe_offload_stage_op.PHASE_DECODE,
    )

    assert runtime.stage_calls == 0
    assert runtime.cached is None


def test_stage_op_confirmed_prefill_overflow_defers_to_b2(monkeypatch):
    """Counterpart to the decode case: a CONFIRMED prefill that overflows slots
    is the legitimate B2 wave handoff and may cache route-stats when explicitly
    enabled, instead of staging the whole working set once.
    """
    import torch

    import vllm_moe_offload_ascend.moe_offload.runtime as runtime_impl
    from vllm_moe_offload_ascend.ops.fused_moe import moe_offload_stage_op

    class FakeRuntime:
        config = SimpleNamespace(
            b2_wave_prefill=True,
            num_slots=2,
            max_num_seqs_hint=0,
        )

        def __init__(self):
            self.cached = None
            self.stage_calls = 0

        def is_static_residency_regime(self, num_logical_experts):
            return False

        def should_use_fixed_slot_plan_for_layer(self, layer_id):
            return True

        def is_layer_registered(self, layer_id):
            return True

        def should_use_b2_pair_waves(self, *, layer_id, active_expert_count):
            return active_expert_count > self.config.num_slots

        def cache_prefill_route_stats(self, **kwargs):
            self.cached = kwargs

        def stage_fixed_slot_plan(self, **kwargs):
            self.stage_calls += 1
            raise AssertionError("prefill overflow must hand off to B2, not stage once")

    runtime = FakeRuntime()
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_ROUTE_STATS_CACHE", "1")
    monkeypatch.setattr(runtime_impl, "_is_current_graph_capturing", lambda: False)
    monkeypatch.setattr(runtime_impl, "get_moe_offload_runtime", lambda: runtime)

    topk_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.int64)
    moe_offload_stage_op._moe_offload_stage_impl(
        topk_ids,
        layer_id=7,
        num_logical_experts=64,
        phase=moe_offload_stage_op.PHASE_PREFILL,
    )

    assert runtime.stage_calls == 0
    assert runtime.cached is not None
    assert runtime.cached["layer_id"] == 7


def test_stage_op_small_topk_counter_uses_python_counts_and_keeps_flat_ids():
    import torch

    from vllm_moe_offload_ascend.ops.fused_moe.moe_offload_stage_op import (
        _count_active_experts_from_cpu_topk,
    )

    counts, flat_ids = _count_active_experts_from_cpu_topk(
        torch.tensor([3, 1, 3, -1, 7, 129], dtype=torch.int64),
        num_logical_experts=128,
    )

    assert counts == {1: 1, 3: 2, 7: 1}
    assert flat_ids == [3, 1, 3, -1, 7, 129]


def test_stage_op_large_topk_counter_does_not_keep_flat_ids():
    import torch

    from vllm_moe_offload_ascend.ops.fused_moe.moe_offload_stage_op import (
        _count_active_experts_from_cpu_topk,
    )

    counts, flat_ids = _count_active_experts_from_cpu_topk(
        torch.tensor([0, 1, 1, 2, 2, 2], dtype=torch.int64),
        num_logical_experts=128,
        small_threshold=2,
    )

    assert counts == {0: 1, 1: 2, 2: 3}
    assert flat_ids is None


# ---------------------------------------------------------------------------
# M4: router fake/real dtype alignment
# ---------------------------------------------------------------------------

def test_moe_mlp_fake_honors_unpadded_output_width():
    import torch

    from vllm_moe_offload_ascend.ops.fused_moe.moe_mlp_op import _moe_mlp_fake

    hidden = torch.empty((4, 16), dtype=torch.bfloat16)
    logits = torch.empty((4, 64), dtype=torch.float32)
    topk_weights = torch.empty((4, 2), dtype=torch.bfloat16)
    topk_ids = torch.empty((4, 2), dtype=torch.int32)

    out = _moe_mlp_fake(
        hidden,
        logits,
        topk_weights,
        topk_ids,
        None,
        None,
        "model.layers.3.mlp.experts",
        12,
    )

    assert out.shape == (4, 12)
    assert out.dtype == hidden.dtype


def test_moe_mlp_shared_fake_has_fixed_tuple_abi_and_unpadded_routed_width():
    import torch
    from vllm_moe_offload_ascend.ops.fused_moe.moe_mlp_op import _moe_mlp_shared_fake

    hidden = torch.empty((4, 16), dtype=torch.bfloat16)
    shared_input = torch.empty((4, 12), dtype=torch.bfloat16)
    shared_out, routed_out = _moe_mlp_shared_fake(
        hidden,
        torch.empty((4, 64)),
        torch.empty((4, 2)),
        torch.empty((4, 2), dtype=torch.int32),
        shared_input,
        None,
        "model.layers.3.mlp.experts",
        12,
    )

    assert shared_out.shape == shared_input.shape
    assert routed_out.shape == (4, 12)
    assert shared_out.dtype == routed_out.dtype == hidden.dtype


def test_moe_router_fake_returns_hidden_states_dtype_not_logits_dtype():
    """M4: the router fake op must proxy topk_weights as hidden_states.dtype
    (matching the real _native_select_experts cast), not router_logits.dtype.
    For indirect-router models the gate output is float32 while hidden is bf16;
    using logits.dtype would mis-specialize the compiled graph."""
    import torch
    from vllm_moe_offload_ascend.ops.fused_moe import moe_router_op

    hidden = torch.empty((4, 16), dtype=torch.bfloat16)
    logits = torch.empty((4, 64), dtype=torch.float32)  # gate output, different dtype

    topk_weights, topk_ids = moe_router_op._moe_router_fake(
        hidden_states=hidden,
        router_logits=logits,
        top_k=2,
        use_grouped_topk=False,
        renormalize=True,
        topk_group=None,
        num_expert_group=None,
        scoring_func="softmax",
        routed_scaling_factor=1.0,
        e_score_correction_bias=None,
        num_experts=64,
    )

    # Must match hidden_states dtype, not router_logits dtype.
    assert topk_weights.dtype == hidden.dtype
    assert topk_ids.dtype == torch.int32
    assert topk_weights.shape == (4, 2)
    assert topk_ids.shape == (4, 2)


def test_moe_router_fake_appends_fused_shared_suffix_shape():
    import torch

    from vllm_moe_offload_ascend.ops.fused_moe import moe_router_op

    topk_weights, topk_ids = moe_router_op._moe_router_fake(
        hidden_states=torch.empty((3, 16), dtype=torch.bfloat16),
        router_logits=torch.empty((3, 64), dtype=torch.float32),
        top_k=2,
        use_grouped_topk=False,
        renormalize=True,
        topk_group=None,
        num_expert_group=None,
        scoring_func="softmax",
        routed_scaling_factor=1.0,
        e_score_correction_bias=None,
        num_experts=64,
        mix_placement=True,
        num_shared_experts=2,
    )

    assert topk_weights.shape == (3, 4)
    assert topk_ids.shape == (3, 4)


def test_moe_router_indirect_uses_ascend_original_routed_scaling_factor(monkeypatch):
    """The seam must match AscendFusedMoE.apply after vLLM rewrites the layer field."""
    pytest.importorskip(
        "vllm_ascend.ops.fused_moe.experts_selector",
        reason="legacy router relocation applies only to MoE seam ABI 1",
    )
    import torch

    from vllm_moe_offload_ascend.ops.fused_moe import moe_router_op

    layer = SimpleNamespace(
        layer_id=7,
        runner=None,
        top_k=2,
        use_grouped_topk=True,
        renormalize=True,
        topk_group=2,
        num_expert_group=8,
        scoring_func="sigmoid",
        routed_scaling_factor=1.0,
        _original_routed_scaling_factor=1.8,
        e_score_correction_bias=torch.zeros(64),
        custom_routing_function=None,
        n_shared_experts=0,
        global_redundant_expert_num=0,
        moe_config=SimpleNamespace(num_experts=64),
    )
    calls = []

    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.runner.moe_runner.get_layer_from_name",
        lambda _name: layer,
    )
    monkeypatch.setattr(
        "vllm_ascend.quantization.methods.base.get_moe_num_logical_experts",
        lambda *_args, **_kwargs: 64,
    )

    def fake_select_experts(**kwargs):
        calls.append(kwargs)
        return torch.ones((4, 2), dtype=torch.bfloat16), torch.zeros(
            (4, 2), dtype=torch.int32
        )

    monkeypatch.setattr(
        "vllm_ascend.ops.fused_moe.experts_selector.select_experts",
        fake_select_experts,
    )

    moe_router_op._moe_router_indirect_impl(
        torch.zeros((4, 16), dtype=torch.bfloat16),
        torch.zeros((4, 64), dtype=torch.float32),
        "fixture.layer.experts",
    )

    assert calls[0]["routed_scaling_factor"] == 1.8


def test_moe_router_indirect_forwards_fused_shared_suffix_arguments(monkeypatch):
    pytest.importorskip(
        "vllm_ascend.ops.fused_moe.experts_selector",
        reason="legacy router relocation applies only to MoE seam ABI 1",
    )
    import torch

    from vllm_moe_offload_ascend.ops.fused_moe import moe_router_op

    layer = SimpleNamespace(
        layer_id=7,
        runner=None,
        top_k=2,
        use_grouped_topk=False,
        renormalize=True,
        topk_group=None,
        num_expert_group=None,
        scoring_func="softmax",
        routed_scaling_factor=1.0,
        e_score_correction_bias=None,
        custom_routing_function=None,
        n_shared_experts=2,
        mix_placement=True,
        global_redundant_expert_num=0,
        moe_config=SimpleNamespace(num_experts=66),
    )
    calls = []
    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.runner.moe_runner.get_layer_from_name",
        lambda _name: layer,
    )
    monkeypatch.setattr(
        "vllm_ascend.quantization.methods.base.get_moe_num_logical_experts",
        lambda *_args, **_kwargs: 64,
    )

    def fake_select_experts(**kwargs):
        calls.append(kwargs)
        return torch.ones((4, 4), dtype=torch.bfloat16), torch.zeros(
            (4, 4), dtype=torch.int32
        )

    monkeypatch.setattr(
        "vllm_ascend.ops.fused_moe.experts_selector.select_experts",
        fake_select_experts,
    )

    weights, ids = moe_router_op._moe_router_indirect_impl(
        torch.zeros((4, 16), dtype=torch.bfloat16),
        torch.zeros((4, 64), dtype=torch.float32),
        "fixture.layer.experts",
    )

    assert weights.shape == ids.shape == (4, 4)
    assert calls[0]["num_experts"] == 64
    assert calls[0]["num_logical_experts"] == 64
    assert calls[0]["mix_placement"] is True
    assert calls[0]["num_shared_experts"] == 2


def test_moe_router_indirect_delegates_to_seam_v2_router(monkeypatch):
    import torch

    from vllm_moe_offload_ascend.ops.fused_moe import moe_router_op

    calls = []

    class Router:
        top_k = 2
        use_grouped_topk = False
        renormalize = True
        topk_group = None
        num_expert_group = None
        scoring_func = "softmax"
        routed_scaling_factor = 1.0
        e_score_correction_bias = None
        custom_routing_function = None

        def select_experts(self, hidden_states, router_logits):
            calls.append((hidden_states, router_logits))
            return (
                torch.ones((4, 2), dtype=torch.bfloat16),
                torch.zeros((4, 2), dtype=torch.int32),
            )

    runner = SimpleNamespace(
        layer_id=7,
        router=Router(),
        gate=None,
        routed_experts=SimpleNamespace(
            n_shared_experts=0,
            mix_placement=False,
        ),
        n_shared_experts=0,
        mix_placement=False,
        global_redundant_expert_num=0,
        moe_config=SimpleNamespace(num_experts=64),
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.runner.moe_runner.get_layer_from_name",
        lambda _name: runner,
    )
    monkeypatch.setattr(
        "vllm_ascend.quantization.methods.base.get_moe_num_logical_experts",
        lambda *_args, **_kwargs: 64,
    )
    hidden = torch.zeros((4, 16), dtype=torch.bfloat16)
    logits = torch.zeros((4, 64), dtype=torch.float32)

    weights, ids = moe_router_op._moe_router_indirect_impl(
        hidden,
        logits,
        "fixture.layer.experts",
    )

    assert len(calls) == 1
    assert calls[0][0] is hidden
    assert calls[0][1] is logits
    assert weights.shape == ids.shape == (4, 2)


def test_seam_v2_unquantized_patch_does_not_require_legacy_selector():
    """ABI 2 routes before apply; its weight/offload hook must install alone."""
    from vllm_moe_offload_ascend.patches import patch_fused_moe

    class SeamV2Method:
        _ascend_moe_offload_runtime_patch = False

        def process_weights_after_loading(self, layer):
            return None

        def apply(self, layer, *args, **kwargs):
            return layer

    module = SimpleNamespace(AscendUnquantizedFusedMoEMethod=SeamV2Method)
    assert not hasattr(module, "select_experts")

    patch_fused_moe._patch_unquantized_moe_method(module)

    assert SeamV2Method._ascend_moe_offload_runtime_patch is True
    assert not hasattr(module, "select_experts")


def test_seam_selection_finalizes_capability_before_graph_forward(monkeypatch):
    """Capability discovery is finalized after weights, outside the graph."""
    from vllm_moe_offload_ascend.patches import patch_fused_moe
    import vllm_moe_offload_ascend.moe_offload.runtime as runtime_mod

    original_forward = object()

    class FakeRunner:
        _ascend_moe_offload_seam_patch = False

        def _select_forward(self):
            return original_forward

    patch_fused_moe._patch_ascend_moe_runner(
        SimpleNamespace(AscendMoERunner=FakeRunner)
    )
    monkeypatch.setattr(
        runtime_mod,
        "get_moe_offload_runtime",
        lambda: SimpleNamespace(
            config=SimpleNamespace(offload_stage_seam=True),
        ),
    )
    runner = FakeRunner()
    runner._seam_config_guards_pass = lambda: True
    resolve_calls = []
    runner._resolve_seam_per_layer_guards = lambda: resolve_calls.append(True) or True

    selected = runner._select_forward()

    assert selected == runner._seam_forward_entry
    assert runner._seam_active is None
    assert resolve_calls == []

    runner._finalize_seam_before_compile()

    assert runner._seam_active is True
    assert resolve_calls == [True]

# ---------------------------------------------------------------------------
# L3: seam guard returns False when layer_id is missing
# ---------------------------------------------------------------------------

def test_seam_guard_returns_false_when_layer_has_no_layer_id(monkeypatch):
    """L3: _resolve_seam_per_layer_guards must bail out (return False) when the
    layer object does not have a layer_id attribute, preventing collision on
    the shared -1 key in the injection/log2phy registries."""
    from vllm_moe_offload_ascend.patches import patch_fused_moe

    class FakeRunner:
        _ascend_moe_offload_seam_patch = False

        def _select_forward(self):
            raise AssertionError("should not be called")

    fake_fused_moe = SimpleNamespace(AscendMoERunner=FakeRunner)
    patch_fused_moe._patch_ascend_moe_runner(fake_fused_moe)

    class LayerWithoutId:
        """Simulates a layer where layer_id was never set."""
        moe_config = SimpleNamespace(
            num_experts=64,
            dp_size=1, ep_size=1, tp_size=1, pcp_size=1,
        )
        top_k = 2
        custom_routing_function = None
        enable_npugraph_ex_static_kernel = False
        zero_expert_num = 0
        zero_expert_type = None
        n_shared_experts = 0
        global_redundant_expert_num = 0
        _shared_experts = None
        # Deliberately no layer_id attribute.

    def fake_get_layer(name):
        return LayerWithoutId()

    def fake_get_num_experts(layer, num_experts, *, global_redundant_expert_num=0,
                              num_shared_experts=0, **kw):
        return num_experts

    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_GB", "14")
    vllm_moe_offload_ascend.register()

    runner = FakeRunner()
    runner.layer_name = "model.layers.0.mlp.experts"
    runner._seam_active = None

    import vllm_moe_offload_ascend.moe_offload.runtime as runtime_mod
    from vllm_moe_offload_ascend.moe_offload.config import MoeOffloadConfig
    from vllm_moe_offload_ascend.moe_offload.runtime import MoeOffloadRuntime
    monkeypatch.setattr(
        runtime_mod, "get_moe_offload_runtime",
        lambda: MoeOffloadRuntime(MoeOffloadConfig(enabled=True, num_slots=4,
                                                    offload_stage_seam=True))
    )

    try:
        from vllm.model_executor.layers.fused_moe.runner import moe_runner as mr
        monkeypatch.setattr(mr, "get_layer_from_name", fake_get_layer)
    except Exception:
        pass
    try:
        from vllm_ascend.quantization.methods import base as qbase
        monkeypatch.setattr(qbase, "get_moe_num_logical_experts", fake_get_num_experts)
    except Exception:
        pass

    result = runner._resolve_seam_per_layer_guards()
    assert result is False, (
        "seam guard must return False for layers without layer_id to avoid "
        "cross-layer collision on key=-1"
    )


def test_seam_guard_resolves_external_gated_internal_corrected_router(monkeypatch):
    """The per-layer guard must select the tuple seam from capabilities alone."""
    import torch
    from vllm_moe_offload_ascend.patches import patch_fused_moe
    import vllm_moe_offload_ascend.moe_offload.runtime as runtime_mod

    class FakeRunner:
        _ascend_moe_offload_seam_patch = False

        def _select_forward(self):
            raise AssertionError("should not be called")

    class SharedExperts(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones((2, 2)))
            self.expert_gate = torch.nn.Linear(2, 1, bias=False)

    registrations = []
    runtime = SimpleNamespace(
        register_resident_shared_weights=lambda **kwargs: registrations.append(kwargs)
    )
    monkeypatch.setattr(runtime_mod, "get_moe_offload_runtime", lambda: runtime)
    patch_fused_moe._patch_ascend_moe_runner(
        SimpleNamespace(AscendMoERunner=FakeRunner)
    )
    layer = SimpleNamespace(
        layer_id=11,
        moe_config=SimpleNamespace(
            num_experts=64, dp_size=1, ep_size=1, tp_size=1, pcp_size=1
        ),
        n_shared_experts=1,
        mix_placement=False,
        top_k=4,
        use_grouped_topk=True,
        topk_group=2,
        num_expert_group=8,
        renormalize=True,
        scoring_func="sigmoid",
        e_score_correction_bias=torch.zeros(64),
        routed_scaling_factor=1.8,
        custom_routing_function=None,
        is_internal_router=True,
        enable_npugraph_ex_static_kernel=False,
        zero_expert_num=0,
        zero_expert_type=None,
    )
    from vllm.model_executor.layers.fused_moe.runner import moe_runner as mr

    monkeypatch.setattr(mr, "get_layer_from_name", lambda _name: layer)
    runner = FakeRunner()
    runner.layer_name = "fixture.layer.experts"
    runner._shared_experts = SharedExperts()
    runner.shared_expert_gate = runner._shared_experts.expert_gate
    runner.gate = object()

    assert runner._resolve_seam_per_layer_guards() is True
    assert runner._seam_capability.output_contract == "shared_routed_tuple"
    assert runner._seam_capability.router_owner == "internal_gate"
    assert runner._seam_capability.selection.correction_bias is True
    assert runner._seam_num_logical_experts == 64
    assert registrations == [
        {
            "layer_id": 11,
            "shared_experts": runner._shared_experts,
            "shared_gate": runner.shared_expert_gate,
        }
    ]


@pytest.mark.parametrize(
    ("layout", "enabled", "blocker"),
    [
        (None, False, "fused_shared_lane_unregistered"),
        (
            SimpleNamespace(
                routed_expert_count=3,
                shared_expert_count=2,
                pinned_logical_ids=(3, 4),
                total_logical_expert_count=5,
            ),
            False,
            "fused_shared_lane_layout_mismatch",
        ),
        (
            SimpleNamespace(
                routed_expert_count=4,
                shared_expert_count=2,
                pinned_logical_ids=(4, 5),
                total_logical_expert_count=6,
            ),
            True,
            None,
        ),
    ],
)
def test_seam_guard_requires_matching_fused_shared_lane(
    monkeypatch,
    layout,
    enabled,
    blocker,
):
    from vllm_moe_offload_ascend.patches import patch_fused_moe
    import vllm_moe_offload_ascend.moe_offload.runtime as runtime_mod

    class FakeRunner:
        _ascend_moe_offload_seam_patch = False

        def _select_forward(self):
            raise AssertionError("should not be called")

    class FakeRuntime:
        def fused_shared_lane_layout(self, layer_id):
            assert int(layer_id) == 11
            return layout

    patch_fused_moe._patch_ascend_moe_runner(
        SimpleNamespace(AscendMoERunner=FakeRunner)
    )
    layer = SimpleNamespace(
        layer_id=11,
        moe_config=SimpleNamespace(
            # Ascend FusedMoE mutates this count to include the two shared rows.
            num_experts=6, dp_size=1, ep_size=1, tp_size=1, pcp_size=1
        ),
        n_shared_experts=2,
        mix_placement=True,
        top_k=2,
        use_grouped_topk=False,
        topk_group=None,
        num_expert_group=None,
        renormalize=True,
        scoring_func="softmax",
        e_score_correction_bias=None,
        routed_scaling_factor=1.0,
        custom_routing_function=None,
        enable_npugraph_ex_static_kernel=False,
        zero_expert_num=0,
        zero_expert_type=None,
    )
    from vllm.model_executor.layers.fused_moe.runner import moe_runner as mr

    monkeypatch.setattr(runtime_mod, "get_moe_offload_runtime", lambda: FakeRuntime())
    monkeypatch.setattr(mr, "get_layer_from_name", lambda _name: layer)
    runner = FakeRunner()
    runner.layer_name = "fixture.layer.experts"
    runner._shared_experts = None
    runner.shared_expert_gate = None
    runner.gate = None

    assert runner._resolve_seam_per_layer_guards() is enabled
    support = runner._seam_capability_support
    if blocker is None:
        assert support.enabled
    else:
        assert not support.enabled
        assert blocker in support.blockers


def test_compile_tracing_probe_is_module_scoped_for_dynamo_capture():
    from vllm_moe_offload_ascend.patches import patch_fused_moe

    assert callable(patch_fused_moe._is_torch_compile_tracing)
    assert "_is_torch_compile_tracing" not in {
        name for name in patch_fused_moe._patch_ascend_moe_runner.__code__.co_varnames
    }


def test_router_compare_skips_data_dependent_ops_during_tracing(monkeypatch):
    import torch

    from vllm_moe_offload_ascend.ops.fused_moe import moe_seam_inject
    from vllm_moe_offload_ascend.patches import patch_fused_moe

    hidden_states = torch.zeros((2, 4), dtype=torch.float32)
    router_logits = torch.zeros((2, 8), dtype=torch.float32)
    native_weights = torch.ones((2, 2), dtype=torch.float32)
    native_ids = torch.zeros((2, 2), dtype=torch.int32)

    class FakeMethod:
        def process_weights_after_loading(self, layer):
            return None

        def apply(self, layer):
            return fake_fused.select_experts(
                hidden_states,
                router_logits,
                hidden_states=hidden_states,
                router_logits=router_logits,
                num_experts=8,
            )

    def native_select(*args, **kwargs):
        return native_weights, native_ids

    fake_fused = SimpleNamespace(
        AscendUnquantizedFusedMoEMethod=FakeMethod,
        select_experts=native_select,
    )
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_COMPARE_ROUTER", "1")
    monkeypatch.setenv(
        "VLLM_ASCEND_MOE_ROUTER_PARITY_PATH",
        "/tmp/should-not-be-created-router-parity.jsonl",
    )
    monkeypatch.setattr(
        moe_seam_inject,
        "peek_injected_topk",
        lambda _layer_id: (native_weights, native_ids),
    )
    monkeypatch.setattr(patch_fused_moe, "_is_torch_compile_tracing", lambda: True)
    monkeypatch.setattr(
        torch,
        "count_nonzero",
        lambda *args, **kwargs: pytest.fail("router compare ran during tracing"),
    )

    patch_fused_moe._patch_unquantized_moe_method(fake_fused)
    FakeMethod().apply(SimpleNamespace(layer_id=3))


def test_graph_tools_default_to_graph_mode_and_reject_forced_eager(monkeypatch):
    from benchmark.scripts import run_fixed_slot_smoke as benchmark_smoke
    from tools import collect_moe_trace, run_fixed_slot_smoke

    for parser, args in ((collect_moe_trace.parse_args, ["tool", "--output", "/tmp/moe-trace.jsonl"]), (run_fixed_slot_smoke.parse_args, ["tool", "--output-dir", "/tmp/smoke"])):
        monkeypatch.setattr(sys, "argv", args)
        assert parser().enforce_eager is False
        monkeypatch.setattr(sys, "argv", [*args, "--enforce-eager"])
        with pytest.raises(SystemExit):
            parser()

    assert benchmark_smoke._smoke_compilation_config(
        mode="fixed_slot_sync",
        enforce_eager=False,
        disable_ascend_norm_quant_fusion=True,
    ) == {
        "cudagraph_mode": "PIECEWISE",
        "pass_config": {"fuse_norm_quant": False},
    }
    assert benchmark_smoke._smoke_compilation_config(
        mode="fixed_slot_sync",
        enforce_eager=True,
        disable_ascend_norm_quant_fusion=False,
    ) is None

    llm_kwargs = benchmark_smoke._build_llm_kwargs(
        SimpleNamespace(
            model="/models/qwen",
            enforce_eager=False,
            gpu_memory_utilization=0.4,
            kv_cache_memory_mb=256,
            max_model_len=64,
            max_num_seqs=1,
            max_num_batched_tokens=2,
            with_native_offload_backend=False,
            disable_ascend_norm_quant_fusion=True,
            npu_profiler_dir=None,
        ),
        {
            "model": {"path": "/unused", "tensor_parallel_size": 1},
            "dataset": {"seed": 42},
        },
        "fixed_slot_sync",
    )
    assert llm_kwargs["enable_prefix_caching"] is False
    assert llm_kwargs["compilation_config"]["cudagraph_mode"] == "PIECEWISE"
    assert llm_kwargs["additional_config"] == {
        "ascend_compilation_config": {"fuse_norm_quant": False}
    }


def test_smoke_runner_records_validated_ascend_additional_config(monkeypatch):
    from benchmark.scripts import run_fixed_slot_smoke as benchmark_smoke

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_fixed_slot_smoke.py",
            "--output-dir",
            "/tmp/smoke",
            "--ascend-additional-config-json",
            '{"mix_placement": true}',
        ],
    )

    args = benchmark_smoke.parse_args()
    assert args.ascend_additional_config == {"mix_placement": True}
    llm_kwargs = benchmark_smoke._build_llm_kwargs(
        SimpleNamespace(
            model="/models/deepseek",
            enforce_eager=False,
            gpu_memory_utilization=0.4,
            kv_cache_memory_mb=256,
            max_model_len=64,
            max_num_seqs=1,
            max_num_batched_tokens=2,
            with_native_offload_backend=False,
            disable_ascend_norm_quant_fusion=False,
            ascend_additional_config=args.ascend_additional_config,
            npu_profiler_dir=None,
        ),
        {
            "model": {"path": "/unused", "tensor_parallel_size": 1},
            "dataset": {"seed": 42},
        },
        "fixed_slot_sync",
    )
    assert llm_kwargs["additional_config"] == {"mix_placement": True}


def test_smoke_runner_can_configure_a_generation_only_npu_profiler(monkeypatch):
    from benchmark.scripts import run_fixed_slot_smoke as benchmark_smoke

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_fixed_slot_smoke.py",
            "--output-dir",
            "/tmp/smoke",
            "--npu-profiler-dir",
            "/tmp/npu-profile",
        ],
    )

    args = benchmark_smoke.parse_args()
    assert args.npu_profiler_dir == "/tmp/npu-profile"
    llm_kwargs = benchmark_smoke._build_llm_kwargs(
        SimpleNamespace(
            model="/models/glm",
            enforce_eager=False,
            gpu_memory_utilization=0.4,
            kv_cache_memory_mb=256,
            max_model_len=64,
            max_num_seqs=1,
            max_num_batched_tokens=2,
            with_native_offload_backend=False,
            disable_ascend_norm_quant_fusion=False,
            ascend_additional_config={},
            npu_profiler_dir=args.npu_profiler_dir,
        ),
        {
            "model": {"path": "/unused", "tensor_parallel_size": 1},
            "dataset": {"seed": 42},
        },
        "fixed_slot_sync",
    )

    assert llm_kwargs["profiler_config"] == {
        "profiler": "torch",
        "torch_profiler_dir": "/tmp/npu-profile",
        "torch_profiler_with_stack": False,
        "ignore_frontend": True,
    }
