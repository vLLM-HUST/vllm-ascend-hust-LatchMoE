import sys
from types import ModuleType, SimpleNamespace

import pytest

import vllm_moe_offload_ascend
from vllm_moe_offload_ascend.patches.patch_fused_moe import (
    _ascend_device_op_is_initializing,
    _install_runtime_patches_when_ready,
    _moe_offload_kv_backstop_active,
    _moe_offload_kv_backstop_hint,
    _patch_kv_cache_capacity_backstop,
    _unpack_mlp_apply_result,
)


def test_runtime_patch_install_waits_for_ascend_device_op(monkeypatch):
    from vllm_moe_offload_ascend.patches import patch_fused_moe

    device_op = ModuleType("vllm_ascend.device.device_op")
    monkeypatch.setitem(sys.modules, "vllm_ascend.device.device_op", device_op)
    monkeypatch.setattr(
        patch_fused_moe,
        "_inject_sys_modules",
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
        "_inject_sys_modules",
        lambda: events.append("inject_sys_modules"),
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
        "inject_sys_modules",
        "install_runtime_module_patches",
    ]


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

    fake_layernorm = ModuleType("vllm_ascend.ops.layernorm")
    fake_layernorm.AscendRMSNorm = FakeRMSNorm
    original_import_module = importlib.import_module

    def import_module(name, package=None):
        if name == "vllm_ascend.ops.layernorm":
            return fake_layernorm
        return original_import_module(name, package)

    fake_torch = ModuleType("torch")
    fake_torch.ops = SimpleNamespace(
        vllm=SimpleNamespace(maybe_chunk_residual=lambda _x, residual: residual)
    )
    fake_torch_npu = ModuleType("torch_npu")
    fake_torch_npu.npu_add_rms_norm = lambda x, residual, weight, eps: (
        ("normalized", x, residual, weight, eps),
        None,
        "next_residual",
    )

    monkeypatch.setattr(importlib, "import_module", import_module)
    monkeypatch.setattr(patch_fused_moe, "_opapi_supports_add_rms_norm_bias", lambda: False)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch_npu", fake_torch_npu)

    patch_fused_moe._patch_rms_norm_bias_cann_compat()

    assert FakeRMSNorm().forward_oot("x", "residual") == (
        ("normalized", "x", "residual", "weight", 1e-6),
        "next_residual",
    )
    assert FakeRMSNorm().forward_oot("x") == ("original", "x", None)


def test_stage_seam_registers_for_full_and_piecewise_cudagraph(monkeypatch):
    import vllm_ascend.platform as platform
    from vllm.config.compilation import CUDAGraphMode

    from vllm_moe_offload_ascend.patches import patch_fused_moe

    class FakeNPUPlatform:
        @classmethod
        def check_and_update_config(cls, _vllm_config):
            return None

    monkeypatch.setattr(platform, "NPUPlatform", FakeNPUPlatform)
    monkeypatch.setattr(patch_fused_moe, "_install_runtime_module_patches", lambda: None)
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_STAGE_SEAM", "1")

    patch_fused_moe._patch_platform_splitting_ops()
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
            splitting_ops=[],
        )
    )

    FakeNPUPlatform.check_and_update_config(config)

    assert "vllm::moe_offload_stage" in config.compilation_config.splitting_ops


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

    monkeypatch.setattr(kv_utils, "_report_kv_cache_config", original_report)
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


def test_register_aliases_plugin_modules_under_vllm_ascend_namespace(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_GB", "14")

    vllm_moe_offload_ascend.register()

    import vllm_ascend
    import vllm_moe_offload_ascend.moe_offload as plugin_pkg
    import vllm_moe_offload_ascend.moe_offload.prefill_residency as prefill_residency
    from vllm_ascend.moe_offload.runtime import get_moe_offload_runtime

    assert sys.modules["vllm_ascend.moe_offload"] is plugin_pkg
    assert sys.modules["vllm_ascend.moe_offload.prefill_residency"] is prefill_residency
    assert vllm_ascend.moe_offload is plugin_pkg
    assert get_moe_offload_runtime.__module__ == "vllm_moe_offload_ascend.moe_offload.runtime"

def test_register_aliases_sew_custom_op_modules(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_GB", "14")

    vllm_moe_offload_ascend.register()

    import vllm_moe_offload_ascend.ops.fused_moe.moe_offload_stage_op as stage_op
    import vllm_moe_offload_ascend.ops.fused_moe.moe_router_op as router_op
    import vllm_ascend.ops.fused_moe as ascend_fused_moe

    assert sys.modules["vllm_ascend.ops.fused_moe.moe_offload_stage_op"] is stage_op
    assert sys.modules["vllm_ascend.ops.fused_moe.moe_router_op"] is router_op
    assert ascend_fused_moe.moe_offload_stage_op is stage_op
    assert ascend_fused_moe.moe_router_op is router_op


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
    import vllm_ascend.moe_offload.runtime as runtime_mod

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
    assert runtime.cached is not None
    assert runtime.cached["layer_id"] == 7
    assert runtime.cached["token_counts_by_expert"] == {
        0: 1,
        1: 1,
        2: 1,
        3: 1,
    }


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
    assert runtime.cached is not None


def test_stage_op_confirmed_prefill_overflow_defers_to_b2(monkeypatch):
    """Counterpart to the decode case: a CONFIRMED prefill that overflows slots
    is the legitimate B2 wave handoff and must cache route-stats, not stage once.
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


# ---------------------------------------------------------------------------
# L3: seam guard returns False when layer_id is missing
# ---------------------------------------------------------------------------

def test_seam_guard_returns_false_when_layer_has_no_layer_id(monkeypatch):
    """L3: _resolve_seam_per_layer_guards must bail out (return False) when the
    layer object does not have a layer_id attribute, preventing collision on
    the shared -1 key in the injection/log2phy registries."""
    import torch
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

    import vllm_ascend.moe_offload.runtime as runtime_mod
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

def test_graph_tools_default_to_graph_mode_and_reject_forced_eager(monkeypatch):
    from tools import collect_moe_trace, run_fixed_slot_smoke

    for parser, args in ((collect_moe_trace.parse_args, ["tool", "--output", "/tmp/moe-trace.jsonl"]), (run_fixed_slot_smoke.parse_args, ["tool", "--output-dir", "/tmp/smoke"])):
        monkeypatch.setattr(sys, "argv", args)
        assert parser().enforce_eager is False
        monkeypatch.setattr(sys, "argv", [*args, "--enforce-eager"])
        with pytest.raises(SystemExit):
            parser()
