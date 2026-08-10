import os
import json

import pytest

from vllm_moe_offload_ascend.moe_offload.autoconfig import (
    MOE_OFFLOAD_GB_ENV,
    apply_moe_offload_defaults,
    apply_profile_guided_residency,
    derive_num_slots_defaults,
    derive_prefetch_defaults,
)
from vllm_moe_offload_ascend.moe_offload.prefill_residency import (
    load_prefill_layer_costs,
    load_prefill_layer_costs_many,
    plan_profile_guided_prefill_residency,
)


QWEN3_MOE_CONFIG = {
    "hidden_size": 2048,
    "moe_intermediate_size": 768,
    "num_experts": 128,
    "num_experts_per_tok": 8,
    "num_hidden_layers": 48,
    "torch_dtype": "bfloat16",
}

AUTOCONFIG_ENV_VARS = (
    MOE_OFFLOAD_GB_ENV,
    "VLLM_ASCEND_MOE_OFFLOAD_SEW_DATAPLANE",
    "VLLM_ASCEND_MOE_OFFLOAD_ENABLED",
    "VLLM_ASCEND_MOE_OFFLOAD_TRACE_ONLY",
    "VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS",
    "VLLM_ASCEND_MOE_OFFLOAD_POLICY",
    "VLLM_ASCEND_MOE_OFFLOAD_ASYNC_LOAD",
    "VLLM_ASCEND_MOE_OFFLOAD_MAX_PHASES",
    "VLLM_ASCEND_MOE_OFFLOAD_LAYERED_RUNTIME",
    "VLLM_ASCEND_MOE_OFFLOAD_FANOUT_THRESHOLD",
    "VLLM_ASCEND_MOE_OFFLOAD_GRAPH_COMPATIBLE",
    "VLLM_ASCEND_MOE_OFFLOAD_STAGE_SEAM",
    "VLLM_ASCEND_MOE_OFFLOAD_B2_WAVE_PREFILL",
    "VLLM_ASCEND_MOE_OFFLOAD_PIN_HOST_MEMORY",
    "VLLM_ASCEND_MOE_OFFLOAD_TRANSFER_AWARE_SCHEDULE",
    "VLLM_ASCEND_MOE_OFFLOAD_PREFILL_PREFETCH_DEPTH",
    "VLLM_ASCEND_MOE_OFFLOAD_PREFILL_BUFFER_COUNT",
    "VLLM_ASCEND_MOE_OFFLOAD_ROUTE_STATS_CACHE",
    "VLLM_ASCEND_MOE_OFFLOAD_RELEASE_ORIGINAL_EXPERT_WEIGHTS",
    "VLLM_ASCEND_MOE_OFFLOAD_RESIDENT_LAYER_IDS",
    "VLLM_ASCEND_MOE_OFFLOAD_PREFILL_RESIDENCY_PROFILE",
    "VLLM_ASCEND_MOE_OFFLOAD_MIN_NET_SAVING_GB",
    "VLLM_ASCEND_MOE_OFFLOAD_MIN_NET_SAVING_RATIO",
    "VLLM_ASCEND_MOE_OFFLOAD_SLOT_HBM_BUDGET_GB",
    "VLLM_ASCEND_MOE_OFFLOAD_SLOT_HBM_FRACTION",
    "VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS_AUTOCONFIG_BOOTSTRAP",
    "VLLM_ASCEND_MOE_OFFLOAD_KV_RESERVE_SEQS",
    "VLLM_ASCEND_MOE_OFFLOAD_KV_RESERVE_CTX",
    "VLLM_ASCEND_MOE_OFFLOAD_SYSTEM_RESERVE_GB",
    "VLLM_ASCEND_MOE_OFFLOAD_ACTIVATION_RESERVE_GB",
)


@pytest.fixture(autouse=True)
def clean_autoconfig_env():
    original = {name: os.environ[name] for name in AUTOCONFIG_ENV_VARS if name in os.environ}
    for name in AUTOCONFIG_ENV_VARS:
        os.environ.pop(name, None)
    yield
    for name in AUTOCONFIG_ENV_VARS:
        os.environ.pop(name, None)
    os.environ.update(original)


def test_apply_defaults_enables_prefetch_layered_runtime(monkeypatch):
    monkeypatch.setenv(MOE_OFFLOAD_GB_ENV, "14")
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_SLOT_HBM_BUDGET_GB", "80")
    engine_args = type(
        "EngineArgsStub",
        (),
        {
            "_ascend_moe_offload_model_config": QWEN3_MOE_CONFIG,
            "offload_backend": "auto",
            "offload_group_size": 0,
            "offload_num_in_group": 1,
            "offload_prefetch_step": 1,
            "offload_params": set(),
            "cpu_offload_gb": 0,
            "cpu_offload_params": set(),
        },
    )()

    assert apply_moe_offload_defaults(engine_args) is True

    assert engine_args.offload_backend == "prefetch"
    assert engine_args.offload_group_size == 4
    assert engine_args.offload_num_in_group == 1
    assert engine_args.offload_prefetch_step == 1
    assert engine_args.offload_params == {"experts"}
    assert engine_args.cpu_offload_gb == 0
    assert engine_args.cpu_offload_params == set()
    assert os.environ["VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS"] == "32"
    assert os.environ["VLLM_ASCEND_MOE_OFFLOAD_LAYERED_RUNTIME"] == "1"
    assert os.environ["VLLM_ASCEND_MOE_OFFLOAD_FANOUT_THRESHOLD"] == "32"
    assert "VLLM_ASCEND_MOE_OFFLOAD_RELEASE_ORIGINAL_EXPERT_WEIGHTS" not in os.environ
    assert os.environ["VLLM_ASCEND_MOE_OFFLOAD_RESIDENT_LAYER_IDS"]
    assert engine_args._ascend_moe_offload_autoconfig_plan["slot_defaults"]["num_slots"] == 32


def test_apply_defaults_preserves_explicit_num_slots_override(monkeypatch):
    monkeypatch.setenv(MOE_OFFLOAD_GB_ENV, "14")
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS", "8")
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_SLOT_HBM_BUDGET_GB", "80")
    engine_args = type(
        "EngineArgsStub",
        (),
        {
            "_ascend_moe_offload_model_config": QWEN3_MOE_CONFIG,
            "offload_backend": "auto",
            "offload_group_size": 0,
            "offload_num_in_group": 1,
            "offload_prefetch_step": 1,
            "offload_params": set(),
            "cpu_offload_gb": 0,
            "cpu_offload_params": set(),
        },
    )()

    assert apply_moe_offload_defaults(engine_args) is True

    assert os.environ["VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS"] == "8"
    assert os.environ["VLLM_ASCEND_MOE_OFFLOAD_FANOUT_THRESHOLD"] == "8"
    assert engine_args._ascend_moe_offload_autoconfig_plan["slot_defaults"]["num_slots"] == 32


def test_num_slots_scales_with_larger_offload_budget():
    os.environ["VLLM_ASCEND_MOE_OFFLOAD_SLOT_HBM_BUDGET_GB"] = "80"
    prefetch_14 = derive_prefetch_defaults(14, QWEN3_MOE_CONFIG)
    slots_14 = derive_num_slots_defaults(
        14,
        QWEN3_MOE_CONFIG,
        prefetch_14,
    )
    prefetch_28 = derive_prefetch_defaults(28, QWEN3_MOE_CONFIG)
    slots_28 = derive_num_slots_defaults(
        28,
        QWEN3_MOE_CONFIG,
        prefetch_28,
    )

    assert prefetch_14["estimated_offloaded_layers"] == 12
    assert slots_14["num_slots"] == 32
    assert slots_14["target_b2_waves"] == 4
    assert prefetch_28["estimated_offloaded_layers"] == 24
    assert slots_28["num_slots"] == 64
    assert slots_28["target_b2_waves"] == 2
    assert slots_28["estimated_b2_waves"] == 2
    assert slots_28["slot_budget_constraints"]["reserve"]["prefill_buffer_count"] == 2
    assert slots_28["slot_plus_b2_stage_gib"] > slots_28["slot_bank_gib"]


def test_num_slots_respects_real_hbm_budget_override():
    os.environ["VLLM_ASCEND_MOE_OFFLOAD_SLOT_HBM_BUDGET_GB"] = "20"
    prefetch = derive_prefetch_defaults(48, QWEN3_MOE_CONFIG)
    slots = derive_num_slots_defaults(
        48,
        QWEN3_MOE_CONFIG,
        prefetch,
    )

    assert prefetch["estimated_offloaded_layers"] == 48
    assert slots["target_b2_waves"] == 2
    assert slots["num_slots"] == 38
    assert slots["slot_bank_gib"] <= 20
    assert slots["slot_plus_b2_stage_gib"] <= 20
    assert slots["slot_budget_constraints"]["real_hbm_slot_budget_gib"] == 20


def test_num_slots_respects_minimum_net_hbm_saving():
    os.environ["VLLM_ASCEND_MOE_OFFLOAD_SLOT_HBM_BUDGET_GB"] = "80"
    os.environ["VLLM_ASCEND_MOE_OFFLOAD_MIN_NET_SAVING_RATIO"] = "0.75"
    prefetch = derive_prefetch_defaults(48, QWEN3_MOE_CONFIG)
    slots = derive_num_slots_defaults(
        48,
        QWEN3_MOE_CONFIG,
        prefetch,
    )

    assert slots["num_slots"] == 23
    assert slots["net_hbm_saving_gib"] >= prefetch["estimated_offloaded_gb"] * 0.75


def test_num_slots_reserves_kv_activation_and_b2_buffers_for_real_qwen_shape():
    config = {
        "hidden_size": 2048,
        "moe_intermediate_size": 768,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "num_hidden_layers": 48,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "max_position_embeddings": 32768,
        "torch_dtype": "bfloat16",
    }
    os.environ["VLLM_ASCEND_MOE_OFFLOAD_SLOT_HBM_BUDGET_GB"] = "80"
    prefetch_14 = derive_prefetch_defaults(14, config)
    slots_14 = derive_num_slots_defaults(14, config, prefetch_14)
    prefetch_28 = derive_prefetch_defaults(28, config)
    slots_28 = derive_num_slots_defaults(28, config, prefetch_28)

    assert slots_14["num_slots"] == 32
    assert slots_28["num_slots"] < 64
    reserve = slots_28["slot_budget_constraints"]["reserve"]
    assert reserve["kv_reserve_seqs"] == 4
    assert reserve["kv_reserve_ctx"] == 8192
    assert reserve["kv_reserve_gib"] > 0
    assert reserve["affordability_layer_equivalent_count"] == (
        prefetch_28["estimated_offloaded_layers"] + reserve["prefill_buffer_count"]
    )


def test_num_slots_uses_physical_slot_fraction_cap_for_64gb_npu(monkeypatch):
    import vllm_moe_offload_ascend.moe_offload.autoconfig as autoconfig

    config = {
        "hidden_size": 2048,
        "moe_intermediate_size": 768,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "num_hidden_layers": 48,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "max_position_embeddings": 40960,
        "torch_dtype": "bfloat16",
    }
    engine_args = type(
        "EngineArgsStub",
        (),
        {
            "max_num_seqs": 1,
            "max_model_len": 4096,
            "gpu_memory_utilization": 0.9,
        },
    )()
    monkeypatch.setattr(
        autoconfig,
        "_read_npu_hbm_budget_gib",
        lambda _engine_args: (
            54.86,
            {
                "source": "torch.npu.mem_get_info",
                "free_gib": 60.62,
                "total_gib": 60.96,
                "used_gib": 0.34,
                "gpu_memory_utilization": 0.9,
                "slot_hbm_budget_gib": 54.86,
            },
        ),
    )

    prefetch = derive_prefetch_defaults(28, config)
    slots = derive_num_slots_defaults(28, config, prefetch, engine_args)

    assert slots["num_slots"] == 32
    physical = slots["slot_budget_constraints"]["physical_slot_fraction"]
    assert physical["slot_hbm_fraction"] == 0.12
    assert slots["slot_plus_b2_stage_gib"] <= physical["slot_fraction_budget_gib"]


def test_apply_defaults_overwrites_bootstrap_num_slots_with_real_model_config(monkeypatch):
    import vllm_moe_offload_ascend.moe_offload.autoconfig as autoconfig

    monkeypatch.setenv(MOE_OFFLOAD_GB_ENV, "28")
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_SEW_DATAPLANE", "1")
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS", "64")
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS_AUTOCONFIG_BOOTSTRAP", "1")
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_SLOT_HBM_FRACTION", "0.12")
    monkeypatch.setattr(
        autoconfig,
        "_read_npu_hbm_budget_gib",
        lambda _engine_args: (
            54.86,
            {
                "source": "torch.npu.mem_get_info",
                "free_gib": 60.62,
                "total_gib": 60.96,
                "used_gib": 0.34,
                "gpu_memory_utilization": 0.9,
                "slot_hbm_budget_gib": 54.86,
            },
        ),
    )
    engine_args = type(
        "EngineArgsStub",
        (),
        {
            "_ascend_moe_offload_model_config": {
                "hidden_size": 2048,
                "moe_intermediate_size": 768,
                "num_experts": 128,
                "num_experts_per_tok": 8,
                "num_hidden_layers": 48,
                "num_attention_heads": 32,
                "num_key_value_heads": 4,
                "head_dim": 128,
                "max_position_embeddings": 40960,
                "torch_dtype": "bfloat16",
            },
            "max_num_seqs": 1,
            "max_model_len": 4096,
            "gpu_memory_utilization": 0.9,
            "offload_backend": "auto",
            "offload_group_size": 0,
            "offload_num_in_group": 1,
            "offload_prefetch_step": 1,
            "offload_params": set(),
            "cpu_offload_gb": 0,
            "cpu_offload_params": set(),
        },
    )()

    assert apply_moe_offload_defaults(engine_args) is True

    assert os.environ["VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS"] == "32"
    assert os.environ["VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS_AUTOCONFIG_BOOTSTRAP"] == "0"


@pytest.mark.parametrize(
    "attrs",
    [
        {"cpu_offload_gb": 1},
        {"offload_backend": "uva"},
    ],
)
def test_apply_defaults_rejects_uva_conflicts(monkeypatch, attrs):
    monkeypatch.setenv(MOE_OFFLOAD_GB_ENV, "14")
    engine_args = type("EngineArgsStub", (), attrs)()

    with pytest.raises(ValueError, match="cpu_offload_gb"):
        apply_moe_offload_defaults(engine_args)


def test_apply_defaults_preserves_explicit_prefetch_values(monkeypatch):
    monkeypatch.setenv(MOE_OFFLOAD_GB_ENV, "14")
    engine_args = type(
        "EngineArgsStub",
        (),
        {
            "_ascend_moe_offload_model_config": QWEN3_MOE_CONFIG,
            "offload_backend": "prefetch",
            "offload_group_size": 6,
            "offload_num_in_group": 2,
            "offload_prefetch_step": 3,
            "offload_params": {"experts.w2_weight"},
            "cpu_offload_gb": 0,
            "cpu_offload_params": set(),
        },
    )()

    assert apply_moe_offload_defaults(engine_args) is True

    assert engine_args.offload_backend == "prefetch"
    assert engine_args.offload_group_size == 6
    assert engine_args.offload_num_in_group == 2
    assert engine_args.offload_prefetch_step == 3
    assert engine_args.offload_params == {"experts.w2_weight"}


def test_apply_defaults_sew_dataplane_skips_native_prefetch(monkeypatch):
    monkeypatch.setenv(MOE_OFFLOAD_GB_ENV, "14")
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_SEW_DATAPLANE", "1")
    engine_args = type(
        "EngineArgsStub",
        (),
        {
            "_ascend_moe_offload_model_config": QWEN3_MOE_CONFIG,
            "offload_backend": "auto",
            "offload_group_size": 0,
            "offload_num_in_group": 1,
            "offload_prefetch_step": 0,
            "offload_params": set(),
            "cpu_offload_gb": 0,
            "cpu_offload_params": set(),
        },
    )()

    assert apply_moe_offload_defaults(engine_args) is True

    assert engine_args.offload_backend == "auto"
    assert engine_args.offload_group_size == 0
    assert engine_args.offload_num_in_group == 1
    assert engine_args.offload_prefetch_step == 0
    assert engine_args.offload_params == set()
    assert os.environ["VLLM_ASCEND_MOE_OFFLOAD_GRAPH_COMPATIBLE"] == "1"
    assert os.environ["VLLM_ASCEND_MOE_OFFLOAD_STAGE_SEAM"] == "1"
    assert os.environ["VLLM_ASCEND_MOE_OFFLOAD_B2_WAVE_PREFILL"] == "1"
    assert os.environ["VLLM_ASCEND_MOE_OFFLOAD_ASYNC_LOAD"] == "1"
    assert os.environ["VLLM_ASCEND_MOE_OFFLOAD_PIN_HOST_MEMORY"] == "1"
    assert os.environ["VLLM_ASCEND_MOE_OFFLOAD_TRANSFER_AWARE_SCHEDULE"] == "1"
    assert os.environ["VLLM_ASCEND_MOE_OFFLOAD_PREFILL_PREFETCH_DEPTH"] == "1"
    assert os.environ["VLLM_ASCEND_MOE_OFFLOAD_PREFILL_BUFFER_COUNT"] == "2"
    assert os.environ["VLLM_ASCEND_MOE_OFFLOAD_ROUTE_STATS_CACHE"] == "0"
    assert os.environ["VLLM_ASCEND_MOE_OFFLOAD_RELEASE_ORIGINAL_EXPERT_WEIGHTS"] == "1"
    assert os.environ["VLLM_ASCEND_MOE_OFFLOAD_LAYERED_RUNTIME"] == "0"
    assert engine_args._ascend_moe_offload_sew_dataplane is True


def test_apply_defaults_sew_dataplane_preserves_release_override(monkeypatch):
    monkeypatch.setenv(MOE_OFFLOAD_GB_ENV, "14")
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_SEW_DATAPLANE", "1")
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_RELEASE_ORIGINAL_EXPERT_WEIGHTS", "0")
    engine_args = type(
        "EngineArgsStub",
        (),
        {
            "_ascend_moe_offload_model_config": QWEN3_MOE_CONFIG,
            "offload_backend": "auto",
            "offload_group_size": 0,
            "offload_num_in_group": 1,
            "offload_prefetch_step": 0,
            "offload_params": set(),
            "cpu_offload_gb": 0,
            "cpu_offload_params": set(),
        },
    )()

    assert apply_moe_offload_defaults(engine_args) is True
    assert os.environ["VLLM_ASCEND_MOE_OFFLOAD_RELEASE_ORIGINAL_EXPERT_WEIGHTS"] == "0"


def test_apply_defaults_sew_dataplane_uses_prefill_profile_for_residency(
    monkeypatch,
    tmp_path,
):
    profile = tmp_path / "prefill_profile.jsonl"
    records = [
        {
            "name": "b2_work_conserving_prefill",
            "layer_id": 3,
            "seconds": 0.1,
            "payload": {"control_ms": {"end_to_end": 100.0}},
        },
        {
            "name": "b2_work_conserving_prefill",
            "layer_id": 0,
            "seconds": 0.01,
            "payload": {"control_ms": {"end_to_end": 10.0}},
        },
    ]
    profile.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    monkeypatch.setenv(MOE_OFFLOAD_GB_ENV, "14")
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_SEW_DATAPLANE", "1")
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_PREFILL_RESIDENCY_PROFILE", str(profile))
    engine_args = type(
        "EngineArgsStub",
        (),
        {
            "_ascend_moe_offload_model_config": QWEN3_MOE_CONFIG,
            "offload_backend": "auto",
            "offload_group_size": 0,
            "offload_num_in_group": 1,
            "offload_prefetch_step": 0,
            "offload_params": set(),
            "cpu_offload_gb": 0,
            "cpu_offload_params": set(),
        },
    )()

    assert apply_moe_offload_defaults(engine_args) is True

    resident_ids = {
        int(part)
        for part in os.environ["VLLM_ASCEND_MOE_OFFLOAD_RESIDENT_LAYER_IDS"].split(",")
        if part
    }
    assert 3 in resident_ids
    assert 0 not in resident_ids
    plan = engine_args._ascend_moe_offload_autoconfig_plan
    assert plan["prefill_residency_plan"]["reason"] == "profile_guided_swaps"
    assert plan["prefill_residency_plan"]["swaps"][0]["make_resident"] == 3
    assert plan["prefill_residency_plan"]["swaps"][0]["make_offloaded"] == 0
    assert plan["prefill_residency_plan"]["swaps"]


def test_apply_defaults_sew_dataplane_merges_multiple_prefill_profiles(
    monkeypatch,
    tmp_path,
):
    offloaded_profile = tmp_path / "b2_profile.jsonl"
    resident_profile = tmp_path / "resident_profile.jsonl"
    offloaded_profile.write_text(
        json.dumps(
            {
                "name": "b2_work_conserving_prefill",
                "layer_id": 3,
                "seconds": 0.1,
                "payload": {"control_ms": {"end_to_end": 100.0}},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    resident_profile.write_text(
        json.dumps(
            {
                "name": "prefill_resident_native",
                "layer_id": 0,
                "seconds": 0.01,
                "payload": {"n_tokens": 128},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(MOE_OFFLOAD_GB_ENV, "14")
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_SEW_DATAPLANE", "1")
    monkeypatch.setenv(
        "VLLM_ASCEND_MOE_OFFLOAD_PREFILL_RESIDENCY_PROFILE",
        os.pathsep.join((str(offloaded_profile), str(resident_profile))),
    )
    engine_args = type(
        "EngineArgsStub",
        (),
        {
            "_ascend_moe_offload_model_config": QWEN3_MOE_CONFIG,
            "offload_backend": "auto",
            "offload_group_size": 0,
            "offload_num_in_group": 1,
            "offload_prefetch_step": 0,
            "offload_params": set(),
            "cpu_offload_gb": 0,
            "cpu_offload_params": set(),
        },
    )()

    assert apply_moe_offload_defaults(engine_args) is True

    plan = engine_args._ascend_moe_offload_autoconfig_plan
    assert plan["prefill_residency_profiles"] == (
        str(offloaded_profile),
        str(resident_profile),
    )
    assert plan["prefill_residency_plan"]["reason"] == "profile_guided_swaps"
    assert 3 in plan["resident_layer_ids"]
    assert 0 in plan["offloaded_layer_ids"]


def test_apply_defaults_sew_dataplane_rejects_native_prefetch(monkeypatch):
    monkeypatch.setenv(MOE_OFFLOAD_GB_ENV, "14")
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_SEW_DATAPLANE", "1")
    engine_args = type(
        "EngineArgsStub",
        (),
        {
            "_ascend_moe_offload_model_config": QWEN3_MOE_CONFIG,
            "offload_backend": "prefetch",
            "offload_group_size": 0,
            "cpu_offload_gb": 0,
            "cpu_offload_params": set(),
        },
    )()

    with pytest.raises(ValueError, match="SEW fixed-slot data plane"):
        apply_moe_offload_defaults(engine_args)


def test_apply_defaults_rejects_layered_runtime_with_released_full_weights(monkeypatch):
    monkeypatch.setenv(MOE_OFFLOAD_GB_ENV, "14")
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_RELEASE_ORIGINAL_EXPERT_WEIGHTS", "1")
    engine_args = type("EngineArgsStub", (), {"_ascend_moe_offload_model_config": QWEN3_MOE_CONFIG})()

    with pytest.raises(ValueError, match="Layered runtime needs the full expert-weight path"):
        apply_moe_offload_defaults(engine_args)


def test_apply_defaults_allows_release_in_pure_fixed_slot_mode(monkeypatch):
    monkeypatch.setenv(MOE_OFFLOAD_GB_ENV, "14")
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_LAYERED_RUNTIME", "0")
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_RELEASE_ORIGINAL_EXPERT_WEIGHTS", "1")
    engine_args = type("EngineArgsStub", (), {"_ascend_moe_offload_model_config": QWEN3_MOE_CONFIG})()

    assert apply_moe_offload_defaults(engine_args) is True


def test_derive_defaults_supports_qwen_moe_aliases():
    config = {
        "hidden_size": 2048,
        "intermediate_size": 768,
        "n_routed_experts": 128,
        "num_hidden_layers": 48,
        "torch_dtype": "bfloat16",
    }

    defaults = derive_prefetch_defaults(14, config)

    assert defaults["offload_group_size"] == 4
    assert defaults["offload_num_in_group"] == 1
    assert defaults["offloaded_layer_ids"][:3] == (3, 7, 11)
    assert 13 <= defaults["estimated_offloaded_gb"] <= 15


def test_apply_profile_guided_residency_swaps_expensive_offloaded_layer(tmp_path):
    profile = tmp_path / "prefill_profile.jsonl"
    records = [
        {
            "name": "b2_work_conserving_prefill",
            "layer_id": 3,
            "seconds": 0.1,
            "payload": {"control_ms": {"end_to_end": 100.0}},
        },
        {
            "name": "b2_work_conserving_prefill",
            "layer_id": 7,
            "seconds": 0.09,
            "payload": {"control_ms": {"end_to_end": 90.0}},
        },
        {
            "name": "b2_work_conserving_prefill",
            "layer_id": 0,
            "seconds": 0.01,
            "payload": {"control_ms": {"end_to_end": 10.0}},
        },
        {
            "name": "b2_work_conserving_prefill",
            "layer_id": 1,
            "seconds": 0.02,
            "payload": {"control_ms": {"end_to_end": 20.0}},
        },
    ]
    profile.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    defaults = derive_prefetch_defaults(14, QWEN3_MOE_CONFIG)
    updated = apply_profile_guided_residency(
        defaults,
        profile_path=str(profile),
        model_config=QWEN3_MOE_CONFIG,
    )

    assert 3 in updated["resident_layer_ids"]
    assert 7 in updated["resident_layer_ids"]
    assert 0 in updated["offloaded_layer_ids"]
    assert 1 in updated["offloaded_layer_ids"]
    assert len(updated["resident_layer_ids"]) == len(defaults["resident_layer_ids"])
    assert len(updated["offloaded_layer_ids"]) == len(defaults["offloaded_layer_ids"])
    assert updated["prefill_residency_plan"]["reason"] == "profile_guided_swaps"


def test_prefill_residency_profile_loader_reads_resident_native_events(tmp_path):
    profile = tmp_path / "prefill_profile.jsonl"
    records = [
        {
            "name": "prefill_resident_native",
            "layer_id": 0,
            "seconds": 0.012,
            "payload": {"n_tokens": 128, "path": "native_fused_moe"},
        },
        {
            "name": "b2_work_conserving_prefill",
            "layer_id": 3,
            "seconds": 0.1,
            "payload": {
                "control_ms": {"end_to_end": 100.0},
                "n_pairs": 256,
                "n_waves": 8,
                "n_active": 64,
            },
        },
    ]
    profile.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    costs = load_prefill_layer_costs(profile)

    assert costs[0].resident_native_calls == 1
    assert costs[0].score_ms == pytest.approx(12.0)
    assert costs[0].pairs == 128
    assert costs[3].b2_calls == 1
    assert costs[3].score_ms == pytest.approx(100.0)


def test_prefill_residency_profile_loader_reads_b2_wave_summary(tmp_path):
    profile = tmp_path / "prefill_profile.jsonl"
    record = {
        "name": "b2_work_conserving_prefill",
        "layer_id": 3,
        "seconds": 0.1,
            "payload": {
                "control_ms": {"end_to_end": 100.0, "scatter_total": 999.0},
                "n_pairs": 256,
                "n_waves": 8,
                "n_active": 64,
                "wave_summary": {
                    "gmm_ms": 23.5,
                    "stage_wait_ms": 7.25,
                    "layer_scatter_ms": 12.5,
                    "h2d_bytes": 4096,
                },
            "waves": [
                {
                    "gmm_ms": 999.0,
                    "stage_wait_ms": 999.0,
                    "h2d_bytes": 999,
                }
            ],
        },
    }
    profile.write_text(json.dumps(record) + "\n", encoding="utf-8")

    costs = load_prefill_layer_costs(profile)

    assert costs[3].gmm_ms == pytest.approx(23.5)
    assert costs[3].stage_wait_ms == pytest.approx(7.25)
    assert costs[3].scatter_ms == pytest.approx(12.5)
    assert costs[3].h2d_bytes == 4096


def test_prefill_residency_profile_loader_merges_many_files(tmp_path):
    offloaded_profile = tmp_path / "b2_profile.jsonl"
    resident_profile = tmp_path / "resident_profile.jsonl"
    offloaded_profile.write_text(
        json.dumps(
            {
                "name": "b2_work_conserving_prefill",
                "layer_id": 3,
                "seconds": 0.1,
                "payload": {"control_ms": {"end_to_end": 100.0}},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    resident_profile.write_text(
        json.dumps(
            {
                "name": "prefill_resident_native",
                "layer_id": 0,
                "seconds": 0.01,
                "payload": {"n_tokens": 128},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    costs = load_prefill_layer_costs_many(
        (str(offloaded_profile), str(resident_profile))
    )

    assert costs[3].b2_calls == 1
    assert costs[0].resident_native_calls == 1
    defaults = derive_prefetch_defaults(14, QWEN3_MOE_CONFIG)
    placement = plan_profile_guided_prefill_residency(
        num_layers=QWEN3_MOE_CONFIG["num_hidden_layers"],
        default_offloaded_layer_ids=tuple(defaults["offloaded_layer_ids"]),
        layer_costs=costs,
    )
    assert placement.reason == "profile_guided_swaps"
    assert placement.swaps[0]["make_resident"] == 3
    assert placement.swaps[0]["make_offloaded"] == 0


def test_prefill_residency_plan_reports_missing_resident_evidence(tmp_path):
    profile = tmp_path / "prefill_profile.jsonl"
    records = [
        {
            "name": "b2_work_conserving_prefill",
            "layer_id": 3,
            "seconds": 0.1,
            "payload": {"control_ms": {"end_to_end": 100.0}},
        },
        {
            "name": "b2_work_conserving_prefill",
            "layer_id": 7,
            "seconds": 0.09,
            "payload": {"control_ms": {"end_to_end": 90.0}},
        },
    ]
    profile.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    defaults = derive_prefetch_defaults(14, QWEN3_MOE_CONFIG)

    placement = plan_profile_guided_prefill_residency(
        num_layers=QWEN3_MOE_CONFIG["num_hidden_layers"],
        default_offloaded_layer_ids=tuple(defaults["offloaded_layer_ids"]),
        layer_costs=load_prefill_layer_costs(profile),
    )

    assert placement.reason == "missing_profiled_resident_layers"
    assert placement.swaps == ()
    assert placement.offloaded_layer_ids == defaults["offloaded_layer_ids"]


def test_derive_defaults_rejects_dense_config():
    with pytest.raises(ValueError, match="requires a MoE model config"):
        derive_prefetch_defaults(
            14,
            {
                "hidden_size": 4096,
                "intermediate_size": 11008,
                "num_hidden_layers": 32,
                "torch_dtype": "bfloat16",
            },
        )
