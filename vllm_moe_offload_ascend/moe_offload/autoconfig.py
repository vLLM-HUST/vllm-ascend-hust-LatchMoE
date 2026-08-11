#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path
from typing import Any

MOE_OFFLOAD_GB_ENV = "VLLM_ASCEND_MOE_OFFLOAD_GB"
_NUM_SLOTS_ENV = "VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS"
_FANOUT_THRESHOLD_ENV = "VLLM_ASCEND_MOE_OFFLOAD_FANOUT_THRESHOLD"
logger = logging.getLogger(__name__)
_BYTES_PER_GIB = 1024**3
_DEFAULT_PREFETCH_GROUP_SIZE = 4
_DEFAULT_TARGET_B2_WAVES = 4
_REFERENCE_OFFLOAD_GB_FOR_SLOTS = 14.0
_MIN_TARGET_B2_WAVES = 2
_MIN_NET_SAVING_GB_ENV = "VLLM_ASCEND_MOE_OFFLOAD_MIN_NET_SAVING_GB"
_MIN_NET_SAVING_RATIO_ENV = "VLLM_ASCEND_MOE_OFFLOAD_MIN_NET_SAVING_RATIO"
_SLOT_HBM_BUDGET_GB_ENV = "VLLM_ASCEND_MOE_OFFLOAD_SLOT_HBM_BUDGET_GB"
_SLOT_HBM_FRACTION_ENV = "VLLM_ASCEND_MOE_OFFLOAD_SLOT_HBM_FRACTION"
_NUM_SLOTS_BOOTSTRAP_ENV = "VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS_AUTOCONFIG_BOOTSTRAP"
_KV_RESERVE_SEQS_ENV = "VLLM_ASCEND_MOE_OFFLOAD_KV_RESERVE_SEQS"
_KV_RESERVE_CTX_ENV = "VLLM_ASCEND_MOE_OFFLOAD_KV_RESERVE_CTX"
_SYSTEM_RESERVE_GB_ENV = "VLLM_ASCEND_MOE_OFFLOAD_SYSTEM_RESERVE_GB"
_ACTIVATION_RESERVE_GB_ENV = "VLLM_ASCEND_MOE_OFFLOAD_ACTIVATION_RESERVE_GB"
_DEFAULT_MIN_NET_SAVING_RATIO = 0.25
_DEFAULT_KV_RESERVE_SEQS = 4
_DEFAULT_SYSTEM_RESERVE_GB = 2.0
_DEFAULT_ACTIVATION_RESERVE_GB = 1.0
_DEFAULT_SLOT_HBM_FRACTION = 0.12
_DEFAULT_QWEN3_30B_A3B_CONFIG = {
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

_DEFAULT_ENGINE_ARGS = {
    "offload_backend": "prefetch",
    "offload_prefetch_step": 1,
    "offload_params": {"experts"},
    "cpu_offload_gb": 0,
    "cpu_offload_params": set(),
}

_DEFAULT_ENV_VARS = {
    "VLLM_ASCEND_MOE_OFFLOAD_ENABLED": "1",
    "VLLM_ASCEND_MOE_OFFLOAD_TRACE_ONLY": "0",
    "VLLM_ASCEND_MOE_OFFLOAD_POLICY": "deadline",
    "VLLM_ASCEND_MOE_OFFLOAD_ASYNC_LOAD": "0",
    "VLLM_ASCEND_MOE_OFFLOAD_MAX_PHASES": "1",
    # Low-fanout decode uses the fixed slot cache. High-fanout prefill falls
    # back to the full expert weights that PrefetchOffloader brings on device.
    "VLLM_ASCEND_MOE_OFFLOAD_LAYERED_RUNTIME": "1",
}
_RESIDENT_LAYER_IDS_ENV = "VLLM_ASCEND_MOE_OFFLOAD_RESIDENT_LAYER_IDS"
_SEW_DATAPLANE_ENV = "VLLM_ASCEND_MOE_OFFLOAD_SEW_DATAPLANE"
_PREFILL_RESIDENCY_PROFILE_ENV = "VLLM_ASCEND_MOE_OFFLOAD_PREFILL_RESIDENCY_PROFILE"
_SEW_DATAPLANE_ENV_VARS = {
    "VLLM_ASCEND_MOE_OFFLOAD_GRAPH_COMPATIBLE": "1",
    "VLLM_ASCEND_MOE_OFFLOAD_STAGE_SEAM": "1",
    "VLLM_ASCEND_MOE_OFFLOAD_B2_WAVE_PREFILL": "1",
    # Multi-wave BF16 accumulation is not token-equivalent to the native
    # single-pass MoE path. Keep it opt-in until it passes the oracle suite.
    "VLLM_ASCEND_MOE_OFFLOAD_B2_OVERFLOW_MODE": "full_layer",
    "VLLM_ASCEND_MOE_OFFLOAD_LAYERED_RUNTIME": "0",
    "VLLM_ASCEND_MOE_OFFLOAD_ASYNC_LOAD": "1",
    "VLLM_ASCEND_MOE_OFFLOAD_PIN_HOST_MEMORY": "1",
    "VLLM_ASCEND_MOE_OFFLOAD_TRANSFER_AWARE_SCHEDULE": "1",
    "VLLM_ASCEND_MOE_OFFLOAD_PREFILL_PREFETCH_DEPTH": "1",
    "VLLM_ASCEND_MOE_OFFLOAD_PREFILL_BUFFER_COUNT": "2",
    "VLLM_ASCEND_MOE_OFFLOAD_DEVICE_PAIR_PLANNING": "1",
    # Correctness-first default. Enable only for the explicit cache matrix.
    "VLLM_ASCEND_MOE_OFFLOAD_ROUTE_STATS_CACHE": "0",
    # SEW is a pure fixed-slot path, so the full NPU expert copies are unused
    # after the host store and slot banks have been initialized.
    "VLLM_ASCEND_MOE_OFFLOAD_RELEASE_ORIGINAL_EXPERT_WEIGHTS": "1",
}


def get_moe_offload_gb() -> float:
    raw_value = os.getenv(MOE_OFFLOAD_GB_ENV)
    if raw_value is None:
        return 0.0
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{MOE_OFFLOAD_GB_ENV} must be a non-negative number, got {raw_value!r}.") from exc
    if value < 0:
        raise ValueError(f"{MOE_OFFLOAD_GB_ENV} must be a non-negative number, got {raw_value!r}.")
    return value


def is_moe_offload_autoconfig_enabled() -> bool:
    return get_moe_offload_gb() > 0


def _sew_dataplane_selected() -> bool:
    return _to_bool_env(_SEW_DATAPLANE_ENV)


class _MoeOffloadGbAction(argparse.Action):

    def __call__(self, parser, namespace, values, option_string=None):
        try:
            value = float(values)
        except ValueError as exc:
            raise argparse.ArgumentError(self, f"{MOE_OFFLOAD_GB_ENV} must be a non-negative number.") from exc
        if value < 0:
            raise argparse.ArgumentError(self, f"{MOE_OFFLOAD_GB_ENV} must be a non-negative number.")
        os.environ[MOE_OFFLOAD_GB_ENV] = str(values)
        setattr(namespace, self.dest, value)


def register_moe_offload_cli_arg(parser: argparse.ArgumentParser) -> None:
    if "--ascend-moe-offload-gb" in parser._option_string_actions:
        return
    group = parser.add_argument_group("Ascend MoE Offload")
    group.add_argument(
        "--ascend-moe-offload-gb",
        dest="ascend_moe_offload_gb",
        default=None,
        metavar="GB",
        action=_MoeOffloadGbAction,
        help=(
            "Enable Ascend MoE expert offload with the vLLM PrefetchOffloader "
            "and fixed-slot MoE runtime. The value is the target expert-weight "
            "offload budget in GiB and is used to derive the prefetch offload "
            "layer fraction. 0 or omitted keeps the normal non-offloaded path. "
            "Do not combine with cpu_offload_gb/UVA."
        ),
    )


def _dtype_size_bytes(torch_dtype: Any) -> int:
    dtype = str(torch_dtype).lower().replace("torch.", "")
    if dtype in ("float32", "fp32"):
        return 4
    if dtype in ("float16", "fp16", "bfloat16", "bf16"):
        return 2
    if dtype in ("float8", "fp8", "int8"):
        return 1
    return 2


def _get_int_config_value(
    model_config: dict[str, Any],
    canonical_name: str,
    aliases: tuple[str, ...] = (),
) -> int:
    for key in (canonical_name, *aliases):
        value = model_config.get(key)
        if value is not None:
            return int(value)
    keys = ", ".join((canonical_name, *aliases))
    raise ValueError(
        f"{MOE_OFFLOAD_GB_ENV} requires a MoE model config with {keys}. "
        "Disable Ascend MoE offload for dense models, or provide a supported "
        "MoE model config."
    )


def _expert_layer_gb(model_config: dict[str, Any]) -> float:
    hidden_size = _get_int_config_value(model_config, "hidden_size")
    moe_intermediate_size = _get_int_config_value(
        model_config,
        "moe_intermediate_size",
        aliases=("intermediate_size",),
    )
    num_experts = _get_int_config_value(
        model_config,
        "num_experts",
        aliases=("n_routed_experts", "moe_num_experts"),
    )
    dtype_size = _dtype_size_bytes(model_config.get("torch_dtype", "bfloat16"))
    # Fused routed experts hold gate/up (2 * I * H) and down (I * H) weights.
    layer_bytes = 3 * hidden_size * moe_intermediate_size * num_experts * dtype_size
    return layer_bytes / _BYTES_PER_GIB


def derive_prefetch_defaults(
    target_offload_gb: float,
    model_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if target_offload_gb <= 0:
        raise ValueError("target_offload_gb must be greater than 0")

    model_config = model_config or _DEFAULT_QWEN3_30B_A3B_CONFIG
    num_layers = _get_int_config_value(model_config, "num_hidden_layers")
    group_size = min(_DEFAULT_PREFETCH_GROUP_SIZE, num_layers)
    num_groups = math.ceil(num_layers / group_size)
    layer_gb = _expert_layer_gb(model_config)
    target_layers_per_group = target_offload_gb / (layer_gb * num_groups)
    offload_num_in_group = min(group_size, max(1, round(target_layers_per_group)))
    offloaded_layer_ids = tuple(
        layer_id
        for layer_id in range(num_layers)
        if layer_id % group_size >= group_size - offload_num_in_group
    )
    offloaded_layer_id_set = set(offloaded_layer_ids)
    resident_layer_ids = tuple(
        layer_id for layer_id in range(num_layers) if layer_id not in offloaded_layer_id_set
    )
    estimated_offloaded_layers = len(offloaded_layer_ids)
    estimated_offloaded_gb = estimated_offloaded_layers * layer_gb

    return {
        "offload_group_size": group_size,
        "offload_num_in_group": offload_num_in_group,
        "offloaded_layer_ids": offloaded_layer_ids,
        "resident_layer_ids": resident_layer_ids,
        "estimated_offloaded_layers": estimated_offloaded_layers,
        "estimated_offloaded_gb": estimated_offloaded_gb,
        "expert_layer_gb": layer_gb,
    }


def _get_optional_int_config_value(
    model_config: dict[str, Any],
    canonical_name: str,
    aliases: tuple[str, ...] = (),
    *,
    default: int,
) -> int:
    for key in (canonical_name, *aliases):
        value = model_config.get(key)
        if value is not None:
            return int(value)
    return int(default)


def _get_engine_int(engine_args: Any | None, field_name: str, default: int) -> int:
    if engine_args is None:
        return int(default)
    value = getattr(engine_args, field_name, None)
    if value is None:
        return int(default)
    try:
        value = int(value)
    except (TypeError, ValueError):
        return int(default)
    return value if value > 0 else int(default)


def _get_engine_float(engine_args: Any | None, field_name: str, default: float) -> float:
    if engine_args is None:
        return float(default)
    value = getattr(engine_args, field_name, None)
    if value is None:
        return float(default)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    return value if value > 0 else float(default)


def _get_optional_env_float(name: str) -> float | None:
    raw_value = os.getenv(name)
    if raw_value is None or str(raw_value).strip() == "":
        return None
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a non-negative number, got {raw_value!r}.") from exc
    if value < 0:
        raise ValueError(f"{name} must be a non-negative number, got {raw_value!r}.")
    return value


def _get_optional_env_int(name: str) -> int | None:
    raw_value = os.getenv(name)
    if raw_value is None or str(raw_value).strip() == "":
        return None
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a non-negative integer, got {raw_value!r}.") from exc
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {raw_value!r}.")
    return value


def _target_b2_waves_for_offload(target_offload_gb: float) -> int:
    """Scale Prefill B2 wave target with the user's offload budget.

    ``14 GiB`` is the measured stable baseline for Qwen3-30B-A3B and maps to
    four waves (32 slots for 128 experts). Larger offload targets are expected
    to buy back Prefill latency by reducing the wave count; e.g. 28 GiB maps to
    two waves (64 slots). We intentionally stop at two waves for auto-config so
    full residency remains an explicit expert override.
    """

    if target_offload_gb <= 0:
        return _DEFAULT_TARGET_B2_WAVES
    scaled = math.ceil(
        _DEFAULT_TARGET_B2_WAVES
        * _REFERENCE_OFFLOAD_GB_FOR_SLOTS
        / float(target_offload_gb)
    )
    return max(_MIN_TARGET_B2_WAVES, int(scaled))


def _slot_reserve_breakdown(
    *,
    advisor: Any,
    serving: Any,
    num_offloaded_layers: int,
    max_model_len: int,
) -> dict[str, Any]:
    from vllm_moe_offload_ascend.moe_offload.autoconfig_advisor import (
        ServingConfig,
    )

    kv_reserve_seqs = _get_optional_env_int(_KV_RESERVE_SEQS_ENV)
    if kv_reserve_seqs is None:
        kv_reserve_seqs = _DEFAULT_KV_RESERVE_SEQS
    kv_reserve_ctx = _get_optional_env_int(_KV_RESERVE_CTX_ENV)
    if kv_reserve_ctx is None:
        kv_reserve_ctx = min(int(max_model_len), 8192)
    kv_reserve_serving = ServingConfig(
        max_batch_size=max(1, int(kv_reserve_seqs)),
        max_seq_len=max(1, int(kv_reserve_ctx)),
        top_k=int(serving.top_k),
        target_b2_waves=int(serving.target_b2_waves),
    )
    kv_reserve_gib = advisor.estimate_kv_cache_gib(kv_reserve_serving)

    activation_estimate_gib = advisor.estimate_activation_gib(serving)
    activation_floor_gib = _get_optional_env_float(_ACTIVATION_RESERVE_GB_ENV)
    if activation_floor_gib is None:
        activation_floor_gib = _DEFAULT_ACTIVATION_RESERVE_GB
    activation_reserve_gib = max(
        float(activation_estimate_gib),
        float(activation_floor_gib),
    )

    system_reserve_gib = _get_optional_env_float(_SYSTEM_RESERVE_GB_ENV)
    if system_reserve_gib is None:
        system_reserve_gib = _DEFAULT_SYSTEM_RESERVE_GB

    prefill_buffer_count = _get_optional_env_int(
        "VLLM_ASCEND_MOE_OFFLOAD_PREFILL_BUFFER_COUNT"
    )
    if prefill_buffer_count is None:
        prefill_buffer_count = int(_SEW_DATAPLANE_ENV_VARS[
            "VLLM_ASCEND_MOE_OFFLOAD_PREFILL_BUFFER_COUNT"
        ])
    prefill_buffer_count = max(0, int(prefill_buffer_count))

    non_slot_reserve_gib = (
        float(kv_reserve_gib)
        + float(activation_reserve_gib)
        + float(system_reserve_gib)
    )
    return {
        "kv_reserve_seqs": int(kv_reserve_seqs),
        "kv_reserve_ctx": int(kv_reserve_ctx),
        "kv_reserve_gib": float(kv_reserve_gib),
        "activation_estimate_gib": float(activation_estimate_gib),
        "activation_reserve_gib": float(activation_reserve_gib),
        "system_reserve_gib": float(system_reserve_gib),
        "non_slot_reserve_gib": float(non_slot_reserve_gib),
        "prefill_buffer_count": int(prefill_buffer_count),
        "affordability_layer_equivalent_count": int(num_offloaded_layers)
        + int(prefill_buffer_count),
    }


def _read_npu_hbm_budget_gib(engine_args: Any | None) -> tuple[float | None, dict[str, Any]]:
    explicit_budget = _get_optional_env_float(_SLOT_HBM_BUDGET_GB_ENV)
    if explicit_budget is not None:
        return explicit_budget, {
            "source": "env_override",
            "env": _SLOT_HBM_BUDGET_GB_ENV,
            "slot_hbm_budget_gib": explicit_budget,
        }

    try:
        import torch

        if not hasattr(torch, "npu"):
            return None, {"source": "unavailable", "reason": "torch_npu_missing"}
        free_bytes, total_bytes = torch.npu.mem_get_info()
    except Exception as exc:
        return None, {"source": "unavailable", "reason": type(exc).__name__}

    free_gib = float(free_bytes) / _BYTES_PER_GIB
    total_gib = float(total_bytes) / _BYTES_PER_GIB
    used_gib = max(0.0, total_gib - free_gib)
    gpu_memory_utilization = min(
        1.0,
        max(0.0, _get_engine_float(engine_args, "gpu_memory_utilization", 1.0)),
    )
    usable_gib = max(0.0, total_gib * gpu_memory_utilization - used_gib)
    return usable_gib, {
        "source": "torch.npu.mem_get_info",
        "free_gib": free_gib,
        "total_gib": total_gib,
        "used_gib": used_gib,
        "gpu_memory_utilization": gpu_memory_utilization,
        "slot_hbm_budget_gib": usable_gib,
    }


def _physical_slot_fraction_budget_gib(real_hbm_info: dict[str, Any]) -> tuple[float | None, dict[str, Any] | None]:
    if real_hbm_info.get("source") != "torch.npu.mem_get_info":
        return None, None
    total_gib = real_hbm_info.get("total_gib")
    if total_gib is None:
        return None, None
    fraction = _get_optional_env_float(_SLOT_HBM_FRACTION_ENV)
    if fraction is None:
        fraction = _DEFAULT_SLOT_HBM_FRACTION
    fraction = min(1.0, max(0.0, float(fraction)))
    budget_gib = max(0.0, float(total_gib) * fraction)
    return budget_gib, {
        "source": (
            "fraction_default"
            if _SLOT_HBM_FRACTION_ENV not in os.environ
            else "fraction_env"
        ),
        "slot_hbm_fraction": fraction,
        "slot_fraction_budget_gib": budget_gib,
    }


def _minimum_net_saving_gib(estimated_offloaded_gb: float) -> tuple[float, dict[str, Any]]:
    explicit_min_saving = _get_optional_env_float(_MIN_NET_SAVING_GB_ENV)
    ratio = _get_optional_env_float(_MIN_NET_SAVING_RATIO_ENV)
    if ratio is None:
        ratio = _DEFAULT_MIN_NET_SAVING_RATIO
    ratio = min(1.0, max(0.0, float(ratio)))
    ratio_min_saving = max(0.0, float(estimated_offloaded_gb) * ratio)
    if explicit_min_saving is None:
        min_saving = ratio_min_saving
        source = "ratio_default" if _MIN_NET_SAVING_RATIO_ENV not in os.environ else "ratio_env"
    else:
        min_saving = max(float(explicit_min_saving), ratio_min_saving)
        source = "max_env_gb_and_ratio"
    return min_saving, {
        "source": source,
        "min_net_saving_gib": min_saving,
        "min_net_saving_ratio": ratio,
        "explicit_min_net_saving_gib": explicit_min_saving,
    }


def derive_num_slots_defaults(
    target_offload_gb: float,
    model_config: dict[str, Any] | None,
    prefetch_defaults: dict[str, Any],
    engine_args: Any | None = None,
) -> dict[str, Any]:
    """Derive startup fixed-slot capacity from the offload budget.

    ``num_slots`` is an internal capacity for the fixed-address expert slot
    pool.  Normal users choose ``ascend-moe-offload-gb``; this function maps
    that budget plus model/serving shape into a stable startup slot count.
    """

    from vllm_moe_offload_ascend.moe_offload.autoconfig_advisor import (
        AutoConfigAdvisor,
        ServingConfig,
    )

    config = model_config or _DEFAULT_QWEN3_30B_A3B_CONFIG
    advisor = AutoConfigAdvisor.from_model_config(config)
    top_k = _get_optional_int_config_value(
        config,
        "num_experts_per_tok",
        aliases=("num_experts_per_token", "moe_top_k", "top_k"),
        default=2,
    )
    max_model_len_default = _get_optional_int_config_value(
        config,
        "max_position_embeddings",
        aliases=("seq_length",),
        default=4096,
    )
    serving = ServingConfig(
        max_batch_size=_get_engine_int(engine_args, "max_num_seqs", 1),
        max_seq_len=_get_engine_int(engine_args, "max_model_len", max_model_len_default),
        top_k=top_k,
        target_b2_waves=_target_b2_waves_for_offload(target_offload_gb),
    )
    num_offloaded_layers = int(prefetch_defaults["estimated_offloaded_layers"])
    estimated_offloaded_gb = float(prefetch_defaults["estimated_offloaded_gb"])
    min_net_saving_gib, min_saving_info = _minimum_net_saving_gib(
        estimated_offloaded_gb,
    )
    net_saving_slot_budget_gib = max(
        0.0,
        estimated_offloaded_gb - min_net_saving_gib,
    )
    real_hbm_budget_gib, real_hbm_info = _read_npu_hbm_budget_gib(engine_args)
    reserve_info = _slot_reserve_breakdown(
        advisor=advisor,
        serving=serving,
        num_offloaded_layers=num_offloaded_layers,
        max_model_len=int(serving.max_seq_len),
    )
    non_slot_reserve_gib = float(reserve_info["non_slot_reserve_gib"])
    net_saving_slot_budget_after_reserve_gib = max(
        0.0,
        net_saving_slot_budget_gib - non_slot_reserve_gib,
    )
    slot_budget_gib = net_saving_slot_budget_after_reserve_gib
    if real_hbm_budget_gib is not None:
        slot_budget_gib = min(
            slot_budget_gib,
            max(0.0, real_hbm_budget_gib - non_slot_reserve_gib),
        )
    physical_slot_budget_gib, physical_slot_budget_info = (
        _physical_slot_fraction_budget_gib(real_hbm_info)
    )
    if physical_slot_budget_gib is not None:
        slot_budget_gib = min(slot_budget_gib, physical_slot_budget_gib)
    affordability_layer_count = int(
        reserve_info["affordability_layer_equivalent_count"]
    )
    num_slots = advisor.suggest_num_slots(
        serving,
        affordability_layer_count,
        slot_budget_gib,
    )
    slot_bank_gib = advisor.slot_bank_gib(int(num_slots), num_offloaded_layers)
    b2_stage_bank_gib = advisor.single_expert_gib() * int(num_slots) * int(
        reserve_info["prefill_buffer_count"]
    )
    return {
        "num_slots": int(num_slots),
        "fanout_threshold": int(num_slots),
        "slot_budget_gib": float(slot_budget_gib),
        "slot_budget_constraints": {
            "net_saving_slot_budget_gib": float(net_saving_slot_budget_gib),
            "net_saving_slot_budget_after_reserve_gib": float(
                net_saving_slot_budget_after_reserve_gib
            ),
            "real_hbm_slot_budget_gib": real_hbm_budget_gib,
            "effective_slot_budget_gib": float(slot_budget_gib),
            "minimum_net_saving": min_saving_info,
            "real_hbm": real_hbm_info,
            "physical_slot_fraction": physical_slot_budget_info,
            "reserve": reserve_info,
        },
        "target_b2_waves": int(serving.target_b2_waves),
        "decode_working_set": min(
            int(serving.top_k) * int(serving.max_batch_size),
            advisor.num_experts,
        ),
        "estimated_b2_waves": math.ceil(advisor.num_experts / max(1, int(num_slots))),
        "slot_bank_gib": slot_bank_gib,
        "b2_stage_bank_gib": b2_stage_bank_gib,
        "slot_plus_b2_stage_gib": slot_bank_gib + b2_stage_bank_gib,
        "net_hbm_saving_gib": max(0.0, estimated_offloaded_gb - slot_bank_gib),
    }


def apply_profile_guided_residency(
    defaults: dict[str, Any],
    *,
    profile_path: str | None,
    model_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Refine resident/offloaded layer ids using a Prefill profile."""

    if not profile_path:
        return defaults

    from vllm_moe_offload_ascend.moe_offload.prefill_residency import (
        load_prefill_layer_costs_many,
        plan_profile_guided_prefill_residency,
    )

    profile_paths = tuple(
        part.strip()
        for part in str(profile_path).split(os.pathsep)
        if part.strip()
    )
    if not profile_paths:
        return defaults

    config = model_config or _DEFAULT_QWEN3_30B_A3B_CONFIG
    num_layers = _get_int_config_value(config, "num_hidden_layers")
    costs = load_prefill_layer_costs_many(profile_paths)
    placement = plan_profile_guided_prefill_residency(
        num_layers=num_layers,
        default_offloaded_layer_ids=tuple(defaults["offloaded_layer_ids"]),
        layer_costs=costs,
    )
    if not placement.profiled_layer_ids:
        logger.warning(
            "%s=%s did not contain Prefill residency profile events; keeping "
            "group-based residency.",
            _PREFILL_RESIDENCY_PROFILE_ENV,
            profile_path,
        )
        return defaults

    updated = dict(defaults)
    updated["resident_layer_ids"] = placement.resident_layer_ids
    updated["offloaded_layer_ids"] = placement.offloaded_layer_ids
    updated["estimated_offloaded_layers"] = len(placement.offloaded_layer_ids)
    updated["estimated_offloaded_gb"] = (
        len(placement.offloaded_layer_ids) * float(defaults["expert_layer_gb"])
    )
    updated["prefill_residency_profile"] = os.pathsep.join(profile_paths)
    updated["prefill_residency_profiles"] = profile_paths
    updated["prefill_residency_plan"] = placement.to_jsonable()
    logger.info(
        "Applied profile-guided Prefill residency from %d profile file(s): "
        "reason=%s, swaps=%d, resident_layers=%d, offloaded_layers=%d.",
        len(profile_paths),
        placement.reason,
        len(placement.swaps),
        len(placement.resident_layer_ids),
        len(placement.offloaded_layer_ids),
    )
    return updated


def _load_model_config_dict(engine_args: Any) -> dict[str, Any] | None:
    test_config = getattr(engine_args, "_ascend_moe_offload_model_config", None)
    if isinstance(test_config, dict):
        return test_config

    model = getattr(engine_args, "model", None)
    if not model:
        return None
    config_path = Path(str(model)) / "config.json"
    if not config_path.is_file():
        return None
    with config_path.open(encoding="utf-8") as f:
        config = json.load(f)
    return config if isinstance(config, dict) else None


def _field_is_unset(current_value: Any, default_value: Any) -> bool:
    if current_value is None:
        return True
    if isinstance(default_value, set):
        return set(current_value) == set()
    return current_value == default_value


def _set_engine_default(engine_args: Any, field_name: str, value: Any) -> None:
    current_value = getattr(engine_args, field_name, None)
    upstream_defaults = {
        "offload_backend": "auto",
        "offload_group_size": 0,
        "offload_num_in_group": 1,
        "offload_prefetch_step": 1,
        "offload_params": set(),
        "cpu_offload_gb": 0,
        "cpu_offload_params": set(),
    }
    if not hasattr(engine_args, field_name) or _field_is_unset(current_value, upstream_defaults[field_name]):
        setattr(engine_args, field_name, value.copy() if isinstance(value, set) else value)


def _to_bool_env(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


def _raise_on_uva_conflict(engine_args: Any) -> None:
    cpu_offload_gb = float(getattr(engine_args, "cpu_offload_gb", 0) or 0)
    offload_backend = getattr(engine_args, "offload_backend", "auto")
    if cpu_offload_gb > 0 or offload_backend == "uva":
        raise ValueError(
            f"{MOE_OFFLOAD_GB_ENV} enables Ascend MoE offload through vLLM "
            "PrefetchOffloader. Remove cpu_offload_gb/UVA settings; they select "
            "the UVA offload backend, which is not the Ascend MoE offload path."
        )


def _raise_on_layered_release_conflict() -> None:
    if not _to_bool_env("VLLM_ASCEND_MOE_OFFLOAD_LAYERED_RUNTIME"):
        return
    if _to_bool_env("VLLM_ASCEND_MOE_OFFLOAD_RELEASE_ORIGINAL_EXPERT_WEIGHTS"):
        raise ValueError(
            "VLLM_ASCEND_MOE_OFFLOAD_RELEASE_ORIGINAL_EXPERT_WEIGHTS=1 is "
            "incompatible with VLLM_ASCEND_MOE_OFFLOAD_LAYERED_RUNTIME=1. "
            "Layered runtime needs the full expert-weight path for high-fanout "
            "batches; unset RELEASE_ORIGINAL_EXPERT_WEIGHTS or set "
            "VLLM_ASCEND_MOE_OFFLOAD_LAYERED_RUNTIME=0 for pure fixed-slot mode."
        )


def _raise_on_sew_native_offload_conflict(engine_args: Any) -> None:
    if not _sew_dataplane_selected():
        return
    offload_backend = getattr(engine_args, "offload_backend", "auto")
    offload_group_size = int(getattr(engine_args, "offload_group_size", 0) or 0)
    if offload_backend == "prefetch" or offload_group_size > 0:
        raise ValueError(
            f"{_SEW_DATAPLANE_ENV}=1 uses the graph-compatible SEW fixed-slot "
            "data plane. Remove native prefetch offload settings such as "
            "offload_backend=prefetch or offload_group_size."
        )


def apply_moe_offload_defaults(engine_args: Any) -> bool:
    target_offload_gb = get_moe_offload_gb()
    if target_offload_gb <= 0:
        return False

    _raise_on_uva_conflict(engine_args)
    sew_dataplane = _sew_dataplane_selected()
    explicit_num_slots = (
        _NUM_SLOTS_ENV in os.environ
        and os.getenv(_NUM_SLOTS_BOOTSTRAP_ENV) != "1"
    )
    explicit_fanout_threshold = _FANOUT_THRESHOLD_ENV in os.environ
    for env_name, value in _DEFAULT_ENV_VARS.items():
        if sew_dataplane and env_name in _SEW_DATAPLANE_ENV_VARS:
            continue
        os.environ.setdefault(env_name, value)
    if sew_dataplane:
        for env_name, value in _SEW_DATAPLANE_ENV_VARS.items():
            os.environ.setdefault(env_name, value)
    _raise_on_layered_release_conflict()
    _raise_on_sew_native_offload_conflict(engine_args)
    model_config = _load_model_config_dict(engine_args)
    prefetch_defaults = derive_prefetch_defaults(
        target_offload_gb,
        model_config,
    )
    if sew_dataplane and _RESIDENT_LAYER_IDS_ENV not in os.environ:
        prefetch_defaults = apply_profile_guided_residency(
            prefetch_defaults,
            profile_path=os.getenv(_PREFILL_RESIDENCY_PROFILE_ENV),
            model_config=model_config,
        )
    slot_defaults = derive_num_slots_defaults(
        target_offload_gb,
        model_config,
        prefetch_defaults,
        engine_args,
    )
    if not explicit_num_slots:
        os.environ[_NUM_SLOTS_ENV] = str(slot_defaults["num_slots"])
        os.environ[_NUM_SLOTS_BOOTSTRAP_ENV] = "0"
    if not explicit_fanout_threshold:
        os.environ[_FANOUT_THRESHOLD_ENV] = os.environ.get(
            _NUM_SLOTS_ENV,
            str(slot_defaults["fanout_threshold"]),
        )
    if _RESIDENT_LAYER_IDS_ENV not in os.environ:
        os.environ[_RESIDENT_LAYER_IDS_ENV] = ",".join(
            str(layer_id) for layer_id in prefetch_defaults["resident_layer_ids"]
        )
    if not sew_dataplane:
        engine_defaults = {
            **_DEFAULT_ENGINE_ARGS,
            "offload_group_size": prefetch_defaults["offload_group_size"],
            "offload_num_in_group": prefetch_defaults["offload_num_in_group"],
        }
        for field_name, value in engine_defaults.items():
            _set_engine_default(engine_args, field_name, value)

    autoconfig_plan = dict(prefetch_defaults)
    autoconfig_plan["slot_defaults"] = slot_defaults
    setattr(engine_args, "_ascend_moe_offload_autoconfig_applied", True)
    setattr(engine_args, "_ascend_moe_offload_sew_dataplane", sew_dataplane)
    logger.info(
        "Enabled Ascend MoE offload autoconfig from %s: target=%.3f GiB, "
        "estimated_offloaded_layers=%d, estimated_offloaded_gb=%.3f, "
        "resident_layers=%d, num_slots=%s%s, slot_bank=%.3f GiB, "
        "estimated_b2_waves=%d, offload_backend=%s.",
        MOE_OFFLOAD_GB_ENV,
        target_offload_gb,
        prefetch_defaults["estimated_offloaded_layers"],
        prefetch_defaults["estimated_offloaded_gb"],
        len(prefetch_defaults["resident_layer_ids"]),
        os.environ.get(_NUM_SLOTS_ENV, ""),
        " (override)" if explicit_num_slots else " (auto)",
        slot_defaults["slot_bank_gib"],
        slot_defaults["estimated_b2_waves"],
        getattr(engine_args, "offload_backend", "auto"),
    )
    setattr(engine_args, "_ascend_moe_offload_autoconfig_plan", autoconfig_plan)
    return True
