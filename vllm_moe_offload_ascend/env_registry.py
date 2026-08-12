"""Environment-variable contract installed by the LatchMoE plugin."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any


EnvLoader = Callable[[], Any]


def _bool(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _optional_float(name: str) -> float | None:
    value = os.getenv(name)
    return None if value is None or not value.strip() else float(value)


def _optional_int(name: str) -> int | None:
    value = os.getenv(name)
    return None if value is None or not value.strip() else int(value)


def _optional_bool(name: str) -> bool | None:
    return None if name not in os.environ else _bool(name)


def _string(name: str, default: str = "") -> EnvLoader:
    return lambda: os.getenv(name, default)


def _boolean(name: str, default: str = "0") -> EnvLoader:
    return lambda: _bool(name, default)


def _integer(name: str, default: str = "0") -> EnvLoader:
    return lambda: int(os.getenv(name, default))


def _floating(name: str, default: str = "0") -> EnvLoader:
    return lambda: float(os.getenv(name, default))


# Keep this as the single inventory for every VLLM_* setting owned by LatchMoE.
# Registration in vllm.envs is required for validation, Ray propagation, and
# compile-cache factors. Registration in vllm_ascend.envs preserves the typed
# attribute API used by the runtime.
ENVIRONMENT_VARIABLES: dict[str, EnvLoader] = {
    "VLLM_ASCEND_MOE_B2_PROFILE_DETAILS": _boolean(
        "VLLM_ASCEND_MOE_B2_PROFILE_DETAILS"
    ),
    "VLLM_ASCEND_MOE_COMPUTE_BUCKET_PLAN_PATH": _string(
        "VLLM_ASCEND_MOE_COMPUTE_BUCKET_PLAN_PATH"
    ),
    "VLLM_ASCEND_MOE_DECODE_PROFILE_SAMPLE_RATE": _integer(
        "VLLM_ASCEND_MOE_DECODE_PROFILE_SAMPLE_RATE", "1"
    ),
    "VLLM_ASCEND_MOE_GMM_BUCKET_PLAN_PATH": _string(
        "VLLM_ASCEND_MOE_GMM_BUCKET_PLAN_PATH"
    ),
    "VLLM_ASCEND_MOE_GMM_PROFILE_PATH": _string(
        "VLLM_ASCEND_MOE_GMM_PROFILE_PATH"
    ),
    "VLLM_ASCEND_MOE_GMM_TRACE_PATH": _string(
        "VLLM_ASCEND_MOE_GMM_TRACE_PATH"
    ),
    "VLLM_ASCEND_MOE_OFFLOAD_ACTIVATION_RESERVE_GB": lambda: _optional_float(
        "VLLM_ASCEND_MOE_OFFLOAD_ACTIVATION_RESERVE_GB"
    ),
    "VLLM_ASCEND_MOE_OFFLOAD_ASYNC_LOAD": _boolean(
        "VLLM_ASCEND_MOE_OFFLOAD_ASYNC_LOAD"
    ),
    "VLLM_ASCEND_MOE_OFFLOAD_B2_AVOID_MIXED_D2D": _boolean(
        "VLLM_ASCEND_MOE_OFFLOAD_B2_AVOID_MIXED_D2D", "1"
    ),
    "VLLM_ASCEND_MOE_OFFLOAD_B2_DIRECT_SCATTER": _boolean(
        "VLLM_ASCEND_MOE_OFFLOAD_B2_DIRECT_SCATTER", "1"
    ),
    "VLLM_ASCEND_MOE_OFFLOAD_B2_OVERFLOW_MODE": _string(
        "VLLM_ASCEND_MOE_OFFLOAD_B2_OVERFLOW_MODE", "full_layer"
    ),
    "VLLM_ASCEND_MOE_OFFLOAD_B2_REFERENCE_FULL_LAYER": _boolean(
        "VLLM_ASCEND_MOE_OFFLOAD_B2_REFERENCE_FULL_LAYER"
    ),
    "VLLM_ASCEND_MOE_OFFLOAD_B2_REFERENCE_FULL_TOKENS": _boolean(
        "VLLM_ASCEND_MOE_OFFLOAD_B2_REFERENCE_FULL_TOKENS"
    ),
    "VLLM_ASCEND_MOE_OFFLOAD_B2_WAVE_PREFILL": _boolean(
        "VLLM_ASCEND_MOE_OFFLOAD_B2_WAVE_PREFILL"
    ),
    "VLLM_ASCEND_MOE_OFFLOAD_COMPARE_LAYER_BOUNDARY": _boolean(
        "VLLM_ASCEND_MOE_OFFLOAD_COMPARE_LAYER_BOUNDARY"
    ),
    "VLLM_ASCEND_MOE_OFFLOAD_COMPARE_ORIGINAL": _boolean(
        "VLLM_ASCEND_MOE_OFFLOAD_COMPARE_ORIGINAL"
    ),
    "VLLM_ASCEND_MOE_OFFLOAD_COMPARE_ROUTER": _boolean(
        "VLLM_ASCEND_MOE_OFFLOAD_COMPARE_ROUTER"
    ),
    "VLLM_ASCEND_MOE_OFFLOAD_COMPAT_ONLY": _boolean(
        "VLLM_ASCEND_MOE_OFFLOAD_COMPAT_ONLY"
    ),
    "VLLM_ASCEND_MOE_OFFLOAD_CPU_FIRST_LOAD": _boolean(
        "VLLM_ASCEND_MOE_OFFLOAD_CPU_FIRST_LOAD"
    ),
    "VLLM_ASCEND_MOE_OFFLOAD_DEVICE_PAIR_PLANNING": _boolean(
        "VLLM_ASCEND_MOE_OFFLOAD_DEVICE_PAIR_PLANNING", "1"
    ),
    "VLLM_ASCEND_MOE_OFFLOAD_ENABLED": _boolean(
        "VLLM_ASCEND_MOE_OFFLOAD_ENABLED"
    ),
    "VLLM_ASCEND_MOE_OFFLOAD_FANOUT_THRESHOLD": _integer(
        "VLLM_ASCEND_MOE_OFFLOAD_FANOUT_THRESHOLD"
    ),
    "VLLM_ASCEND_MOE_OFFLOAD_GB": _floating(
        "VLLM_ASCEND_MOE_OFFLOAD_GB"
    ),
    "VLLM_ASCEND_MOE_OFFLOAD_GRAPH_COMPATIBLE": _boolean(
        "VLLM_ASCEND_MOE_OFFLOAD_GRAPH_COMPATIBLE"
    ),
    "VLLM_ASCEND_MOE_OFFLOAD_KV_RESERVE_CTX": lambda: _optional_int(
        "VLLM_ASCEND_MOE_OFFLOAD_KV_RESERVE_CTX"
    ),
    "VLLM_ASCEND_MOE_OFFLOAD_KV_RESERVE_SEQS": lambda: _optional_int(
        "VLLM_ASCEND_MOE_OFFLOAD_KV_RESERVE_SEQS"
    ),
    "VLLM_ASCEND_MOE_OFFLOAD_LAYERED_RUNTIME": _boolean(
        "VLLM_ASCEND_MOE_OFFLOAD_LAYERED_RUNTIME", "1"
    ),
    "VLLM_ASCEND_MOE_OFFLOAD_MAX_NUM_SEQS_HINT": _integer(
        "VLLM_ASCEND_MOE_OFFLOAD_MAX_NUM_SEQS_HINT"
    ),
    "VLLM_ASCEND_MOE_OFFLOAD_MAX_PHASES": _integer(
        "VLLM_ASCEND_MOE_OFFLOAD_MAX_PHASES", "2"
    ),
    "VLLM_ASCEND_MOE_OFFLOAD_MIN_NET_SAVING_GB": lambda: _optional_float(
        "VLLM_ASCEND_MOE_OFFLOAD_MIN_NET_SAVING_GB"
    ),
    "VLLM_ASCEND_MOE_OFFLOAD_MIN_NET_SAVING_RATIO": lambda: _optional_float(
        "VLLM_ASCEND_MOE_OFFLOAD_MIN_NET_SAVING_RATIO"
    ),
    "VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS": _integer(
        "VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS"
    ),
    "VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS_AUTOCONFIG_BOOTSTRAP": _boolean(
        "VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS_AUTOCONFIG_BOOTSTRAP"
    ),
    "VLLM_ASCEND_MOE_OFFLOAD_ONDEMAND_STAGE": _boolean(
        "VLLM_ASCEND_MOE_OFFLOAD_ONDEMAND_STAGE"
    ),
    "VLLM_ASCEND_MOE_OFFLOAD_PHASE_SPLIT": _boolean(
        "VLLM_ASCEND_MOE_OFFLOAD_PHASE_SPLIT"
    ),
    "VLLM_ASCEND_MOE_OFFLOAD_PIN_HOST_MEMORY": lambda: _optional_bool(
        "VLLM_ASCEND_MOE_OFFLOAD_PIN_HOST_MEMORY"
    ),
    "VLLM_ASCEND_MOE_OFFLOAD_POLICY": _string(
        "VLLM_ASCEND_MOE_OFFLOAD_POLICY", "deadline"
    ),
    "VLLM_ASCEND_MOE_OFFLOAD_PREFILL_BUFFER_COUNT": _integer(
        "VLLM_ASCEND_MOE_OFFLOAD_PREFILL_BUFFER_COUNT", "2"
    ),
    "VLLM_ASCEND_MOE_OFFLOAD_PREFILL_PREFETCH_DEPTH": _integer(
        "VLLM_ASCEND_MOE_OFFLOAD_PREFILL_PREFETCH_DEPTH", "1"
    ),
    "VLLM_ASCEND_MOE_OFFLOAD_PREFILL_RESIDENCY_PROFILE": _string(
        "VLLM_ASCEND_MOE_OFFLOAD_PREFILL_RESIDENCY_PROFILE"
    ),
    "VLLM_ASCEND_MOE_OFFLOAD_PROFILE_PATH": _string(
        "VLLM_ASCEND_MOE_OFFLOAD_PROFILE_PATH"
    ),
    "VLLM_ASCEND_MOE_OFFLOAD_RELEASE_ORIGINAL_EXPERT_WEIGHTS": _boolean(
        "VLLM_ASCEND_MOE_OFFLOAD_RELEASE_ORIGINAL_EXPERT_WEIGHTS"
    ),
    "VLLM_ASCEND_MOE_OFFLOAD_RESIDENT_LAYER_IDS": _string(
        "VLLM_ASCEND_MOE_OFFLOAD_RESIDENT_LAYER_IDS"
    ),
    "VLLM_ASCEND_MOE_OFFLOAD_ROUTE_STATS_CACHE": _boolean(
        "VLLM_ASCEND_MOE_OFFLOAD_ROUTE_STATS_CACHE"
    ),
    "VLLM_ASCEND_MOE_OFFLOAD_SEW_DATAPLANE": _boolean(
        "VLLM_ASCEND_MOE_OFFLOAD_SEW_DATAPLANE"
    ),
    "VLLM_ASCEND_MOE_OFFLOAD_SLOT_HBM_BUDGET_GB": lambda: _optional_float(
        "VLLM_ASCEND_MOE_OFFLOAD_SLOT_HBM_BUDGET_GB"
    ),
    "VLLM_ASCEND_MOE_OFFLOAD_SLOT_HBM_FRACTION": lambda: _optional_float(
        "VLLM_ASCEND_MOE_OFFLOAD_SLOT_HBM_FRACTION"
    ),
    "VLLM_ASCEND_MOE_OFFLOAD_STAGE_SEAM": _boolean(
        "VLLM_ASCEND_MOE_OFFLOAD_STAGE_SEAM"
    ),
    "VLLM_ASCEND_MOE_OFFLOAD_SYSTEM_RESERVE_GB": lambda: _optional_float(
        "VLLM_ASCEND_MOE_OFFLOAD_SYSTEM_RESERVE_GB"
    ),
    "VLLM_ASCEND_MOE_OFFLOAD_TRACE_MAX_RECORDS": _integer(
        "VLLM_ASCEND_MOE_OFFLOAD_TRACE_MAX_RECORDS", "4096"
    ),
    "VLLM_ASCEND_MOE_OFFLOAD_TRACE_ONLY": _boolean(
        "VLLM_ASCEND_MOE_OFFLOAD_TRACE_ONLY"
    ),
    "VLLM_ASCEND_MOE_OFFLOAD_TRACE_PATH": _string(
        "VLLM_ASCEND_MOE_OFFLOAD_TRACE_PATH"
    ),
    "VLLM_ASCEND_MOE_OFFLOAD_TRANSFER_AWARE_SCHEDULE": _boolean(
        "VLLM_ASCEND_MOE_OFFLOAD_TRANSFER_AWARE_SCHEDULE", "1"
    ),
    "VLLM_ASCEND_MOE_PIPELINE_PROFILING": _boolean(
        "VLLM_ASCEND_MOE_PIPELINE_PROFILING"
    ),
    "VLLM_ASCEND_MOE_PROFILE_EXPERT_LISTS": _boolean(
        "VLLM_ASCEND_MOE_PROFILE_EXPERT_LISTS", "1"
    ),
    "VLLM_ASCEND_MOE_PROFILE_FLUSH_EVERY": _integer(
        "VLLM_ASCEND_MOE_PROFILE_FLUSH_EVERY", "1"
    ),
}


def _register(module: Any, registry_name: str) -> int:
    registry = getattr(module, registry_name, None)
    if not isinstance(registry, dict):
        raise RuntimeError(
            f"{module.__name__}.{registry_name} is unavailable; "
            "the installed host version is incompatible with LatchMoE"
        )
    added = 0
    for name, loader in ENVIRONMENT_VARIABLES.items():
        if name not in registry:
            registry[name] = loader
            added += 1
    getattr_fn = getattr(module, "__getattr__", None)
    cache_clear = getattr(getattr_fn, "cache_clear", None)
    if callable(cache_clear):
        cache_clear()
    return added


def register_environment_variables() -> tuple[int, int]:
    """Register every plugin variable in both host environment registries."""

    import vllm.envs as vllm_envs
    import vllm_ascend.envs as ascend_envs

    return (
        _register(vllm_envs, "environment_variables"),
        _register(ascend_envs, "env_variables"),
    )
