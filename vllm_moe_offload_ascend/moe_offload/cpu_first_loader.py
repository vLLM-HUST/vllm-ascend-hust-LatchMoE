#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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

from typing import Any

import torch


CPU_FIRST_LOAD_ENV = "VLLM_ASCEND_MOE_OFFLOAD_CPU_FIRST_LOAD"
CPU_FIRST_MARKER = "_ascend_moe_cpu_first_load"
CPU_FIRST_PROCESSED_MARKER = "_ascend_moe_cpu_first_processed"


def should_cpu_first_load_layer(layer: torch.nn.Module, runtime: Any) -> bool:
    config = getattr(runtime, "config", None)
    if not bool(getattr(config, "cpu_first_load", False)):
        return False
    layer_id = _layer_id(layer)
    if layer_id < 0:
        return False
    return bool(runtime.should_use_fixed_slot_plan_for_layer(layer_id))


def maybe_create_unquantized_cpu_first_weights(
    method: Any,
    layer: torch.nn.Module,
    *,
    runtime: Any,
    num_experts: int,
    hidden_size: int,
    intermediate_size_per_partition: int,
    params_dtype: torch.dtype,
    extra_weight_attrs: dict[str, Any],
) -> bool:
    """Create CPU-resident unquantized expert parameters for offloaded layers."""

    if not should_cpu_first_load_layer(layer, runtime):
        return False
    moe = getattr(method, "moe", None)
    if bool(getattr(moe, "has_bias", False)):
        return False

    from vllm.model_executor.utils import set_weight_attrs

    pin_memory = bool(getattr(getattr(runtime, "config", None), "should_pin_host_memory", False))
    w13_up_dim = (
        2 * int(intermediate_size_per_partition)
        if bool(getattr(moe, "is_act_and_mul", True))
        else int(intermediate_size_per_partition)
    )
    w13_weight = torch.nn.Parameter(
        _empty_cpu_tensor(
            (int(num_experts), w13_up_dim, int(hidden_size)),
            dtype=params_dtype,
            pin_memory=pin_memory,
        ),
        requires_grad=False,
    )
    layer.register_parameter("w13_weight", w13_weight)
    set_weight_attrs(w13_weight, extra_weight_attrs)

    w2_weight = torch.nn.Parameter(
        _empty_cpu_tensor(
            (int(num_experts), int(hidden_size), int(intermediate_size_per_partition)),
            dtype=params_dtype,
            pin_memory=pin_memory,
        ),
        requires_grad=False,
    )
    layer.register_parameter("w2_weight", w2_weight)
    set_weight_attrs(w2_weight, extra_weight_attrs)

    setattr(layer, CPU_FIRST_MARKER, True)
    return True


def maybe_process_unquantized_cpu_first_weights(
    method: Any,
    layer: torch.nn.Module,
    *,
    runtime: Any,
) -> bool:
    """Format CPU-first weights and register the fixed-slot host store.

    If an NPU is available, each full expert tensor is temporarily moved to NPU
    for Ascend format conversion and moved back to CPU immediately. This bounds
    startup HBM pressure to one MoE layer instead of the whole model.
    """

    if not bool(getattr(layer, CPU_FIRST_MARKER, False)):
        return False
    if bool(getattr(layer, CPU_FIRST_PROCESSED_MARKER, False)):
        return True
    if not should_cpu_first_load_layer(layer, runtime):
        return False

    w13_weight = getattr(layer, "w13_weight")
    w2_weight = getattr(layer, "w2_weight")
    if not _can_process_cpu_first_weight_pair(w13_weight, w2_weight):
        return False

    slot_device = _current_npu_device_or_cpu()
    w13_data = _format_ascend_unquantized_weight(
        method,
        w13_weight.data,
        slot_device=slot_device,
    )
    w2_data = _format_ascend_unquantized_weight(
        method,
        w2_weight.data,
        slot_device=slot_device,
    )

    if bool(getattr(getattr(runtime, "config", None), "should_pin_host_memory", False)):
        w13_data = _pin_if_possible(w13_data)
        w2_data = _pin_if_possible(w2_data)

    layer.w13_weight = torch.nn.Parameter(w13_data, requires_grad=False)
    layer.w2_weight = torch.nn.Parameter(w2_data, requires_grad=False)
    setattr(layer, CPU_FIRST_PROCESSED_MARKER, True)

    maybe_register_unquantized_fixed_slot_weights(
        layer,
        runtime=runtime,
        slot_device=slot_device,
    )
    return True


def maybe_register_unquantized_fixed_slot_weights(
    layer: torch.nn.Module,
    *,
    runtime: Any,
    slot_device: torch.device | None = None,
) -> bool:
    """Register formatted unquantized weights for fixed-slot execution.

    CPU-first loading performs formatting itself, while the normal loading path
    delegates formatting to vLLM-Ascend. Both paths must converge here before
    the first profile forward can execute the stage seam.
    """

    layer_id = ensure_moe_layer_id(layer)
    if layer_id < 0 or not runtime.should_use_fixed_slot_plan_for_layer(layer_id):
        return False
    if not hasattr(layer, "w13_weight") or not hasattr(layer, "w2_weight"):
        return False

    if not runtime.is_layer_registered(layer_id):
        if slot_device is None:
            slot_device = getattr(layer, "w13_weight").device
        runtime.register_layer_for_fixed_slots(layer, slot_device=slot_device)
    if bool(
        getattr(
            getattr(runtime, "config", None),
            "release_original_expert_weights",
            False,
        )
    ):
        runtime.release_original_expert_weights_if_ready(layer)
    return True


def is_cpu_first_layer(layer: torch.nn.Module) -> bool:
    return bool(getattr(layer, CPU_FIRST_MARKER, False))


def _can_process_cpu_first_weight_pair(
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
) -> bool:
    """Accept CPU weights and vLLM's temporary NPU loading-context weights.

    ``process_weights_after_loading`` moves CPU parameters to the target
    accelerator before invoking the quant method. The formatted copies still
    return to CPU before fixed-slot host-store registration, so accepting NPU
    here does not turn CPU-first loading into resident-weight loading.
    """

    w13_device = w13_weight.device.type
    w2_device = w2_weight.device.type
    return w13_device == w2_device and w13_device in {"cpu", "npu"}



def _format_ascend_unquantized_weight(
    method: Any,
    weight: torch.Tensor,
    *,
    slot_device: torch.device,
) -> torch.Tensor:
    work = weight
    if slot_device.type != "cpu":
        work = work.to(slot_device)
    work = method._maybe_pad_weight(work).transpose(1, 2).contiguous()
    if slot_device.type != "cpu":
        work = _maybe_ascend_format_cast(work)
    work = work.to("cpu")
    _empty_npu_cache_if_available()
    return work


def _maybe_ascend_format_cast(weight: torch.Tensor) -> torch.Tensor:
    try:
        import torch_npu
        import vllm_ascend.envs as envs_ascend
        from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ, maybe_trans_nz
    except Exception:
        return weight
    if bool(getattr(envs_ascend, "VLLM_ASCEND_ENABLE_FUSED_MC2", False)):
        return torch_npu.npu_format_cast(weight, ACL_FORMAT_FRACTAL_NZ)
    return maybe_trans_nz(weight)


def _empty_cpu_tensor(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    pin_memory: bool,
) -> torch.Tensor:
    if pin_memory:
        try:
            return torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
        except Exception:
            pass
    return torch.empty(shape, dtype=dtype, device="cpu")


def _pin_if_possible(tensor: torch.Tensor) -> torch.Tensor:
    try:
        if hasattr(tensor, "is_pinned") and tensor.is_pinned():
            return tensor
        pinned = tensor.pin_memory()
        if not hasattr(pinned, "is_pinned") or pinned.is_pinned():
            return pinned
    except Exception:
        pass
    return tensor


def _current_npu_device_or_cpu() -> torch.device:
    npu = getattr(torch, "npu", None)
    if npu is None:
        return torch.device("cpu")
    try:
        if hasattr(npu, "is_available") and not npu.is_available():
            return torch.device("cpu")
        return torch.device("npu", npu.current_device())
    except Exception:
        return torch.device("cpu")


def _empty_npu_cache_if_available() -> None:
    npu = getattr(torch, "npu", None)
    if npu is None or not hasattr(npu, "empty_cache"):
        return
    try:
        npu.empty_cache()
    except Exception:
        pass


def _layer_id(layer: torch.nn.Module) -> int:
    try:
        return int(getattr(layer, "layer_id", -1))
    except Exception:
        return -1


def ensure_moe_layer_id(layer: torch.nn.Module) -> int:
    """Restore the layer index for vLLM RoutedExperts containers.

    Newer vLLM versions keep ``layer_name`` on RoutedExperts but expose the
    ``layer_id`` property only on the outer MoERunner. The Ascend offload hook
    receives RoutedExperts during weight creation and post-processing, so
    cache the index there before CPU-first and fixed-slot decisions run.
    """

    layer_id = _layer_id(layer)
    if layer_id >= 0:
        return layer_id

    layer_name = getattr(layer, "layer_name", "")
    if layer_name:
        try:
            from vllm.model_executor.models.utils import extract_layer_index

            layer_id = int(extract_layer_index(str(layer_name)))
        except (AttributeError, TypeError, ValueError):
            layer_id = -1
        if layer_id >= 0:
            setattr(layer, "layer_id", layer_id)
            return layer_id

    # Weight creation runs before Qwen's RoutedExperts container receives its
    # path-derived layer_name. FusedMoE already assigns this stable index in its
    # constructor, so use it to enable CPU-first allocation at that earlier
    # lifecycle point.
    try:
        layer_id = int(getattr(layer, "moe_layer_id", -1))
    except (TypeError, ValueError):
        return -1
    if layer_id < 0:
        return -1
    setattr(layer, "layer_id", layer_id)
    return layer_id
