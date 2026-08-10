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

from dataclasses import dataclass

import torch

from vllm_moe_offload_ascend.moe_offload.expert_key import ExpertKey


@dataclass(frozen=True)
class ExpertWeightBundle:
    layer_id: int
    expert_id: int
    w13: torch.Tensor
    w2: torch.Tensor
    w13_scale: torch.Tensor | None = None
    w2_scale: torch.Tensor | None = None

    @property
    def key(self) -> ExpertKey:
        return ExpertKey(self.layer_id, self.expert_id)


@dataclass(frozen=True)
class HostExpertLayerBuffer:
    layer_id: int
    w13: torch.Tensor
    w2: torch.Tensor

    @property
    def num_experts(self) -> int:
        return int(self.w13.shape[0])

    @property
    def total_bytes(self) -> int:
        return _tensor_nbytes(self.w13) + _tensor_nbytes(self.w2)

    def bundle(self, expert_id: int) -> ExpertWeightBundle:
        normalized_expert_id = int(expert_id)
        return ExpertWeightBundle(
            layer_id=int(self.layer_id),
            expert_id=normalized_expert_id,
            w13=self.w13[normalized_expert_id],
            w2=self.w2[normalized_expert_id],
        )


@dataclass(frozen=True)
class HostExpertLayerSignature:
    layer_id: int
    num_experts: int
    w13_shape: tuple[int, ...]
    w13_dtype: torch.dtype
    w13_stride: tuple[int, ...]
    w2_shape: tuple[int, ...]
    w2_dtype: torch.dtype
    w2_stride: tuple[int, ...]


@dataclass(frozen=True)
class HostStoreCompletenessReport:
    complete: bool
    layers_checked: tuple[int, ...]
    blockers: tuple[str, ...]


@dataclass(frozen=True)
class HostStoreRegisterReport:
    layer_id: int
    num_experts: int
    pin_memory_requested: bool
    pinned_tensors: int
    pin_failures: tuple[str, ...]

    @property
    def pin_memory_enabled(self) -> bool:
        return self.pin_memory_requested and not self.pin_failures

    def to_jsonable(self) -> dict[str, object]:
        return {
            "layer_id": int(self.layer_id),
            "num_experts": int(self.num_experts),
            "pin_memory_requested": bool(self.pin_memory_requested),
            "pin_memory_enabled": bool(self.pin_memory_enabled),
            "pinned_tensors": int(self.pinned_tensors),
            "pin_failures": list(self.pin_failures),
        }


class HostExpertStore:
    def __init__(self) -> None:
        self._weights: dict[ExpertKey, ExpertWeightBundle] = {}
        self._weights_by_layer: dict[int, tuple[ExpertWeightBundle, ...]] = {}
        self._layer_buffers: dict[int, HostExpertLayerBuffer] = {}
        self._layer_signatures: dict[int, HostExpertLayerSignature] = {}

    def register_layer(
        self,
        layer: torch.nn.Module,
        *,
        pin_memory: bool = False,
        clone_tensors: bool = True,
    ) -> HostStoreRegisterReport:
        layer_id = int(getattr(layer, "layer_id", -1))
        w13_weight = getattr(layer, "w13_weight")
        w2_weight = getattr(layer, "w2_weight")
        if w13_weight.shape[0] != w2_weight.shape[0]:
            raise ValueError("w13_weight and w2_weight must have the same number of experts")

        num_experts = int(w13_weight.shape[0])
        if num_experts <= 0:
            raise ValueError("host expert store requires at least one expert")
        self._weights = {key: bundle for key, bundle in self._weights.items() if key.layer_id != layer_id}
        self._weights_by_layer.pop(layer_id, None)
        self._layer_buffers.pop(layer_id, None)
        w13_host, w13_pinned, w13_error = _to_host_layer_tensor(
            w13_weight,
            clone=clone_tensors,
            pin_memory=pin_memory,
        )
        w2_host, w2_pinned, w2_error = _to_host_layer_tensor(
            w2_weight,
            clone=clone_tensors,
            pin_memory=pin_memory,
        )
        layer_buffer = HostExpertLayerBuffer(
            layer_id=layer_id,
            w13=w13_host,
            w2=w2_host,
        )
        self._layer_buffers[layer_id] = layer_buffer
        self._layer_signatures[layer_id] = HostExpertLayerSignature(
            layer_id=layer_id,
            num_experts=num_experts,
            w13_shape=tuple(int(dim) for dim in w13_host.shape[1:]),
            w13_dtype=w13_host.dtype,
            w13_stride=_expert_stride(w13_host),
            w2_shape=tuple(int(dim) for dim in w2_host.shape[1:]),
            w2_dtype=w2_host.dtype,
            w2_stride=_expert_stride(w2_host),
        )
        pinned_tensors = int(w13_pinned) + int(w2_pinned)
        pin_failures: list[str] = []
        if w13_error is not None:
            pin_failures.append(f"w13_layer:{w13_error}")
        if w2_error is not None:
            pin_failures.append(f"w2_layer:{w2_error}")
        layer_bundles: list[ExpertWeightBundle] = []
        for expert_id in range(num_experts):
            bundle = layer_buffer.bundle(expert_id)
            self._weights[bundle.key] = bundle
            layer_bundles.append(bundle)
        self._weights_by_layer[layer_id] = tuple(layer_bundles)
        return HostStoreRegisterReport(
            layer_id=layer_id,
            num_experts=num_experts,
            pin_memory_requested=bool(pin_memory),
            pinned_tensors=pinned_tensors,
            pin_failures=tuple(pin_failures[:8]),
        )

    def get(self, layer_id: int, expert_id: int) -> ExpertWeightBundle:
        normalized_layer_id = int(layer_id)
        normalized_expert_id = int(expert_id)
        layer_bundles = self._weights_by_layer.get(normalized_layer_id)
        if layer_bundles is not None:
            return layer_bundles[normalized_expert_id]
        return self._weights[ExpertKey(normalized_layer_id, normalized_expert_id)]

    def get_layer_buffer(self, layer_id: int) -> HostExpertLayerBuffer:
        normalized_layer_id = int(layer_id)
        try:
            return self._layer_buffers[normalized_layer_id]
        except KeyError as exc:
            raise KeyError(
                f"layer {normalized_layer_id} is not registered in the host expert store"
            ) from exc

    def validate_complete_layers(self, expected_layer_ids: tuple[int, ...]) -> HostStoreCompletenessReport:
        normalized_layer_ids = tuple(int(layer_id) for layer_id in expected_layer_ids)
        blockers: list[str] = []

        missing_layers = [
            layer_id for layer_id in normalized_layer_ids if layer_id not in self._layer_signatures
        ]
        if missing_layers:
            blockers.append(f"host_store_missing_layers:{missing_layers}")

        for layer_id in normalized_layer_ids:
            signature = self._layer_signatures.get(layer_id)
            if signature is None:
                continue

            missing_expert_ids: list[int] = []
            for expert_id in range(signature.num_experts):
                key = ExpertKey(layer_id, expert_id)
                bundle = self._weights.get(key)
                if bundle is None:
                    missing_expert_ids.append(expert_id)
                    continue

                blockers.extend(_layout_mismatch_blockers(signature, bundle))

            if missing_expert_ids:
                blockers.append(f"host_store_missing_experts:layer={layer_id},experts={missing_expert_ids}")

        return HostStoreCompletenessReport(
            complete=not blockers,
            layers_checked=normalized_layer_ids,
            blockers=tuple(blockers),
        )

    @property
    def total_bytes(self) -> int:
        total = 0
        for layer_buffer in self._layer_buffers.values():
            total += layer_buffer.total_bytes
        return total

    def __len__(self) -> int:
        return len(self._weights)


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def _maybe_pin_tensor(tensor: torch.Tensor) -> tuple[torch.Tensor, bool, str | None]:
    try:
        if hasattr(tensor, "is_pinned") and tensor.is_pinned():
            return tensor, True, None
    except Exception:
        pass
    try:
        pinned = tensor.pin_memory()
    except Exception as exc:
        return tensor, False, f"{type(exc).__name__}:{str(exc)[:120]}"
    is_pinned = bool(pinned.is_pinned()) if hasattr(pinned, "is_pinned") else True
    return pinned, is_pinned, None if is_pinned else "pin_memory_returned_unpinned"


def _to_host_layer_tensor(
    tensor: torch.Tensor,
    *,
    clone: bool,
    pin_memory: bool,
) -> tuple[torch.Tensor, bool, str | None]:
    host = tensor.detach()
    if host.device.type != "cpu":
        host = host.cpu()
        clone = True
    if clone:
        host = host.clone()
    if not host.is_contiguous():
        host = host.contiguous()
    if not pin_memory:
        return host, False, None
    pinned, pinned_ok, pin_error = _maybe_pin_tensor(host)
    return pinned, pinned_ok, pin_error


def _expert_stride(tensor: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(value) for value in tensor[0].stride())


def _layout_mismatch_blockers(
    signature: HostExpertLayerSignature,
    bundle: ExpertWeightBundle,
) -> tuple[str, ...]:
    blockers: list[str] = []
    if tuple(int(dim) for dim in bundle.w13.shape) != signature.w13_shape:
        blockers.append(f"host_store_layout_mismatch:layer={signature.layer_id},expert={bundle.expert_id},w13")
    if bundle.w13.dtype != signature.w13_dtype:
        blockers.append(f"host_store_layout_mismatch:layer={signature.layer_id},expert={bundle.expert_id},w13_dtype")
    if tuple(int(value) for value in bundle.w13.stride()) != signature.w13_stride:
        blockers.append(f"host_store_layout_mismatch:layer={signature.layer_id},expert={bundle.expert_id},w13_stride")
    if bundle.w13.device.type != "cpu":
        blockers.append(
            f"host_store_device_mismatch:layer={signature.layer_id},expert={bundle.expert_id},w13={bundle.w13.device.type}"
        )
    if tuple(int(dim) for dim in bundle.w2.shape) != signature.w2_shape:
        blockers.append(f"host_store_layout_mismatch:layer={signature.layer_id},expert={bundle.expert_id},w2")
    if bundle.w2.dtype != signature.w2_dtype:
        blockers.append(f"host_store_layout_mismatch:layer={signature.layer_id},expert={bundle.expert_id},w2_dtype")
    if tuple(int(value) for value in bundle.w2.stride()) != signature.w2_stride:
        blockers.append(f"host_store_layout_mismatch:layer={signature.layer_id},expert={bundle.expert_id},w2_stride")
    if bundle.w2.device.type != "cpu":
        blockers.append(
            f"host_store_device_mismatch:layer={signature.layer_id},expert={bundle.expert_id},w2={bundle.w2.device.type}"
        )
    return tuple(blockers)
