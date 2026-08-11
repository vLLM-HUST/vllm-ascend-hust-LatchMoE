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

from vllm_moe_offload_ascend.moe_offload.host_store import ExpertWeightBundle
from vllm_moe_offload_ascend.moe_offload.layout import LayoutValidator
from vllm_moe_offload_ascend.moe_offload.slot_bank import (
    ExpertSlot,
    SlotLease,
    SlotState,
)


@dataclass
class TransferReadyEvent:
    """Transfer-stream event plus the slots it makes consumable."""

    event: object | None
    leases: tuple[tuple[ExpertSlot, SlotLease], ...]
    completed: bool = False
    consumer_wait_installed: bool = False

    def install_consumer_dependency(self, consumer_stream) -> None:
        """Order the copy before one consumer stream, then publish READY."""

        if self.completed:
            return
        self._validate_loading_leases()
        if self.event is not None:
            consumer_stream.wait_event(self.event)
        self.consumer_wait_installed = True
        self.mark_ready()

    def mark_ready(self) -> None:
        if self.completed:
            return
        self._validate_loading_leases()
        for slot, _ in self.leases:
            slot.state = SlotState.READY
        self.completed = True

    def _validate_loading_leases(self) -> None:
        for slot, lease in self.leases:
            if not slot.matches_lease(lease):
                raise RuntimeError(
                    "stale H2D completion for slot "
                    f"{slot.slot_id}: expected {lease.expert_key} v{lease.version}, "
                    f"found {slot.expert_key} v{slot.version}"
                )
            if slot.state != SlotState.LOADING:
                raise RuntimeError(
                    f"H2D completion for slot {slot.slot_id} requires loading state, "
                    f"found {slot.state.value}"
                )


class TransferEngine:
    def __init__(self) -> None:
        self._h2d_stream = None
        self._pending_ready_events: list[TransferReadyEvent] = []

    def load_sync(self, bundle: ExpertWeightBundle, slot: ExpertSlot) -> None:
        LayoutValidator.validate_copy_compatible(bundle, slot.as_bundle())
        _copy_loads(((bundle, slot),), non_blocking=False)
        slot.state = SlotState.READY

    def load_many_sync(
        self,
        loads: list[tuple[ExpertWeightBundle, ExpertSlot]],
        *,
        validate_layout: bool = True,
    ) -> None:
        if not loads:
            return
        if validate_layout:
            for bundle, slot in loads:
                LayoutValidator.validate_copy_compatible(bundle, slot.as_bundle())
        for _, slot in loads:
            slot.state = SlotState.LOADING
        _copy_loads(tuple(loads), non_blocking=False)
        for _, slot in loads:
            slot.state = SlotState.READY

    def load_async(
        self,
        bundle: ExpertWeightBundle,
        slot: ExpertSlot,
        *,
        wait_event=None,
        record_event: bool = True,
    ):
        """Queue a host-to-device expert load on a dedicated transfer stream."""
        import torch

        LayoutValidator.validate_copy_compatible(bundle, slot.as_bundle())
        if _loads_target_cpu(((bundle, slot),)):
            _wait_for_event(wait_event)
            slot.state = SlotState.LOADING
            _copy_loads(((bundle, slot),), non_blocking=False)
            slot.state = SlotState.READY
            return None

        stream = self._get_h2d_stream()
        if wait_event is not None:
            stream.wait_event(getattr(wait_event, "event", wait_event))
        slot.state = SlotState.LOADING
        with torch.npu.stream(stream):
            _copy_loads(((bundle, slot),), non_blocking=True)
            ready_event = None
            if record_event:
                ready_event = torch.npu.Event()
                ready_event.record(stream)
        if ready_event is None:
            stream.synchronize()
            slot.state = SlotState.READY
            return None
        transfer_event = TransferReadyEvent(
            ready_event,
            ((slot, slot.lease()),),
        )
        self._pending_ready_events.append(transfer_event)
        return transfer_event

    def load_many_async(
        self,
        loads: list[tuple[ExpertWeightBundle, ExpertSlot]],
        *,
        wait_event=None,
        record_event: bool = True,
        validate_layout: bool = True,
    ):
        """Queue several expert loads inside one transfer-stream context."""
        import torch

        if not loads:
            return None
        if validate_layout:
            for bundle, slot in loads:
                LayoutValidator.validate_copy_compatible(bundle, slot.as_bundle())

        if _loads_target_cpu(tuple(loads)):
            _wait_for_event(wait_event)
            for _, slot in loads:
                slot.state = SlotState.LOADING
            _copy_loads(tuple(loads), non_blocking=False)
            for _, slot in loads:
                slot.state = SlotState.READY
            return None

        stream = self._get_h2d_stream()
        if wait_event is not None:
            stream.wait_event(getattr(wait_event, "event", wait_event))
        with torch.npu.stream(stream):
            for bundle, slot in loads:
                slot.state = SlotState.LOADING
            _copy_loads(tuple(loads), non_blocking=True)
            ready_event = None
            if record_event:
                ready_event = torch.npu.Event()
                ready_event.record(stream)
        slots = tuple(slot for _, slot in loads)
        if ready_event is None:
            stream.synchronize()
            for slot in slots:
                slot.state = SlotState.READY
            return None
        transfer_event = TransferReadyEvent(
            ready_event,
            tuple((slot, slot.lease()) for slot in slots),
        )
        self._pending_ready_events.append(transfer_event)
        return transfer_event

    def synchronize(self) -> None:
        """Wait for all queued H2D copies on the transfer stream to finish."""
        if self._h2d_stream is not None:
            self._h2d_stream.synchronize()
        pending, self._pending_ready_events = self._pending_ready_events, []
        for ready_event in pending:
            ready_event.mark_ready()

    def _get_h2d_stream(self):
        if self._h2d_stream is None:
            import torch

            self._h2d_stream = torch.npu.Stream()
        return self._h2d_stream


def _copy_loads(
    loads: tuple[tuple[ExpertWeightBundle, ExpertSlot], ...],
    *,
    non_blocking: bool,
) -> None:
    for run in _contiguous_load_runs(loads):
        if len(run) <= 1:
            bundle, slot = run[0]
            slot.w13.copy_(bundle.w13, non_blocking=non_blocking)
            slot.w2.copy_(bundle.w2, non_blocking=non_blocking)
            continue

        src_w13 = _try_batch_view(tuple(bundle.w13 for bundle, _ in run))
        src_w2 = _try_batch_view(tuple(bundle.w2 for bundle, _ in run))
        dst_w13 = _try_batch_view(tuple(slot.w13 for _, slot in run))
        dst_w2 = _try_batch_view(tuple(slot.w2 for _, slot in run))
        if src_w13 is None or src_w2 is None or dst_w13 is None or dst_w2 is None:
            for bundle, slot in run:
                slot.w13.copy_(bundle.w13, non_blocking=non_blocking)
                slot.w2.copy_(bundle.w2, non_blocking=non_blocking)
            continue

        dst_w13.copy_(src_w13, non_blocking=non_blocking)
        dst_w2.copy_(src_w2, non_blocking=non_blocking)


def _loads_target_cpu(
    loads: tuple[tuple[ExpertWeightBundle, ExpertSlot], ...],
) -> bool:
    return all(slot.w13.device.type == "cpu" and slot.w2.device.type == "cpu" for _, slot in loads)


def _wait_for_event(wait_event) -> None:
    if wait_event is None:
        return
    event = getattr(wait_event, "event", wait_event)
    synchronize = getattr(event, "synchronize", None)
    if callable(synchronize):
        synchronize()


def _contiguous_load_runs(
    loads: tuple[tuple[ExpertWeightBundle, ExpertSlot], ...],
) -> tuple[tuple[tuple[ExpertWeightBundle, ExpertSlot], ...], ...]:
    runs: list[list[tuple[ExpertWeightBundle, ExpertSlot]]] = []
    current: list[tuple[ExpertWeightBundle, ExpertSlot]] = []
    previous_bundle: ExpertWeightBundle | None = None
    previous_slot: ExpertSlot | None = None
    for bundle, slot in loads:
        can_extend = previous_bundle is not None and previous_slot is not None
        if can_extend:
            can_extend = _can_batch_after(
                previous_bundle,
                previous_slot,
                bundle,
                slot,
            )
        if not can_extend and current:
            runs.append(current)
            current = []
        current.append((bundle, slot))
        previous_bundle = bundle
        previous_slot = slot
    if current:
        runs.append(current)
    return tuple(tuple(run) for run in runs)


def _can_batch_after(
    previous_bundle: ExpertWeightBundle,
    previous_slot: ExpertSlot,
    bundle: ExpertWeightBundle,
    slot: ExpertSlot,
) -> bool:
    return (
        int(bundle.layer_id) == int(previous_bundle.layer_id)
        and int(bundle.expert_id) == int(previous_bundle.expert_id) + 1
        and int(slot.slot_id) == int(previous_slot.slot_id) + 1
        and _tensor_is_next_contiguous(previous_bundle.w13, bundle.w13)
        and _tensor_is_next_contiguous(previous_bundle.w2, bundle.w2)
        and _tensor_is_next_contiguous(previous_slot.w13, slot.w13)
        and _tensor_is_next_contiguous(previous_slot.w2, slot.w2)
    )


def _tensor_is_next_contiguous(previous, current) -> bool:
    if current.device != previous.device or current.dtype != previous.dtype:
        return False
    if tuple(current.shape) != tuple(previous.shape):
        return False
    if tuple(current.stride()) != tuple(previous.stride()):
        return False
    element_size = int(previous.element_size())
    delta_bytes = int(current.data_ptr()) - int(previous.data_ptr())
    return (
        delta_bytes > 0
        and delta_bytes % element_size == 0
        and int(delta_bytes // element_size) == int(previous.numel())
    )


def _try_batch_view(tensors: tuple[object, ...]):
    if not tensors:
        return None
    first = tensors[0]
    if len(tensors) == 1:
        return first.unsqueeze(0)
    element_size = int(first.element_size())
    first_ptr = int(first.data_ptr())
    previous_ptr = first_ptr
    delta_elems: int | None = None
    for index, tensor in enumerate(tensors):
        if tensor.device != first.device or tensor.dtype != first.dtype:
            return None
        if tuple(tensor.shape) != tuple(first.shape):
            return None
        if tuple(tensor.stride()) != tuple(first.stride()):
            return None
        ptr = int(tensor.data_ptr())
        if index == 0:
            continue
        delta_bytes = ptr - previous_ptr
        if delta_bytes <= 0 or delta_bytes % element_size != 0:
            return None
        current_delta_elems = delta_bytes // element_size
        if int(current_delta_elems) != int(first.numel()):
            return None
        if delta_elems is None:
            delta_elems = int(current_delta_elems)
        elif int(current_delta_elems) != int(delta_elems):
            return None
        previous_ptr = ptr
    if delta_elems is None:
        return first.unsqueeze(0)
    return first.as_strided(
        (len(tensors), *tuple(first.shape)),
        (int(delta_elems), *tuple(first.stride())),
    )
