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
from enum import Enum

import torch

from vllm_moe_offload_ascend.moe_offload.expert_key import ExpertKey
from vllm_moe_offload_ascend.moe_offload.host_store import ExpertWeightBundle


class SlotState(str, Enum):
    EMPTY = "empty"
    LOADING = "loading"
    READY = "ready"
    COMPUTING = "computing"


@dataclass(frozen=True)
class SlotLease:
    """Identity of one immutable slot ownership interval."""

    slot_id: int
    expert_key: ExpertKey
    version: int


@dataclass
class ExpertSlot:
    slot_id: int
    w13: torch.Tensor
    w2: torch.Tensor
    state: SlotState = SlotState.EMPTY
    expert_key: ExpertKey | None = None
    version: int = 0
    last_used_step: int = -1

    def as_bundle(self) -> ExpertWeightBundle:
        key = self.expert_key or ExpertKey(-1, -1)
        return ExpertWeightBundle(
            layer_id=key.layer_id,
            expert_id=key.expert_id,
            w13=self.w13,
            w2=self.w2,
        )

    def lease(self) -> SlotLease:
        if self.expert_key is None:
            raise RuntimeError(f"slot {self.slot_id} has no owner")
        return SlotLease(
            slot_id=int(self.slot_id),
            expert_key=self.expert_key,
            version=int(self.version),
        )

    def matches_lease(self, lease: SlotLease) -> bool:
        return (
            int(self.slot_id) == int(lease.slot_id)
            and self.expert_key == lease.expert_key
            and int(self.version) == int(lease.version)
        )


class ExpertSlotBank:
    def __init__(
        self,
        num_slots: int,
        w13_shape: tuple[int, ...],
        w2_shape: tuple[int, ...],
        *,
        dtype: torch.dtype,
        device: torch.device,
        pinned_logical_ids: tuple[int, ...] = (),
    ) -> None:
        if num_slots <= 0:
            raise ValueError("num_slots must be greater than 0")
        normalized_pinned = tuple(int(expert_id) for expert_id in pinned_logical_ids)
        if len(set(normalized_pinned)) != len(normalized_pinned):
            raise ValueError("pinned logical expert ids must be unique")
        if any(expert_id < 0 for expert_id in normalized_pinned):
            raise ValueError("pinned logical expert ids must be non-negative")

        # A fused/mix-placement layer exposes shared experts in the same backend
        # weight tensor as routed experts. Keep those rows in a stable suffix of
        # the physical tensor, while ``slots`` continues to describe *only* the
        # dynamic routed cache. This makes it impossible for LRU/victim logic to
        # evict a pinned shared expert by accident.
        self.num_dynamic_slots = int(num_slots)
        self.pinned_logical_ids = normalized_pinned
        self._pinned_slot_by_logical_id = {
            logical_id: int(num_slots) + offset
            for offset, logical_id in enumerate(normalized_pinned)
        }
        physical_expert_count = int(num_slots) + len(normalized_pinned)
        self.w13_slots = torch.empty(
            (physical_expert_count, *w13_shape), dtype=dtype, device=device
        )
        self.w2_slots = torch.empty(
            (physical_expert_count, *w2_shape), dtype=dtype, device=device
        )
        self.slots = [
            ExpertSlot(
                slot_id=slot_id,
                w13=self.w13_slots[slot_id],
                w2=self.w2_slots[slot_id],
            )
            for slot_id in range(num_slots)
        ]
        self._resident: dict[ExpertKey, int] = {}
        self._resident_by_expert_id: dict[int, int] = {}

    @property
    def physical_expert_count(self) -> int:
        return int(self.w13_slots.shape[0])

    @property
    def pinned_slot_ids(self) -> tuple[int, ...]:
        return tuple(
            int(self._pinned_slot_by_logical_id[logical_id])
            for logical_id in self.pinned_logical_ids
        )

    @property
    def dynamic_bytes(self) -> int:
        return _tensor_nbytes(self.w13_slots[: self.num_dynamic_slots]) + _tensor_nbytes(
            self.w2_slots[: self.num_dynamic_slots]
        )

    @property
    def pinned_bytes(self) -> int:
        return _tensor_nbytes(self.w13_slots[self.num_dynamic_slots :]) + _tensor_nbytes(
            self.w2_slots[self.num_dynamic_slots :]
        )

    def pinned_slot_for_logical_id(self, logical_id: int) -> int | None:
        return self._pinned_slot_by_logical_id.get(int(logical_id))

    def install_pinned_weights(
        self,
        w13: torch.Tensor,
        w2: torch.Tensor,
    ) -> None:
        """Copy always-resident shared rows into the immutable suffix."""

        pinned_count = len(self.pinned_logical_ids)
        if pinned_count == 0:
            if w13.numel() or w2.numel():
                raise ValueError("cannot install pinned weights into a routed-only bank")
            return
        expected_w13 = tuple(int(dim) for dim in self.w13_slots.shape[1:])
        expected_w2 = tuple(int(dim) for dim in self.w2_slots.shape[1:])
        if (
            int(w13.shape[0]) != pinned_count
            or tuple(int(dim) for dim in w13.shape[1:]) != expected_w13
            or int(w2.shape[0]) != pinned_count
            or tuple(int(dim) for dim in w2.shape[1:]) != expected_w2
        ):
            raise ValueError(
                "pinned shared weight layout does not match slot bank: "
                f"w13={tuple(w13.shape)} expected=({pinned_count}, {expected_w13}); "
                f"w2={tuple(w2.shape)} expected=({pinned_count}, {expected_w2})"
            )
        # Weight registration must not attach the permanent slot storage to the
        # checkpoint Parameter's autograd graph. Inference weights may still be
        # Parameters even though the runtime never differentiates them.
        with torch.no_grad():
            self.w13_slots[self.num_dynamic_slots :].copy_(w13, non_blocking=False)
            self.w2_slots[self.num_dynamic_slots :].copy_(w2, non_blocking=False)

    def copy_pinned_weights_from(self, source: "ExpertSlotBank") -> None:
        """Clone a compatible pinned lane into a temporary prefill bank."""

        if self.pinned_logical_ids != source.pinned_logical_ids:
            raise ValueError("cannot copy pinned weights across different logical layouts")
        if not self.pinned_logical_ids:
            return
        with torch.no_grad():
            self.w13_slots[self.num_dynamic_slots :].copy_(
                source.w13_slots[source.num_dynamic_slots :], non_blocking=False
            )
            self.w2_slots[self.num_dynamic_slots :].copy_(
                source.w2_slots[source.num_dynamic_slots :], non_blocking=False
            )

    def install_pinned_log2phy(self, log2phy: torch.Tensor) -> None:
        """Write immutable shared logical->physical entries into a mapping."""

        for logical_id, slot_id in self._pinned_slot_by_logical_id.items():
            if logical_id >= int(log2phy.numel()):
                raise ValueError(
                    "log2phy does not cover pinned logical expert "
                    f"{logical_id}; size={int(log2phy.numel())}"
                )
            log2phy[int(logical_id)] = int(slot_id)

    def allocate_for(self, expert_key: ExpertKey, *, step_id: int) -> ExpertSlot:
        if expert_key in self._resident:
            slot = self.slots[self._resident[expert_key]]
            slot.last_used_step = int(step_id)
            return slot

        slot = self._first_empty_slot()
        if slot is None:
            slot = self._lru_evictable_slot()
        if slot is None:
            raise RuntimeError(f"no evictable expert slots; states={self.state_counts()}")

        if slot.expert_key is not None:
            self._resident.pop(slot.expert_key, None)
            self._resident_by_expert_id.pop(int(slot.expert_key.expert_id), None)
        slot.expert_key = expert_key
        slot.state = SlotState.LOADING
        slot.version += 1
        slot.last_used_step = int(step_id)
        self._resident[expert_key] = slot.slot_id
        self._resident_by_expert_id[int(expert_key.expert_id)] = slot.slot_id
        return slot

    def mark_ready(self, slot_id: int) -> None:
        self.slots[int(slot_id)].state = SlotState.READY

    def mark_computing(self, slot_id: int) -> None:
        self.slots[int(slot_id)].state = SlotState.COMPUTING

    def mark_released(self, slot_id: int) -> None:
        self.slots[int(slot_id)].state = SlotState.EMPTY

    def lookup(self, expert_key: ExpertKey) -> ExpertSlot | None:
        slot_id = self._resident.get(expert_key)
        return None if slot_id is None else self.slots[slot_id]

    def lookup_expert_id(self, expert_id: int) -> ExpertSlot | None:
        slot_id = self._resident_by_expert_id.get(int(expert_id))
        return None if slot_id is None else self.slots[slot_id]

    def assign_slot(self, slot_id: int, expert_key: ExpertKey, *, step_id: int) -> ExpertSlot:
        slot = self.slots[int(slot_id)]
        self._require_reassignable(slot)
        if slot.expert_key is not None and slot.expert_key != expert_key:
            self._resident.pop(slot.expert_key, None)
            self._resident_by_expert_id.pop(int(slot.expert_key.expert_id), None)
        existing_slot_id = self._resident.get(expert_key)
        if existing_slot_id is not None and existing_slot_id != int(slot_id):
            old_slot = self.slots[int(existing_slot_id)]
            old_slot.expert_key = None
            old_slot.state = SlotState.EMPTY
            self._resident_by_expert_id.pop(int(expert_key.expert_id), None)
        slot.expert_key = expert_key
        slot.version += 1
        slot.last_used_step = int(step_id)
        slot.state = SlotState.LOADING
        self._resident[expert_key] = int(slot_id)
        self._resident_by_expert_id[int(expert_key.expert_id)] = int(slot_id)
        return slot

    def assign_transient_slot(
        self,
        slot_id: int,
        expert_key: ExpertKey,
        *,
        step_id: int,
    ) -> ExpertSlot:
        """Assign a temporary wave slot without updating the lookup index.

        Prefill staging banks are short-lived double buffers. The B2 dataplane
        already receives an explicit logical->physical mapping, so maintaining
        the resident lookup dictionary for every staged wave is pure control
        overhead.
        """
        slot = self.slots[int(slot_id)]
        self._require_reassignable(slot)
        if slot.expert_key is not None:
            self._resident.pop(slot.expert_key, None)
            self._resident_by_expert_id.pop(int(slot.expert_key.expert_id), None)
        slot.expert_key = expert_key
        slot.version += 1
        slot.last_used_step = int(step_id)
        slot.state = SlotState.LOADING
        return slot

    def clear_slot(self, slot_id: int, *, force: bool = False) -> None:
        slot = self.slots[int(slot_id)]
        if not force and slot.state == SlotState.COMPUTING:
            raise RuntimeError(
                f"slot {slot.slot_id} is computing and cannot be reassigned"
            )
        if slot.expert_key is not None:
            self._resident.pop(slot.expert_key, None)
            self._resident_by_expert_id.pop(int(slot.expert_key.expert_id), None)
        slot.expert_key = None
        slot.state = SlotState.EMPTY

    @property
    def total_bytes(self) -> int:
        return _tensor_nbytes(self.w13_slots) + _tensor_nbytes(self.w2_slots)

    def state_counts(self) -> dict[str, int]:
        counts = {state.value: 0 for state in SlotState}
        for slot in self.slots:
            counts[str(slot.state.value)] = counts.get(str(slot.state.value), 0) + 1
        return counts

    def _first_empty_slot(self) -> ExpertSlot | None:
        for slot in self.slots:
            if slot.state == SlotState.EMPTY:
                return slot
        return None

    def _lru_evictable_slot(self) -> ExpertSlot | None:
        candidates = [slot for slot in self.slots if slot.state == SlotState.READY]
        if not candidates:
            return None
        return min(candidates, key=lambda slot: (slot.last_used_step, slot.slot_id))

    @staticmethod
    def _require_reassignable(slot: ExpertSlot) -> None:
        if slot.state in (SlotState.LOADING, SlotState.COMPUTING):
            raise RuntimeError(
                f"slot {slot.slot_id} is {slot.state.value} and cannot be reassigned"
            )


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())
