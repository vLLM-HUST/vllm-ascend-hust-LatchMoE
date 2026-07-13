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


class ExpertSlotBank:
    def __init__(
        self,
        num_slots: int,
        w13_shape: tuple[int, ...],
        w2_shape: tuple[int, ...],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if num_slots <= 0:
            raise ValueError("num_slots must be greater than 0")
        self.w13_slots = torch.empty((num_slots, *w13_shape), dtype=dtype, device=device)
        self.w2_slots = torch.empty((num_slots, *w2_shape), dtype=dtype, device=device)
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
        if slot.expert_key is not None:
            self._resident.pop(slot.expert_key, None)
            self._resident_by_expert_id.pop(int(slot.expert_key.expert_id), None)
        slot.expert_key = expert_key
        slot.version += 1
        slot.last_used_step = int(step_id)
        slot.state = SlotState.LOADING
        return slot

    def clear_slot(self, slot_id: int) -> None:
        slot = self.slots[int(slot_id)]
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


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())
