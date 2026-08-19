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

import os

import torch

from vllm_moe_offload_ascend.moe_offload.expert_key import ExpertKey
from vllm_moe_offload_ascend.moe_offload.host_store import ExpertWeightBundle
from vllm_moe_offload_ascend.moe_offload.layout import LayoutValidator
from vllm_moe_offload_ascend.moe_offload.slot_bank import ExpertSlotBank, SlotState


def _dedupe_preserve_order(values: tuple[int, ...]) -> tuple[int, ...]:
    seen: set[int] = set()
    deduped: list[int] = []
    for value in values:
        int_value = int(value)
        if int_value not in seen:
            seen.add(int_value)
            deduped.append(int_value)
    return tuple(deduped)


@dataclass(frozen=True)
class ExpertSlotMapping:
    layer_id: int
    active_experts: tuple[int, ...]
    logical_to_physical: torch.Tensor
    slot_to_expert: tuple[int | None, ...]
    active_slot_ids: tuple[int, ...]
    # Whether ``logical_to_physical`` is a complete, current logical->physical
    # table that may be used to remap topk_ids.  Set False when the producer
    # deliberately skipped rebuilding the buffer (e.g. ``build_log2phy=False`` on
    # the B2 wave fast path, which reuses a persistent per-buffer-index tensor
    # and routes via precomputed wave-plan physical ids instead).  Consumers that
    # actually index ``logical_to_physical`` must check this flag — the buffer
    # otherwise holds a STALE mapping from a previous wave.
    log2phy_valid: bool = True

    @classmethod
    def from_slot_bank(
        cls,
        *,
        layer_id: int,
        active_experts: tuple[int, ...],
        num_logical_experts: int,
        slot_bank: ExpertSlotBank,
        device: torch.device,
        dtype: torch.dtype = torch.int32,
    ) -> "ExpertSlotMapping":
        if num_logical_experts <= 0:
            raise ValueError("num_logical_experts must be greater than 0")

        layer_id = int(layer_id)
        unique_active_experts = _dedupe_preserve_order(tuple(int(expert_id) for expert_id in active_experts))
        logical_to_physical = torch.full(
            (int(num_logical_experts),),
            fill_value=-1,
            dtype=dtype,
            device=device,
        )
        slot_to_expert: list[int | None] = [None] * slot_bank.physical_expert_count
        active_slot_ids: list[int] = []

        # Fused/mix-placement shared experts occupy immutable physical suffix
        # rows. They are not part of the dynamic active set, but their logical
        # IDs are present in every backend top-k and must map on every call.
        slot_bank.install_pinned_log2phy(logical_to_physical)
        for logical_id in slot_bank.pinned_logical_ids:
            pinned_slot = slot_bank.pinned_slot_for_logical_id(logical_id)
            assert pinned_slot is not None
            slot_to_expert[int(pinned_slot)] = int(logical_id)

        for expert_id in unique_active_experts:
            if expert_id < 0 or expert_id >= num_logical_experts:
                raise ValueError(f"active expert {expert_id} is outside num_logical_experts={num_logical_experts}")

            slot = slot_bank.lookup(ExpertKey(layer_id, expert_id))
            if slot is None:
                raise RuntimeError(f"active expert {expert_id} is not resident in layer {layer_id}")
            if slot.state != SlotState.READY:
                raise RuntimeError(f"active expert {expert_id} slot {slot.slot_id} is not ready")

            logical_to_physical[expert_id] = int(slot.slot_id)
            active_slot_ids.append(int(slot.slot_id))

        for slot in slot_bank.slots:
            if slot.expert_key is not None:
                slot_to_expert[slot.slot_id] = int(slot.expert_key.expert_id)

        return cls(
            layer_id=layer_id,
            active_experts=unique_active_experts,
            logical_to_physical=logical_to_physical,
            slot_to_expert=tuple(slot_to_expert),
            active_slot_ids=tuple(active_slot_ids),
        )

    def remap_topk_ids(self, topk_ids: torch.Tensor) -> torch.Tensor:
        if not self.log2phy_valid:
            raise RuntimeError(
                "remap_topk_ids called on a mapping whose logical_to_physical was "
                "not rebuilt (log2phy_valid=False); use the wave-plan physical ids "
                "for this path instead of this stale buffer"
            )
        remapped = self.logical_to_physical[topk_ids]
        # Fail-closed when any active expert lacks a ready slot (log2phy == -1).
        # On CPU we can check cheaply.  On NPU the .item() sync is too costly for
        # the per-step hot path, so it stays opt-in via SEW_LOG2PHY_VALIDATE; when
        # disabled, an unstaged expert silently indexes slot[-1] (wrong weights /
        # MTE out-of-range), so the caller MUST stage the complete active set.
        if remapped.device.type == "cpu":
            if bool((remapped < 0).any().item()):
                raise RuntimeError("topk_ids contain experts without ready slots")
        elif os.environ.get("SEW_LOG2PHY_VALIDATE"):
            if bool((remapped < 0).any().item()):
                missing = (
                    topk_ids[remapped < 0].detach().cpu().unique().tolist()
                )
                raise RuntimeError(
                    f"topk_ids contain experts without ready slots: {missing}"
                )
        return remapped


@dataclass(frozen=True)
class PreparedSlotWeights:
    w1: torch.Tensor
    w2: torch.Tensor
    log2phy: torch.Tensor
    physical_expert_count: int
    mapping: ExpertSlotMapping

    def validate_backend_ready(self, *, expected_device_type: str) -> None:
        if self.physical_expert_count <= 0:
            raise ValueError("physical_expert_count must be greater than 0")
        if self.w1.shape[0] != self.physical_expert_count:
            raise ValueError(
                "w1 physical expert count mismatch: "
                f"{self.w1.shape[0]} != {self.physical_expert_count}"
            )
        if self.w2.shape[0] != self.physical_expert_count:
            raise ValueError(
                "w2 physical expert count mismatch: "
                f"{self.w2.shape[0]} != {self.physical_expert_count}"
            )
        LayoutValidator.validate_backend_ready(
            ExpertWeightBundle(
                layer_id=self.mapping.layer_id,
                expert_id=-1,
                w13=self.w1,
                w2=self.w2,
            ),
            expected_device_type=expected_device_type,
        )

    @classmethod
    def from_slot_bank(
        cls,
        *,
        slot_bank: ExpertSlotBank,
        mapping: ExpertSlotMapping,
    ) -> "PreparedSlotWeights":
        return cls(
            w1=slot_bank.w13_slots,
            w2=slot_bank.w2_slots,
            log2phy=mapping.logical_to_physical,
            physical_expert_count=slot_bank.physical_expert_count,
            mapping=mapping,
        )
