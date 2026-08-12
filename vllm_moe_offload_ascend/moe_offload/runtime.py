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

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from itertools import count
import logging
import os
from pathlib import Path
import threading
from time import perf_counter

import torch

logger = logging.getLogger(__name__)

from vllm_ascend import envs
from vllm_moe_offload_ascend.moe_offload.config import MoeOffloadConfig
from vllm_moe_offload_ascend.moe_offload.cpu_first_loader import is_cpu_first_layer
from vllm_moe_offload_ascend.moe_offload.compute_bucket import (
    ComputeBucketClassifier,
    ComputeBucketDecision,
    load_compute_bucket_classifier,
)
from vllm_moe_offload_ascend.moe_offload.expert_key import ExpertKey
from vllm_moe_offload_ascend.moe_offload.host_store import HostExpertStore
from vllm_moe_offload_ascend.moe_offload.placement_plan import (
    PlacementPlanProvider,
    resolve_placement_plan,
)
from vllm_moe_offload_ascend.moe_offload.profile_io import append_jsonl
from vllm_moe_offload_ascend.moe_offload.slot_bank import (
    ExpertSlotBank,
    SlotLease,
    SlotState,
)
from vllm_moe_offload_ascend.moe_offload.slot_mapping import ExpertSlotMapping, PreparedSlotWeights
from vllm_moe_offload_ascend.moe_offload.trace_collector import TraceCollector, TraceRecord
from vllm_moe_offload_ascend.moe_offload.expert_weight_release import release_layer_original_expert_weights
from vllm_moe_offload_ascend.moe_offload.tiered_residency import TieredResidencyPolicy
from vllm_moe_offload_ascend.moe_offload.transfer_engine import TransferEngine


def _env_value(name: str, default: str = "") -> str:
    return str(os.getenv(name, getattr(envs, name, default)) or "")


@dataclass(frozen=True)
class MoeOffloadMemoryLedger:
    registered_layers: int
    host_experts: int
    original_expert_weight_bytes: int
    host_store_bytes: int
    slot_bank_bytes: int
    prefill_stage_bank_count: int = 0
    prefill_stage_bank_bytes: int = 0
    prefill_stage_mapping_bytes: int = 0
    full_layer_prefill_bank_count: int = 0
    full_layer_prefill_bank_bytes: int = 0
    full_layer_prefill_mapping_bytes: int = 0

    @property
    def original_expert_weights_retained(self) -> bool:
        return self.original_expert_weight_bytes > 0

    @property
    def total_managed_bytes(self) -> int:
        return (
            self.original_expert_weight_bytes
            + self.host_store_bytes
            + self.slot_bank_bytes
            + self.prefill_stage_bank_bytes
            + self.prefill_stage_mapping_bytes
            + self.full_layer_prefill_bank_bytes
            + self.full_layer_prefill_mapping_bytes
        )

    @property
    def total_npu_slot_bytes(self) -> int:
        return (
            self.slot_bank_bytes
            + self.prefill_stage_bank_bytes
            + self.prefill_stage_mapping_bytes
            + self.full_layer_prefill_bank_bytes
            + self.full_layer_prefill_mapping_bytes
        )

    def to_jsonable(self) -> dict[str, int | bool]:
        return {
            "registered_layers": int(self.registered_layers),
            "host_experts": int(self.host_experts),
            "original_expert_weight_bytes": int(self.original_expert_weight_bytes),
            "host_store_bytes": int(self.host_store_bytes),
            "slot_bank_bytes": int(self.slot_bank_bytes),
            "prefill_stage_bank_count": int(self.prefill_stage_bank_count),
            "prefill_stage_bank_bytes": int(self.prefill_stage_bank_bytes),
            "prefill_stage_mapping_bytes": int(self.prefill_stage_mapping_bytes),
            "full_layer_prefill_bank_count": int(
                self.full_layer_prefill_bank_count
            ),
            "full_layer_prefill_bank_bytes": int(
                self.full_layer_prefill_bank_bytes
            ),
            "full_layer_prefill_mapping_bytes": int(
                self.full_layer_prefill_mapping_bytes
            ),
            "total_npu_slot_bytes": int(self.total_npu_slot_bytes),
            "original_expert_weights_retained": self.original_expert_weights_retained,
            "total_managed_bytes": int(self.total_managed_bytes),
        }


@dataclass(frozen=True)
class MoeExpertReleasePlan:
    ready: bool
    layers_ready: tuple[int, ...]
    blockers: tuple[str, ...]


class MoeOffloadDecisionPath(str, Enum):
    FULL_WEIGHT_PATH = "full_weight_path"
    SLOT_CACHE_PATH = "slot_cache_path"
    FAIL_CLOSED = "fail_closed"


@dataclass(frozen=True)
class MoeOffloadPathDecision:
    path: MoeOffloadDecisionPath
    layer_id: int
    active_expert_count: int
    active_experts: tuple[int, ...]
    fanout_threshold: int
    full_weights_available: bool
    slot_cache_ready: bool
    reason: str

    def to_jsonable(self) -> dict[str, object]:
        return {
            "path": self.path.value,
            "layer_id": self.layer_id,
            "active_expert_count": self.active_expert_count,
            "active_experts": list(self.active_experts),
            "fanout_threshold": self.fanout_threshold,
            "full_weights_available": self.full_weights_available,
            "slot_cache_ready": self.slot_cache_ready,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class MoeOffloadProfileEvent:
    name: str
    layer_id: int | None
    seconds: float
    memory_ledger: MoeOffloadMemoryLedger
    payload: dict[str, object] | None = None

    def to_jsonable(self) -> dict[str, object]:
        data = {
            "name": self.name,
            "layer_id": self.layer_id,
            "seconds": self.seconds,
            "memory_ledger": self.memory_ledger.to_jsonable(),
        }
        if self.payload is not None:
            data["payload"] = self.payload
        return data


@dataclass(frozen=True)
class MoePrefillRouteStats:
    layer_id: int
    topk_cache_key: tuple[object, ...]
    token_counts_by_expert: dict[int, int]
    active_experts: tuple[int, ...]
    num_tokens: int
    top_k: int
    pair_offsets_by_expert: dict[int, tuple[int, ...]] | None = None


@dataclass(frozen=True)
class SlotComputeHandle:
    layer_id: int
    slot_ids: tuple[int, ...]
    leases: tuple[SlotLease, ...]


@dataclass(frozen=True)
class SlotAddressFingerprint:
    w13_data_ptr: int
    w2_data_ptr: int
    log2phy_data_ptr: int
    w13_shape: tuple[int, ...]
    w2_shape: tuple[int, ...]
    log2phy_shape: tuple[int, ...]
    device: str
    dtype: str


def _slot_address_fingerprint_payload(
    fingerprint: SlotAddressFingerprint,
) -> dict[str, object]:
    return {
        "w13_data_ptr": int(fingerprint.w13_data_ptr),
        "w2_data_ptr": int(fingerprint.w2_data_ptr),
        "log2phy_data_ptr": int(fingerprint.log2phy_data_ptr),
        "w13_shape": [int(dim) for dim in fingerprint.w13_shape],
        "w2_shape": [int(dim) for dim in fingerprint.w2_shape],
        "log2phy_shape": [int(dim) for dim in fingerprint.log2phy_shape],
        "device": str(fingerprint.device),
        "dtype": str(fingerprint.dtype),
    }


class MoeOffloadRuntime:
    def __init__(self, config: MoeOffloadConfig | None = None) -> None:
        self.config = config if config is not None else MoeOffloadConfig.from_env()
        self.trace_collector = TraceCollector(max_records=self.config.trace_max_records)
        self._step_counter = count()
        self._pending_trace_step_by_layer: dict[int, int] = {}
        self._host_store = HostExpertStore()
        self._slot_banks: dict[int, ExpertSlotBank] = {}
        # Physical B2 stage buffers are layout-scoped, not layer-scoped. Layers
        # execute serially through the MoE splitting seam, so equal-layout
        # layers can safely lease the same double buffer when H2D waits for the
        # previous compute-stream release event.
        self._prefill_stage_pool: dict[tuple[object, ...], list[ExpertSlotBank]] = {}
        self._prefill_stage_banks: dict[int, list[ExpertSlotBank]] = {}
        self._prefill_stage_pool_release_events: dict[
            tuple[tuple[object, ...], int], object
        ] = {}
        self._prefill_stage_log2phy_buffers: dict[int, list[torch.Tensor]] = {}
        self._full_layer_prefill_pool: dict[
            tuple[object, ...], ExpertSlotBank
        ] = {}
        self._full_layer_prefill_log2phy_buffers: dict[
            tuple[object, ...], torch.Tensor
        ] = {}
        self._original_expert_weight_bytes_by_layer: dict[int, int] = {}
        self._expert_weight_bytes_by_layer: dict[int, int] = {}
        self._slot_expert_weight_bytes_by_layer: dict[int, int] = {}
        self._released_original_weight_layers: set[int] = set()
        # Option-2 (graph-compatible offload): persistent per-layer log2phy buffer.
        # Fixed address, allocated once at register time, updated in-place by the
        # eager staging step. The captured graph reads this stable buffer, so the
        # data-dependent decision never enters the captured op stream.
        self._log2phy_buffers: dict[int, torch.Tensor] = {}
        self._log2phy_slot_by_expert: dict[int, dict[int, int]] = {}
        self._graph_slot_address_fingerprints: dict[int, SlotAddressFingerprint] = {}
        self._graph_slot_address_lock_evidence: set[int] = set()
        self._graph_slot_address_validation_evidence: set[int] = set()
        self._slot_generation_protection_evidence: set[int] = set()
        self._transfer_engine = TransferEngine()
        self._profile_events: list[MoeOffloadProfileEvent] = []
        self._memory_ledger_cache: MoeOffloadMemoryLedger | None = None
        self._compute_bucket_classifier: ComputeBucketClassifier | None = None
        self._compute_bucket_classifier_loaded = False
        self._prefill_route_stats_by_layer: dict[int, MoePrefillRouteStats] = {}
        self._active_slot_ids_by_layer: dict[int, tuple[int, ...]] = {}
        self._slot_compute_done_events: dict[tuple[int, int, int], object] = {}
        self._placement_plan_provider: PlacementPlanProvider | None = None
        self._fixed_slot_execution_state = threading.local()

    def set_placement_plan_provider(
        self,
        provider: PlacementPlanProvider | None,
    ) -> None:
        """Install a policy-neutral placement-plan provider.

        The provider has no access to slot storage and may only reorder the
        experts already selected by the router. This preserves LatchMoE's
        ownership of address-stable slot allocation and transfer lifecycle.
        """
        self._placement_plan_provider = provider

    def _resolve_placement_plan(
        self,
        *,
        layer_id: int,
        active_experts: tuple[int, ...],
        num_logical_experts: int,
    ) -> tuple[int, ...]:
        return resolve_placement_plan(
            self._placement_plan_provider,
            layer_id=int(layer_id),
            routed_experts=active_experts,
            num_logical_experts=int(num_logical_experts),
        )

    def trace_routing(
        self,
        *,
        layer_id: int,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        num_experts: int,
        mode: str = "unknown",
        step_id: int | None = None,
        **_: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.trace_logical_active_experts(
            layer_id=layer_id,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            num_logical_experts=num_experts,
            mode=mode,
            step_id=step_id,
        )

    def next_step_id(self) -> int:
        """Compatibility hook for older moe-offload hook branches."""
        return int(next(self._step_counter))

    def _prefill_route_stats_key(
        self,
        *,
        layer_id: int,
        topk_ids: torch.Tensor,
    ) -> tuple[object, ...]:
        return (
            int(layer_id),
            str(topk_ids.device),
            str(topk_ids.dtype),
            tuple(int(dim) for dim in topk_ids.shape),
            tuple(int(stride) for stride in topk_ids.stride()),
            int(topk_ids.storage_offset()),
            int(topk_ids.numel()),
            int(topk_ids.data_ptr()),
            int(getattr(topk_ids, "_version", 0)),
        )

    def cache_prefill_route_stats(
        self,
        *,
        layer_id: int,
        topk_ids: torch.Tensor,
        token_counts_by_expert: dict[int, int],
        pair_offsets_by_expert: dict[int, tuple[int, ...]] | None = None,
    ) -> MoePrefillRouteStats:
        """Remember route statistics already read by the eager SEW seam.

        B2 Prefill needs the same ``topk_ids`` CPU statistics to plan waves. The
        staging seam has already paid that host-sync cost to decide whether B2
        should defer single-shot staging, so the downstream B2 path can consume
        this cache and avoid a second D2H read for the same tensor.
        """
        normalized_layer_id = int(layer_id)
        counts = {
            int(expert_id): int(count)
            for expert_id, count in token_counts_by_expert.items()
            if int(count) > 0
        }
        normalized_offsets = None
        if pair_offsets_by_expert is not None:
            normalized_offsets = {
                int(expert_id): tuple(int(offset) for offset in offsets)
                for expert_id, offsets in pair_offsets_by_expert.items()
                if int(expert_id) in counts
            }
        top_k = int(topk_ids.shape[1]) if topk_ids.ndim > 1 else 1
        stats = MoePrefillRouteStats(
            layer_id=normalized_layer_id,
            topk_cache_key=self._prefill_route_stats_key(
                layer_id=normalized_layer_id,
                topk_ids=topk_ids,
            ),
            token_counts_by_expert=counts,
            active_experts=tuple(sorted(counts)),
            num_tokens=int(topk_ids.shape[0]) if topk_ids.ndim > 0 else 0,
            top_k=top_k,
            pair_offsets_by_expert=normalized_offsets,
        )
        self._prefill_route_stats_by_layer[normalized_layer_id] = stats
        return stats

    def consume_prefill_route_stats_record(
        self,
        *,
        layer_id: int,
        topk_ids: torch.Tensor,
    ) -> MoePrefillRouteStats | None:
        normalized_layer_id = int(layer_id)
        stats = self._prefill_route_stats_by_layer.get(normalized_layer_id)
        if stats is None:
            return None
        expected_key = self._prefill_route_stats_key(
            layer_id=normalized_layer_id,
            topk_ids=topk_ids,
        )
        # Drop stale entries rather than risking reuse after a tensor address is
        # recycled for a different route-id buffer.
        self._prefill_route_stats_by_layer.pop(normalized_layer_id, None)
        if stats.topk_cache_key != expected_key:
            return None
        return stats

    def consume_prefill_route_stats(
        self,
        *,
        layer_id: int,
        topk_ids: torch.Tensor,
    ) -> dict[int, int] | None:
        stats = self.consume_prefill_route_stats_record(
            layer_id=layer_id,
            topk_ids=topk_ids,
        )
        if stats is None:
            return None
        return dict(stats.token_counts_by_expert)

    def trace_logical_active_experts(
        self,
        *,
        layer_id: int,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        num_logical_experts: int,
        mode: str = "unknown",
        step_id: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.config.should_trace:
            normalized_layer_id = int(layer_id)
            normalized_step_id = (
                int(step_id)
                if step_id is not None and int(step_id) >= 0
                else int(next(self._step_counter))
            )
            self._pending_trace_step_by_layer[normalized_layer_id] = normalized_step_id
            record = self.trace_collector.record_logical(
                layer_id=normalized_layer_id,
                step_id=normalized_step_id,
                topk_ids=topk_ids,
                num_logical_experts=num_logical_experts,
                mode=mode,
            )
            self._append_trace_record_jsonl(record)
        return topk_ids, topk_weights

    def trace_grouped_active_experts(
        self,
        *,
        layer_id: int,
        group_list: torch.Tensor | None,
        group_list_type: int | None,
        physical_expert_count: int | None = None,
        mode: str = "unknown",
    ) -> torch.Tensor | None:
        if not self.config.should_trace_gmm or group_list is None or group_list_type is None:
            return group_list
        if _is_current_graph_capturing():
            return group_list

        normalized_layer_id = int(layer_id)
        step_id = self._pending_trace_step_by_layer.pop(normalized_layer_id, None)
        if step_id is None:
            step_id = next(self._step_counter)
        record = self.trace_collector.record_grouped(
            layer_id=normalized_layer_id,
            step_id=step_id,
            group_list=group_list,
            group_list_type=group_list_type,
            physical_expert_count=physical_expert_count,
            mode=mode,
        )
        self._append_trace_record_jsonl(record)
        return group_list

    def classify_grouped_compute_bucket(
        self,
        *,
        layer_id: int,
        group_list: torch.Tensor | None,
        group_list_type: int | None,
        phase: str = "unknown",
    ) -> ComputeBucketDecision | None:
        if _is_current_graph_capturing():
            return None
        classifier = self._get_compute_bucket_classifier()
        if classifier is None or not classifier.enabled:
            return None
        start = perf_counter()
        decision = classifier.classify(
            group_list=group_list,
            group_list_type=group_list_type,
            phase=phase,
        )
        self._record_profile_event(
            "compute_bucket_decision",
            layer_id=int(layer_id),
            start=start,
            payload=decision.to_jsonable(),
        )
        return decision

    def record_compute_bucket_fast_path_gate(
        self,
        *,
        layer_id: int | None,
        enabled: bool,
        reason: str,
        bucket_id: int | None = None,
        signature: str = "",
        original_expert_count: int = 0,
        compact_expert_count: int = 0,
    ) -> None:
        start = perf_counter()
        self._record_profile_event(
            "compute_bucket_fast_path_gate",
            layer_id=layer_id,
            start=start,
            payload={
                "enabled": bool(enabled),
                "reason": str(reason),
                "bucket_id": bucket_id,
                "signature": str(signature),
                "original_expert_count": int(original_expert_count),
                "compact_expert_count": int(compact_expert_count),
            },
        )

    def export_trace(self, path: str | Path) -> int:
        return self.trace_collector.write_jsonl(path)

    @property
    def should_use_fixed_slots(self) -> bool:
        return self.config.enabled and not self.config.trace_only and self.config.num_slots > 0

    @property
    def should_use_layered_runtime(self) -> bool:
        return self.should_use_fixed_slots and self.config.layered_runtime

    def register_layer_for_fixed_slots(
        self,
        layer: torch.nn.Module,
        *,
        slot_device: torch.device | None = None,
    ) -> None:
        layer_id = int(getattr(layer, "layer_id", -1))
        if layer_id < 0:
            raise ValueError("layer.layer_id is required for fixed-slot registration")
        if layer_id in self._graph_slot_address_fingerprints:
            raise RuntimeError(
                f"layer {layer_id} fixed-slot addresses are already captured; "
                "re-registering would invalidate an ACLGraph replay"
            )

        start = perf_counter()
        cpu_first_layer = is_cpu_first_layer(layer)
        host_store_report = self._host_store.register_layer(
            layer,
            pin_memory=self.config.should_pin_host_memory,
            clone_tensors=not cpu_first_layer,
        )
        w13_weight = getattr(layer, "w13_weight")
        w2_weight = getattr(layer, "w2_weight")
        self._original_expert_weight_bytes_by_layer[layer_id] = _tensor_nbytes(w13_weight) + _tensor_nbytes(w2_weight)
        self._expert_weight_bytes_by_layer[layer_id] = (
            _tensor_nbytes(w13_weight[0]) + _tensor_nbytes(w2_weight[0])
        )
        device = slot_device if slot_device is not None else w13_weight.device
        self._slot_banks[layer_id] = ExpertSlotBank(
            self.config.num_slots,
            tuple(int(dim) for dim in w13_weight.shape[1:]),
            tuple(int(dim) for dim in w2_weight.shape[1:]),
            dtype=w13_weight.dtype,
            device=device,
        )
        self._slot_expert_weight_bytes_by_layer[layer_id] = (
            _tensor_nbytes(self._slot_banks[layer_id].w13_slots[0])
            + _tensor_nbytes(self._slot_banks[layer_id].w2_slots[0])
        )
        self._prefill_stage_banks.pop(layer_id, None)
        self._prefill_stage_log2phy_buffers.pop(layer_id, None)
        # Option-2: allocate the persistent log2phy buffer once, fixed address.
        # Size = num_logical_experts (from the original weight's expert dim).
        num_logical_experts = int(w13_weight.shape[0])
        self._log2phy_buffers[layer_id] = torch.full(
            (num_logical_experts,),
            fill_value=-1,
            dtype=torch.int32,
            device=device,
        )
        self._log2phy_slot_by_expert[layer_id] = {}
        self._invalidate_memory_ledger_cache()
        self._record_profile_event(
            "register_layer_for_fixed_slots",
            layer_id=layer_id,
            start=start,
            payload={
                "host_store": host_store_report.to_jsonable(),
            },
        )

    def is_layer_registered(self, layer_id: int) -> bool:
        return int(layer_id) in self._slot_banks

    def slot_readiness_for_experts(
        self,
        *,
        layer_id: int,
        expert_ids: tuple[int, ...],
    ) -> dict[int, bool]:
        """Return whether each logical expert already has a READY fixed slot."""
        readiness, _ = self.slot_readiness_and_ready_slot_ids_for_experts(
            layer_id=layer_id,
            expert_ids=expert_ids,
        )
        return readiness

    def slot_readiness_and_ready_slot_ids_for_experts(
        self,
        *,
        layer_id: int,
        expert_ids: tuple[int, ...],
    ) -> tuple[dict[int, bool], dict[int, int]]:
        """Return READY status and main-slot ids with one slot-bank scan."""
        slot_bank = self._slot_banks.get(int(layer_id))
        if slot_bank is None:
            return {int(expert_id): False for expert_id in expert_ids}, {}
        readiness: dict[int, bool] = {}
        ready_slot_ids: dict[int, int] = {}
        for expert_id in expert_ids:
            expert_id = int(expert_id)
            slot = slot_bank.lookup_expert_id(expert_id)
            is_ready = slot is not None and slot.state == SlotState.READY
            readiness[expert_id] = is_ready
            if is_ready:
                ready_slot_ids[expert_id] = int(slot.slot_id)
        return readiness, ready_slot_ids

    def ready_slot_ids_for_experts(
        self,
        *,
        layer_id: int,
        expert_ids: tuple[int, ...],
    ) -> dict[int, int]:
        """Return logical expert -> READY main-slot id for already resident experts."""
        slot_bank = self._slot_banks.get(int(layer_id))
        if slot_bank is None:
            return {}
        slot_ids: dict[int, int] = {}
        for expert_id in expert_ids:
            expert_id = int(expert_id)
            slot = slot_bank.lookup_expert_id(expert_id)
            if slot is not None and slot.state == SlotState.READY:
                slot_ids[expert_id] = int(slot.slot_id)
        return slot_ids

    def estimate_expert_weight_bytes(
        self,
        *,
        layer_id: int,
        expert_id: int,
    ) -> int:
        """Return host-store bytes for one expert, used for B2 transfer profiling."""
        bundle = self._host_store.get(int(layer_id), int(expert_id))
        total = _tensor_nbytes(bundle.w13) + _tensor_nbytes(bundle.w2)
        if bundle.w13_scale is not None:
            total += _tensor_nbytes(bundle.w13_scale)
        if bundle.w2_scale is not None:
            total += _tensor_nbytes(bundle.w2_scale)
        return int(total)

    def cached_layer_expert_weight_bytes(self, *, layer_id: int) -> int:
        """Return the registered per-expert host bytes for uniform expert weights.

        This is a fast-path estimate for B2 scheduling/profile aggregation.  It
        avoids repeated host-store bundle lookups when all experts in a layer
        share the same unquantized weight layout; callers that need exact
        per-expert accounting should use ``estimate_expert_weight_bytes``.
        """
        return int(self._expert_weight_bytes_by_layer.get(int(layer_id), 0))

    def estimate_slot_expert_weight_bytes(
        self,
        *,
        layer_id: int,
        expert_id: int,
    ) -> int:
        """Return bytes copied when a ready slot is staged into a temp bank."""
        slot_bank = self._slot_banks.get(int(layer_id))
        if slot_bank is None:
            return 0
        slot = slot_bank.lookup(ExpertKey(int(layer_id), int(expert_id)))
        if slot is None or slot.state != SlotState.READY:
            return 0
        return int(_tensor_nbytes(slot.w13) + _tensor_nbytes(slot.w2))

    def _allocate_slot_with_loading_fallback(
        self,
        slot_bank: ExpertSlotBank,
        key: ExpertKey,
        *,
        step_id: int,
    ):
        try:
            return slot_bank.allocate_for(key, step_id=step_id)
        except RuntimeError:
            if not any(
                slot.state in (SlotState.LOADING, SlotState.COMPUTING)
                for slot in slot_bank.slots
            ):
                raise
            self._drain_slot_bank_inflight_work(slot_bank)
            return slot_bank.allocate_for(key, step_id=step_id)

    def _drain_slot_bank_inflight_work(self, slot_bank: ExpertSlotBank) -> None:
        if any(slot.state == SlotState.LOADING for slot in slot_bank.slots):
            self._transfer_engine.synchronize()
            loading_slots = [
                slot.slot_id
                for slot in slot_bank.slots
                if slot.state == SlotState.LOADING
            ]
            if loading_slots:
                raise RuntimeError(
                    "H2D synchronization completed without a matching ready event "
                    f"for slots {loading_slots}"
                )

        computing_slots = [
            slot for slot in slot_bank.slots if slot.state == SlotState.COMPUTING
        ]
        events_to_sync: list[object] = []
        seen_events: set[int] = set()
        leases: list[SlotLease] = []
        for slot in computing_slots:
            lease = slot.lease()
            leases.append(lease)
            event_key = (
                int(lease.expert_key.layer_id),
                int(lease.slot_id),
                int(lease.version),
            )
            event = self._slot_compute_done_events.pop(event_key, None)
            if event is None:
                raise RuntimeError(
                    "computing slot has no completion event: "
                    f"layer={lease.expert_key.layer_id} slot={lease.slot_id} "
                    f"version={lease.version}"
                )
            if id(event) not in seen_events:
                seen_events.add(id(event))
                events_to_sync.append(event)
        for event in events_to_sync:
            self._synchronize_event(event)
        for slot, lease in zip(computing_slots, leases, strict=True):
            if not slot.matches_lease(lease):
                raise RuntimeError(
                    f"slot {lease.slot_id} ownership changed while compute was in flight"
                )
            if slot.state == SlotState.COMPUTING:
                slot.state = SlotState.READY
        protected_layers = sorted(
            {
                int(lease.expert_key.layer_id)
                for lease in leases
                if int(lease.expert_key.layer_id)
                not in self._slot_generation_protection_evidence
            }
        )
        for layer_id in protected_layers:
            layer_leases = [
                lease
                for lease in leases
                if int(lease.expert_key.layer_id) == layer_id
            ]
            self._slot_generation_protection_evidence.add(layer_id)
            self._record_profile_event(
                "slot_generation_protected_until_compute_complete",
                layer_id=layer_id,
                start=perf_counter(),
                payload={
                    "slot_ids": [int(lease.slot_id) for lease in layer_leases],
                    "generations": [int(lease.version) for lease in layer_leases],
                    "expert_ids": [
                        int(lease.expert_key.expert_id) for lease in layer_leases
                    ],
                    "completion_events_synchronized": len(events_to_sync),
                    "leases_still_match": True,
                },
            )

    def _synchronize_event(self, event: object) -> None:
        synchronize = getattr(event, "synchronize", None)
        if callable(synchronize):
            synchronize()
            return
        wait = getattr(event, "wait", None)
        if callable(wait):
            wait()
            return
        try:
            import torch

            torch.npu.current_stream().wait_event(event)
        except Exception:
            return

    def _release_allocated_loads(
        self,
        slot_bank: ExpertSlotBank,
        loads: list[tuple[object, object]],
    ) -> None:
        for _, slot in loads:
            if getattr(slot, "state", None) == SlotState.LOADING:
                slot_bank.clear_slot(int(slot.slot_id), force=True)

    def begin_slot_compute(
        self,
        prepared: PreparedSlotWeights,
    ) -> SlotComputeHandle | None:
        """Protect main-bank slots while the compute stream reads them."""
        layer_id = int(prepared.mapping.layer_id)
        slot_bank = self._slot_banks.get(layer_id)
        if slot_bank is None:
            return None
        if prepared.w1.data_ptr() != slot_bank.w13_slots.data_ptr():
            return None
        slot_ids = tuple(
            sorted({int(slot_id) for slot_id in prepared.mapping.active_slot_ids})
        )
        if not slot_ids:
            slot_ids = tuple(
                int(slot_id)
                for slot_id in self._active_slot_ids_by_layer.get(layer_id, ())
            )
        if not slot_ids:
            return None
        is_graph_capture = _is_current_graph_capturing()
        leases: list[SlotLease] = []
        for slot_id in slot_ids:
            slot = slot_bank.slots[int(slot_id)]
            if int(slot_id) >= len(prepared.mapping.slot_to_expert):
                raise RuntimeError(
                    f"prepared slot mapping has no owner entry for slot {slot_id}"
                )
            expected_expert = prepared.mapping.slot_to_expert[int(slot_id)]
            expected_key = (
                ExpertKey(layer_id, int(expected_expert))
                if expected_expert is not None
                else None
            )
            if slot.expert_key != expected_key:
                raise RuntimeError(
                    f"prepared mapping owner mismatch for slot {slot_id}: "
                    f"expected {expected_key}, found {slot.expert_key}"
                )
            if slot.state == SlotState.COMPUTING and is_graph_capture:
                # vLLM-Ascend enters capture on a dedicated stream after making
                # that stream wait for the preceding default stream. Transfer
                # this matching completion record to the capture stream instead
                # of synchronizing the host from inside graph capture.
                lease = slot.lease()
                event_key = (
                    int(lease.expert_key.layer_id),
                    int(lease.slot_id),
                    int(lease.version),
                )
                if self._slot_compute_done_events.pop(event_key, None) is None:
                    raise RuntimeError(
                        "capturing slot has no completion event from the "
                        "preceding eager compute: "
                        f"layer={lease.expert_key.layer_id} slot={lease.slot_id} "
                        f"version={lease.version}"
                    )
                if not slot.matches_lease(lease):
                    raise RuntimeError(
                        f"slot {lease.slot_id} ownership changed before capture handoff"
                    )
                slot.state = SlotState.READY
            if slot.state != SlotState.READY:
                raise RuntimeError(
                    f"prepared slot {slot_id} is not ready for compute: {slot.state.value}"
                )
            leases.append(slot.lease())
        for slot_id in slot_ids:
            slot_bank.mark_computing(int(slot_id))
        return SlotComputeHandle(
            layer_id=layer_id,
            slot_ids=slot_ids,
            leases=tuple(leases),
        )

    def end_slot_compute(self, handle: SlotComputeHandle | None) -> None:
        if handle is None:
            return
        slot_bank = self._slot_banks.get(int(handle.layer_id))
        if slot_bank is None:
            return
        for lease in handle.leases:
            slot = slot_bank.slots[int(lease.slot_id)]
            if not slot.matches_lease(lease):
                raise RuntimeError(
                    f"slot {lease.slot_id} ownership changed before compute completed"
                )
            if slot.state != SlotState.COMPUTING:
                raise RuntimeError(
                    f"slot {lease.slot_id} is not computing at compute completion: "
                    f"{slot.state.value}"
                )
        event = self._record_current_stream_event_or_none()
        for lease in handle.leases:
            slot = slot_bank.slots[int(lease.slot_id)]
            if event is None:
                slot.state = SlotState.READY
            else:
                self._slot_compute_done_events[
                    (
                        int(handle.layer_id),
                        int(lease.slot_id),
                        int(lease.version),
                    )
                ] = event

    def _record_current_stream_event_or_none(self):
        try:
            import torch

            npu = getattr(torch, "npu", None)
            if npu is None or not hasattr(npu, "current_stream"):
                return None
            stream = npu.current_stream()
            record_event = getattr(stream, "record_event", None)
            if callable(record_event):
                return record_event()
            event_cls = getattr(npu, "Event", None)
            if event_cls is None:
                return None
            event = event_cls()
            event.record(stream)
            return event
        except Exception:
            return None

    def is_resident_layer(self, layer_id: int) -> bool:
        return self.config.tiered_residency.is_resident_layer(int(layer_id))

    @contextmanager
    def suspend_fixed_slot_execution(self) -> Iterator[None]:
        """Temporarily force the current thread through the native MoE path."""
        state = self._fixed_slot_execution_state
        depth = int(getattr(state, "suspend_depth", 0))
        state.suspend_depth = depth + 1
        try:
            yield
        finally:
            if depth == 0:
                try:
                    del state.suspend_depth
                except AttributeError:
                    pass
            else:
                state.suspend_depth = depth

    def should_use_fixed_slot_plan_for_layer(self, layer_id: int) -> bool:
        if int(
            getattr(self._fixed_slot_execution_state, "suspend_depth", 0)
        ) > 0:
            return False
        if not self.should_use_fixed_slots:
            return False
        return not self.is_resident_layer(int(layer_id))

    def is_static_residency_regime(self, num_logical_experts: int) -> bool:
        """Regime A iff num_slots >= num_logical_experts.

        Under Regime A every logical expert owns a fixed slot, so the
        logical->physical (log2phy) mapping is *static* (step-independent): it is
        staged ONCE for all experts before ACLGraph capture
        (``stage_full_residency_slot_plan``) and must NOT be re-derived per step.
        The per-step ``moe_offload_stage`` seam (which overwrites log2phy with
        only the current active subset, resetting inactive experts to -1) is a
        no-op here — restaging would corrupt the static mapping and make the
        captured gather read slot[-1] (MTE out-of-range) for any expert active in
        a later step but not the staging step.

        Regime B (num_slots < num_logical_experts) is the inverse: the mapping is
        data-dependent, full-residency staging is rejected by the working-set
        guard, and the per-step seam owns staging.
        """
        return int(self.config.num_slots) >= int(num_logical_experts)

    def should_use_b2_wave_prefill(
        self,
        *,
        layer_id: int,
        active_expert_count: int,
        is_prefill: bool,
    ) -> bool:
        """Gate for B2 wave-streamed prefill (capacity-bounded waves).

        True iff ALL of:
          * config.b2_wave_prefill is on (default off),
          * this is a prefill call (decode keeps the single-wave B1 path),
          * the layer is offloaded under fixed slots (resident layers untouched),
          * the call's distinct active expert set exceeds num_slots (otherwise B1
            single-wave already fits and is cheaper).

        When False the caller keeps its existing path (B1 single wave, or the
        fail-closed working-set guard). This predicate performs no device work and
        is pure-Python testable.
        """
        if not self.config.b2_wave_prefill:
            return False
        if not is_prefill:
            return False
        if not self.should_use_fixed_slot_plan_for_layer(int(layer_id)):
            return False
        return int(active_expert_count) > int(self.config.num_slots)

    def should_use_b2_pair_waves(
        self,
        *,
        layer_id: int,
        active_expert_count: int,
    ) -> bool:
        """Gate the exact capacity-bounded executor for any serving phase.

        Prefill, decode, and a scheduler-produced mixed batch all have the same
        correctness requirement when their distinct expert union exceeds the
        physical slot capacity: execute every routed pair exactly once across
        bounded waves and scatter-add it back to its original token.
        """
        if not self.config.b2_wave_prefill:
            return False
        if not self.should_use_fixed_slot_plan_for_layer(int(layer_id)):
            return False
        return int(active_expert_count) > int(self.config.num_slots)

    def memory_ledger(self) -> MoeOffloadMemoryLedger:
        cached = self._memory_ledger_cache
        if cached is not None:
            return cached
        original_bytes = sum(
            bytes_
            for layer_id, bytes_ in self._original_expert_weight_bytes_by_layer.items()
            if int(layer_id) not in self._released_original_weight_layers
        )
        # A later shared-pool implementation may expose the same physical bank
        # through multiple layer entries. Count storage identity, not references.
        unique_stage_banks = {
            id(bank): bank
            for banks in self._prefill_stage_banks.values()
            for bank in banks
        }
        unique_stage_mappings = {
            int(buffer.data_ptr()): buffer
            for buffers in self._prefill_stage_log2phy_buffers.values()
            for buffer in buffers
        }
        unique_full_layer_banks = {
            id(bank): bank for bank in self._full_layer_prefill_pool.values()
        }
        unique_full_layer_mappings = {
            int(buffer.data_ptr()): buffer
            for buffer in self._full_layer_prefill_log2phy_buffers.values()
        }
        ledger = MoeOffloadMemoryLedger(
            registered_layers=len(self._slot_banks),
            host_experts=len(self._host_store),
            original_expert_weight_bytes=original_bytes,
            host_store_bytes=self._host_store.total_bytes,
            slot_bank_bytes=sum(slot_bank.total_bytes for slot_bank in self._slot_banks.values()),
            prefill_stage_bank_count=len(unique_stage_banks),
            prefill_stage_bank_bytes=sum(
                bank.total_bytes for bank in unique_stage_banks.values()
            ),
            prefill_stage_mapping_bytes=sum(
                _tensor_nbytes(buffer)
                for buffer in unique_stage_mappings.values()
            ),
            full_layer_prefill_bank_count=len(unique_full_layer_banks),
            full_layer_prefill_bank_bytes=sum(
                bank.total_bytes for bank in unique_full_layer_banks.values()
            ),
            full_layer_prefill_mapping_bytes=sum(
                _tensor_nbytes(buffer)
                for buffer in unique_full_layer_mappings.values()
            ),
        )
        self._memory_ledger_cache = ledger
        return ledger

    def _invalidate_memory_ledger_cache(self) -> None:
        self._memory_ledger_cache = None

    def profiling_summary(self) -> dict[str, object]:
        total_seconds_by_event: dict[str, float] = {}
        for event in self._profile_events:
            total_seconds_by_event[event.name] = total_seconds_by_event.get(event.name, 0.0) + event.seconds
        return {
            "events": [event.to_jsonable() for event in self._profile_events],
            "total_seconds_by_event": total_seconds_by_event,
            "memory_ledger": self.memory_ledger().to_jsonable(),
        }

    def original_expert_weights_available_for_layer(self, layer_id: int) -> bool:
        return int(layer_id) not in self._released_original_weight_layers

    def decide_layered_path(
        self,
        *,
        layer_id: int,
        active_experts: tuple[int, ...],
        step_id: int | None = None,
        **_: object,
    ) -> MoeOffloadPathDecision:
        normalized_layer_id = int(layer_id)
        unique_active_experts = _dedupe_preserve_order(active_experts)
        fanout_threshold = int(self.config.effective_fanout_threshold)
        full_weights_available = self.original_expert_weights_available_for_layer(normalized_layer_id)
        slot_cache_ready = self.should_use_fixed_slot_plan_for_layer(normalized_layer_id) and (
            normalized_layer_id in self._slot_banks
        )

        if not self.should_use_layered_runtime:
            path = MoeOffloadDecisionPath.SLOT_CACHE_PATH
            reason = "layered_runtime_disabled"
        elif len(unique_active_experts) > fanout_threshold:
            if full_weights_available:
                path = MoeOffloadDecisionPath.FULL_WEIGHT_PATH
                reason = "high_fanout_full_weights_available"
            else:
                path = MoeOffloadDecisionPath.FAIL_CLOSED
                reason = "high_fanout_full_weights_unavailable"
        elif slot_cache_ready:
            path = MoeOffloadDecisionPath.SLOT_CACHE_PATH
            reason = "low_fanout_slot_cache_ready"
        else:
            path = MoeOffloadDecisionPath.FAIL_CLOSED
            reason = "low_fanout_slot_cache_unavailable"

        decision = MoeOffloadPathDecision(
            path=path,
            layer_id=normalized_layer_id,
            active_expert_count=len(unique_active_experts),
            active_experts=unique_active_experts,
            fanout_threshold=fanout_threshold,
            full_weights_available=full_weights_available,
            slot_cache_ready=slot_cache_ready,
            reason=reason,
        )
        self._record_profile_event(
            "layered_path_decision",
            layer_id=normalized_layer_id,
            start=perf_counter(),
            payload=decision.to_jsonable(),
        )
        return decision

    def plan_original_weight_release(
        self,
        *,
        expected_layer_ids: tuple[int, ...],
        default_path_preserved: bool,
        host_store_is_complete: bool | None = None,
        allow_retained_original_weights: bool = False,
    ) -> MoeExpertReleasePlan:
        normalized_layer_ids = tuple(int(layer_id) for layer_id in expected_layer_ids)
        blockers: list[str] = []
        if not normalized_layer_ids:
            blockers.append("no_expected_layers")

        missing_layers = tuple(layer_id for layer_id in normalized_layer_ids if layer_id not in self._slot_banks)
        if missing_layers:
            blockers.append(f"layers_not_registered:{list(missing_layers)}")

        if not default_path_preserved:
            blockers.append("default_path_not_preserved")
        if host_store_is_complete is False:
            blockers.append("host_store_not_marked_complete")

        host_store_report = self._host_store.validate_complete_layers(normalized_layer_ids)
        blockers.extend(host_store_report.blockers)
        if self.memory_ledger().original_expert_weights_retained and not allow_retained_original_weights:
            blockers.append("original_expert_weights_still_retained")

        layers_ready = () if blockers else normalized_layer_ids
        return MoeExpertReleasePlan(
            ready=not blockers,
            layers_ready=layers_ready,
            blockers=tuple(blockers),
        )

    def release_original_expert_weights_if_ready(
        self,
        layer: torch.nn.Module,
        *,
        default_path_preserved: bool = True,
    ) -> MoeExpertReleasePlan:
        """Opt-in partial release for a single non-resident layer after host store is complete."""
        if not self.config.release_original_expert_weights:
            return MoeExpertReleasePlan(
                ready=False,
                layers_ready=(),
                blockers=("release_original_expert_weights_disabled",),
            )
        if not self.should_use_fixed_slots:
            return MoeExpertReleasePlan(
                ready=False,
                layers_ready=(),
                blockers=("fixed_slots_disabled",),
            )

        layer_id = int(getattr(layer, "layer_id", -1))
        if layer_id < 0:
            return MoeExpertReleasePlan(
                ready=False,
                layers_ready=(),
                blockers=("invalid_layer_id",),
            )
        if self.is_resident_layer(layer_id):
            return MoeExpertReleasePlan(
                ready=False,
                layers_ready=(),
                blockers=(f"resident_layer:{layer_id}",),
            )
        if layer_id in self._released_original_weight_layers:
            return MoeExpertReleasePlan(ready=True, layers_ready=(layer_id,), blockers=())

        plan = self.plan_original_weight_release(
            expected_layer_ids=(layer_id,),
            default_path_preserved=default_path_preserved,
            allow_retained_original_weights=True,
        )
        if not plan.ready:
            return plan

        start = perf_counter()
        release_layer_original_expert_weights(layer)
        self._released_original_weight_layers.add(layer_id)
        self._original_expert_weight_bytes_by_layer[layer_id] = 0
        self._invalidate_memory_ledger_cache()
        self._record_profile_event(
            "release_original_expert_weights",
            layer_id=layer_id,
            start=start,
        )
        return MoeExpertReleasePlan(ready=True, layers_ready=(layer_id,), blockers=())

    def prepare_fixed_slot_plan(
        self,
        *,
        layer_id: int,
        active_experts: tuple[int, ...],
        num_logical_experts: int,
        device: torch.device,
        step_id: int | None = None,
        record_stage_profile: bool = False,
        **_: object,
    ) -> PreparedSlotWeights:
        if not self.should_use_fixed_slots:
            raise RuntimeError("fixed-slot plan requested while moe offload fixed slots are disabled")

        layer_id = int(layer_id)
        if self.is_resident_layer(layer_id):
            raise RuntimeError(
                f"fixed-slot plan must not run on resident layer {layer_id}; use original NPU expert weights"
            )
        unique_active_experts = _dedupe_preserve_order(active_experts)
        _validate_active_expert_ids(
            layer_id=layer_id,
            active_experts=unique_active_experts,
            num_logical_experts=num_logical_experts,
        )
        if len(unique_active_experts) > self.config.num_slots:
            raise RuntimeError(
                f"active expert working set size {len(unique_active_experts)} exceeds num_slots={self.config.num_slots}"
            )

        slot_bank = self._slot_banks.get(layer_id)
        if slot_bank is None:
            raise RuntimeError(f"layer {layer_id} is not registered for fixed-slot execution")

        # Complete the preceding execution before reserving this plan's misses.
        # A synchronous batch marks new slots LOADING until the batch is issued
        # after this loop; draining only after those reservations can mistake
        # unsubmitted work for an in-flight H2D transfer.
        if any(
            slot.state in (SlotState.LOADING, SlotState.COMPUTING)
            for slot in slot_bank.slots
        ):
            self._drain_slot_bank_inflight_work(slot_bank)

        step_id = (
            int(step_id)
            if step_id is not None and int(step_id) >= 0
            else int(next(self._step_counter))
        )
        collect_profile = bool(record_stage_profile) and bool(
            self.config.gmm_profile_path
            or _env_value("VLLM_ASCEND_MOE_GMM_PROFILE_PATH")
            or _env_value("VLLM_ASCEND_MOE_OFFLOAD_PROFILE_PATH")
        )
        profile_sample_rate = _decode_profile_sample_rate()
        collect_profile = bool(collect_profile) and _should_sample_decode_profile(
            step_id=step_id,
            sample_rate=profile_sample_rate,
        )
        collect_profile_details = bool(collect_profile) and _profile_expert_lists_enabled()
        hit_experts: list[int] = []
        miss_experts: list[int] = []
        h2d_bytes = 0
        load_sync_ms = 0.0
        sync_loads = []
        _n_hits = 0
        _n_misses = 0
        stage_start = perf_counter() if collect_profile else 0.0
        expert_weight_bytes = (
            int(self._expert_weight_bytes_by_layer.get(layer_id, 0))
            if collect_profile
            else 0
        )
        for expert_id in unique_active_experts:
            key = ExpertKey(layer_id, int(expert_id))
            slot = slot_bank.lookup(key)
            if slot is not None and slot.state != SlotState.READY:
                self._drain_slot_bank_inflight_work(slot_bank)
                slot = slot_bank.lookup(key)
            if slot is not None and slot.state == SlotState.READY:
                slot.last_used_step = int(step_id)
                _n_hits += 1
                if collect_profile_details:
                    hit_experts.append(int(expert_id))
                continue

            _n_misses += 1
            try:
                slot = self._allocate_slot_with_loading_fallback(
                    slot_bank,
                    key,
                    step_id=step_id,
                )
            except RuntimeError as exc:
                raise RuntimeError(
                    f"failed to allocate expert slot for layer {layer_id} "
                    f"expert {int(expert_id)} with num_slots={self.config.num_slots}; "
                    f"async_load={self.config.async_load}. If all slots are LOADING "
                    "or COMPUTING, wait for the transfer/compute stage to finish "
                    "before eviction or increase startup slot capacity."
                ) from exc
            bundle = self._host_store.get(layer_id, int(expert_id))
            if collect_profile:
                if expert_weight_bytes <= 0:
                    expert_weight_bytes = int(
                        self.estimate_expert_weight_bytes(
                            layer_id=layer_id,
                            expert_id=int(expert_id),
                        )
                    )
                h2d_bytes += int(expert_weight_bytes)
            if collect_profile_details:
                miss_experts.append(int(expert_id))
            sync_loads.append((bundle, slot))

        if sync_loads:
            load_start = perf_counter() if collect_profile else 0.0
            try:
                self._transfer_engine.load_many_sync(
                    sync_loads,
                    validate_layout=False,
                )
            except Exception:
                self._release_allocated_loads(slot_bank, sync_loads)
                raise
            if collect_profile:
                load_sync_ms += (perf_counter() - load_start) * 1000.0

        mapping_start = perf_counter() if collect_profile else 0.0
        mapping = ExpertSlotMapping.from_slot_bank(
            layer_id=layer_id,
            active_experts=unique_active_experts,
            num_logical_experts=num_logical_experts,
            slot_bank=slot_bank,
            device=device,
        )
        mapping_ms = (perf_counter() - mapping_start) * 1000.0 if collect_profile else 0.0
        if collect_profile:
            payload = {
                "n_active": int(len(unique_active_experts)),
                "n_hits": int(_n_hits),
                "n_misses": int(_n_misses),
                "hit_rate": (
                    round(float(_n_hits) / float(len(unique_active_experts)), 6)
                    if unique_active_experts
                    else 0.0
                ),
                "h2d_bytes": int(h2d_bytes),
                "stage_ms": round(
                    (perf_counter() - stage_start) * 1000.0,
                    3,
                ),
                "load_sync_ms": round(float(load_sync_ms), 3),
                "mapping_ms": round(float(mapping_ms), 3),
                "step_id": int(step_id),
                "num_slots": int(self.config.num_slots),
                "profile_sample_rate": int(profile_sample_rate),
            }
            if collect_profile_details:
                payload.update(
                    {
                        "active_experts": [int(e) for e in unique_active_experts],
                        "hit_experts": hit_experts,
                        "miss_experts": miss_experts,
                    }
                )
            else:
                payload["expert_lists"] = "omitted"
            self._record_profile_event(
                "decode_fixed_slot_stage",
                layer_id=layer_id,
                start=mapping_start,
                payload=payload,
            )
        return PreparedSlotWeights.from_slot_bank(slot_bank=slot_bank, mapping=mapping)

    def prepare_fixed_slot_plan_into_log2phy(
        self,
        *,
        layer_id: int,
        active_experts: tuple[int, ...],
        num_logical_experts: int,
        log2phy: torch.Tensor,
        step_id: int | None = None,
        record_stage_profile: bool = False,
    ) -> PreparedSlotWeights:
        """Stage decode slots and write the mapping directly into ``log2phy``.

        This is the decode hot path: unlike ``prepare_fixed_slot_plan`` it avoids
        allocating a temporary logical->physical tensor only to copy it into the
        persistent ACLGraph-visible buffer.
        """
        if not self.should_use_fixed_slots:
            raise RuntimeError("fixed-slot plan requested while moe offload fixed slots are disabled")

        layer_id = int(layer_id)
        if self.is_resident_layer(layer_id):
            raise RuntimeError(
                f"fixed-slot plan must not run on resident layer {layer_id}; use original NPU expert weights"
            )
        unique_active_experts = _dedupe_preserve_order(active_experts)
        _validate_active_expert_ids(
            layer_id=layer_id,
            active_experts=unique_active_experts,
            num_logical_experts=num_logical_experts,
        )
        if len(unique_active_experts) > self.config.num_slots:
            raise RuntimeError(
                f"active expert working set size {len(unique_active_experts)} exceeds num_slots={self.config.num_slots}"
            )

        if int(log2phy.numel()) != int(num_logical_experts):
            raise RuntimeError(
                f"log2phy buffer for layer {layer_id} has size {int(log2phy.numel())}, "
                f"expected {int(num_logical_experts)}"
            )

        slot_bank = self._slot_banks.get(layer_id)
        if slot_bank is None:
            raise RuntimeError(f"layer {layer_id} is not registered for fixed-slot execution")

        # Complete the preceding execution before reserving this plan's misses.
        # A synchronous batch marks new slots LOADING until the batch is issued
        # after this loop; draining only after those reservations can mistake
        # unsubmitted work for an in-flight H2D transfer.
        if any(
            slot.state in (SlotState.LOADING, SlotState.COMPUTING)
            for slot in slot_bank.slots
        ):
            self._drain_slot_bank_inflight_work(slot_bank)

        step_id = (
            int(step_id)
            if step_id is not None and int(step_id) >= 0
            else int(next(self._step_counter))
        )
        collect_profile = bool(record_stage_profile) and bool(
            self.config.gmm_profile_path
            or _env_value("VLLM_ASCEND_MOE_GMM_PROFILE_PATH")
            or _env_value("VLLM_ASCEND_MOE_OFFLOAD_PROFILE_PATH")
        )
        profile_sample_rate = _decode_profile_sample_rate()
        collect_profile = bool(collect_profile) and _should_sample_decode_profile(
            step_id=step_id,
            sample_rate=profile_sample_rate,
        )
        collect_profile_details = bool(collect_profile) and _profile_expert_lists_enabled()
        hit_experts: list[int] = []
        miss_experts: list[int] = []
        h2d_bytes = 0
        load_sync_ms = 0.0
        load_enqueue_ms = 0.0
        ready_wait_ms = 0.0
        ready_event = None
        consumer_dependency_installed = False
        async_loads = []
        sync_loads = []
        _n_hits = 0
        _n_misses = 0
        stage_start = perf_counter() if collect_profile else 0.0
        expert_weight_bytes = (
            int(self._expert_weight_bytes_by_layer.get(layer_id, 0))
            if collect_profile
            else 0
        )

        active_slot_ids: list[int] = []
        slot_to_expert: list[int | None] = [None] * len(slot_bank.slots)
        for expert_id in unique_active_experts:
            key = ExpertKey(layer_id, int(expert_id))
            slot = slot_bank.lookup(key)
            if slot is not None and slot.state != SlotState.READY:
                self._drain_slot_bank_inflight_work(slot_bank)
                slot = slot_bank.lookup(key)
            if slot is not None and slot.state == SlotState.READY:
                slot.last_used_step = int(step_id)
                _n_hits += 1
                if collect_profile_details:
                    hit_experts.append(int(expert_id))
            else:
                _n_misses += 1
                try:
                    slot = self._allocate_slot_with_loading_fallback(
                        slot_bank,
                        key,
                        step_id=step_id,
                    )
                except RuntimeError as exc:
                    raise RuntimeError(
                        f"failed to allocate expert slot for layer {layer_id} "
                        f"expert {int(expert_id)} with num_slots={self.config.num_slots}; "
                        f"async_load={self.config.async_load}. If all slots are LOADING "
                        "or COMPUTING, wait for the transfer/compute stage to finish "
                        "before eviction or increase startup slot capacity."
                    ) from exc
                bundle = self._host_store.get(layer_id, int(expert_id))
                if collect_profile:
                    if expert_weight_bytes <= 0:
                        expert_weight_bytes = int(
                            self.estimate_expert_weight_bytes(
                                layer_id=layer_id,
                                expert_id=int(expert_id),
                            )
                        )
                    h2d_bytes += int(expert_weight_bytes)
                if collect_profile_details:
                    miss_experts.append(int(expert_id))
                if self.config.async_load:
                    async_loads.append((bundle, slot))
                else:
                    sync_loads.append((bundle, slot))
            active_slot_ids.append(int(slot.slot_id))

        if sync_loads:
            load_start = perf_counter() if collect_profile else 0.0
            try:
                self._transfer_engine.load_many_sync(
                    sync_loads,
                    validate_layout=False,
                )
            except Exception:
                self._release_allocated_loads(slot_bank, sync_loads)
                raise
            if collect_profile:
                load_sync_ms += (perf_counter() - load_start) * 1000.0

        if async_loads:
            load_start = perf_counter() if collect_profile else 0.0
            try:
                ready_event = self._transfer_engine.load_many_async(
                    async_loads,
                    record_event=True,
                    validate_layout=False,
                )
            except Exception:
                self._release_allocated_loads(slot_bank, async_loads)
                raise
            if collect_profile:
                load_enqueue_ms = (perf_counter() - load_start) * 1000.0

        # Install the transfer dependency on the exact consumer stream before
        # publishing the logical mapping. A mapping must never name a slot whose
        # copy is not yet ordered before its consumer.
        if ready_event is not None:
            wait_start = perf_counter() if collect_profile else 0.0
            self._wait_transfer_event(ready_event)
            consumer_dependency_installed = True
            if collect_profile:
                ready_wait_ms = (perf_counter() - wait_start) * 1000.0
                load_sync_ms += load_enqueue_ms + ready_wait_ms

        mapping_start = perf_counter() if collect_profile else 0.0
        active_slot_ids_tuple = tuple(active_slot_ids)
        log2phy_update_experts, log2phy_update_slots = (
            self._changed_log2phy_entries(
                layer_id=layer_id,
                expert_ids=unique_active_experts,
                slot_ids=active_slot_ids_tuple,
            )
        )
        if log2phy_update_experts:
            _update_log2phy_entries(
                log2phy,
                expert_ids=log2phy_update_experts,
                slot_ids=log2phy_update_slots,
            )
            self._remember_log2phy_entries(
                layer_id=layer_id,
                expert_ids=log2phy_update_experts,
                slot_ids=log2phy_update_slots,
            )
        for expert_id, slot_id in zip(unique_active_experts, active_slot_ids, strict=True):
            slot_to_expert[int(slot_id)] = int(expert_id)
        mapping = ExpertSlotMapping(
            layer_id=layer_id,
            active_experts=unique_active_experts,
            logical_to_physical=log2phy,
            slot_to_expert=tuple(slot_to_expert),
            active_slot_ids=active_slot_ids_tuple,
        )  # log2phy updated in-place above; always valid on this path
        self._active_slot_ids_by_layer[layer_id] = active_slot_ids_tuple
        mapping_ms = (perf_counter() - mapping_start) * 1000.0 if collect_profile else 0.0
        if collect_profile:
            if _n_misses <= 0:
                stage_mode = "main_slot_hit"
            elif async_loads:
                stage_mode = "async_decode_load_many"
            else:
                stage_mode = "sync_decode_load"
            payload = {
                "n_active": int(len(unique_active_experts)),
                "n_hits": int(_n_hits),
                "n_misses": int(_n_misses),
                "hit_rate": (
                    round(float(_n_hits) / float(len(unique_active_experts)), 6)
                    if unique_active_experts
                    else 0.0
                ),
                "h2d_bytes": int(h2d_bytes),
                "stage_ms": round(
                    (perf_counter() - stage_start) * 1000.0,
                    3,
                ),
                "load_sync_ms": round(float(load_sync_ms), 3),
                "load_enqueue_ms": round(float(load_enqueue_ms), 3),
                "ready_wait_ms": round(float(ready_wait_ms), 3),
                "mapping_ms": round(float(mapping_ms), 3),
                "step_id": int(step_id),
                "num_slots": int(self.config.num_slots),
                "mapping_mode": "persistent_log2phy",
                "log2phy_update_count": int(len(log2phy_update_experts)),
                "stage_mode": stage_mode,
                "consumer_dependency_installed": (
                    ready_event is None or consumer_dependency_installed
                ),
                "mapping_published_after_ready": True,
                "profile_sample_rate": int(profile_sample_rate),
            }
            if collect_profile_details:
                payload.update(
                    {
                        "active_experts": [int(e) for e in unique_active_experts],
                        "hit_experts": hit_experts,
                        "miss_experts": miss_experts,
                    }
                )
            else:
                payload["expert_lists"] = "omitted"
            self._record_profile_event(
                "decode_fixed_slot_stage",
                layer_id=layer_id,
                start=mapping_start,
                payload=payload,
            )
        return PreparedSlotWeights.from_slot_bank(slot_bank=slot_bank, mapping=mapping)

    def _changed_log2phy_entries(
        self,
        *,
        layer_id: int,
        expert_ids: tuple[int, ...],
        slot_ids: tuple[int, ...],
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        if len(expert_ids) != len(slot_ids):
            raise ValueError(
                f"expert_ids and slot_ids length mismatch: {len(expert_ids)} != {len(slot_ids)}"
            )
        cached = self._log2phy_slot_by_expert.setdefault(int(layer_id), {})
        update_experts: list[int] = []
        update_slots: list[int] = []
        for expert_id, slot_id in zip(expert_ids, slot_ids, strict=True):
            expert_id = int(expert_id)
            slot_id = int(slot_id)
            if cached.get(expert_id) == slot_id:
                continue
            update_experts.append(expert_id)
            update_slots.append(slot_id)
        return tuple(update_experts), tuple(update_slots)

    def _remember_log2phy_entries(
        self,
        *,
        layer_id: int,
        expert_ids: tuple[int, ...],
        slot_ids: tuple[int, ...],
    ) -> None:
        cached = self._log2phy_slot_by_expert.setdefault(int(layer_id), {})
        for expert_id, slot_id in zip(expert_ids, slot_ids, strict=True):
            cached[int(expert_id)] = int(slot_id)

    def prepare_ready_slot_plan(
        self,
        *,
        layer_id: int,
        active_experts: tuple[int, ...],
        num_logical_experts: int,
        device: torch.device,
        step_id: int | None = None,
        build_log2phy: bool = True,
        **_: object,
    ) -> PreparedSlotWeights:
        """Build a zero-copy plan for experts already READY in the main slot bank.

        This is the Prefill B2 hit-only fast path: unlike
        ``prepare_prefill_stage_plan`` it does not copy READY slot contents into a
        temporary wave bank, and unlike ``prepare_fixed_slot_plan`` it never
        allocates or loads missing experts. Any miss fails closed so callers do
        not accidentally turn a hit wave into synchronous H2D.
        """
        if not self.should_use_fixed_slots:
            raise RuntimeError("ready-slot plan requested while fixed slots are disabled")

        layer_id = int(layer_id)
        if self.is_resident_layer(layer_id):
            raise RuntimeError(
                f"ready-slot plan must not run on resident layer {layer_id}; use original NPU expert weights"
            )
        unique_active_experts = _dedupe_preserve_order(active_experts)
        _validate_active_expert_ids(
            layer_id=layer_id,
            active_experts=unique_active_experts,
            num_logical_experts=num_logical_experts,
        )
        if len(unique_active_experts) > self.config.num_slots:
            raise RuntimeError(
                f"ready expert working set size {len(unique_active_experts)} exceeds num_slots={self.config.num_slots}"
            )

        slot_bank = self._slot_banks.get(layer_id)
        if slot_bank is None:
            raise RuntimeError(f"layer {layer_id} is not registered for fixed-slot execution")

        step_id = (
            int(step_id)
            if step_id is not None and int(step_id) >= 0
            else int(next(self._step_counter))
        )
        missing_experts: list[int] = []
        slot_to_expert: list[int | None] = [None] * len(slot_bank.slots)
        active_slot_ids: list[int] = []
        for expert_id in unique_active_experts:
            key = ExpertKey(layer_id, int(expert_id))
            slot = slot_bank.lookup(key)
            if slot is None or slot.state != SlotState.READY:
                missing_experts.append(int(expert_id))
                continue
            slot.last_used_step = int(step_id)
            active_slot_ids.append(int(slot.slot_id))
        if missing_experts:
            raise RuntimeError(
                f"ready-slot plan requested for non-ready experts in layer {layer_id}: {missing_experts}"
            )

        if build_log2phy:
            mapping = ExpertSlotMapping.from_slot_bank(
                layer_id=layer_id,
                active_experts=unique_active_experts,
                num_logical_experts=num_logical_experts,
                slot_bank=slot_bank,
                device=device,
            )
        else:
            for slot in slot_bank.slots:
                if slot.expert_key is not None:
                    slot_to_expert[int(slot.slot_id)] = int(slot.expert_key.expert_id)
            mapping = ExpertSlotMapping(
                layer_id=layer_id,
                active_experts=unique_active_experts,
                logical_to_physical=torch.empty(
                    0,
                    dtype=torch.int32,
                    device=device,
                ),
                slot_to_expert=tuple(slot_to_expert),
                active_slot_ids=tuple(active_slot_ids),
                log2phy_valid=False,
            )
        return PreparedSlotWeights.from_slot_bank(slot_bank=slot_bank, mapping=mapping)

    def prepare_prefill_stage_plan(
        self,
        *,
        layer_id: int,
        active_experts: tuple[int, ...],
        num_logical_experts: int,
        device: torch.device,
        buffer_index: int,
        async_load: bool,
        wait_event=None,
        step_id: int | None = None,
        build_log2phy: bool = True,
        known_miss: bool = False,
    ) -> tuple[PreparedSlotWeights, object | None, dict[str, object]]:
        """Stage one B2 prefill wave into a dedicated temporary slot bank.

        These banks are separate from the decode slot cache, so prefetching wave
        k+1 cannot overwrite the fixed slots still used by wave k compute.
        """
        if not self.should_use_fixed_slots:
            raise RuntimeError("prefill stage plan requested while fixed slots are disabled")

        self._validate_prefill_buffer_index(int(buffer_index))

        layer_id = int(layer_id)
        if self.is_resident_layer(layer_id):
            raise RuntimeError(
                f"prefill stage plan must not run on resident layer {layer_id}"
            )
        unique_active_experts = _dedupe_preserve_order(active_experts)
        _validate_active_expert_ids(
            layer_id=layer_id,
            active_experts=unique_active_experts,
            num_logical_experts=num_logical_experts,
        )
        if len(unique_active_experts) > self.config.num_slots:
            raise RuntimeError(
                f"prefill wave size {len(unique_active_experts)} exceeds num_slots={self.config.num_slots}"
            )

        src_bank = self._slot_banks.get(layer_id)
        if src_bank is None:
            raise RuntimeError(f"layer {layer_id} is not registered for fixed-slot execution")
        stage_bank = self._get_prefill_stage_bank(
            layer_id=layer_id,
            buffer_index=int(buffer_index),
            template_bank=src_bank,
        )
        if wait_event is None:
            wait_event = self.prefill_stage_buffer_release_event(
                layer_id=layer_id,
                buffer_index=int(buffer_index),
            )
        log2phy = self._get_prefill_stage_log2phy_buffer(
            layer_id=layer_id,
            buffer_index=int(buffer_index),
            num_logical_experts=num_logical_experts,
            device=device,
        )
        step_id = (
            int(step_id)
            if step_id is not None and int(step_id) >= 0
            else int(next(self._step_counter))
        )

        hit_experts: list[int] = []
        miss_experts: list[int] = []
        active_slot_ids: list[int] = []
        async_loads = []
        sync_loads = []
        queued_async_load = False
        collect_profile = bool(
            self.config.gmm_profile_path
            or _env_value("VLLM_ASCEND_MOE_GMM_PROFILE_PATH")
            or _env_value("VLLM_ASCEND_MOE_OFFLOAD_PROFILE_PATH")
        )
        profile_ms: dict[str, float] = {}

        def _mark(name: str, start: float) -> None:
            if collect_profile:
                profile_ms[name] = profile_ms.get(name, 0.0) + (
                    perf_counter() - start
                ) * 1000.0

        build_log2phy = bool(build_log2phy)
        known_miss = bool(known_miss)
        if build_log2phy:
            timer = perf_counter() if collect_profile else 0.0
            log2phy.fill_(-1)
            _mark("log2phy_fill", timer)
        for slot_id, expert_id in enumerate(unique_active_experts):
            key = ExpertKey(layer_id, int(expert_id))
            timer = perf_counter() if collect_profile else 0.0
            if known_miss:
                stage_slot = stage_bank.assign_transient_slot(
                    slot_id,
                    key,
                    step_id=int(step_id),
                )
            else:
                stage_slot = stage_bank.assign_slot(
                    slot_id,
                    key,
                    step_id=int(step_id),
                )
            if build_log2phy:
                log2phy[int(expert_id)] = int(slot_id)
            active_slot_ids.append(int(slot_id))
            _mark("assign_and_map", timer)
            timer = perf_counter() if collect_profile else 0.0
            src_slot = None if known_miss else src_bank.lookup(key)
            bundle = None
            if src_slot is not None and src_slot.state == SlotState.READY:
                hit_experts.append(int(expert_id))
                bundle = src_slot.as_bundle()
            else:
                miss_experts.append(int(expert_id))
                bundle = self._host_store.get(layer_id, int(expert_id))
            _mark("resolve_bundle", timer)

            if async_load:
                async_loads.append((bundle, stage_slot))
                queued_async_load = True
            else:
                sync_loads.append((bundle, stage_slot))

        if not async_load and sync_loads:
            timer = perf_counter() if collect_profile else 0.0
            try:
                self._transfer_engine.load_many_sync(
                    sync_loads,
                    validate_layout=False,
                )
            except Exception:
                self._release_allocated_loads(stage_bank, sync_loads)
                raise
            _mark("load_enqueue", timer)

        ready_event = None
        if async_load and async_loads:
            timer = perf_counter() if collect_profile else 0.0
            try:
                ready_event = self._transfer_engine.load_many_async(
                    async_loads,
                    wait_event=wait_event,
                    record_event=True,
                    validate_layout=False,
                )
            except Exception:
                self._release_allocated_loads(stage_bank, async_loads)
                raise
            _mark("load_enqueue", timer)

        timer = perf_counter() if collect_profile else 0.0
        for slot_id in range(len(unique_active_experts), len(stage_bank.slots)):
            stage_bank.clear_slot(slot_id)
        _mark("clear_tail", timer)

        timer = perf_counter() if collect_profile else 0.0
        if async_load and not queued_async_load:
            ready_event = wait_event
        _mark("ready_event", timer)

        timer = perf_counter() if collect_profile else 0.0
        slot_to_expert: list[int | None] = [None] * len(stage_bank.slots)
        for slot_id, expert_id in enumerate(unique_active_experts):
            slot_to_expert[int(slot_id)] = int(expert_id)
        mapping = ExpertSlotMapping(
            layer_id=layer_id,
            active_experts=unique_active_experts,
            logical_to_physical=log2phy,
            slot_to_expert=tuple(slot_to_expert),
            active_slot_ids=tuple(active_slot_ids),
            log2phy_valid=build_log2phy,
        )
        prepared = PreparedSlotWeights.from_slot_bank(
            slot_bank=stage_bank,
            mapping=mapping,
        )
        _mark("mapping", timer)
        timer = perf_counter() if collect_profile else 0.0
        expert_weight_bytes = int(self._expert_weight_bytes_by_layer.get(layer_id, 0))
        if expert_weight_bytes <= 0 and miss_experts:
            expert_weight_bytes = int(
                self.estimate_expert_weight_bytes(
                    layer_id=layer_id,
                    expert_id=miss_experts[0],
                )
            )
        slot_expert_weight_bytes = int(
            self._slot_expert_weight_bytes_by_layer.get(layer_id, 0)
        )
        if slot_expert_weight_bytes <= 0 and hit_experts:
            slot_expert_weight_bytes = int(
                self.estimate_slot_expert_weight_bytes(
                    layer_id=layer_id,
                    expert_id=hit_experts[0],
                )
            )
        _mark("payload_bytes", timer)
        payload = {
            "buffer_index": int(buffer_index),
            "hit_experts": hit_experts,
            "miss_experts": miss_experts,
            "h2d_bytes": int(expert_weight_bytes * len(miss_experts)),
            "d2d_bytes": int(slot_expert_weight_bytes * len(hit_experts)),
            "log2phy_built": build_log2phy,
            "known_miss": known_miss,
        }
        if collect_profile:
            payload["profile_ms"] = {
                str(name): round(float(value), 3)
                for name, value in profile_ms.items()
            }
        return prepared, ready_event, payload

    def prepare_full_layer_prefill_plan(
        self,
        *,
        layer_id: int,
        num_logical_experts: int,
        device: torch.device,
        step_id: int | None = None,
    ) -> tuple[PreparedSlotWeights, dict[str, object]]:
        """Synchronously stage one complete expert layer into a shared NPU bank.

        This is a correctness reference for capacity-overflow prefill. It keeps
        the native top-k shape intact and reuses one layout-scoped bank across
        serial MoE layers, so only one complete layer is resident transiently.
        """
        if not self.should_use_fixed_slots:
            raise RuntimeError(
                "full-layer prefill plan requested while fixed slots are disabled"
            )

        layer_id = int(layer_id)
        num_logical_experts = int(num_logical_experts)
        if self.is_resident_layer(layer_id):
            raise RuntimeError(
                f"full-layer prefill plan must not run on resident layer {layer_id}"
            )
        if num_logical_experts <= 0:
            raise ValueError("num_logical_experts must be greater than 0")

        template_bank = self._slot_banks.get(layer_id)
        if template_bank is None:
            raise RuntimeError(
                f"layer {layer_id} is not registered for fixed-slot execution"
            )
        layer_buffer = self._host_store.get_layer_buffer(layer_id)
        if int(layer_buffer.num_experts) != num_logical_experts:
            raise RuntimeError(
                f"host expert count for layer {layer_id} is "
                f"{layer_buffer.num_experts}, expected {num_logical_experts}"
            )

        pool_key = self._full_layer_prefill_pool_key(
            template_bank,
            num_logical_experts=num_logical_experts,
        )
        pool_reused = pool_key in self._full_layer_prefill_pool
        bank = self._get_full_layer_prefill_bank(
            layer_id=layer_id,
            template_bank=template_bank,
            num_logical_experts=num_logical_experts,
            pool_key=pool_key,
        )
        log2phy = self._get_full_layer_prefill_log2phy_buffer(
            pool_key=pool_key,
            num_logical_experts=num_logical_experts,
            device=device,
        )
        step_id = (
            int(step_id)
            if step_id is not None and int(step_id) >= 0
            else int(next(self._step_counter))
        )

        for expert_id in range(num_logical_experts):
            bank.assign_transient_slot(
                expert_id,
                ExpertKey(layer_id, expert_id),
                step_id=step_id,
            )

        stage_start = perf_counter()
        try:
            bank.w13_slots.copy_(layer_buffer.w13, non_blocking=False)
            bank.w2_slots.copy_(layer_buffer.w2, non_blocking=False)
        except Exception:
            for slot_id in range(num_logical_experts):
                bank.clear_slot(slot_id, force=True)
            raise
        for slot_id in range(num_logical_experts):
            bank.mark_ready(slot_id)
        stage_ms = (perf_counter() - stage_start) * 1000.0

        all_experts = tuple(range(num_logical_experts))
        mapping = ExpertSlotMapping(
            layer_id=layer_id,
            active_experts=all_experts,
            logical_to_physical=log2phy,
            slot_to_expert=all_experts,
            active_slot_ids=all_experts,
        )
        prepared = PreparedSlotWeights.from_slot_bank(
            slot_bank=bank,
            mapping=mapping,
        )
        payload = {
            "execution_mode": "reference_full_layer",
            "num_logical_experts": num_logical_experts,
            "physical_expert_count": int(prepared.physical_expert_count),
            "h2d_bytes": int(layer_buffer.total_bytes),
            "stage_ms": round(stage_ms, 3),
            "pool_reused": bool(pool_reused),
        }
        return prepared, payload

    def wait_prefill_stage_plan(self, ready_event) -> None:
        self._wait_transfer_event(ready_event)

    def _wait_transfer_event(self, ready_event) -> None:
        if ready_event is None:
            return
        import torch

        install_dependency = getattr(
            ready_event,
            "install_consumer_dependency",
            None,
        )
        if callable(install_dependency):
            install_dependency(torch.npu.current_stream())
            return

        event = getattr(ready_event, "event", ready_event)
        has_ready_handle = hasattr(ready_event, "mark_ready")
        if event is not None:
            try:
                torch.npu.current_stream().wait_event(event)
            except TypeError:
                if has_ready_handle or not isinstance(event, (str, bytes)):
                    raise
        if has_ready_handle:
            ready_event.mark_ready()

    def _get_prefill_stage_bank(
        self,
        *,
        layer_id: int,
        buffer_index: int,
        template_bank: ExpertSlotBank,
    ) -> ExpertSlotBank:
        self._validate_prefill_buffer_index(buffer_index)
        pool_key = self._prefill_stage_pool_key(template_bank)
        banks = self._prefill_stage_pool.setdefault(pool_key, [])
        self._prefill_stage_banks[int(layer_id)] = banks
        while len(banks) <= int(buffer_index):
            allocation_start = perf_counter()
            hbm_before = self._npu_hbm_snapshot(template_bank.w13_slots.device)
            banks.append(
                ExpertSlotBank(
                    len(template_bank.slots),
                    tuple(int(dim) for dim in template_bank.w13_slots.shape[1:]),
                    tuple(int(dim) for dim in template_bank.w2_slots.shape[1:]),
                    dtype=template_bank.w13_slots.dtype,
                    device=template_bank.w13_slots.device,
                )
            )
            self._invalidate_memory_ledger_cache()
            hbm_after = self._npu_hbm_snapshot(template_bank.w13_slots.device)
            self._record_profile_event(
                "prefill_stage_bank_allocate",
                layer_id=int(layer_id),
                start=allocation_start,
                payload={
                    "buffer_index": int(len(banks) - 1),
                    "bank_bytes": int(banks[-1].total_bytes),
                    "hbm_before": hbm_before,
                    "hbm_after": hbm_after,
                    "allocated_delta_bytes": int(
                        hbm_after.get("allocated_bytes", 0)
                        - hbm_before.get("allocated_bytes", 0)
                    ),
                    "free_delta_bytes": int(
                        hbm_after.get("free_bytes", 0)
                        - hbm_before.get("free_bytes", 0)
                    ),
                },
            )
        return banks[int(buffer_index)]

    def _full_layer_prefill_pool_key(
        self,
        template_bank: ExpertSlotBank,
        *,
        num_logical_experts: int,
    ) -> tuple[object, ...]:
        return (
            str(template_bank.w13_slots.device),
            str(template_bank.w13_slots.dtype),
            int(num_logical_experts),
            tuple(int(dim) for dim in template_bank.w13_slots.shape[1:]),
            tuple(int(dim) for dim in template_bank.w2_slots.shape[1:]),
            tuple(int(stride) for stride in template_bank.w13_slots[0].stride()),
            tuple(int(stride) for stride in template_bank.w2_slots[0].stride()),
        )

    def _get_full_layer_prefill_bank(
        self,
        *,
        layer_id: int,
        template_bank: ExpertSlotBank,
        num_logical_experts: int,
        pool_key: tuple[object, ...],
    ) -> ExpertSlotBank:
        bank = self._full_layer_prefill_pool.get(pool_key)
        if bank is not None:
            return bank

        allocation_start = perf_counter()
        hbm_before = self._npu_hbm_snapshot(template_bank.w13_slots.device)
        bank = ExpertSlotBank(
            int(num_logical_experts),
            tuple(int(dim) for dim in template_bank.w13_slots.shape[1:]),
            tuple(int(dim) for dim in template_bank.w2_slots.shape[1:]),
            dtype=template_bank.w13_slots.dtype,
            device=template_bank.w13_slots.device,
        )
        self._full_layer_prefill_pool[pool_key] = bank
        self._invalidate_memory_ledger_cache()
        hbm_after = self._npu_hbm_snapshot(template_bank.w13_slots.device)
        self._record_profile_event(
            "full_layer_prefill_bank_allocate",
            layer_id=int(layer_id),
            start=allocation_start,
            payload={
                "bank_bytes": int(bank.total_bytes),
                "physical_expert_count": int(num_logical_experts),
                "hbm_before": hbm_before,
                "hbm_after": hbm_after,
            },
        )
        return bank

    def _get_full_layer_prefill_log2phy_buffer(
        self,
        *,
        pool_key: tuple[object, ...],
        num_logical_experts: int,
        device: torch.device,
    ) -> torch.Tensor:
        buffer = self._full_layer_prefill_log2phy_buffers.get(pool_key)
        if buffer is None:
            buffer = torch.arange(
                int(num_logical_experts),
                dtype=torch.int32,
                device=device,
            )
            self._full_layer_prefill_log2phy_buffers[pool_key] = buffer
            self._invalidate_memory_ledger_cache()
        if int(buffer.numel()) != int(num_logical_experts):
            raise RuntimeError(
                "full-layer prefill identity mapping size changed for a shared layout"
            )
        if buffer.device != device:
            raise RuntimeError(
                f"full-layer prefill identity mapping is on {buffer.device}, "
                f"expected {device}"
            )
        return buffer

    def _prefill_stage_pool_key(
        self,
        template_bank: ExpertSlotBank,
    ) -> tuple[object, ...]:
        return (
            str(template_bank.w13_slots.device),
            str(template_bank.w13_slots.dtype),
            len(template_bank.slots),
            tuple(int(dim) for dim in template_bank.w13_slots.shape[1:]),
            tuple(int(dim) for dim in template_bank.w2_slots.shape[1:]),
            tuple(int(stride) for stride in template_bank.w13_slots.stride()),
            tuple(int(stride) for stride in template_bank.w2_slots.stride()),
        )

    def prefill_stage_buffer_release_event(
        self,
        *,
        layer_id: int,
        buffer_index: int,
    ) -> object | None:
        self._validate_prefill_buffer_index(buffer_index)
        template_bank = self._slot_banks.get(int(layer_id))
        if template_bank is None:
            return None
        key = self._prefill_stage_pool_key(template_bank)
        return self._prefill_stage_pool_release_events.get(
            (key, int(buffer_index))
        )

    def remember_prefill_stage_buffer_release(
        self,
        *,
        layer_id: int,
        buffer_index: int,
        event: object | None,
    ) -> None:
        self._validate_prefill_buffer_index(buffer_index)
        if event is None:
            return
        template_bank = self._slot_banks.get(int(layer_id))
        if template_bank is None:
            return
        key = self._prefill_stage_pool_key(template_bank)
        self._prefill_stage_pool_release_events[(key, int(buffer_index))] = event

    def _get_prefill_stage_log2phy_buffer(
        self,
        *,
        layer_id: int,
        buffer_index: int,
        num_logical_experts: int,
        device: torch.device,
    ) -> torch.Tensor:
        self._validate_prefill_buffer_index(buffer_index)
        if num_logical_experts <= 0:
            raise ValueError("num_logical_experts must be greater than 0")
        buffers = self._prefill_stage_log2phy_buffers.setdefault(int(layer_id), [])
        while len(buffers) <= int(buffer_index):
            buffers.append(
                torch.empty(
                    (int(num_logical_experts),),
                    dtype=torch.int32,
                    device=device,
                )
            )
            self._invalidate_memory_ledger_cache()
        buf = buffers[int(buffer_index)]
        if int(buf.numel()) != int(num_logical_experts):
            raise RuntimeError(
                f"prefill log2phy buffer for layer {layer_id} buffer "
                f"{buffer_index} has size {int(buf.numel())}, expected "
                f"{int(num_logical_experts)}"
            )
        if buf.device != device:
            raise RuntimeError(
                f"prefill log2phy buffer for layer {layer_id} buffer "
                f"{buffer_index} is on {buf.device}, expected {device}"
            )
        return buf

    def _validate_prefill_buffer_index(self, buffer_index: int) -> None:
        count = int(self.config.effective_prefill_buffer_count)
        if int(buffer_index) < 0 or int(buffer_index) >= count:
            raise RuntimeError(
                f"prefill buffer index {buffer_index} is outside configured capacity "
                f"prefill_buffer_count={count}"
            )

    def _npu_hbm_snapshot(self, device: torch.device) -> dict[str, object]:
        """Best-effort allocator/physical-HBM snapshot for allocation evidence."""
        if device.type != "npu":
            return {}
        try:
            npu = getattr(torch, "npu", None)
            if npu is None:
                return {}
            index = device.index
            if index is None:
                index = int(npu.current_device())
            free_bytes, total_bytes = npu.mem_get_info(int(index))
            return {
                "device_index": int(index),
                "allocated_bytes": int(npu.memory_allocated(int(index))),
                "reserved_bytes": int(npu.memory_reserved(int(index))),
                "free_bytes": int(free_bytes),
                "total_bytes": int(total_bytes),
            }
        except Exception as exc:
            return {"snapshot_error": str(exc)}

    # --- Option 2: graph-compatible offload via decision/execution decoupling ---

    def log2phy_buffer(self, layer_id: int) -> torch.Tensor | None:
        """Return the persistent (fixed-address) log2phy buffer for a layer.

        The captured graph reads this stable tensor; only its *contents* change
        between replays, written in-place by ``stage_fixed_slot_plan``.
        """
        return self._log2phy_buffers.get(int(layer_id))

    def stage_fixed_slot_plan(
        self,
        *,
        layer_id: int,
        active_experts: tuple[int, ...],
        num_logical_experts: int,
    ) -> "PreparedSlotWeights":
        """Eager pre-replay staging: host decision + H2D + in-place log2phy write.

        This is the data-dependent / host-sync work hoisted OUT of the captured
        region. It must run eager (outside stream capture). It (1) decides which
        experts occupy which slots, (2) synchronously loads miss experts into the
        fixed slot tensors, and (3) writes the logical->physical mapping in-place
        into the persistent ``log2phy`` buffer (fixed address). The captured graph
        then only reads fixed slot tensors + the fixed log2phy buffer.

        Returns a ``PreparedSlotWeights`` whose ``log2phy`` IS the persistent
        buffer (not a fresh allocation), so the address is stable across steps.
        """
        if _is_current_graph_capturing():
            raise RuntimeError(
                "stage_fixed_slot_plan must run eager (outside graph capture); "
                "it performs host decision + H2D staging"
            )
        active_experts = self._resolve_placement_plan(
            layer_id=int(layer_id),
            active_experts=active_experts,
            num_logical_experts=int(num_logical_experts),
        )
        self._assert_locked_slot_addresses(int(layer_id))
        buf = self._log2phy_buffers[int(layer_id)]
        prepared = self.prepare_fixed_slot_plan_into_log2phy(
            layer_id=int(layer_id),
            active_experts=active_experts,
            num_logical_experts=int(num_logical_experts),
            log2phy=buf,
            record_stage_profile=True,
        )
        return PreparedSlotWeights(
            w1=prepared.w1,
            w2=prepared.w2,
            log2phy=buf,
            physical_expert_count=prepared.physical_expert_count,
            mapping=prepared.mapping,
        )

    def capture_safe_slot_weights(self, *, layer_id: int) -> "PreparedSlotWeights | None":
        """Capture-path plan: fixed slot tensors + fixed log2phy buffer, NO host sync.

        Used during graph capture (dummy run) where the real routing decision is
        irrelevant — capture only records the op sequence against fixed addresses.
        Performs zero device->host sync and zero conditional H2D, so the captured
        stream contains no forbidden synchronize/memcpy. Returns ``None`` if the
        layer is not registered for fixed-slot execution.
        """
        layer_id = int(layer_id)
        slot_bank = self._slot_banks.get(layer_id)
        buf = self._log2phy_buffers.get(layer_id)
        if slot_bank is None or buf is None:
            return None
        self._lock_graph_slot_addresses(layer_id)
        from vllm_moe_offload_ascend.moe_offload.slot_mapping import ExpertSlotMapping

        mapping = ExpertSlotMapping(
            layer_id=layer_id,
            active_experts=(),
            logical_to_physical=buf,
            slot_to_expert=tuple(
                int(slot.expert_key.expert_id) if slot.expert_key is not None else None
                for slot in slot_bank.slots
            ),
            active_slot_ids=(),
        )
        return PreparedSlotWeights.from_slot_bank(slot_bank=slot_bank, mapping=mapping)

    def _slot_address_fingerprint(self, layer_id: int) -> SlotAddressFingerprint:
        slot_bank = self._slot_banks.get(int(layer_id))
        log2phy = self._log2phy_buffers.get(int(layer_id))
        if slot_bank is None or log2phy is None:
            raise RuntimeError(
                f"layer {layer_id} has no fixed-slot tensors to fingerprint"
            )
        return SlotAddressFingerprint(
            w13_data_ptr=int(slot_bank.w13_slots.data_ptr()),
            w2_data_ptr=int(slot_bank.w2_slots.data_ptr()),
            log2phy_data_ptr=int(log2phy.data_ptr()),
            w13_shape=tuple(int(dim) for dim in slot_bank.w13_slots.shape),
            w2_shape=tuple(int(dim) for dim in slot_bank.w2_slots.shape),
            log2phy_shape=tuple(int(dim) for dim in log2phy.shape),
            device=str(slot_bank.w13_slots.device),
            dtype=str(slot_bank.w13_slots.dtype),
        )

    def _lock_graph_slot_addresses(self, layer_id: int) -> None:
        start = perf_counter()
        fingerprint = self._slot_address_fingerprint(layer_id)
        existing = self._graph_slot_address_fingerprints.get(int(layer_id))
        if existing is None:
            self._graph_slot_address_fingerprints[int(layer_id)] = fingerprint
            return
        if existing != fingerprint:
            raise RuntimeError(
                f"fixed-slot address fingerprint changed for captured layer {layer_id}"
            )
        if (
            not _is_current_graph_capturing()
            and int(layer_id) not in self._graph_slot_address_validation_evidence
        ):
            if int(layer_id) not in self._graph_slot_address_lock_evidence:
                self._graph_slot_address_lock_evidence.add(int(layer_id))
                self._record_profile_event(
                    "graph_slot_address_lock",
                    layer_id=int(layer_id),
                    start=start,
                    payload=_slot_address_fingerprint_payload(existing),
                )
            self._graph_slot_address_validation_evidence.add(int(layer_id))
            self._record_profile_event(
                "graph_slot_address_validate",
                layer_id=int(layer_id),
                start=start,
                payload={
                    **_slot_address_fingerprint_payload(fingerprint),
                    "matches_capture_fingerprint": True,
                },
            )

    def _assert_locked_slot_addresses(self, layer_id: int) -> None:
        if int(layer_id) not in self._graph_slot_address_fingerprints:
            return
        self._lock_graph_slot_addresses(int(layer_id))

    def stage_full_residency_slot_plan(self, *, layer_id: int) -> bool:
        """Regime A staging hook: one-time fill of slots + log2phy before capture.

        Precondition (Regime A): ``num_slots >= num_logical_experts`` so every
        logical expert owns a fixed slot and the log2phy mapping is *static*
        (independent of any step's active set). Under this condition the
        control-plane/data-plane ring dependency (need active_experts to stage,
        need replay to learn active_experts) is broken: we can stage ALL experts
        once, after weight loading and before ACLGraph capture.

        This is the missing wire that makes the captured graph token-correct: it
        writes the real logical->physical mapping into the persistent (fixed
        address) log2phy buffer that ``capture_safe_slot_weights`` exposes to the
        captured gather. Without it the buffer stays at its ``-1`` init and the
        captured graph mis-routes offloaded layers.

        Returns ``True`` if staging ran, ``False`` if it was a no-op (feature off,
        resident layer, layer not registered, or not graph-compatible mode). Only
        valid in Regime A; ``num_slots < num_logical_experts`` is rejected by the
        underlying ``prepare_fixed_slot_plan`` working-set guard (fail-closed).

        Must run eager (outside graph capture) — it performs host decision + H2D.
        """
        layer_id = int(layer_id)
        if not (self.should_use_fixed_slots and self.config.graph_compatible_offload):
            return False
        if self.is_resident_layer(layer_id):
            return False
        if not self.is_layer_registered(layer_id):
            return False
        if _is_current_graph_capturing():
            # Staging performs host decision + H2D; forbidden on a captured
            # stream. In the canonical flow staging already ran eager at load
            # time, so during capture this is a safe no-op.
            # If this is the FIRST call for this layer (log2phy still all -1),
            # fail in Python before the captured graph records a bad gather.
            buf = self._log2phy_buffers.get(layer_id)
            if buf is not None and bool((buf < 0).all().item()):
                raise RuntimeError(
                    f"stage_full_residency_slot_plan called during graph capture "
                    f"for layer {layer_id} but log2phy buffer is still all -1. Staging "
                    "must run eager BEFORE capture to populate the buffer; the "
                    "captured graph would mis-route this layer."
                )
            return False
        buf = self._log2phy_buffers.get(layer_id)
        if buf is None:
            return False
        num_logical_experts = int(buf.numel())
        if not self.is_static_residency_regime(num_logical_experts):
            self._record_profile_event(
                "skip_full_residency_slot_plan",
                layer_id=layer_id,
                start=perf_counter(),
                payload={
                    "reason": "regime_b_num_slots_lt_logical_experts",
                    "num_slots": int(self.config.num_slots),
                    "num_logical_experts": int(num_logical_experts),
                },
            )
            return False
        self.stage_fixed_slot_plan(
            layer_id=layer_id,
            active_experts=tuple(range(num_logical_experts)),
            num_logical_experts=num_logical_experts,
        )
        return True

    def prepare_weights_for_execution(
        self,
        *,
        layer_id: int,
        active_experts: tuple[int, ...],
    ) -> None:
        del layer_id, active_experts
        if not self.should_use_fixed_slots:
            return None
        raise NotImplementedError(
            "fixed-slot execution requires num_logical_experts and backend wiring; "
            "use prepare_fixed_slot_plan() for the current safe planning path"
        )

    def _get_compute_bucket_classifier(self) -> ComputeBucketClassifier | None:
        if self._compute_bucket_classifier_loaded:
            return self._compute_bucket_classifier
        self._compute_bucket_classifier_loaded = True
        plan_path = self.config.compute_bucket_plan_path
        if not plan_path:
            return None
        try:
            self._compute_bucket_classifier = load_compute_bucket_classifier(plan_path)
        except Exception as exc:
            self._record_profile_event(
                "compute_bucket_plan_load_failed",
                layer_id=None,
                start=perf_counter(),
                payload={
                    "plan_path": str(plan_path),
                    "reason": str(exc),
                },
            )
            self._compute_bucket_classifier = None
        return self._compute_bucket_classifier

    def _record_profile_event(
        self,
        name: str,
        *,
        layer_id: int | None,
        start: float,
        payload: dict[str, object] | None = None,
    ) -> None:
        event = MoeOffloadProfileEvent(
            name=name,
            layer_id=layer_id,
            seconds=perf_counter() - start,
            memory_ledger=self.memory_ledger(),
            payload=payload,
        )
        self._profile_events.append(event)
        self._append_profile_event_jsonl(event)

    def _append_profile_event_jsonl(self, event: MoeOffloadProfileEvent) -> None:
        profile_path = (
            self.config.gmm_profile_path
            or _env_value("VLLM_ASCEND_MOE_GMM_PROFILE_PATH")
            or _env_value("VLLM_ASCEND_MOE_OFFLOAD_PROFILE_PATH")
        )
        if not profile_path:
            return
        append_jsonl(profile_path, event.to_jsonable())

    def _append_trace_record_jsonl(self, record: TraceRecord) -> None:
        trace_path = (
            self.config.gmm_trace_path
            or _env_value("VLLM_ASCEND_MOE_GMM_TRACE_PATH")
            or _env_value("VLLM_ASCEND_MOE_OFFLOAD_TRACE_PATH")
        )
        if not trace_path:
            return
        append_jsonl(trace_path, record.to_jsonable())


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def _to_bool_env(name: str, default: str = "0") -> bool:
    value = os.getenv(name, default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _profile_expert_lists_enabled() -> bool:
    value = os.getenv("VLLM_ASCEND_MOE_PROFILE_EXPERT_LISTS", "1")
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


def _decode_profile_sample_rate() -> int:
    value = os.getenv("VLLM_ASCEND_MOE_DECODE_PROFILE_SAMPLE_RATE", "1")
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return 1


def _should_sample_decode_profile(*, step_id: int, sample_rate: int) -> bool:
    rate = max(1, int(sample_rate))
    return rate <= 1 or int(step_id) % rate == 0


def _is_current_graph_capturing() -> bool:
    try:
        from vllm_ascend.ascend_forward_context import _EXTRA_CTX

        if bool(getattr(_EXTRA_CTX, "capturing", False)):
            return True
    except Exception:
        pass
    try:
        return bool(torch.npu.is_current_stream_capturing())
    except Exception:
        return False


_runtime: MoeOffloadRuntime | None = None


def get_moe_offload_runtime() -> MoeOffloadRuntime:
    global _runtime
    if _runtime is None:
        _runtime = MoeOffloadRuntime()
    return _runtime


def reset_moe_offload_runtime() -> None:
    global _runtime
    _runtime = None


def _dedupe_preserve_order(values: tuple[int, ...]) -> tuple[int, ...]:
    seen: set[int] = set()
    deduped: list[int] = []
    for value in values:
        int_value = int(value)
        if int_value not in seen:
            seen.add(int_value)
            deduped.append(int_value)
    return tuple(deduped)


def _validate_active_expert_ids(
    *,
    layer_id: int,
    active_experts: tuple[int, ...],
    num_logical_experts: int,
) -> None:
    invalid_expert_ids = [
        int(expert_id)
        for expert_id in active_experts
        if int(expert_id) < 0 or int(expert_id) >= int(num_logical_experts)
    ]
    if invalid_expert_ids:
        raise ValueError(
            "fixed-slot active expert id out of range: "
            f"layer_id={int(layer_id)}, "
            f"num_logical_experts={int(num_logical_experts)}, "
            f"expert_ids={invalid_expert_ids}"
        )


def _update_log2phy_entries(
    log2phy: torch.Tensor,
    *,
    expert_ids: list[int] | tuple[int, ...],
    slot_ids: list[int] | tuple[int, ...],
) -> None:
    """Update the active logical->physical entries in one tensor op.

    Decode staging used to write one scalar ``log2phy[expert] = slot`` per
    active expert. On NPU that becomes several tiny device writes on every
    offloaded layer, and profiling shows this fixed mapping cost dominates even
    hit-only decode steps. Batched ``index_copy_`` preserves the active-only
    update contract while cutting the number of enqueued mapping ops.
    """
    if len(expert_ids) != len(slot_ids):
        raise ValueError(
            f"expert_ids and slot_ids length mismatch: {len(expert_ids)} != {len(slot_ids)}"
        )
    if not expert_ids:
        return
    if len(expert_ids) == 1:
        log2phy[int(expert_ids[0])] = int(slot_ids[0])
        return
    index = torch.as_tensor(
        tuple(int(expert_id) for expert_id in expert_ids),
        dtype=torch.long,
        device=log2phy.device,
    )
    values = torch.as_tensor(
        tuple(int(slot_id) for slot_id in slot_ids),
        dtype=log2phy.dtype,
        device=log2phy.device,
    )
    log2phy.index_copy_(0, index, values)
