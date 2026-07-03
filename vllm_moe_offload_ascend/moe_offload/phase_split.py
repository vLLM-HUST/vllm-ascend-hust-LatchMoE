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

"""MVP-D.11: Post-dispatch phase split semantic prototype.

This module provides the contracts, slicing, phase planning, partial MLP
execution, and scatter/gather logic for splitting the post-dispatch MoE MLP
compute into multiple phases (hit-first, then miss).  D.11 is a **semantic
prototype**: it proves that slicing + per-phase grouped matmul + gather
produces element-wise identical results to a single-phase run.  It does NOT
introduce async transfer, performance optimisation, or changes to router /
top-k / token count.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

from vllm_ascend import envs

if TYPE_CHECKING:
    import torch

    from vllm_ascend.ops.fused_moe.moe_stage_contracts import (
        MoEMlpComputeInput,
        MoEWeights,
    )


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MoEPhase:
    """A single phase of expert MLP execution.

    Each phase covers a contiguous (after extraction) range of tokens for a
    subset of the active experts.
    """

    phase_index: int
    # Logical expert ids covered by this phase (order as they appear in the
    # original sorted-hidden-states layout).  Used for STAGING (which logical
    # experts to load into slots), NOT for indexing group_list / weights.
    expert_indices: tuple[int, ...]
    # Start / end (exclusive) offsets in the *original* sorted hidden states
    # that this phase will extract.  Because experts may be interleaved with
    # experts from other phases the extracted region is not necessarily
    # contiguous in the original buffer – the executor is responsible for
    # gathering the individual expert slices.
    token_slices: tuple[tuple[int, int], ...]
    # True when all experts in this phase are resident / slot-ready (hit).
    is_hit: bool
    # Compact POSITION index of each expert within the layer's active-expert
    # order (i.e. the row index into the compact ``group_list`` / ``weights``
    # tensors), aligned 1:1 with ``expert_indices`` / ``token_slices``.  This is
    # distinct from ``expert_indices`` (logical ids): the two coincide only when
    # the active set is the dense identity ``(0,1,2,...)``.  ``_compute_wave``
    # must index group_list/weights with ``slice_positions``, never with the
    # logical ``expert_indices`` (doing so silently selects the wrong experts'
    # counts/weights, or raises IndexError, whenever the active set is sparse).
    # Defaults to ``()`` for backward-compatible construction; callers that omit
    # it implicitly assert a dense-identity active set.
    slice_positions: tuple[int, ...] = ()

    def resolved_slice_positions(self) -> tuple[int, ...]:
        """Positions to index compact group_list/weights for this phase.

        Falls back to ``expert_indices`` when ``slice_positions`` was not set,
        preserving the legacy dense-identity behaviour.
        """
        if self.slice_positions:
            return self.slice_positions
        return self.expert_indices

    @property
    def total_tokens(self) -> int:
        return sum(end - start for start, end in self.token_slices)

    def to_jsonable(self) -> dict[str, object]:
        return {
            "phase_index": self.phase_index,
            "expert_indices": list(self.expert_indices),
            "token_slices": [[int(s), int(e)] for s, e in self.token_slices],
            "total_tokens": self.total_tokens,
            "is_hit": self.is_hit,
        }


@dataclass(frozen=True)
class MoEPhasePlan:
    """Complete phase-split plan for one MoE layer invocation."""

    phases: tuple[MoEPhase, ...]
    total_phases: int
    hit_phases: int
    miss_phases: int
    total_tokens: int
    # Optional human-readable reason for diagnostics.
    reason: str = ""

    def to_jsonable(self) -> dict[str, object]:
        return {
            "total_phases": self.total_phases,
            "hit_phases": self.hit_phases,
            "miss_phases": self.miss_phases,
            "total_tokens": self.total_tokens,
            "reason": self.reason,
            "phases": [p.to_jsonable() for p in self.phases],
        }


@dataclass(frozen=True)
class B2WavePlan:
    """Capacity-bounded prefill wave plan keyed by logical expert ids."""

    waves: tuple[tuple[int, ...], ...]
    token_counts_by_expert: dict[int, int]

    @property
    def active_expert_count(self) -> int:
        return len(self.token_counts_by_expert)

    @property
    def total_pairs(self) -> int:
        return sum(int(v) for v in self.token_counts_by_expert.values())

    def wave_tokens(self, wave: tuple[int, ...]) -> int:
        return sum(int(self.token_counts_by_expert.get(int(e), 0)) for e in wave)

    def to_jsonable(self) -> dict[str, object]:
        return {
            "n_active": self.active_expert_count,
            "n_pairs": self.total_pairs,
            "n_waves": len(self.waves),
            "waves": [
                {
                    "experts": [int(e) for e in wave],
                    "tokens": self.wave_tokens(wave),
                    "per_expert_tokens": {
                        str(int(e)): int(self.token_counts_by_expert.get(int(e), 0))
                        for e in wave
                    },
                }
                for wave in self.waves
            ],
        }


@dataclass(frozen=True)
class B2PrefillAsyncSchedule:
    """Compute order plus early staging order for B2 prefill waves.

    Without transfer-aware estimates, compute order stays equal to the planner's
    wave order. With estimates, hit waves are allowed to run first as compute
    cover while staged waves are issued in long-transfer-first order. Staging
    order contains only waves with at least one miss, so their H2D can be issued
    before hit-only waves are computed.
    """

    compute_order: tuple[int, ...]
    hit_wave_indices: tuple[int, ...]
    staged_wave_indices: tuple[int, ...]
    staged_issue_order: tuple[int, ...]
    prefetch_depth: int = 1
    buffer_count: int = 2
    initial_stage_count: int = 0
    transfer_aware: bool = False
    estimated_h2d_bytes_by_wave: tuple[int, ...] = ()
    estimated_compute_cost_by_wave: tuple[float, ...] = ()

    def to_jsonable(self) -> dict[str, object]:
        return {
            "compute_order": [int(i) for i in self.compute_order],
            "hit_wave_indices": [int(i) for i in self.hit_wave_indices],
            "staged_wave_indices": [int(i) for i in self.staged_wave_indices],
            "staged_issue_order": [int(i) for i in self.staged_issue_order],
            "prefetch_depth": int(self.prefetch_depth),
            "buffer_count": int(self.buffer_count),
            "initial_stage_count": int(self.initial_stage_count),
            "transfer_aware": bool(self.transfer_aware),
            "estimated_h2d_bytes_by_wave": [
                int(v) for v in self.estimated_h2d_bytes_by_wave
            ],
            "estimated_compute_cost_by_wave": [
                float(v) for v in self.estimated_compute_cost_by_wave
            ],
        }


@dataclass(frozen=True)
class B2PairMicrobatch:
    """A work-conserving wave microbatch of routed (token, expert) pairs."""

    hidden_states: "torch.Tensor"
    topk_ids: "torch.Tensor"
    topk_weights: "torch.Tensor"
    restore_token_indices: "torch.Tensor"
    logical_expert_ids: "torch.Tensor"

    @property
    def num_pairs(self) -> int:
        return int(self.restore_token_indices.numel())


@dataclass(frozen=True)
class B2RoutedPairIndex:
    """Reusable routed-pair index for all B2 waves in one prefill layer call."""

    topk_ids: "torch.Tensor"
    topk_weights: "torch.Tensor"
    num_tokens: int
    top_k: int
    pair_offsets_by_expert: dict[int, tuple[int, ...]]

    @property
    def total_pairs(self) -> int:
        return sum(len(offsets) for offsets in self.pair_offsets_by_expert.values())


@dataclass(frozen=True)
class B2WaveMicrobatchPlan:
    """Precomputed routed-pair tensors for one B2 wave."""

    wave_experts: tuple[int, ...]
    token_indices: "torch.Tensor"
    topk_positions: "torch.Tensor"
    logical_expert_ids: "torch.Tensor"
    physical_slot_ids: "torch.Tensor"
    top_k: int = 1
    layer_pair_start: int = 0
    layer_pair_end: int = 0
    layer_token_indices: "torch.Tensor | None" = None
    layer_topk_positions: "torch.Tensor | None" = None
    layer_logical_expert_ids: "torch.Tensor | None" = None
    layer_physical_slot_ids: "torch.Tensor | None" = None

    @property
    def num_pairs(self) -> int:
        return int(self.token_indices.numel())

    @property
    def pair_offsets(self):
        return self.token_indices * int(self.top_k) + self.topk_positions


@dataclass(frozen=True)
class B2DirectScatterPayload:
    """Per-wave GMM output before AllGather token unpermute.

    B2 pair waves force ``top_k=1`` and build a microbatch containing only the
    routed pairs assigned to the wave.  The AllGather dispatcher's
    ``expanded_row_idx`` is the unpermute index: ``abs(expanded_row_idx)[i]``
    points to the permuted/GMM row for microbatch row ``i``. Layer-level scatter
    can directly add those weighted microbatch rows into the full prefill output.
    """

    permuted_tokens: "torch.Tensor"
    expanded_row_idx: "torch.Tensor"
    topk_weights: "torch.Tensor"
    restore_token_indices: "torch.Tensor"


# ---------------------------------------------------------------------------
# Expert token slicing from group_list
# ---------------------------------------------------------------------------


def compute_expert_token_slices(
    group_list: "torch.Tensor",
    group_list_type: int,
) -> list[tuple[int, int]]:
    """Return ``[(start, end), ...]`` for each expert in *group_list* order.

    *group_list_type*:
        - 1 (count):  ``group_list[i]`` = number of tokens for expert *i*.
        - 0 (cumsum): ``group_list[i]`` = cumulative end offset for expert *i*.
    """
    if group_list_type == 1:
        counts = [int(c) for c in group_list.cpu().tolist()]
    elif group_list_type == 0:
        cumsum = [int(c) for c in group_list.cpu().tolist()]
        counts = [cumsum[0]]
        for i in range(1, len(cumsum)):
            counts.append(cumsum[i] - cumsum[i - 1])
    else:
        raise ValueError(f"Unsupported group_list_type={group_list_type}; D.11 supports 0 or 1 only.")

    slices: list[tuple[int, int]] = []
    offset = 0
    for count in counts:
        slices.append((offset, offset + count))
        offset += count
    return slices


# ---------------------------------------------------------------------------
# Phase planner
# ---------------------------------------------------------------------------


def _build_single_phase_plan(
    expert_slices: list[tuple[int, int]],
    active_expert_ids: tuple[int, ...],
    reason: str = "",
) -> MoEPhasePlan:
    """Fallback: every active expert in one phase."""
    total = sum(end - start for start, end in expert_slices)
    phase = MoEPhase(
        phase_index=0,
        expert_indices=active_expert_ids,
        token_slices=tuple(expert_slices),
        is_hit=True,
        slice_positions=tuple(range(len(expert_slices))),
    )
    return MoEPhasePlan(
        phases=(phase,),
        total_phases=1,
        hit_phases=1,
        miss_phases=0,
        total_tokens=total,
        reason=reason,
    )


def plan_hit_miss_phases(
    expert_slices: list[tuple[int, int]],
    active_expert_ids: tuple[int, ...],
    slot_readiness: dict[int, bool] | None = None,
    max_phases: int = 2,
) -> MoEPhasePlan:
    """Split active experts into hit (ready) / miss phases.

    Parameters
    ----------
    expert_slices:
        Per-expert ``(start, end)`` offsets in the original sorted hidden
        states, aligned with *active_expert_ids*.
    active_expert_ids:
        Logical expert ids in the order they appear in ``group_list`` (which
        is also the order of *expert_slices*).
    slot_readiness:
        ``expert_id -> bool`` mapping.  Experts not in the map are treated as
        *ready* (hit).  Pass ``None`` to force single-phase.
    max_phases:
        Upper bound on the number of phases (default 2 = hit + miss).
    """
    if slot_readiness is None or max_phases <= 1:
        return _build_single_phase_plan(expert_slices, active_expert_ids, reason="single_phase")

    if len(active_expert_ids) != len(expert_slices):
        raise ValueError(
            f"Mismatched lengths: active_expert_ids={len(active_expert_ids)}, "
            f"expert_slices={len(expert_slices)}"
        )

    hit_pairs: list[tuple[int, int]] = []  # (expert_id, slice_index)
    miss_pairs: list[tuple[int, int]] = []

    for slice_idx, expert_id in enumerate(active_expert_ids):
        if slot_readiness.get(int(expert_id), True):
            hit_pairs.append((int(expert_id), slice_idx))
        else:
            miss_pairs.append((int(expert_id), slice_idx))

    phases: list[MoEPhase] = []

    def _make_phase(idx: int, pairs: list[tuple[int, int]], is_hit: bool) -> MoEPhase | None:
        if not pairs:
            return None
        return MoEPhase(
            phase_index=idx,
            expert_indices=tuple(eid for eid, _ in pairs),
            token_slices=tuple(expert_slices[si] for _, si in pairs),
            is_hit=is_hit,
            slice_positions=tuple(int(si) for _, si in pairs),
        )

    hit_phase = _make_phase(0, hit_pairs, True)
    if hit_phase is not None:
        phases.append(hit_phase)

    miss_phase = _make_phase(len(phases), miss_pairs, False)
    if miss_phase is not None:
        phases.append(miss_phase)

    if not phases:
        return _build_single_phase_plan(expert_slices, active_expert_ids, reason="empty_phases_fallback")

    total_tokens = sum(end - start for start, end in expert_slices)
    return MoEPhasePlan(
        phases=tuple(phases),
        total_phases=len(phases),
        hit_phases=sum(1 for p in phases if p.is_hit),
        miss_phases=sum(1 for p in phases if not p.is_hit),
        total_tokens=total_tokens,
        reason="hit_miss_split" if miss_pairs else "all_hit",
    )


def build_b2_wave_routing(
    physical_topk_ids: "torch.Tensor",
    topk_weights: "torch.Tensor",
):
    """B2 per-wave routing on the OFFLOAD path (log2phy-remapped topk_ids).

    The offload dispatch path remaps ``topk_ids`` through the wave's ``log2phy``
    buffer BEFORE this point: experts staged into this wave's slots get a valid
    physical slot id (>=0); every other (non-wave) logical expert maps to -1.
    Unlike the EP path, offload dispatch does NOT auto-zero dropped experts, so
    here we make the wave self-contained:

      * ``safe_ids`` = physical_topk_ids with -1 replaced by 0 (an in-range slot)
        so ``npu_moe_init_routing`` never indexes out of bounds. Those tokens are
        routed into slot 0 but contribute nothing because...
      * ``masked_weights`` = topk_weights zeroed wherever physical id was -1, so
        the combine (unpermute by probs) adds 0 for non-wave (token,expert) pairs.

    Summing each wave's combined output reproduces the full MoE output (addition
    over disjoint expert subsets), per the wave-accumulate keystone.

    Returns ``(safe_ids, masked_weights)`` -- both same shape as the inputs.
    """
    import torch

    kept = physical_topk_ids != -1
    safe_ids = torch.where(kept, physical_topk_ids, torch.zeros_like(physical_topk_ids))
    masked_weights = topk_weights * kept.to(topk_weights.dtype)
    return safe_ids, masked_weights


def count_routed_tokens_by_expert(topk_ids: "torch.Tensor") -> dict[int, int]:
    """Count routed (token, top-k slot) pairs for each logical expert."""
    if topk_ids.numel() == 0:
        return {}

    flat = topk_ids.detach().reshape(-1)
    if flat.device.type != "cpu":
        flat = flat.cpu()
    unique, counts = flat.to(dtype=flat.dtype).unique(return_counts=True)
    return {
        int(expert_id): int(count)
        for expert_id, count in zip(unique.tolist(), counts.tolist(), strict=True)
        if int(expert_id) >= 0 and int(count) > 0
    }


def plan_balanced_b2_waves(
    token_counts_by_expert: dict[int, int],
    num_slots: int,
    *,
    slot_readiness: dict[int, bool] | None = None,
) -> B2WavePlan:
    """Plan capacity-bounded waves, balanced by routed token count.

    Experts already present in slots are scheduled first so hit waves can start
    computing before miss-heavy waves in future async variants. Within the hit
    and miss groups, First-Fit Decreasing by routed pair count avoids packing
    several hot experts into one long-tail wave.
    """
    if num_slots <= 0:
        raise ValueError(f"num_slots must be greater than 0, got {num_slots}")

    active = {
        int(expert_id): int(count)
        for expert_id, count in token_counts_by_expert.items()
        if int(count) > 0
    }
    if not active:
        return B2WavePlan(waves=(), token_counts_by_expert={})

    readiness = slot_readiness or {}
    def _pack(expert_ids: list[int]) -> list[list[int]]:
        if not expert_ids:
            return []
        sorted_experts = sorted(
            expert_ids,
            key=lambda expert_id: (-active[int(expert_id)], int(expert_id)),
        )
        min_waves = (len(sorted_experts) + num_slots - 1) // num_slots
        bins: list[list[int]] = [[] for _ in range(min_waves)]
        bin_loads: list[int] = [0 for _ in range(min_waves)]
        for expert_id in sorted_experts:
            best_idx = None
            best_load = None
            for idx, wave in enumerate(bins):
                if len(wave) >= num_slots:
                    continue
                load = bin_loads[idx]
                if best_load is None or load < best_load:
                    best_idx = idx
                    best_load = load
            if best_idx is None:
                bins.append([int(expert_id)])
                bin_loads.append(active[int(expert_id)])
            else:
                bins[best_idx].append(int(expert_id))
                bin_loads[best_idx] += active[int(expert_id)]
        return bins

    hit_experts = [
        int(expert_id)
        for expert_id in active
        if readiness.get(int(expert_id), False)
    ]
    miss_experts = [
        int(expert_id)
        for expert_id in active
        if not readiness.get(int(expert_id), False)
    ]

    hit_bins = _pack(hit_experts)
    miss_bins = _pack(miss_experts)

    # Preserve full hit-only waves: these can compute directly from the main slot
    # cache and are useful cover for async miss-wave staging. Partial hit waves are
    # different: keeping a tiny hit tail separate increases wave count and repeats
    # dispatch/GMM/scatter overhead. Fold that tail into miss waves when there is
    # spare capacity; the staged path will copy those hit experts device-to-device.
    full_hit_bins: list[list[int]] = []
    partial_hit_experts: list[int] = []
    for wave in hit_bins:
        if len(wave) >= num_slots:
            full_hit_bins.append(wave)
        else:
            partial_hit_experts.extend(int(expert_id) for expert_id in wave)

    if miss_bins and partial_hit_experts:
        miss_loads = [sum(active[int(expert_id)] for expert_id in wave) for wave in miss_bins]
        remaining_hit_experts = sorted(
            partial_hit_experts,
            key=lambda expert_id: (-active[int(expert_id)], int(expert_id)),
        )
        unplaced_hit_experts: list[int] = []
        for expert_id in remaining_hit_experts:
            best_idx = None
            best_load = None
            for idx, wave in enumerate(miss_bins):
                if len(wave) >= num_slots:
                    continue
                load = miss_loads[idx]
                if best_load is None or load < best_load:
                    best_idx = idx
                    best_load = load
            if best_idx is None:
                unplaced_hit_experts.append(int(expert_id))
                continue
            miss_bins[best_idx].append(int(expert_id))
            miss_loads[best_idx] += active[int(expert_id)]
        partial_hit_bins = _pack(unplaced_hit_experts)
    else:
        partial_hit_bins = _pack(partial_hit_experts)

    waves = tuple(tuple(wave) for wave in (full_hit_bins + miss_bins + partial_hit_bins))
    return B2WavePlan(waves=waves, token_counts_by_expert=active)


def plan_b2_prefill_async_schedule(
    waves: tuple[tuple[int, ...], ...],
    *,
    slot_readiness: dict[int, bool] | None = None,
    prefetch_depth: int = 1,
    buffer_count: int = 2,
    h2d_bytes_by_wave: dict[int, int] | tuple[int, ...] | None = None,
    compute_cost_by_wave: dict[int, float] | tuple[float, ...] | None = None,
    transfer_aware: bool = False,
) -> B2PrefillAsyncSchedule:
    """Plan staging order without changing B2 wave compute order.

    Hit-only waves use the main slot cache and do not consume temp stage buffers.
    Waves with any miss need temp staging, so they are listed in
    ``staged_issue_order`` and can be primed before hit-only compute starts. When
    ``transfer_aware`` is enabled, long H2D waves are staged first and high-cost
    hit waves are computed first to provide cover for those transfers.
    """
    readiness = slot_readiness or {}
    hit_wave_indices: list[int] = []
    staged_wave_indices: list[int] = []
    for wave_index, wave in enumerate(waves):
        if all(readiness.get(int(expert_id), False) for expert_id in wave):
            hit_wave_indices.append(int(wave_index))
        else:
            staged_wave_indices.append(int(wave_index))
    effective_prefetch_depth = max(0, int(prefetch_depth))
    effective_buffer_count = max(1, int(buffer_count))
    initial_stage_count = min(
        effective_prefetch_depth,
        effective_buffer_count,
        len(staged_wave_indices),
    )
    h2d_estimates = _normalize_wave_int_estimates(
        h2d_bytes_by_wave,
        wave_count=len(waves),
    )
    compute_estimates = _normalize_wave_float_estimates(
        compute_cost_by_wave,
        wave_count=len(waves),
    )
    use_transfer_aware = (
        bool(transfer_aware)
        and bool(waves)
        and effective_prefetch_depth > 0
        and effective_buffer_count > 1
    )
    if use_transfer_aware:
        hit_compute_order = tuple(
            sorted(
                hit_wave_indices,
                key=lambda wave_idx: (
                    -float(compute_estimates[int(wave_idx)]),
                    int(wave_idx),
                ),
            )
        )
        staged_issue_order = tuple(
            sorted(
                staged_wave_indices,
                key=lambda wave_idx: (
                    -int(h2d_estimates[int(wave_idx)]),
                    -float(compute_estimates[int(wave_idx)]),
                    int(wave_idx),
                ),
            )
        )
        compute_order = hit_compute_order + staged_issue_order
    else:
        staged_issue_order = tuple(staged_wave_indices)
        compute_order = tuple(range(len(waves)))

    return B2PrefillAsyncSchedule(
        compute_order=compute_order,
        hit_wave_indices=tuple(hit_wave_indices),
        staged_wave_indices=tuple(staged_wave_indices),
        staged_issue_order=staged_issue_order,
        prefetch_depth=effective_prefetch_depth,
        buffer_count=effective_buffer_count,
        initial_stage_count=initial_stage_count,
        transfer_aware=use_transfer_aware,
        estimated_h2d_bytes_by_wave=h2d_estimates,
        estimated_compute_cost_by_wave=compute_estimates,
    )


def _normalize_wave_int_estimates(
    estimates: dict[int, int] | tuple[int, ...] | None,
    *,
    wave_count: int,
) -> tuple[int, ...]:
    if estimates is None:
        return tuple(0 for _ in range(int(wave_count)))
    if isinstance(estimates, dict):
        return tuple(
            max(0, int(estimates.get(int(wave_idx), 0)))
            for wave_idx in range(int(wave_count))
        )
    values = tuple(int(v) for v in estimates)
    return tuple(
        max(0, int(values[wave_idx])) if wave_idx < len(values) else 0
        for wave_idx in range(int(wave_count))
    )


def _normalize_wave_float_estimates(
    estimates: dict[int, float] | tuple[float, ...] | None,
    *,
    wave_count: int,
) -> tuple[float, ...]:
    if estimates is None:
        return tuple(0.0 for _ in range(int(wave_count)))
    if isinstance(estimates, dict):
        return tuple(
            max(0.0, float(estimates.get(int(wave_idx), 0.0)))
            for wave_idx in range(int(wave_count))
        )
    values = tuple(float(v) for v in estimates)
    return tuple(
        max(0.0, float(values[wave_idx])) if wave_idx < len(values) else 0.0
        for wave_idx in range(int(wave_count))
    )


def simulate_b2_prefill_issue_log(
    schedule: B2PrefillAsyncSchedule,
    *,
    prefetch_depth: int | None = None,
    buffer_count: int | None = None,
) -> tuple[tuple[str, int], ...]:
    """Return the issue/compute order for the B2 prefill async scheduler.

    This pure helper mirrors the fused runtime loop without touching tensors or
    streams. It is intentionally small, so tests can lock in the scheduling
    invariant: miss waves are staged ahead, hit-only waves are issued lazily from
    the main slot cache when their compute turn arrives.
    """
    log: list[tuple[str, int]] = []
    staged_order = list(schedule.staged_issue_order)
    staged_cursor = 0
    issued: set[int] = set()
    if prefetch_depth is None:
        prefetch_depth = int(schedule.prefetch_depth)
    if buffer_count is None:
        buffer_count = int(schedule.buffer_count)
    initial_stage_count = min(
        max(prefetch_depth, 0),
        max(buffer_count, 1),
        len(staged_order),
    )

    stage_records: dict[int, int] = {}

    def _next_free_buffer(current_buffer: int | None = None) -> int | None:
        busy = set(stage_records.values())
        if current_buffer is not None:
            busy.add(int(current_buffer))
        for buffer_index in range(max(buffer_count, 1)):
            if buffer_index not in busy:
                return int(buffer_index)
        return None

    def _issue_next_stage(current_buffer: int | None = None) -> bool:
        nonlocal staged_cursor
        if staged_cursor >= len(staged_order):
            return False
        buffer_index = _next_free_buffer(current_buffer)
        if buffer_index is None:
            return False
        wave_index = int(staged_order[staged_cursor])
        staged_cursor += 1
        issued.add(wave_index)
        stage_records[wave_index] = int(buffer_index)
        log.append(("issue_stage", wave_index))
        return True

    while staged_cursor < initial_stage_count:
        if not _issue_next_stage():
            break

    def _prefetch_ahead(current_buffer: int | None = None) -> None:
        if prefetch_depth <= 0:
            return
        occupied_by_current = 1 if current_buffer is not None else 0
        target_future = min(
            max(prefetch_depth, 0),
            max(buffer_count, 1) - occupied_by_current,
        )
        while len(stage_records) < target_future:
            if not _issue_next_stage(current_buffer):
                break

    staged_wave_indices = set(int(i) for i in schedule.staged_wave_indices)
    for wave_index in schedule.compute_order:
        wave_index = int(wave_index)
        if wave_index not in issued:
            if wave_index in staged_wave_indices:
                if not _issue_next_stage():
                    raise RuntimeError(
                        f"unable to issue staged wave {wave_index}; no buffer available"
                    )
            else:
                issued.add(wave_index)
                log.append(("issue_hit", wave_index))
        current_buffer = stage_records.pop(wave_index, None)
        _prefetch_ahead(current_buffer)
        log.append(("compute", wave_index))
        _prefetch_ahead()

    return tuple(log)


def build_b2_pair_microbatch(
    hidden_states: "torch.Tensor",
    topk_ids: "torch.Tensor",
    topk_weights: "torch.Tensor",
    logical_to_physical: "torch.Tensor",
    wave_experts: tuple[int, ...],
) -> B2PairMicrobatch:
    """Build a top_k=1 routed-pair microbatch for one B2 wave.

    The returned tensors contain only routed pairs whose logical expert belongs
    to ``wave_experts``. ``topk_ids`` are remapped to physical fixed-slot ids and
    shaped ``[num_pairs, 1]`` so dispatch/GMM/combine do no full-prompt masked
    work for this wave.
    """
    import torch

    if topk_ids.shape != topk_weights.shape:
        raise ValueError(
            "topk_ids and topk_weights must have identical shape, got "
            f"{tuple(topk_ids.shape)} vs {tuple(topk_weights.shape)}"
        )
    if hidden_states.shape[0] != topk_ids.shape[0]:
        raise ValueError(
            "hidden_states and topk_ids disagree on token count, got "
            f"{hidden_states.shape[0]} vs {topk_ids.shape[0]}"
        )
    if not wave_experts:
        empty_idx = torch.empty(0, dtype=torch.long, device=hidden_states.device)
        return B2PairMicrobatch(
            hidden_states=hidden_states.index_select(0, empty_idx),
            topk_ids=torch.empty(0, 1, dtype=topk_ids.dtype, device=topk_ids.device),
            topk_weights=torch.empty(0, 1, dtype=topk_weights.dtype, device=topk_weights.device),
            restore_token_indices=empty_idx,
            logical_expert_ids=torch.empty(0, dtype=topk_ids.dtype, device=topk_ids.device),
        )

    expert_tensor = torch.tensor(
        [int(expert_id) for expert_id in wave_experts],
        dtype=topk_ids.dtype,
        device=topk_ids.device,
    )
    pair_mask = (topk_ids.unsqueeze(-1) == expert_tensor).any(dim=-1)
    token_idx, topk_pos = torch.nonzero(pair_mask, as_tuple=True)

    if token_idx.numel() == 0:
        empty_idx = torch.empty(0, dtype=torch.long, device=hidden_states.device)
        return B2PairMicrobatch(
            hidden_states=hidden_states.index_select(0, empty_idx),
            topk_ids=torch.empty(0, 1, dtype=topk_ids.dtype, device=topk_ids.device),
            topk_weights=torch.empty(0, 1, dtype=topk_weights.dtype, device=topk_weights.device),
            restore_token_indices=empty_idx,
            logical_expert_ids=torch.empty(0, dtype=topk_ids.dtype, device=topk_ids.device),
        )

    logical_ids = topk_ids[token_idx, topk_pos]
    physical_ids = logical_to_physical[logical_ids]
    # This invariant is useful when debugging wave planning, but the CPU sync
    # required to prove it would serialize every prefill wave and break transfer
    # / compute overlap. Keep it behind an explicit validation probe.
    import os

    if os.environ.get("SEW_B2_VALIDATE"):
        if bool((physical_ids < 0).detach().cpu().any().item()):
            missing = logical_ids[physical_ids < 0].detach().cpu().unique().tolist()
            raise RuntimeError(f"B2 wave has unstaged experts in logical_to_physical: {missing}")

    restore_token_indices = token_idx.to(device=hidden_states.device, dtype=torch.long)
    return B2PairMicrobatch(
        hidden_states=hidden_states.index_select(0, restore_token_indices).contiguous(),
        topk_ids=physical_ids.reshape(-1, 1).contiguous(),
        topk_weights=topk_weights[token_idx, topk_pos].reshape(-1, 1).contiguous(),
        restore_token_indices=restore_token_indices,
        logical_expert_ids=logical_ids.contiguous(),
    )


def build_b2_routed_pair_index(
    topk_ids: "torch.Tensor",
    topk_weights: "torch.Tensor",
    *,
    pair_offsets_by_expert: dict[int, tuple[int, ...]] | None = None,
) -> B2RoutedPairIndex:
    """Build a reusable routed-pair index for wave microbatch construction.

    The original B2 microbatch builder re-scans ``topk_ids`` for every wave. In
    prefill that means 7-10 full prompt scans per offloaded layer. This helper
    builds the expert -> flat pair offsets map once; each wave then concatenates
    the already-known offsets for its experts.
    """
    import torch

    if topk_ids.shape != topk_weights.shape:
        raise ValueError(
            "topk_ids and topk_weights must have identical shape, got "
            f"{tuple(topk_ids.shape)} vs {tuple(topk_weights.shape)}"
        )
    if topk_ids.ndim == 0:
        raise ValueError("topk_ids must have at least one dimension")

    top_k = int(topk_ids.shape[1]) if topk_ids.ndim > 1 else 1
    num_tokens = int(topk_ids.shape[0])
    if pair_offsets_by_expert is None:
        flat = topk_ids.detach().reshape(-1)
        if flat.device.type != "cpu":
            flat = flat.cpu()
        buckets: dict[int, list[int]] = {}
        for pair_offset, expert_id in enumerate(flat.tolist()):
            expert_id = int(expert_id)
            if expert_id < 0:
                continue
            buckets.setdefault(expert_id, []).append(int(pair_offset))
        pair_offsets_by_expert = {
            int(expert_id): tuple(offsets)
            for expert_id, offsets in buckets.items()
            if offsets
        }
    else:
        pair_offsets_by_expert = {
            int(expert_id): tuple(int(offset) for offset in offsets)
            for expert_id, offsets in pair_offsets_by_expert.items()
            if offsets
        }

    max_offset = int(topk_ids.numel())
    for expert_id, offsets in pair_offsets_by_expert.items():
        for offset in offsets:
            if offset < 0 or offset >= max_offset:
                raise ValueError(
                    f"pair offset {offset} for expert {expert_id} outside "
                    f"topk_ids flat size {max_offset}"
                )

    return B2RoutedPairIndex(
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        num_tokens=num_tokens,
        top_k=top_k,
        pair_offsets_by_expert=pair_offsets_by_expert,
    )


def build_b2_pair_microbatch_from_index(
    hidden_states: "torch.Tensor",
    pair_index: B2RoutedPairIndex,
    logical_to_physical: "torch.Tensor",
    wave_experts: tuple[int, ...],
) -> B2PairMicrobatch:
    """Build one B2 wave microbatch using a precomputed routed-pair index."""
    import os
    import torch

    if hidden_states.shape[0] != pair_index.num_tokens:
        raise ValueError(
            "hidden_states and routed-pair index disagree on token count, got "
            f"{hidden_states.shape[0]} vs {pair_index.num_tokens}"
        )
    if not wave_experts:
        empty_idx = torch.empty(0, dtype=torch.long, device=hidden_states.device)
        return B2PairMicrobatch(
            hidden_states=hidden_states.index_select(0, empty_idx),
            topk_ids=torch.empty(0, 1, dtype=pair_index.topk_ids.dtype, device=pair_index.topk_ids.device),
            topk_weights=torch.empty(0, 1, dtype=pair_index.topk_weights.dtype, device=pair_index.topk_weights.device),
            restore_token_indices=empty_idx,
            logical_expert_ids=torch.empty(0, dtype=pair_index.topk_ids.dtype, device=pair_index.topk_ids.device),
        )

    offsets: list[int] = []
    for expert_id in wave_experts:
        offsets.extend(pair_index.pair_offsets_by_expert.get(int(expert_id), ()))
    offsets.sort()
    if not offsets:
        empty_idx = torch.empty(0, dtype=torch.long, device=hidden_states.device)
        return B2PairMicrobatch(
            hidden_states=hidden_states.index_select(0, empty_idx),
            topk_ids=torch.empty(0, 1, dtype=pair_index.topk_ids.dtype, device=pair_index.topk_ids.device),
            topk_weights=torch.empty(0, 1, dtype=pair_index.topk_weights.dtype, device=pair_index.topk_weights.device),
            restore_token_indices=empty_idx,
            logical_expert_ids=torch.empty(0, dtype=pair_index.topk_ids.dtype, device=pair_index.topk_ids.device),
        )

    pair_offsets = torch.tensor(
        offsets,
        dtype=torch.long,
        device=pair_index.topk_ids.device,
    )
    topk_pos = pair_offsets % int(pair_index.top_k)
    token_idx = pair_offsets // int(pair_index.top_k)
    logical_ids = pair_index.topk_ids.reshape(-1).index_select(0, pair_offsets)
    physical_ids = logical_to_physical[logical_ids]

    if os.environ.get("SEW_B2_VALIDATE"):
        if bool((physical_ids < 0).detach().cpu().any().item()):
            missing = logical_ids[physical_ids < 0].detach().cpu().unique().tolist()
            raise RuntimeError(f"B2 wave has unstaged experts in logical_to_physical: {missing}")

    restore_token_indices = token_idx.to(device=hidden_states.device, dtype=torch.long)
    return B2PairMicrobatch(
        hidden_states=hidden_states.index_select(0, restore_token_indices).contiguous(),
        topk_ids=physical_ids.reshape(-1, 1).contiguous(),
        topk_weights=pair_index.topk_weights[token_idx, topk_pos].reshape(-1, 1).contiguous(),
        restore_token_indices=restore_token_indices,
        logical_expert_ids=logical_ids.contiguous(),
    )


def build_b2_wave_microbatch_plan(
    pair_index: B2RoutedPairIndex,
    wave_experts: tuple[int, ...],
    *,
    physical_slot_by_expert: dict[int, int] | None = None,
) -> B2WaveMicrobatchPlan:
    """Precompute all wave-specific routed-pair tensors except log2phy remap."""
    plans = build_b2_wave_microbatch_plans(
        pair_index,
        (wave_experts,),
        physical_slot_by_expert=physical_slot_by_expert,
    )
    return plans[0]


def build_b2_wave_microbatch_plans(
    pair_index: B2RoutedPairIndex,
    waves: tuple[tuple[int, ...], ...],
    *,
    physical_slot_by_expert: dict[int, int] | None = None,
) -> tuple[B2WaveMicrobatchPlan, ...]:
    """Precompute routed-pair tensors for all B2 waves with one device tensor.

    Prefill B2 executes one layer as several expert waves. Building each wave
    plan separately creates one small device tensor per wave and repeats the
    same arithmetic kernels. This batched builder keeps each wave's sorted pair
    order identical to :func:`build_b2_wave_microbatch_plan`, but concatenates
    all offsets first so the hot path does one tensor materialization and slices
    views for the individual wave plans. ``physical_slot_by_expert`` lets hit
    waves reuse main-slot physical ids without building a full log2phy buffer;
    experts not present in that mapping keep the temporary-wave slot id.
    """
    import torch

    normalized_waves = tuple(tuple(int(e) for e in wave) for wave in waves)
    physical_slots = (
        {int(expert_id): int(slot_id) for expert_id, slot_id in physical_slot_by_expert.items()}
        if physical_slot_by_expert is not None
        else {}
    )
    wave_sizes: list[int] = []
    all_token_indices: list[int] = []
    all_topk_positions: list[int] = []
    all_logical_ids: list[int] = []
    all_slot_ids: list[int] = []
    for wave in normalized_waves:
        wave_pairs: list[tuple[int, int, int]] = []
        # Main-bank physical ids are valid ONLY when the whole wave is a pure-hit
        # wave executed against the main slot bank (zero-copy path).  A mixed wave
        # (any miss) is staged into a temporary stage bank at positional slots
        # 0..k-1; applying a main-bank slot id to a hit expert in that wave indexes
        # the wrong position in the stage bank and produces silent weight corruption.
        wave_is_pure_hit = bool(physical_slots) and all(
            int(e) in physical_slots for e in wave
        )
        for slot_id, expert_id in enumerate(wave):
            physical_id = (
                int(physical_slots[int(expert_id)])
                if wave_is_pure_hit
                else int(slot_id)
            )
            wave_pairs.extend(
                (int(offset), physical_id, int(expert_id))
                for offset in pair_index.pair_offsets_by_expert.get(
                    int(expert_id),
                    (),
                )
            )
        wave_pairs.sort(key=lambda item: item[0])
        wave_sizes.append(len(wave_pairs))
        for offset, slot_id, expert_id in wave_pairs:
            all_token_indices.append(int(offset) // int(pair_index.top_k))
            all_topk_positions.append(int(offset) % int(pair_index.top_k))
            all_logical_ids.append(int(expert_id))
            all_slot_ids.append(int(slot_id))

    all_token_indices_tensor = torch.tensor(
        all_token_indices,
        dtype=torch.long,
        device=pair_index.topk_ids.device,
    )
    all_topk_positions_tensor = torch.tensor(
        all_topk_positions,
        dtype=torch.long,
        device=pair_index.topk_ids.device,
    )
    all_logical_ids_tensor = torch.tensor(
        all_logical_ids,
        dtype=pair_index.topk_ids.dtype,
        device=pair_index.topk_ids.device,
    )
    all_physical_slot_ids = torch.tensor(
        all_slot_ids,
        dtype=pair_index.topk_ids.dtype,
        device=pair_index.topk_ids.device,
    )
    if all_token_indices_tensor.numel() == 0:
        empty_long = all_token_indices_tensor
        empty_ids = torch.empty(
            0,
            dtype=pair_index.topk_ids.dtype,
            device=pair_index.topk_ids.device,
        )
        return tuple(
            B2WaveMicrobatchPlan(
                wave_experts=wave,
                token_indices=empty_long,
                topk_positions=empty_long,
                logical_expert_ids=empty_ids,
                physical_slot_ids=empty_ids,
                top_k=int(pair_index.top_k),
                layer_pair_start=0,
                layer_pair_end=0,
                layer_token_indices=empty_long,
                layer_topk_positions=empty_long,
                layer_logical_expert_ids=empty_ids,
                layer_physical_slot_ids=empty_ids,
            )
            for wave in normalized_waves
        )

    plans: list[B2WaveMicrobatchPlan] = []
    cursor = 0
    for wave, size in zip(normalized_waves, wave_sizes, strict=True):
        end = cursor + size
        plans.append(
            B2WaveMicrobatchPlan(
                wave_experts=wave,
                token_indices=all_token_indices_tensor[cursor:end],
                topk_positions=all_topk_positions_tensor[cursor:end],
                logical_expert_ids=all_logical_ids_tensor[cursor:end],
                physical_slot_ids=all_physical_slot_ids[cursor:end],
                top_k=int(pair_index.top_k),
                layer_pair_start=cursor,
                layer_pair_end=end,
                layer_token_indices=all_token_indices_tensor,
                layer_topk_positions=all_topk_positions_tensor,
                layer_logical_expert_ids=all_logical_ids_tensor,
                layer_physical_slot_ids=all_physical_slot_ids,
            )
        )
        cursor = end
    return tuple(plans)


def build_b2_pair_microbatch_from_plan(
    hidden_states: "torch.Tensor",
    topk_weights: "torch.Tensor",
    logical_to_physical: "torch.Tensor | None",
    plan: B2WaveMicrobatchPlan,
) -> B2PairMicrobatch:
    """Build one B2 wave microbatch from a precomputed wave plan.

    ``logical_to_physical`` may be ``None`` for staged B2 waves because the temp
    bank always assigns ``wave_experts[i]`` to physical slot ``i``. In that case
    the plan's precomputed slot ids avoid all per-wave log2phy device writes.
    """
    import os
    import torch

    if plan.num_pairs == 0:
        empty_idx = torch.empty(0, dtype=torch.long, device=hidden_states.device)
        topk_dtype = (
            logical_to_physical.dtype
            if logical_to_physical is not None
            else plan.physical_slot_ids.dtype
        )
        topk_device = (
            logical_to_physical.device
            if logical_to_physical is not None
            else plan.physical_slot_ids.device
        )
        return B2PairMicrobatch(
            hidden_states=hidden_states.index_select(0, empty_idx),
            topk_ids=torch.empty(0, 1, dtype=topk_dtype, device=topk_device),
            topk_weights=torch.empty(0, 1, dtype=topk_weights.dtype, device=topk_weights.device),
            restore_token_indices=empty_idx,
            logical_expert_ids=torch.empty(0, dtype=plan.logical_expert_ids.dtype, device=plan.logical_expert_ids.device),
        )

    physical_ids = (
        logical_to_physical[plan.logical_expert_ids]
        if logical_to_physical is not None
        else plan.physical_slot_ids
    )
    if os.environ.get("SEW_B2_VALIDATE"):
        if bool((physical_ids < 0).detach().cpu().any().item()):
            missing = plan.logical_expert_ids[physical_ids < 0].detach().cpu().unique().tolist()
            raise RuntimeError(f"B2 wave has unstaged experts in logical_to_physical: {missing}")

    restore_token_indices = plan.token_indices.to(device=hidden_states.device, dtype=torch.long)
    return B2PairMicrobatch(
        hidden_states=hidden_states.index_select(0, restore_token_indices).contiguous(),
        topk_ids=physical_ids.reshape(-1, 1).contiguous(),
        topk_weights=topk_weights[
            plan.token_indices,
            plan.topk_positions,
        ].reshape(-1, 1).contiguous(),
        restore_token_indices=restore_token_indices,
        logical_expert_ids=plan.logical_expert_ids.contiguous(),
    )


def materialize_b2_pair_microbatches_from_plans(
    hidden_states: "torch.Tensor",
    topk_weights: "torch.Tensor",
    plans: tuple[B2WaveMicrobatchPlan, ...],
) -> tuple[B2PairMicrobatch, ...]:
    """Materialize all wave pair microbatches with one layer-level gather.

    The per-wave builder is semantically simple, but it launches small gathers
    for every wave. Prefill usually has several waves per offloaded layer, so
    this helper gathers all routed pair hidden states and router weights once,
    then returns per-wave views. It preserves each wave's pair order and physical
    slot ids from ``B2WaveMicrobatchPlan``.
    """
    import torch

    if not plans:
        return ()

    total_pairs = sum(int(plan.num_pairs) for plan in plans)
    if total_pairs == 0:
        empty_idx = torch.empty(0, dtype=torch.long, device=hidden_states.device)
        microbatches: list[B2PairMicrobatch] = []
        for plan in plans:
            microbatches.append(
                B2PairMicrobatch(
                    hidden_states=hidden_states.index_select(0, empty_idx),
                    topk_ids=torch.empty(
                        0,
                        1,
                        dtype=plan.physical_slot_ids.dtype,
                        device=plan.physical_slot_ids.device,
                    ),
                    topk_weights=torch.empty(
                        0,
                        1,
                        dtype=topk_weights.dtype,
                        device=topk_weights.device,
                    ),
                    restore_token_indices=empty_idx,
                    logical_expert_ids=torch.empty(
                        0,
                        dtype=plan.logical_expert_ids.dtype,
                        device=plan.logical_expert_ids.device,
                    ),
                )
            )
        return tuple(microbatches)

    first_plan = plans[0]
    can_reuse_layer_tensors = (
        first_plan.layer_token_indices is not None
        and first_plan.layer_topk_positions is not None
        and first_plan.layer_physical_slot_ids is not None
        and first_plan.layer_logical_expert_ids is not None
        and int(first_plan.layer_pair_start) == 0
        and int(plans[-1].layer_pair_end) == int(total_pairs)
        and all(
            plan.layer_token_indices is first_plan.layer_token_indices
            and plan.layer_topk_positions is first_plan.layer_topk_positions
            and plan.layer_physical_slot_ids is first_plan.layer_physical_slot_ids
            and plan.layer_logical_expert_ids is first_plan.layer_logical_expert_ids
            for plan in plans
        )
    )
    if can_reuse_layer_tensors:
        all_token_indices = first_plan.layer_token_indices
        all_topk_positions = first_plan.layer_topk_positions
        all_physical_slot_ids = first_plan.layer_physical_slot_ids
        all_logical_expert_ids = first_plan.layer_logical_expert_ids
    else:
        all_token_indices = torch.cat(tuple(plan.token_indices for plan in plans))
        all_topk_positions = torch.cat(tuple(plan.topk_positions for plan in plans))
        all_physical_slot_ids = torch.cat(tuple(plan.physical_slot_ids for plan in plans))
        all_logical_expert_ids = torch.cat(tuple(plan.logical_expert_ids for plan in plans))
    restore_token_indices = all_token_indices.to(
        device=hidden_states.device,
        dtype=torch.long,
    )
    materialized_hidden = hidden_states.index_select(
        0,
        restore_token_indices,
    ).contiguous()
    materialized_topk_ids = all_physical_slot_ids.reshape(-1, 1).contiguous()
    materialized_topk_weights = topk_weights[
        all_token_indices,
        all_topk_positions,
    ].reshape(-1, 1).contiguous()

    microbatches: list[B2PairMicrobatch] = []
    cursor = 0
    for plan in plans:
        if can_reuse_layer_tensors:
            start = int(plan.layer_pair_start)
            end = int(plan.layer_pair_end)
        else:
            start = cursor
            end = start + int(plan.num_pairs)
            cursor = end
        microbatches.append(
            B2PairMicrobatch(
                hidden_states=materialized_hidden[start:end],
                topk_ids=materialized_topk_ids[start:end],
                topk_weights=materialized_topk_weights[start:end],
                restore_token_indices=restore_token_indices[start:end],
                logical_expert_ids=all_logical_expert_ids[start:end],
            )
        )
    return tuple(microbatches)


def scatter_add_b2_pair_output(
    full_output: "torch.Tensor",
    pair_output: "torch.Tensor",
    restore_token_indices: "torch.Tensor",
) -> "torch.Tensor":
    """Accumulate pair-level wave output back to full token output."""
    if pair_output.numel() == 0:
        return full_output
    full_output.index_add_(0, restore_token_indices.to(full_output.device), pair_output)
    return full_output


def scatter_add_b2_pair_outputs(
    full_output: "torch.Tensor",
    pair_outputs: tuple["torch.Tensor", ...],
    restore_token_indices: tuple["torch.Tensor", ...],
) -> "torch.Tensor":
    """Accumulate several wave outputs with one index_add_ when possible."""
    if len(pair_outputs) != len(restore_token_indices):
        raise ValueError(
            "pair_outputs and restore_token_indices must have the same length, "
            f"got {len(pair_outputs)} vs {len(restore_token_indices)}"
        )
    non_empty = [
        (output, indices)
        for output, indices in zip(pair_outputs, restore_token_indices, strict=True)
        if output.numel() > 0
    ]
    if not non_empty:
        return full_output
    if len(non_empty) == 1:
        output, indices = non_empty[0]
        return scatter_add_b2_pair_output(full_output, output, indices)

    import torch

    outputs = torch.cat(tuple(output for output, _ in non_empty), dim=0)
    indices = torch.cat(
        tuple(indices.to(full_output.device) for _, indices in non_empty),
        dim=0,
    )
    full_output.index_add_(0, indices, outputs)
    return full_output


def direct_scatter_add_b2_permuted_output(
    full_output: "torch.Tensor",
    *,
    permuted_tokens: "torch.Tensor",
    expanded_row_idx: "torch.Tensor",
    topk_weights: "torch.Tensor",
    restore_token_indices: "torch.Tensor",
) -> "torch.Tensor":
    """Fuse AllGather top_k=1 unpermute with B2 scatter-add.

    For the AllGather dispatcher in B2 waves we force ``top_k=1`` and pass a
    microbatch containing only routed pairs in the wave. ``expanded_row_idx``
    is the same unpermute index consumed by ``npu_moe_token_unpermute``:
    ``abs(expanded_row_idx)[i]`` points to the permuted/GMM row for microbatch
    row ``i``. We gather those rows, multiply by router weights, and index-add
    directly into the full prefill output, avoiding a separate wave-local output
    tensor.
    """
    import torch

    if permuted_tokens.numel() == 0:
        return full_output

    expanded = expanded_row_idx.reshape(-1).abs().to(
        device=permuted_tokens.device,
        dtype=torch.long,
    )
    weights = topk_weights.reshape(-1).to(
        device=permuted_tokens.device,
        dtype=permuted_tokens.dtype,
    )
    gathered = permuted_tokens.index_select(0, expanded)
    weighted = gathered * weights.unsqueeze(-1)
    full_output.index_add_(
        0,
        restore_token_indices.to(full_output.device),
        weighted.to(full_output.device),
    )
    return full_output


def direct_scatter_add_b2_permuted_outputs(
    full_output: "torch.Tensor",
    payloads: tuple[B2DirectScatterPayload, ...],
) -> "torch.Tensor":
    """Batch direct B2 unpermute/scatter for all waves in one layer.

    This is the work-conserving counterpart to ``scatter_add_b2_pair_outputs``:
    instead of materializing one wave-local combined output per wave, it offsets
    every wave-local unpermute index into one concatenated permuted-token buffer,
    applies the top-k weights once, and performs one layer-level ``index_add_``.
    """
    non_empty = [payload for payload in payloads if payload.permuted_tokens.numel() > 0]
    if not non_empty:
        return full_output

    if len(non_empty) == 1:
        payload = non_empty[0]
        return direct_scatter_add_b2_permuted_output(
            full_output,
            permuted_tokens=payload.permuted_tokens,
            expanded_row_idx=payload.expanded_row_idx,
            topk_weights=payload.topk_weights,
            restore_token_indices=payload.restore_token_indices,
        )

    import torch

    weighted_chunks = []
    restore_chunks = []
    for payload in non_empty:
        expanded = payload.expanded_row_idx.reshape(-1).abs().to(
            device=payload.permuted_tokens.device,
            dtype=torch.long,
        )
        weights = payload.topk_weights.reshape(-1).to(
            device=payload.permuted_tokens.device,
            dtype=payload.permuted_tokens.dtype,
        )
        gathered = payload.permuted_tokens.index_select(0, expanded)
        weighted_chunks.append(
            (gathered * weights.unsqueeze(-1)).to(full_output.device)
        )
        restore_chunks.append(payload.restore_token_indices.to(full_output.device))

    if not weighted_chunks:
        return full_output
    full_output.index_add_(
        0,
        torch.cat(tuple(restore_chunks), dim=0),
        torch.cat(tuple(weighted_chunks), dim=0),
    )
    return full_output


def build_wave_expert_map(
    wave_logical_experts: tuple[int, ...],
    num_logical_experts: int,
) -> "torch.Tensor":
    """Build a per-wave ``expert_map`` for B2 wave-streamed prefill.

    The AllGather token dispatcher drops experts whose ``expert_map`` entry is -1
    (``mask = expert_map[topk_ids] != -1; topk_weights = topk_weights * mask``).
    For one wave we therefore map ONLY that wave's logical experts to physical slot
    positions ``0..k-1`` (their order within the wave) and map every other logical
    expert to -1. Running dispatch->matmul->combine with this map computes exactly
    that wave's ``(token, expert)`` contributions (others contribute 0); summing
    across waves reproduces the full MoE output (see the wave-accumulate keystone).

    Returns an int32 tensor of shape ``[num_logical_experts]``.
    """
    import torch

    expert_map = torch.full((int(num_logical_experts),), -1, dtype=torch.int32)
    for slot_position, logical_expert in enumerate(wave_logical_experts):
        expert_map[int(logical_expert)] = slot_position
    return expert_map


def plan_capacity_bounded_phases(
    expert_slices: list[tuple[int, int]],
    active_expert_ids: tuple[int, ...],
    num_slots: int,
) -> MoEPhasePlan:
    """B2: split active experts into capacity-bounded waves of <= num_slots each.

    Unlike :func:`plan_hit_miss_phases` (which splits on slot *readiness*), this
    planner splits on slot *capacity*: when an offloaded layer's active expert
    set exceeds ``num_slots`` (the fixed HBM slot budget), a single grouped
    matmul cannot run because not all experts can be resident at once. We instead
    emit ``ceil(N / num_slots)`` waves, each covering a contiguous chunk of at
    most ``num_slots`` experts. The executor stages each wave's experts into the
    slot bank (eager prefill only), runs a partial grouped matmul over just that
    wave's tokens, and scatters the result back. Because every token belongs to
    exactly one expert and every expert to exactly one wave, the per-wave scatters
    are disjoint and cover all tokens -> the concatenated result is element-wise
    identical to a single-phase run.

    Parameters mirror :func:`plan_hit_miss_phases`: ``expert_slices[i]`` is the
    ``(start, end)`` token range for ``active_expert_ids[i]`` in the sorted hidden
    states. ``num_slots`` is the per-layer fixed slot count.

    Waves are marked ``is_hit=False`` because every wave requires staging (no wave
    is resident up front under B2's capacity pressure).
    """
    if num_slots <= 0:
        raise ValueError(f"num_slots must be greater than 0, got {num_slots}")
    if len(active_expert_ids) != len(expert_slices):
        raise ValueError(
            f"Mismatched lengths: active_expert_ids={len(active_expert_ids)}, "
            f"expert_slices={len(expert_slices)}"
        )

    total_tokens = sum(end - start for start, end in expert_slices)

    # Fits in one wave -> degenerate to a single phase (same as B1's slot path).
    if len(active_expert_ids) <= num_slots:
        return _build_single_phase_plan(
            expert_slices, active_expert_ids, reason="capacity_single_wave"
        )

    phases: list[MoEPhase] = []
    for wave_index, start in enumerate(range(0, len(active_expert_ids), num_slots)):
        chunk_ids = active_expert_ids[start : start + num_slots]
        chunk_slices = tuple(expert_slices[start : start + num_slots])
        phases.append(
            MoEPhase(
                phase_index=wave_index,
                expert_indices=tuple(int(e) for e in chunk_ids),
                token_slices=chunk_slices,
                is_hit=False,
                slice_positions=tuple(range(start, start + len(chunk_ids))),
            )
        )

    return MoEPhasePlan(
        phases=tuple(phases),
        total_phases=len(phases),
        hit_phases=0,
        miss_phases=len(phases),
        total_tokens=total_tokens,
        reason="capacity_bounded_waves",
    )


# ---------------------------------------------------------------------------
# Partial MLP execution helpers
# ---------------------------------------------------------------------------


def _extract_phase_tokens(
    hidden_states: "torch.Tensor",
    token_slices: tuple[tuple[int, int], ...],
) -> "torch.Tensor":
    """Extract and concatenate tokens for a phase from sorted hidden states."""
    import torch

    if not token_slices:
        return torch.empty(0, hidden_states.size(1), dtype=hidden_states.dtype, device=hidden_states.device)

    chunks = [hidden_states[start:end] for start, end in token_slices]
    if len(chunks) == 1:
        return chunks[0].contiguous()
    return torch.cat(chunks, dim=0).contiguous()


def _build_phase_group_list(
    group_list: "torch.Tensor",
    group_list_type: int,
    expert_indices: tuple[int, ...],
) -> "torch.Tensor":
    """Build a new group_list tensor covering only *expert_indices*."""
    import torch

    if group_list_type == 1:
        selected = [group_list[i] for i in expert_indices]
        return torch.stack(selected) if selected else torch.empty(0, dtype=group_list.dtype, device=group_list.device)
    elif group_list_type == 0:
        # cumulative mode: need to reconstruct per-expert counts
        cumsum = group_list.cpu().tolist()
        prev = 0
        counts = []
        for i, end in enumerate(cumsum):
            counts.append(end - prev)
            prev = end
        selected = [counts[i] for i in expert_indices]
        result = torch.tensor(selected, dtype=group_list.dtype, device=group_list.device)
        # Convert back to cumulative
        return torch.cumsum(result, dim=0)
    else:
        raise ValueError(f"Unsupported group_list_type={group_list_type}")


def _slice_expert_weights(
    weights: "MoEWeights",
    expert_indices: tuple[int, ...],
) -> "MoEWeights":
    """Return a MoEWeights view sliced to *expert_indices*."""
    from vllm_ascend.ops.fused_moe.moe_stage_contracts import MoEWeights as _MoEWeights

    def _index(w):
        if w is None:
            return None
        if isinstance(w, list):
            return [t[list(expert_indices)] for t in w]
        return w[list(expert_indices)]

    return _MoEWeights(
        w1=_index(weights.w1),
        w2=_index(weights.w2),
        w1_bias=_index(weights.w1_bias),
        w2_bias=_index(weights.w2_bias),
        w1_scale=_index(weights.w1_scale),
        w2_scale=_index(weights.w2_scale),
        w1_scale_bias=_index(weights.w1_scale_bias),
        w2_scale_bias=_index(weights.w2_scale_bias),
        w1_offset=_index(weights.w1_offset),
        w2_offset=_index(weights.w2_offset),
    )


# ---------------------------------------------------------------------------
# Scatter / gather
# ---------------------------------------------------------------------------


def _scatter_phase_output(
    full_output: "torch.Tensor",
    phase_output: "torch.Tensor",
    token_slices: tuple[tuple[int, int], ...],
) -> "torch.Tensor":
    """Write *phase_output* back into *full_output* at the given slices.

    Returns *full_output* (modified in-place).
    """
    offset = 0
    for start, end in token_slices:
        length = end - start
        if length > 0:
            full_output[start:end] = phase_output[offset : offset + length]
            offset += length
    return full_output


# ---------------------------------------------------------------------------
# Top-level phased MLP orchestrator
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Wave staging interface (overlap-ready: separates transfer from compute)
# ---------------------------------------------------------------------------


class WaveStager:
    """Two-phase staging contract for capacity-bounded waves.

    The contract deliberately splits "move this wave's experts into HBM" into two
    calls so the executor can pipeline transfer (MTE) against compute (Cube):

      * ``issue(wave_index, expert_indices)`` -- start staging the wave's experts
        into a fixed slot buffer. May be asynchronous (return before the H2D copy
        finishes); the serial implementation does it synchronously.
      * ``wait(wave_index)`` -- block until that wave's slots are READY to be read
        by the grouped matmul. The serial implementation is a no-op (issue already
        finished the copy).

    A serial (``prefetch_depth=0``) run calls ``issue`` then ``wait`` then compute
    for each wave in turn -> identical to the original single-buffer staging. An
    overlapped run (``prefetch_depth>=1``, double/N-buffered) issues wave k+1
    BEFORE computing wave k, so the next wave's H2D rides under the current wave's
    matmul. ``buffer_count`` declares how many waves may be in flight; the executor
    guarantees it never issues a wave into a buffer whose prior occupant has not
    yet been consumed (so a single-buffer stager is never asked to overlap).

    This base class is the serial reference. NPU async (separate transfer stream +
    SetFlag/WaitFlag) is a drop-in subclass that overrides issue/wait -- no change
    to the executor or the planner.
    """

    #: How many waves may be resident concurrently. 1 == single buffer (serial).
    buffer_count: int = 1

    def issue(self, wave_index: int, expert_indices: tuple[int, ...]) -> None:
        raise NotImplementedError

    def wait(self, wave_index: int) -> None:
        raise NotImplementedError


class _CallbackWaveStager(WaveStager):
    """Serial stager adapting the simple ``stage_wave_fn(expert_indices)`` callback.

    ``issue`` runs the callback synchronously (the H2D completes before it
    returns); ``wait`` is a no-op. ``buffer_count=1`` so the executor keeps strict
    serial order -- preserving the original ``stage_wave_fn`` semantics exactly.
    """

    buffer_count = 1

    def __init__(self, stage_wave_fn):
        self._stage_wave_fn = stage_wave_fn

    def issue(self, wave_index: int, expert_indices: tuple[int, ...]) -> None:
        self._stage_wave_fn(expert_indices)

    def wait(self, wave_index: int) -> None:
        return None


def execute_phased_mlp(
    *,
    mlp_compute_input: "MoEMlpComputeInput",
    phase_plan: MoEPhasePlan,
    _apply_mlp_fn=None,
    stage_wave_fn=None,
    wave_stager=None,
    prefetch_depth: int = 0,
) -> "torch.Tensor":
    """Execute MoE MLP in phases according to *phase_plan*.

    This replaces a single ``_apply_mlp(mlp_compute_input)`` call with one
    ``_apply_mlp`` call per phase, each operating on a contiguous subset of
    tokens / experts, then scatters the results back into a full output buffer.

    Parameters
    ----------
    mlp_compute_input:
        The full (single-phase) MLP compute input.
    phase_plan:
        Pre-computed phase split plan.
    _apply_mlp_fn:
        Callable ``(MoEMlpComputeInput) -> Tensor``.  Defaults to
        ``unified_apply_mlp``.
    stage_wave_fn:
        Optional callable ``(expert_indices: tuple[int, ...]) -> None`` invoked
        before each non-empty phase's MLP, used by B2 capacity-bounded waves to
        stage that wave's experts into the fixed slot bank (eager prefill only).
        ``None`` (default) preserves the original behavior: weights are assumed
        already resident. Wrapped in a serial ``_CallbackWaveStager``. Mutually
        exclusive with ``wave_stager``.
    wave_stager:
        Optional :class:`WaveStager` giving the overlap-ready two-phase
        (issue/wait) staging contract. Lets a subclass run transfer on a separate
        stream so wave k+1's H2D overlaps wave k's matmul. ``buffer_count`` caps
        in-flight waves.
    prefetch_depth:
        How many waves to issue ahead of the one being computed. ``0`` (default)
        == serial (issue, wait, compute per wave). ``>=1`` enables software
        pipelining; clamped so issued-ahead waves never exceed the stager's
        ``buffer_count`` (no buffer is reused before its wave is consumed).

    Returns
    -------
    Tensor with the same shape / layout as a single-phase ``_apply_mlp`` call.
    """
    import torch

    if stage_wave_fn is not None and wave_stager is not None:
        raise ValueError("pass at most one of stage_wave_fn / wave_stager")
    if wave_stager is None and stage_wave_fn is not None:
        wave_stager = _CallbackWaveStager(stage_wave_fn)

    if _apply_mlp_fn is None:
        from vllm_ascend.ops.fused_moe.moe_mlp import unified_apply_mlp as _default

        _apply_mlp_fn = _default

    # Single phase → fast-path (no slicing overhead). Skipped when a stager is
    # present: B2 needs the per-wave stage call to run even for a lone wave.
    if phase_plan.total_phases == 1 and wave_stager is None:
        return _apply_mlp_fn(mlp_compute_input=mlp_compute_input)

    from vllm_ascend.ops.fused_moe.moe_stage_contracts import MoEMlpComputeInput as _MoEMlpComputeInput

    hidden_states = mlp_compute_input.hidden_states
    group_list = mlp_compute_input.group_list
    group_list_type = mlp_compute_input.group_list_type
    hidden_size = hidden_states.size(-1)
    device = hidden_states.device
    dtype = hidden_states.dtype

    full_output = torch.empty(
        phase_plan.total_tokens,
        hidden_size,
        dtype=dtype,
        device=device,
    )

    # Only non-empty waves participate (0-token waves are pure no-ops).
    waves = [p for p in phase_plan.phases if p.total_tokens > 0]

    def _compute_wave(phase) -> None:
        # group_list / weights are COMPACT (one row per active expert, in the
        # layer's active-expert order), so they must be indexed by the compact
        # position, not by the logical expert id.  See MoEPhase.slice_positions.
        slice_positions = phase.resolved_slice_positions()
        phase_hidden = _extract_phase_tokens(hidden_states, phase.token_slices)
        phase_group_list = _build_phase_group_list(group_list, group_list_type, slice_positions)
        phase_weights = _slice_expert_weights(mlp_compute_input.weights, slice_positions)
        phase_input = _MoEMlpComputeInput(
            hidden_states=phase_hidden,
            group_list=phase_group_list,
            group_list_type=group_list_type,
            dynamic_scale=mlp_compute_input.dynamic_scale,
            topk_scales=mlp_compute_input.topk_scales,
            weights=phase_weights,
            quant=mlp_compute_input.quant,
            fusion=mlp_compute_input.fusion,
            activation=mlp_compute_input.activation,
            need_trans=mlp_compute_input.need_trans,
            dynamic_eplb=mlp_compute_input.dynamic_eplb,
        )
        phase_output = _apply_mlp_fn(mlp_compute_input=phase_input)
        _scatter_phase_output(full_output, phase_output, phase.token_slices)

    if wave_stager is None:
        # No staging contract: plain per-wave compute (weights already resident).
        for phase in waves:
            _compute_wave(phase)
        return full_output

    # Software-pipelined staging. ``ahead`` = how many waves are issued but not yet
    # computed; capped by both prefetch_depth and the stager's buffer_count so a
    # buffer is never reused while its wave is still in flight (correctness guard
    # that lets a single-buffer/serial stager stay strictly serial).
    max_in_flight = min(max(prefetch_depth, 0) + 1, max(wave_stager.buffer_count, 1))
    issued = 0
    # Prime: issue the first ``max_in_flight`` waves.
    while issued < min(max_in_flight, len(waves)):
        wave_stager.issue(issued, waves[issued].expert_indices)
        issued += 1
    for compute_idx in range(len(waves)):
        wave_stager.wait(compute_idx)
        _compute_wave(waves[compute_idx])
        # After consuming this wave's buffer, issue the next not-yet-issued wave.
        if issued < len(waves):
            wave_stager.issue(issued, waves[issued].expert_indices)
            issued += 1

    return full_output


# ---------------------------------------------------------------------------
# Observability helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PhaseSplitProfileEvent:
    name: str
    layer_id: int
    seconds: float
    phase_plan_jsonable: dict[str, object] | None = None
    fail_reason: str | None = None

    def to_jsonable(self) -> dict[str, object]:
        data: dict[str, object] = {
            "event": "phase_split",
            "name": self.name,
            "layer_id": self.layer_id,
            "seconds": round(self.seconds, 6),
        }
        if self.phase_plan_jsonable is not None:
            data["phase_plan"] = self.phase_plan_jsonable
        if self.fail_reason is not None:
            data["fail_reason"] = self.fail_reason
        return data


def _write_phase_split_profile_jsonl(event: PhaseSplitProfileEvent) -> None:
    profile_path = envs.VLLM_ASCEND_MOE_OFFLOAD_PROFILE_PATH
    if not profile_path:
        return
    path = Path(profile_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event.to_jsonable(), sort_keys=True) + "\n")
