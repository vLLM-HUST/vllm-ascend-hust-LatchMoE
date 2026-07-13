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

"""Pipe-level npu.Event timing for the MoE execution pipeline (P0: trace-only).

This module records Stage T (transfer), Stage R (routing reorder),
Stage C (compute), and Stage M (combine) elapsed times using
``torch.npu.Event`` without changing execution or introducing overlap.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm_ascend import envs
from vllm_moe_offload_ascend.moe_offload.profile_io import append_jsonl

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class MoePipelineTiming:
    layer_id: int
    step_id: int

    # Raw npu.Event elapsed times in milliseconds.
    stage_t_ms: float  # _maybe_apply_moe_offload_plan (transfer + slot plan)
    stage_r_ms: float  # token_dispatch (routing reorder)
    stage_c_ms: float  # _apply_mlp (grouped matmul + activation)
    stage_m_ms: float  # token_combine

    # Reference: no-offload baseline fields (populated when running no_offload mode).
    baseline_stage_r_ms: float | None = None
    baseline_stage_c_ms: float | None = None
    baseline_stage_m_ms: float | None = None

    # Derived ratios (computed by caller, not the profiler).
    t_total_ms: float | None = None  # T + R + C + M
    t_frac: float | None = None  # T / (T+R+C+M)
    r_frac: float | None = None
    c_frac: float | None = None
    m_frac: float | None = None
    # Overlap potential: how much of T could be hidden by (R+C) if they ran in parallel.
    overlap_potential_ratio: float | None = None  # min(1, (R+C)/T)
    total_ms_excl_t: float | None = None  # R+C+M (what could overlap with T)

    def to_jsonable(self) -> dict[str, object]:
        return {
            "layer_id": self.layer_id,
            "step_id": self.step_id,
            "stage_t_ms": round(self.stage_t_ms, 4),
            "stage_r_ms": round(self.stage_r_ms, 4),
            "stage_c_ms": round(self.stage_c_ms, 4),
            "stage_m_ms": round(self.stage_m_ms, 4),
            **(
                {
                    "baseline_stage_r_ms": round(self.baseline_stage_r_ms, 4),
                    "baseline_stage_c_ms": round(self.baseline_stage_c_ms, 4),
                    "baseline_stage_m_ms": round(self.baseline_stage_m_ms, 4),
                }
                if self.baseline_stage_r_ms is not None
                else {}
            ),
            **(
                {
                    "t_total_ms": round(self.t_total_ms, 4),
                    "t_frac": round(self.t_frac, 4),
                    "r_frac": round(self.r_frac, 4),
                    "c_frac": round(self.c_frac, 4),
                    "m_frac": round(self.m_frac, 4),
                    "overlap_potential_ratio": round(self.overlap_potential_ratio, 4),
                    "total_ms_excl_t": round(self.total_ms_excl_t, 4),
                }
                if self.t_total_ms is not None
                else {}
            ),
        }


@dataclass(frozen=True)
class MoePipelineDetailTiming:
    layer_id: int
    step_id: int
    name: str
    duration_ms: float
    source: str = "npu_event"
    payload: dict[str, object] | None = None

    def to_jsonable(self) -> dict[str, object]:
        data: dict[str, object] = {
            "layer_id": self.layer_id,
            "step_id": self.step_id,
            "name": self.name,
            "duration_ms": round(self.duration_ms, 4),
            "source": self.source,
        }
        if self.payload:
            data["payload"] = self.payload
        return data


@dataclass
class _MoePipelineDetailEvent:
    name: str
    start: "torch.npu.Event"
    end: "torch.npu.Event"
    payload: dict[str, object] | None = None


@dataclass
class _MoePipelineWallDetailEvent:
    name: str
    duration_ms: float
    payload: dict[str, object] | None = None


@dataclass
class _MoePipelineDetailContext:
    layer_id: int
    step_id: int
    events: list[_MoePipelineDetailEvent]
    wall_events: list[_MoePipelineWallDetailEvent]


_detail_context: ContextVar[_MoePipelineDetailContext | None] = ContextVar(
    "moe_pipeline_detail_context",
    default=None,
)


class MoePipelineProfiler:
    """Collect npu.Event-based stage timing for the MoE fused_experts pipeline.

    Usage inside ``MoECommMethod.fused_experts()``::

        profiler = get_moe_pipeline_profiler()
        e0 = profiler.record()  # before offload plan
        # ... _maybe_apply_moe_offload_plan ...
        e1 = profiler.record()  # after offload plan / before dispatch
        # ... token_dispatch ...
        e2 = profiler.record()  # after dispatch / before compute
        # ... _apply_mlp ...
        e3 = profiler.record()  # after compute / before combine
        # ... token_combine ...
        e4 = profiler.record()  # after combine

        profiler.commit(layer_id=..., step_id=..., events=(e0,e1,e2,e3,e4))
    """

    def __init__(self) -> None:
        self._timings: list[MoePipelineTiming] = []
        self._detail_timings: list[MoePipelineDetailTiming] = []

    @property
    def enabled(self) -> bool:
        return bool(getattr(envs, "VLLM_ASCEND_MOE_PIPELINE_PROFILING", False))

    def record(self) -> "torch.npu.Event":
        import torch

        event = torch.npu.Event(enable_timing=True)
        torch.npu.current_stream().record_event(event)
        return event

    @contextmanager
    def detail_context(self, *, layer_id: int, step_id: int):
        ctx = _MoePipelineDetailContext(
            layer_id=int(layer_id),
            step_id=int(step_id),
            events=[],
            wall_events=[],
        )
        token = _detail_context.set(ctx)
        try:
            yield ctx
        finally:
            _detail_context.reset(token)

    def add_detail_event(
        self,
        name: str,
        *,
        start: "torch.npu.Event",
        end: "torch.npu.Event",
        payload: dict[str, object] | None = None,
    ) -> None:
        if not self.enabled:
            return
        ctx = _detail_context.get()
        if ctx is None:
            return
        ctx.events.append(
            _MoePipelineDetailEvent(
                name=str(name),
                start=start,
                end=end,
                payload=payload,
            )
        )

    def add_wall_detail_timing(
        self,
        name: str,
        *,
        duration_ms: float,
        payload: dict[str, object] | None = None,
    ) -> None:
        if not self.enabled:
            return
        ctx = _detail_context.get()
        if ctx is None:
            return
        ctx.wall_events.append(
            _MoePipelineWallDetailEvent(
                name=str(name),
                duration_ms=float(duration_ms),
                payload=payload,
            )
        )

    def commit_detail_context(self, ctx: _MoePipelineDetailContext | None) -> list[MoePipelineDetailTiming]:
        if not self.enabled or ctx is None:
            return []

        timings: list[MoePipelineDetailTiming] = []
        for event in ctx.events:
            duration_ms = self._elapsed_ms(event.start, event.end)
            timing = MoePipelineDetailTiming(
                layer_id=ctx.layer_id,
                step_id=ctx.step_id,
                name=event.name,
                duration_ms=duration_ms,
                payload=event.payload,
            )
            timings.append(timing)
            self._detail_timings.append(timing)
            self._write_detail_jsonl(timing)
        for event in ctx.wall_events:
            timing = MoePipelineDetailTiming(
                layer_id=ctx.layer_id,
                step_id=ctx.step_id,
                name=event.name,
                duration_ms=event.duration_ms,
                source="cpu_wall",
                payload=event.payload,
            )
            timings.append(timing)
            self._detail_timings.append(timing)
            self._write_detail_jsonl(timing)
        return timings

    def commit(
        self,
        *,
        layer_id: int,
        step_id: int,
        events: tuple[
            "torch.npu.Event",  # e0: before transfer
            "torch.npu.Event",  # e1: after transfer / before dispatch
            "torch.npu.Event",  # e2: after dispatch / before compute
            "torch.npu.Event",  # e3: after compute / before combine
            "torch.npu.Event",  # e4: after combine
        ],
    ) -> MoePipelineTiming | None:
        if not self.enabled or len(events) != 5:
            return None

        e0, e1, e2, e3, e4 = events

        timing = MoePipelineTiming(
            layer_id=int(layer_id),
            step_id=int(step_id),
            stage_t_ms=self._elapsed_ms(e0, e1),
            stage_r_ms=self._elapsed_ms(e1, e2),
            stage_c_ms=self._elapsed_ms(e2, e3),
            stage_m_ms=self._elapsed_ms(e3, e4),
        )
        self._timings.append(timing)
        self._write_jsonl(timing)
        return timing

    def summarize(self) -> dict[str, object]:
        if not self._timings:
            return {"count": 0, "stages": {}}

        def _mean(values: list[float]) -> float:
            return sum(values) / len(values) if values else 0.0

        t_list = [t.stage_t_ms for t in self._timings]
        r_list = [t.stage_r_ms for t in self._timings]
        c_list = [t.stage_c_ms for t in self._timings]
        m_list = [t.stage_m_ms for t in self._timings]
        total_list = [a + b + c + d for a, b, c, d in zip(t_list, r_list, c_list, m_list)]

        mean_total = _mean(total_list)
        mean_t = _mean(t_list)
        mean_r = _mean(r_list)
        mean_c = _mean(c_list)
        mean_m = _mean(m_list)
        mean_excl_t = _mean([r + c + m for r, c, m in zip(r_list, c_list, m_list)])

        return {
            "count": len(self._timings),
            "layer_ids": sorted({t.layer_id for t in self._timings}),
            "stages": {
                "stage_t_ms": {"mean": round(mean_t, 4)},
                "stage_r_ms": {"mean": round(mean_r, 4)},
                "stage_c_ms": {"mean": round(mean_c, 4)},
                "stage_m_ms": {"mean": round(mean_m, 4)},
            },
            "total_pipeline_ms": {"mean": round(mean_total, 4)},
            "total_excl_transfer_ms": {"mean": round(mean_excl_t, 4)},
            "fractions": {
                "t_frac": round(mean_t / mean_total, 4) if mean_total > 0 else 0,
                "r_frac": round(mean_r / mean_total, 4) if mean_total > 0 else 0,
                "c_frac": round(mean_c / mean_total, 4) if mean_total > 0 else 0,
                "m_frac": round(mean_m / mean_total, 4) if mean_total > 0 else 0,
            },
            "overlap_potential": {
                "r_plus_c_ms": round(mean_r + mean_c, 4),
                "r_plus_c_ratio_to_t": round((mean_r + mean_c) / mean_t, 4) if mean_t > 0 else 0,
                "assessment": (
                    "good_overlap_candidate"
                    if mean_t > 0 and (mean_r + mean_c) >= mean_t * 0.5
                    else "marginal_overlap"
                    if mean_t > 0 and (mean_r + mean_c) >= mean_t * 0.2
                    else "transfer_dominated"
                ),
            },
            "detail_count": len(self._detail_timings),
        }

    @staticmethod
    def _elapsed_ms(start: "torch.npu.Event", end: "torch.npu.Event") -> float:
        try:
            return float(start.elapsed_time(end))
        except Exception:
            return -1.0

    @staticmethod
    def _write_jsonl(timing: MoePipelineTiming) -> None:
        profile_path = envs.VLLM_ASCEND_MOE_OFFLOAD_PROFILE_PATH
        if not profile_path:
            return
        entry = {"event": "moe_pipeline_timing", **timing.to_jsonable()}
        append_jsonl(profile_path, entry)

    @staticmethod
    def _write_detail_jsonl(timing: MoePipelineDetailTiming) -> None:
        profile_path = envs.VLLM_ASCEND_MOE_OFFLOAD_PROFILE_PATH
        if not profile_path:
            return
        entry = {"event": "moe_pipeline_detail_timing", **timing.to_jsonable()}
        append_jsonl(profile_path, entry)


_pipeline_profiler: MoePipelineProfiler | None = None


def get_moe_pipeline_profiler() -> MoePipelineProfiler:
    global _pipeline_profiler
    if _pipeline_profiler is None:
        _pipeline_profiler = MoePipelineProfiler()
    return _pipeline_profiler


def reset_moe_pipeline_profiler() -> None:
    global _pipeline_profiler
    _pipeline_profiler = None
