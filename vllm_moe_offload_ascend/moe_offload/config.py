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

from vllm_ascend import envs
from vllm_moe_offload_ascend.moe_offload.tiered_residency import (
    TieredResidencyPolicy,
    parse_comma_separated_ints,
)


def _bool_env(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class MoeOffloadConfig:
    enabled: bool = False
    trace_only: bool = False
    num_slots: int = 0
    policy: str = "deadline"
    max_phases: int = 2
    async_load: bool = False
    trace_max_records: int = 4096
    # MVP-D.9: tiered residency (default off / empty)
    resident_layer_ids: frozenset[int] = frozenset()
    release_original_expert_weights: bool = False
    # MVP-D.10: dynamic-count layered runtime path selector (default off).
    layered_runtime: bool = False
    fanout_threshold: int = 0
    # MVP-D.11: post-dispatch phase split semantic prototype (default off).
    phase_split_enabled: bool = False
    # Option 2: graph-compatible offload via decision/execution decoupling.
    # When on, capture uses the no-host-sync capture-safe slot plan so ACLGraph
    # can record the MoE region; the data-dependent decision + H2D staging is
    # done eager before replay. Default off => current eager-only behavior.
    graph_compatible_offload: bool = False
    # Regime B path ①: per-step staging seam (vllm::moe_offload_stage splitting
    # op between router and grouped MLP). Supports num_slots < n. When on, the
    # load-time full-residency hook is skipped for offloaded layers and staging
    # happens per-step eager between captured pieces. Default off => Regime A.
    offload_stage_seam: bool = False
    # Regime B "B2": wave-streamed prefill when eager-prefill active union exceeds
    # num_slots (capacity-bounded waves instead of fail-closed). Prefill+eager only;
    # decode keeps the single-wave B1 path. Default off.
    b2_wave_prefill: bool = False
    # Stage-1 out-of-core loading: allocate offloaded experts on CPU during
    # model init and never materialize the full expert tensors on NPU.
    cpu_first_load: bool = False
    # Pin CPU expert tensors so async CPU->NPU copies can be host-nonblocking.
    # Defaults to async_load when the env var is absent.
    pin_host_memory: bool | None = None
    # B2 Prefill software pipeline knobs. Depth=1 with two buffers is the current
    # double-buffer behavior; larger values are useful for measuring H2D/MLP
    # overlap on workloads with short per-wave compute.
    transfer_aware_wave_schedule: bool = True
    prefill_prefetch_depth: int = 1
    prefill_buffer_count: int = 2
    # Prefer extra hit-only waves over copying READY main-slot experts into a
    # temporary miss wave. This preserves the fixed HBM slot budget; it only
    # changes B2 wave membership to avoid device-to-device staging traffic.
    b2_avoid_mixed_wave_d2d: bool = True
    # P1-C scaffold: optional profiling-suite plan for stable grouped compute buckets.
    compute_bucket_plan_path: str = ""
    # Non-offload MoE GroupedMatmul diagnostics and plan inputs.
    gmm_trace_path: str = ""
    gmm_profile_path: str = ""
    gmm_bucket_plan_path: str = ""
    # Optional serving-shape hint used only to keep B2 overflow fallback narrow
    # when vLLM profile/dummy prefill does not expose phase metadata.
    max_num_seqs_hint: int = 0

    def __post_init__(self) -> None:
        if self.gmm_bucket_plan_path:
            object.__setattr__(self, "compute_bucket_plan_path", self.gmm_bucket_plan_path)

    @classmethod
    def from_env(cls) -> "MoeOffloadConfig":
        return cls(
            enabled=envs.VLLM_ASCEND_MOE_OFFLOAD_ENABLED,
            trace_only=envs.VLLM_ASCEND_MOE_OFFLOAD_TRACE_ONLY,
            num_slots=envs.VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS,
            policy=envs.VLLM_ASCEND_MOE_OFFLOAD_POLICY,
            max_phases=envs.VLLM_ASCEND_MOE_OFFLOAD_MAX_PHASES,
            async_load=envs.VLLM_ASCEND_MOE_OFFLOAD_ASYNC_LOAD,
            trace_max_records=envs.VLLM_ASCEND_MOE_OFFLOAD_TRACE_MAX_RECORDS,
            resident_layer_ids=parse_comma_separated_ints(
                envs.VLLM_ASCEND_MOE_OFFLOAD_RESIDENT_LAYER_IDS
            ),
            release_original_expert_weights=envs.VLLM_ASCEND_MOE_OFFLOAD_RELEASE_ORIGINAL_EXPERT_WEIGHTS,
            layered_runtime=envs.VLLM_ASCEND_MOE_OFFLOAD_LAYERED_RUNTIME,
            fanout_threshold=envs.VLLM_ASCEND_MOE_OFFLOAD_FANOUT_THRESHOLD,
            phase_split_enabled=envs.VLLM_ASCEND_MOE_OFFLOAD_PHASE_SPLIT,
            graph_compatible_offload=envs.VLLM_ASCEND_MOE_OFFLOAD_GRAPH_COMPATIBLE,
            offload_stage_seam=envs.VLLM_ASCEND_MOE_OFFLOAD_STAGE_SEAM,
            b2_wave_prefill=envs.VLLM_ASCEND_MOE_OFFLOAD_B2_WAVE_PREFILL,
            cpu_first_load=envs.VLLM_ASCEND_MOE_OFFLOAD_CPU_FIRST_LOAD,
            pin_host_memory=envs.VLLM_ASCEND_MOE_OFFLOAD_PIN_HOST_MEMORY,
            transfer_aware_wave_schedule=envs.VLLM_ASCEND_MOE_OFFLOAD_TRANSFER_AWARE_SCHEDULE,
            prefill_prefetch_depth=envs.VLLM_ASCEND_MOE_OFFLOAD_PREFILL_PREFETCH_DEPTH,
            prefill_buffer_count=envs.VLLM_ASCEND_MOE_OFFLOAD_PREFILL_BUFFER_COUNT,
            b2_avoid_mixed_wave_d2d=_bool_env(
                "VLLM_ASCEND_MOE_OFFLOAD_B2_AVOID_MIXED_D2D",
                "1",
            ),
            compute_bucket_plan_path=envs.VLLM_ASCEND_MOE_COMPUTE_BUCKET_PLAN_PATH,
            gmm_trace_path=envs.VLLM_ASCEND_MOE_GMM_TRACE_PATH,
            gmm_profile_path=envs.VLLM_ASCEND_MOE_GMM_PROFILE_PATH,
            gmm_bucket_plan_path=envs.VLLM_ASCEND_MOE_GMM_BUCKET_PLAN_PATH,
            max_num_seqs_hint=int(
                os.getenv(
                    "VLLM_ASCEND_MOE_OFFLOAD_MAX_NUM_SEQS_HINT",
                    "0",
                )
            ),
        )

    @property
    def tiered_residency(self) -> TieredResidencyPolicy:
        return TieredResidencyPolicy(
            resident_layer_ids=self.resident_layer_ids,
            release_original_expert_weights=self.release_original_expert_weights,
        )

    @property
    def should_trace(self) -> bool:
        return self.enabled and self.trace_only

    @property
    def should_trace_gmm(self) -> bool:
        return bool(self.gmm_trace_path) or self.should_trace

    @property
    def effective_fanout_threshold(self) -> int:
        if self.fanout_threshold > 0:
            return self.fanout_threshold
        return self.num_slots

    @property
    def should_pin_host_memory(self) -> bool:
        if self.pin_host_memory is None:
            return bool(self.async_load)
        return bool(self.pin_host_memory)

    @property
    def effective_prefill_prefetch_depth(self) -> int:
        return max(0, int(self.prefill_prefetch_depth))

    @property
    def effective_prefill_buffer_count(self) -> int:
        return max(1, int(self.prefill_buffer_count))
