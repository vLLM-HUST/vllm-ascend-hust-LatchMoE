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
"""Graph-compatible MoE offload: the ROUTER op (SEW-Offload, Option B piece 1).

Part of the three-way decomposition of the opaque ``vllm::moe_forward`` custom
op. The monolithic
``moe_forward`` hides ``select_experts`` *inside* its body, so a staging seam
placed there is invisible to the FX graph splitter (R3 negative result). To make
staging a real top-level split point we hoist the router decision into this
opaque op:

    [vllm::moe_router]  ->  vllm::moe_offload_stage  ->  [vllm::moe_mlp]
     (captured piece 1)      (splitting op, EAGER)        (captured piece 2)

This op is a FAITHFUL wrapper of the apply-path ``select_experts`` call
(vllm_ascend/ops/fused_moe/fused_moe.py: ``AscendUnquantizedFusedMoEMethod.apply``
lines 244-257). It does NOT change router semantics -- same logits, top-k, group,
renormalize, scoring, e_score_bias, num_experts. The ONLY change is *where* the
single ``select_experts`` call happens (now a top-level op instead of buried in
the quant method). Bit-equivalence of the produced ``(topk_weights, topk_ids)``
is asserted by UT (CPU, native path) and NPU capture validation (V-B/V-C).

First-version constraints (see design doc decisions):
  - ``custom_routing_function`` MUST be None (a Callable cannot cross an op
    boundary). Callers with a custom routing function fall back to the
    monolithic ``super().forward()`` path.
  - mirrors the apply-path call exactly, including fused/mix-placement shared
    suffixes when the materialized layer enables them.

Current integration: the explicit router core and the layer-name indirect
router are registered for the SEW router -> stage -> MLP dataplane. The seam
path is still guarded; unsupported routing forms fall back to the monolithic
``moe_forward`` path before graph capture.
"""

from __future__ import annotations

import os

import torch

from vllm.utils.torch_utils import direct_register_custom_op

_PROBE_CALLS = {"router_indirect": 0, "router_impl": 0}


def _capture_state() -> str:
    try:
        from vllm_moe_offload_ascend.moe_offload.runtime import (
            _is_current_graph_capturing,
        )

        return str(bool(_is_current_graph_capturing()))
    except Exception:
        return "unknown"


def _moe_router_impl(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    top_k: int,
    use_grouped_topk: bool,
    renormalize: bool,
    topk_group: int | None,
    num_expert_group: int | None,
    scoring_func: str,
    routed_scaling_factor: float,
    e_score_correction_bias: torch.Tensor | None,
    num_experts: int,
    mix_placement: bool = False,
    num_shared_experts: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run expert selection exactly as the apply-path does, returning
    ``(topk_weights, topk_ids)``.

    Faithful wrapper: every argument is forwarded 1:1 to ``select_experts`` with
    the same keyword it carries in
    ``AscendUnquantizedFusedMoEMethod.apply``. ``custom_routing_function`` is
    pinned to ``None`` (first-version constraint) and ``num_experts`` here is the
    ``num_logical_experts`` value the apply path computes before the call.
    """
    # Lazy import keeps this module importable without pulling the full fused_moe
    # stack at registration time (e.g. in unit tests that only check the schema).
    from vllm_ascend.ops.fused_moe.experts_selector import select_experts

    if os.environ.get("SEW_DECODE_PROBE"):
        _PROBE_CALLS["router_impl"] += 1
        print(
            f"SEW_ROUTER branch=IMPL count={_PROBE_CALLS['router_impl']} "
            f"tokens={int(hidden_states.shape[0])} capturing={_capture_state()}",
            flush=True,
        )

    topk_weights, topk_ids = select_experts(
        hidden_states=hidden_states,
        router_logits=router_logits,
        top_k=top_k,
        use_grouped_topk=use_grouped_topk,
        renormalize=renormalize,
        topk_group=topk_group,
        num_expert_group=num_expert_group,
        custom_routing_function=None,
        scoring_func=scoring_func,
        routed_scaling_factor=routed_scaling_factor,
        e_score_correction_bias=e_score_correction_bias,
        num_experts=num_experts,
        mix_placement=bool(mix_placement),
        num_logical_experts=int(num_experts),
        num_shared_experts=int(num_shared_experts),
    )
    return topk_weights, topk_ids


def _moe_router_fake(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    top_k: int,
    use_grouped_topk: bool,
    renormalize: bool,
    topk_group: int | None,
    num_expert_group: int | None,
    scoring_func: str,
    routed_scaling_factor: float,
    e_score_correction_bias: torch.Tensor | None,
    num_experts: int,
    mix_placement: bool = False,
    num_shared_experts: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Trace-time shape/dtype proxy. Fused mix-placement appends one shared pair
    # per shared expert to the routed top-k suffix.
    #   - topk_weights: real _native_select_experts casts to hidden_states.dtype
    #     (experts_selector.py line ~322: `topk_weights = topk_weights.to(x.dtype)`).
    #     Using router_logits.dtype was wrong for indirect-router models where the
    #     gate output is float32 but hidden_states is bf16, causing torch.compile
    #     to specialize on the wrong dtype and potentially mis-codegen downstream.
    #   - topk_ids: int32 selected logical-expert ids.
    num_tokens = hidden_states.shape[0]
    topk_weights = torch.empty(
        (num_tokens, top_k + (int(num_shared_experts) if mix_placement else 0)),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    topk_ids = torch.empty(
        (num_tokens, top_k + (int(num_shared_experts) if mix_placement else 0)),
        dtype=torch.int32,
        device=hidden_states.device,
    )
    return topk_weights, topk_ids


direct_register_custom_op(
    op_name="moe_router",
    op_func=_moe_router_impl,
    fake_impl=_moe_router_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)


# ---------------------------------------------------------------------------
# Layer-name-indirect router op (P2a).
#
# The explicit-scalar ``vllm::moe_router`` above is the tested numerical core.
# But the seam entry (a per-layer runner method, P2c) cannot read the routing
# scalars at trace time without a trace-time layer lookup, and Ascend's
# apply-path reads those scalars from the *layer* (AscendFusedMoE: self.top_k,
# self.scoring_func, ...), not the runner. So we mirror the proven moe_forward
# indirection: this op takes ``layer_name``, resolves the layer at runtime via
# ``get_layer_from_name``, reads the SAME scalars the apply-path reads
# (fused_moe.py:690-711), computes ``num_logical_experts`` the SAME way
# (get_moe_num_logical_experts), and delegates to the explicit-scalar core.
#
# IMPORTANT (index safety): pass the REAL layer name string here. The top-level
# seam entry consumes the "from_forward_context" sentinel exactly once; router
# and MLP then use direct real-name lookups without incrementing the index again.
# ---------------------------------------------------------------------------
def _moe_router_indirect_impl(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import (
        get_layer_from_name,
    )

    from vllm_ascend.quantization.methods.base import get_moe_num_logical_experts

    layer = get_layer_from_name(layer_name)

    if os.environ.get("SEW_DECODE_PROBE"):
        _PROBE_CALLS["router_indirect"] += 1
        print(
            f"SEW_ROUTER branch=INDIRECT count={_PROBE_CALLS['router_indirect']} "
            f"layer={getattr(layer, 'layer_id', '?')} "
            f"tokens={int(hidden_states.shape[0])} capturing={_capture_state()}",
            flush=True,
        )

    # custom_routing_function is a Callable and cannot cross an op boundary; the
    # seam path is only selected when it is None (guarded in _select_forward).
    assert getattr(layer, "custom_routing_function", None) is None, (
        "moe_router seam requires custom_routing_function=None"
    )

    # Internal-router models (e.g. Qwen3-MoE: is_internal_router == gate is not
    # None) hold the gate on the runner and pass router_logits == hidden_states as
    # a PLACEHOLDER (qwen3_moe.py:240). The real logits are produced by the gate
    # inside _forward_impl (moe_runner.py:710): `router_logits, _ = gate(h)`. To
    # route correctly here we must apply the SAME gate to hidden_states BEFORE
    # select_experts. This is a faithful relocation of that exact matmul (same
    # gate module, same input) into the captured router piece -- NOT a router-
    # semantics change. _forward_impl recomputes the identical logits later; the
    # B1 topk injection makes that recompute dead weight on the seam path.
    runner = getattr(layer, "runner", None)
    gate = getattr(runner, "gate", None)
    if gate is not None:
        router_logits, _ = gate(hidden_states)

    num_shared_experts = getattr(layer, "n_shared_experts", 0) or 0
    mix_placement = bool(getattr(layer, "mix_placement", False))
    num_logical_experts = get_moe_num_logical_experts(
        layer,
        layer.moe_config.num_experts,
        global_redundant_expert_num=getattr(layer, "global_redundant_expert_num", 0),
        num_shared_experts=num_shared_experts,
    )

    # AscendFusedMoE preserves the model value here because vLLM may rewrite
    # ``layer.routed_scaling_factor`` to 1.0 when the scale is applied at the
    # output.  The native apply path uses this preserved value, so the seam
    # must use it as well to keep router weights bit-for-bit aligned.
    routed_scaling_factor = getattr(
        layer,
        "_original_routed_scaling_factor",
        getattr(layer, "routed_scaling_factor", 1.0),
    )
    topk_weights, topk_ids = _moe_router_impl(
        hidden_states=hidden_states,
        router_logits=router_logits,
        top_k=layer.top_k,
        use_grouped_topk=layer.use_grouped_topk,
        renormalize=layer.renormalize,
        topk_group=layer.topk_group,
        num_expert_group=layer.num_expert_group,
        scoring_func=layer.scoring_func,
        routed_scaling_factor=routed_scaling_factor,
        e_score_correction_bias=layer.e_score_correction_bias,
        num_experts=num_logical_experts,
        mix_placement=mix_placement,
        num_shared_experts=num_shared_experts,
    )
    if os.getenv("VLLM_ASCEND_MOE_ROUTER_PARITY_PATH") and _capture_state() != "True":
        from vllm_moe_offload_ascend.moe_offload.router_parity import (
            record_router_snapshot,
        )

        record_router_snapshot(
            role="seam",
            layer_id=int(getattr(layer, "layer_id", -1)),
            router_logits=router_logits,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )
    return topk_weights, topk_ids


def _moe_router_indirect_fake(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import (
        get_layer_from_name,
    )

    layer = get_layer_from_name(layer_name)
    return _moe_router_fake(
        hidden_states=hidden_states,
        router_logits=router_logits,
        top_k=layer.top_k,
        use_grouped_topk=layer.use_grouped_topk,
        renormalize=layer.renormalize,
        topk_group=layer.topk_group,
        num_expert_group=layer.num_expert_group,
        scoring_func=layer.scoring_func,
        routed_scaling_factor=getattr(
            layer,
            "_original_routed_scaling_factor",
            getattr(layer, "routed_scaling_factor", 1.0),
        ),
        e_score_correction_bias=layer.e_score_correction_bias,
        num_experts=layer.moe_config.num_experts,
        mix_placement=bool(getattr(layer, "mix_placement", False)),
        num_shared_experts=int(getattr(layer, "n_shared_experts", 0) or 0),
    )


direct_register_custom_op(
    op_name="moe_router_indirect",
    op_func=_moe_router_indirect_impl,
    fake_impl=_moe_router_indirect_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)
