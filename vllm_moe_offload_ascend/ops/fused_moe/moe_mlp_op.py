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
"""Graph-compatible MoE offload: the MLP op (SEW-Offload, Option B piece 3).

Third piece of the ``moe_forward`` decomposition (design doc 13):

    [vllm::moe_router]  ->  vllm::moe_offload_stage  ->  [vllm::moe_mlp]
     (captured piece 1)      (splitting op, EAGER)        (captured piece 2)

``moe_mlp`` runs the SAME computation the opaque ``moe_forward`` op runs --
dispatch / gather log2phy[topk_ids] / grouped matmul / combine -- by delegating
to the layer's ``runner._forward_impl`` (the exact call ``_moe_forward`` makes,
moe_runner.py:98). The ONLY difference is that the router decision was already
made by ``vllm::moe_router`` at the top level; this op feeds those
``(topk_weights, topk_ids)`` into the apply-path via the B1 injection registry
so apply consumes them instead of recomputing select_experts.

The staged ``topk_ids`` tensor is the same address-stable tensor that crosses
the side-effect-only ``vllm::moe_offload_stage`` splitting op. Taking it as an
explicit input keeps the ordering dependency real so torch.compile cannot
dead-code-eliminate or reorder the staging op.

First-version constraint: only the ``_shared_experts is None`` path (returns a
single tensor). Shared-expert layers fall back to monolithic moe_forward (guarded
in _select_forward).

Current integration: registered as the captured MLP piece of the SEW dataplane
and wired through the B1 injection registry. Unsupported shared-expert forms
fall back before selecting this path.
"""

from __future__ import annotations

import os

import torch

from vllm.utils.torch_utils import direct_register_custom_op

_PROBE_CALLS = {"mlp": 0}


def _capture_state() -> str:
    try:
        from vllm_moe_offload_ascend.moe_offload.runtime import (
            _is_current_graph_capturing,
        )

        return str(bool(_is_current_graph_capturing()))
    except Exception:
        return "unknown"


def _moe_mlp_impl(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    layer_name: str,
) -> torch.Tensor:
    """Run the routed-experts MLP using the router's precomputed topk.

    Pass the REAL layer name (never the "from_forward_context" sentinel) so the
    lookup does not increment moe_layer_index (index safety in the three-op seam).
    """
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import (
        get_layer_from_name,
    )

    from vllm_moe_offload_ascend.ops.fused_moe import moe_seam_inject

    layer = get_layer_from_name(layer_name)
    layer_id = int(getattr(layer, "layer_id", -1))
    if layer_id < 0:
        import warnings
        warnings.warn(
            f"moe_mlp seam: layer {layer_name!r} has no layer_id attribute; "
            "topk injection will target key=-1 and may collide with other "
            "unidentified layers. Set layer.layer_id to suppress this warning.",
            stacklevel=2,
        )

    if os.environ.get("SEW_DECODE_PROBE"):
        _PROBE_CALLS["mlp"] += 1
        print(
            f"SEW_MLP branch=IMPL count={_PROBE_CALLS['mlp']} "
            f"layer={layer_id} tokens={int(hidden_states.shape[0])} "
            f"capturing={_capture_state()}",
            flush=True,
        )

    moe_seam_inject.set_injected_topk(layer_id, topk_weights, topk_ids)
    try:
        # Same delegation as _moe_forward (moe_runner.py:98). With the injection
        # set, the apply-path short-circuit consumes (topk_weights, topk_ids)
        # instead of recomputing select_experts.
        result = layer.runner._forward_impl(
            layer,
            hidden_states,
            router_logits,
            shared_experts_input,
            input_ids,
        )
    finally:
        moe_seam_inject.clear_injected_topk(layer_id)

    # First-version constraint: _shared_experts is None -> single tensor.
    assert not isinstance(result, tuple), (
        "moe_mlp seam supports only the _shared_experts is None path"
    )
    return result


def _moe_mlp_fake(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    layer_name: str,
) -> torch.Tensor:
    # Routed output has the same shape/dtype as hidden_states (mirrors
    # _moe_forward_fake, moe_runner.py:114).
    return torch.empty_like(hidden_states)


direct_register_custom_op(
    op_name="moe_mlp",
    op_func=_moe_mlp_impl,
    fake_impl=_moe_mlp_fake,
    mutates_args=["hidden_states"],
    dispatch_key="PrivateUse1",
)
