from __future__ import annotations

import json
from types import SimpleNamespace

from vllm_moe_offload_ascend.moe_offload.capabilities import (
    describe_checkpoint_config,
    describe_layer_capability,
    evaluate_support,
)


class UnquantizedMethod:
    pass


def _layer(**overrides):
    values = {
        "moe_config": SimpleNamespace(num_experts=64, dp_size=1, ep_size=1, tp_size=1, pcp_size=1),
        "quant_method": UnquantizedMethod(),
        "n_shared_experts": 0,
        "mix_placement": False,
        "top_k": 4,
        "use_grouped_topk": False,
        "topk_group": None,
        "num_expert_group": None,
        "renormalize": True,
        "scoring_func": "softmax",
        "e_score_correction_bias": None,
        "routed_scaling_factor": 1.0,
        "custom_routing_function": None,
        "is_internal_router": False,
        "multistream_overlap_gate": False,
        "multistream_overlap_shared_expert": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _runner(**overrides):
    values = {"_shared_experts": None, "shared_expert_gate": None, "gate": None}
    values.update(overrides)
    return SimpleNamespace(**values)


def test_routed_only_descriptor_is_serializable_and_implemented():
    descriptor = describe_layer_capability(_layer(), _runner())

    assert descriptor.output_contract == "routed_tensor"
    assert descriptor.shared_mode == "none"
    assert descriptor.router_owner == "external_logits"
    assert evaluate_support(descriptor).state == "implemented"
    assert json.loads(descriptor.fingerprint())["routed_expert_count"] == 64


def test_external_shared_complex_router_descriptor_is_implemented():
    shared_gate = object()
    shared = SimpleNamespace(expert_gate=shared_gate)
    descriptor = describe_layer_capability(
        _layer(
            n_shared_experts=1,
            use_grouped_topk=True,
            topk_group=2,
            num_expert_group=8,
            scoring_func="sigmoid",
            e_score_correction_bias=object(),
            routed_scaling_factor=1.8,
        ),
        _runner(_shared_experts=shared),
    )

    assert descriptor.output_contract == "shared_routed_tuple"
    assert descriptor.shared_mode == "external_resident"
    assert descriptor.shared_activation == "gated"
    assert descriptor.selection.scoring_function == "sigmoid"
    assert descriptor.selection.correction_bias is True
    assert descriptor.selection.routed_scaling_factor == 1.8
    assert evaluate_support(descriptor).enabled is True


def test_external_shared_multistream_descriptor_is_implemented():
    descriptor = describe_layer_capability(
        _layer(
            n_shared_experts=1,
            multistream_overlap_shared_expert=True,
        ),
        _runner(_shared_experts=SimpleNamespace(expert_gate=object())),
    )

    assert descriptor.shared_mode == "external_resident"
    assert descriptor.overlap_mode == "shared_expert_multistream"
    assert evaluate_support(descriptor).enabled is True


def test_other_overlap_modes_remain_fail_closed():
    descriptor = describe_layer_capability(
        _layer(
            n_shared_experts=1,
            multistream_overlap_gate=True,
        ),
        _runner(_shared_experts=SimpleNamespace(expert_gate=object())),
    )

    support = evaluate_support(descriptor)

    assert support.state == "unsupported"
    assert "unsupported_overlap_mode:shared_gate_multistream" in support.blockers


def test_internal_router_and_gated_shared_do_not_depend_on_model_name():
    descriptor = describe_layer_capability(
        _layer(n_shared_experts=1),
        _runner(_shared_experts=SimpleNamespace(expert_gate=object()), gate=object()),
    )

    assert descriptor.router_owner == "internal_gate"
    assert descriptor.shared_activation == "gated"
    assert evaluate_support(descriptor).state == "implemented"


def test_fused_mix_placement_is_described_and_supported_by_its_own_lane():
    descriptor = describe_layer_capability(
        _layer(
            n_shared_experts=2,
            mix_placement=True,
            moe_config=SimpleNamespace(
                # FusedMoE materializes routed + shared rows, while retaining
                # num_logical_experts as the routed-domain contract.
                num_experts=66,
                num_logical_experts=64,
                dp_size=1,
                ep_size=1,
                tp_size=1,
                pcp_size=1,
            ),
        ),
        _runner(),
    )
    support = evaluate_support(descriptor)

    assert descriptor.shared_mode == "fused_mix_placement"
    assert descriptor.routed_expert_count == 64
    assert support.state == "implemented"
    assert support.blockers == ()


def test_fused_mix_placement_derives_routed_count_from_materialized_suffix():
    descriptor = describe_layer_capability(
        _layer(
            n_shared_experts=2,
            mix_placement=True,
            moe_config=SimpleNamespace(
                num_experts=66,
                dp_size=1,
                ep_size=1,
                tp_size=1,
                pcp_size=1,
            ),
        ),
        _runner(),
    )

    assert descriptor.routed_expert_count == 64


def test_python_router_callable_and_multicard_are_fail_closed():
    descriptor = describe_layer_capability(
        _layer(
            custom_routing_function=lambda *_args: None,
            moe_config=SimpleNamespace(num_experts=64, dp_size=1, ep_size=1, tp_size=2, pcp_size=1),
        ),
        _runner(),
    )
    support = evaluate_support(descriptor)

    assert support.state == "unsupported"
    assert "unsupported_router_owner:unknown" in support.blockers
    assert "unsupported_parallel_mode:multi_npu" in support.blockers


def test_checkpoint_config_describes_glm_without_a_model_name_rule():
    descriptor = describe_checkpoint_config(
        {
            "n_routed_experts": 64,
            "n_shared_experts": 1,
            "num_experts_per_tok": 4,
            "topk_method": "noaux_tc",
            "topk_group": 1,
            "n_group": 1,
            "norm_topk_prob": True,
            "routed_scaling_factor": 1.8,
        }
    )

    assert descriptor.shared_mode == "external_resident"
    assert descriptor.selection.scoring_function == "sigmoid"
    assert descriptor.selection.correction_bias is True
    assert descriptor.router_owner == "unknown"
    assert evaluate_support(descriptor).state == "unsupported"
