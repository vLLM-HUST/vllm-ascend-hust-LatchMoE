import pytest
import torch

from vllm_moe_offload_ascend.moe_offload.config import MoeOffloadConfig
from vllm_moe_offload_ascend.moe_offload.placement_plan import PlacementPlan
from vllm_moe_offload_ascend.moe_offload.runtime import MoeOffloadRuntime


class TinyLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer_id = 7
        self.w13_weight = torch.nn.Parameter(
            torch.arange(4 * 2 * 3, dtype=torch.float32).reshape(4, 2, 3)
        )
        self.w2_weight = torch.nn.Parameter(
            torch.arange(4 * 3 * 2, dtype=torch.float32).reshape(4, 3, 2)
        )


def test_external_placement_plan_can_reorder_without_changing_route_coverage():
    runtime = MoeOffloadRuntime(MoeOffloadConfig(enabled=True, num_slots=3))
    runtime.register_layer_for_fixed_slots(TinyLayer(), slot_device=torch.device("cpu"))
    requests = []

    def provider(request):
        requests.append(request)
        return PlacementPlan(
            layer_id=request.layer_id,
            ordered_experts=(2, 0, 1),
            source="mock-control",
        )

    runtime.set_placement_plan_provider(provider)
    prepared = runtime.stage_fixed_slot_plan(
        layer_id=7,
        active_experts=(0, 1, 2),
        num_logical_experts=4,
    )

    assert requests[0].routed_experts == (0, 1, 2)
    assert prepared.mapping.active_experts == (2, 0, 1)
    assert set(prepared.mapping.active_experts) == {0, 1, 2}
    assert all(prepared.log2phy[expert_id].item() >= 0 for expert_id in (0, 1, 2))


def test_external_placement_plan_cannot_drop_or_add_router_experts():
    runtime = MoeOffloadRuntime(MoeOffloadConfig(enabled=True, num_slots=2))
    runtime.register_layer_for_fixed_slots(TinyLayer(), slot_device=torch.device("cpu"))
    runtime.set_placement_plan_provider(
        lambda request: PlacementPlan(
            layer_id=request.layer_id,
            ordered_experts=(0, 3),
        )
    )

    with pytest.raises(RuntimeError, match="must be a permutation"):
        runtime.stage_fixed_slot_plan(
            layer_id=7,
            active_experts=(0, 1),
            num_logical_experts=4,
        )
