"""Runtime-only placement-plan adapter for external MoE controllers.

The adapter accepts an ordered permutation of the experts selected by the
router. It deliberately does not encode a placement policy: a controller may
provide an ordering, while LatchMoE owns slot allocation, H2D, and graph-safe
lifecycle checks.
"""

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class PlacementPlanRequest:
    """Router facts exposed to an optional external placement provider."""

    layer_id: int
    routed_experts: tuple[int, ...]
    num_logical_experts: int


@dataclass(frozen=True)
class PlacementPlan:
    """A policy-neutral ordering for the current routed expert working set."""

    layer_id: int
    ordered_experts: tuple[int, ...]
    source: str = "external"


PlacementPlanProvider = Callable[[PlacementPlanRequest], PlacementPlan | None]


def resolve_placement_plan(
    provider: PlacementPlanProvider | None,
    *,
    layer_id: int,
    routed_experts: tuple[int, ...],
    num_logical_experts: int,
) -> tuple[int, ...]:
    """Resolve and validate an optional controller-supplied plan.

    A plan can change the staging order only. Requiring an exact permutation of
    router-selected experts prevents a controller from dropping a routed expert
    or materializing an unrelated expert during graph replay.
    """
    routed = tuple(dict.fromkeys(int(expert_id) for expert_id in routed_experts))
    if provider is None:
        return routed

    plan = provider(
        PlacementPlanRequest(
            layer_id=int(layer_id),
            routed_experts=routed,
            num_logical_experts=int(num_logical_experts),
        )
    )
    if plan is None:
        return routed
    if not isinstance(plan, PlacementPlan):
        raise TypeError("placement-plan provider must return PlacementPlan or None")
    if int(plan.layer_id) != int(layer_id):
        raise RuntimeError(
            f"placement plan targets layer {plan.layer_id}, expected {layer_id}"
        )

    ordered = tuple(int(expert_id) for expert_id in plan.ordered_experts)
    if len(ordered) != len(routed) or set(ordered) != set(routed):
        raise RuntimeError(
            "placement plan must be a permutation of the routed expert set; "
            f"routed={routed}, planned={ordered}"
        )
    return ordered
