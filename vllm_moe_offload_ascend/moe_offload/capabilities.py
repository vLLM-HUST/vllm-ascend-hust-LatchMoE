"""Model-generic capability contracts for the LatchMoE graph seam.

The seam is deliberately selected from observable layer capabilities instead
of architecture or checkpoint names.  This keeps unsupported combinations out
of graph capture and makes the exact capability tuple reproducible in a run
manifest.
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Literal


OutputContract = Literal["routed_tensor", "shared_routed_tuple"]
SharedMode = Literal["none", "external_resident", "fused_mix_placement"]
RouterOwner = Literal["external_logits", "internal_gate", "registered_custom", "unknown"]
SupportState = Literal["implemented", "npu_qualified", "unsupported"]


@dataclass(frozen=True)
class RouterSelection:
    top_k: int
    grouped_top_k: bool
    topk_group: int | None
    num_expert_group: int | None
    renormalize: bool
    scoring_function: str
    correction_bias: bool
    routed_scaling_factor: float


@dataclass(frozen=True)
class MoeCapabilityDescriptor:
    """A serializable description of one MoE execution capability tuple."""

    schema_version: str
    output_contract: OutputContract
    shared_mode: SharedMode
    shared_activation: str
    shared_expert_count: int
    routed_expert_count: int
    router_owner: RouterOwner
    router_adapter: str
    selection: RouterSelection
    combine_owner: str
    weight_lifecycle: str
    weight_mode: str
    parallel_mode: str
    overlap_mode: str

    def to_jsonable(self) -> dict[str, object]:
        return asdict(self)

    def fingerprint(self) -> str:
        return json.dumps(self.to_jsonable(), sort_keys=True, separators=(",", ":"))

    def digest(self) -> str:
        """Return a stable identity suitable for manifests and artifact names."""

        return hashlib.sha256(self.fingerprint().encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CapabilitySupport:
    state: SupportState
    blockers: tuple[str, ...]

    @property
    def enabled(self) -> bool:
        return self.state in {"implemented", "npu_qualified"}

    def to_jsonable(self) -> dict[str, object]:
        return {"state": self.state, "blockers": list(self.blockers)}


class LatchMoECapabilityError(RuntimeError):
    """A fail-closed error which can be copied into a run manifest."""

    def __init__(
        self,
        descriptor: MoeCapabilityDescriptor,
        support: CapabilitySupport,
    ) -> None:
        self.descriptor = descriptor
        self.support = support
        payload = {
            "event": "latchmoe_capability_rejected",
            "descriptor": descriptor.to_jsonable(),
            "support": support.to_jsonable(),
        }
        super().__init__(json.dumps(payload, sort_keys=True))


def describe_layer_capability(layer: Any, runner: Any) -> MoeCapabilityDescriptor:
    """Describe a materialized FusedMoE layer without model-name checks."""

    moe_config = getattr(layer, "moe_config", None) or getattr(runner, "moe_config", None)
    routed_expert_count = _positive_int(
        getattr(moe_config, "num_experts", None),
        _positive_int(getattr(layer, "num_experts", None), 0),
    )
    external_shared = getattr(runner, "_shared_experts", None)
    mix_placement = bool(getattr(layer, "mix_placement", False))
    shared_expert_count = _positive_int(getattr(layer, "n_shared_experts", None), 0)
    if external_shared is not None:
        shared_mode: SharedMode = "external_resident"
        output_contract: OutputContract = "shared_routed_tuple"
        if shared_expert_count == 0:
            shared_expert_count = 1
    elif mix_placement:
        shared_mode = "fused_mix_placement"
        output_contract = "routed_tensor"
    else:
        shared_mode = "none"
        output_contract = "routed_tensor"

    shared_gate = getattr(runner, "shared_expert_gate", None)
    if shared_gate is None:
        shared_gate = _find_shared_gate(external_shared)
    shared_activation = "gated" if shared_gate is not None else (
        "always_on" if shared_mode != "none" else "none"
    )

    custom_router = getattr(layer, "custom_routing_function", None)
    named_adapter = getattr(layer, "latchmoe_router_adapter", None)
    if custom_router is not None:
        router_owner: RouterOwner = "registered_custom" if isinstance(named_adapter, str) else "unknown"
        router_adapter = str(named_adapter or "python_callable")
    elif getattr(runner, "gate", None) is not None or bool(
        getattr(layer, "is_internal_router", False)
    ):
        router_owner = "internal_gate"
        router_adapter = "builtin"
    else:
        router_owner = "external_logits"
        router_adapter = "builtin"

    selection = RouterSelection(
        top_k=_positive_int(getattr(layer, "top_k", None), 0),
        grouped_top_k=bool(getattr(layer, "use_grouped_topk", False)),
        topk_group=_optional_int(getattr(layer, "topk_group", None)),
        num_expert_group=_optional_int(getattr(layer, "num_expert_group", None)),
        renormalize=bool(getattr(layer, "renormalize", False)),
        scoring_function=str(getattr(layer, "scoring_func", "softmax") or "softmax"),
        correction_bias=getattr(layer, "e_score_correction_bias", None) is not None,
        routed_scaling_factor=float(
            getattr(
                layer,
                "_original_routed_scaling_factor",
                getattr(layer, "routed_scaling_factor", 1.0),
            )
            or 1.0
        ),
    )
    return MoeCapabilityDescriptor(
        schema_version="latchmoe-capability-v1",
        output_contract=output_contract,
        shared_mode=shared_mode,
        shared_activation=shared_activation,
        shared_expert_count=shared_expert_count,
        routed_expert_count=routed_expert_count,
        router_owner=router_owner,
        router_adapter=router_adapter,
        selection=selection,
        combine_owner="native_runner",
        weight_lifecycle="routed_cpu_first_offloaded_shared_resident",
        weight_mode=_weight_mode(layer, runner),
        parallel_mode=_parallel_mode(moe_config),
        overlap_mode=_overlap_mode(layer),
    )


def describe_checkpoint_config(config: dict[str, Any]) -> MoeCapabilityDescriptor:
    """Produce a Phase-A descriptor from config.json only.

    Checkpoint config files do not always encode whether vLLM places a gate on
    the runner.  Those fields intentionally stay ``unknown`` until the native
    model-construction preflight resolves them.
    """

    routed_experts = _positive_int(
        config.get("n_routed_experts"),
        _positive_int(config.get("num_experts"), 0),
    )
    shared_experts = _positive_int(config.get("n_shared_experts"), 0)
    if shared_experts == 0 and _positive_int(config.get("shared_expert_intermediate_size"), 0):
        shared_experts = 1
    topk_method = str(config.get("topk_method") or "")
    scoring = str(config.get("scoring_func") or "")
    if not scoring:
        scoring = "sigmoid" if topk_method.startswith("noaux") else "softmax"
    return MoeCapabilityDescriptor(
        schema_version="latchmoe-capability-v1",
        output_contract="shared_routed_tuple" if shared_experts else "routed_tensor",
        shared_mode="external_resident" if shared_experts else "none",
        shared_activation=(
            "gated" if _positive_int(config.get("shared_expert_intermediate_size"), 0) else
            ("always_on" if shared_experts else "none")
        ),
        shared_expert_count=shared_experts,
        routed_expert_count=routed_experts,
        router_owner="unknown",
        router_adapter="unresolved_until_native_preflight",
        selection=RouterSelection(
            top_k=_positive_int(config.get("num_experts_per_tok"), 0),
            grouped_top_k=bool(config.get("topk_group") not in (None, 1)),
            topk_group=_optional_int(config.get("topk_group")),
            num_expert_group=_optional_int(config.get("n_group")),
            renormalize=bool(config.get("norm_topk_prob", False)),
            scoring_function=scoring,
            correction_bias=topk_method.startswith("noaux"),
            routed_scaling_factor=float(config.get("routed_scaling_factor") or 1.0),
        ),
        combine_owner="native_runner",
        weight_lifecycle="routed_cpu_first_offloaded_shared_resident",
        weight_mode="bf16_unquantized" if not config.get("quantization_config") else "quantized",
        parallel_mode="single_npu",
        overlap_mode="none",
    )


def evaluate_support(descriptor: MoeCapabilityDescriptor) -> CapabilitySupport:
    """Return an explicit support state; unknown paths are never implicitly on."""

    blockers: list[str] = []
    if descriptor.output_contract not in {"routed_tensor", "shared_routed_tuple"}:
        blockers.append(f"unknown_output_contract:{descriptor.output_contract}")
    if descriptor.shared_mode not in {"none", "external_resident"}:
        blockers.append(f"unsupported_shared_mode:{descriptor.shared_mode}")
    if descriptor.router_owner not in {"external_logits", "internal_gate"}:
        blockers.append(f"unsupported_router_owner:{descriptor.router_owner}")
    if descriptor.router_adapter != "builtin":
        blockers.append(f"unsupported_router_adapter:{descriptor.router_adapter}")
    if descriptor.selection.top_k <= 0:
        blockers.append("invalid_router_top_k")
    if descriptor.routed_expert_count <= 0:
        blockers.append("invalid_routed_expert_count")
    if descriptor.selection.scoring_function not in {"softmax", "sigmoid"}:
        blockers.append(f"unsupported_scoring:{descriptor.selection.scoring_function}")
    if descriptor.weight_mode != "bf16_unquantized":
        blockers.append(f"unsupported_weight_mode:{descriptor.weight_mode}")
    if descriptor.parallel_mode != "single_npu":
        blockers.append(f"unsupported_parallel_mode:{descriptor.parallel_mode}")
    if descriptor.overlap_mode != "none":
        blockers.append(f"unsupported_overlap_mode:{descriptor.overlap_mode}")
    return CapabilitySupport(
        state="unsupported" if blockers else "implemented",
        blockers=tuple(blockers),
    )


def _find_shared_gate(shared_experts: Any) -> Any | None:
    if shared_experts is None:
        return None
    for candidate in (
        shared_experts,
        getattr(shared_experts, "_experts", None),
        getattr(shared_experts, "experts", None),
    ):
        gate = getattr(candidate, "expert_gate", None)
        if gate is not None:
            return gate
    return None


def _weight_mode(layer: Any, runner: Any) -> str:
    method = getattr(layer, "quant_method", None) or getattr(runner, "_quant_method", None)
    if method is None:
        # The seam is attached only to the unquantized method.  Keeping this
        # default makes capability probing robust while weight creation is in
        # progress; known quantized method names are rejected below.
        return "bf16_unquantized"
    name = type(method).__name__.lower()
    return "bf16_unquantized" if "unquantized" in name else "quantized"


def _parallel_mode(moe_config: Any) -> str:
    if moe_config is None:
        return "single_npu"
    for name in ("dp_size", "ep_size", "tp_size", "pcp_size"):
        if _positive_int(getattr(moe_config, name, 1), 1) > 1:
            return "multi_npu"
    return "single_npu"


def _overlap_mode(layer: Any) -> str:
    if bool(getattr(layer, "multistream_overlap_gate", False)):
        return "shared_gate_multistream"
    if bool(getattr(layer, "multistream_overlap_shared_expert", False)):
        return "shared_expert_multistream"
    return "none"


def _positive_int(value: Any, default: int) -> int:
    try:
        converted = int(value)
    except (TypeError, ValueError):
        return int(default)
    return converted if converted > 0 else int(default)


def _optional_int(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None
