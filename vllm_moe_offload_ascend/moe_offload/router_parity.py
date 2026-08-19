"""Router-boundary parity artifacts for native-vs-LatchMoE diagnostics."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

from vllm_moe_offload_ascend.moe_offload.profile_io import append_jsonl


ROUTER_PARITY_PATH_ENV = "VLLM_ASCEND_MOE_ROUTER_PARITY_PATH"
ROUTER_PARITY_MAX_TOKENS_ENV = "VLLM_ASCEND_MOE_ROUTER_PARITY_MAX_TOKENS"


def record_router_snapshot(
    *,
    role: str,
    layer_id: int,
    router_logits: Any,
    topk_ids: Any,
    topk_weights: Any,
) -> None:
    """Write a bounded eager-only snapshot when a parity artifact is requested."""

    path = os.getenv(ROUTER_PARITY_PATH_ENV)
    if not path or not _is_bounded_tensor(router_logits):
        return
    if not _is_bounded_tensor(topk_ids) or not _is_bounded_tensor(topk_weights):
        return
    append_jsonl(
        path,
        {
            "kind": "latchmoe_router_parity_v1",
            "role": str(role),
            "layer_id": int(layer_id),
            "router_logits": _tensor_to_nested_list(router_logits),
            "topk_ids": _tensor_to_nested_list(topk_ids),
            "topk_weights": _tensor_to_nested_list(topk_weights),
        },
    )


def compare_router_artifacts(
    native_records: list[dict[str, Any]],
    seam_records: list[dict[str, Any]],
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> dict[str, Any]:
    """Compare records in layer/occurrence order and return the first mismatch."""

    native = _ordered_records(native_records)
    seam = _ordered_records(seam_records)
    if not native:
        return {"status": "failed", "reason": "no_native_records"}
    if not seam:
        return {"status": "failed", "reason": "no_seam_records"}
    if [key for key, _ in native] != [key for key, _ in seam]:
        return {
            "status": "failed",
            "reason": "record_key_mismatch",
            "native_keys": [list(key) for key, _ in native],
            "seam_keys": [list(key) for key, _ in seam],
        }
    for key, native_record in native:
        seam_record = dict(seam)[key]
        mismatch = _compare_record(native_record, seam_record, atol=atol, rtol=rtol)
        if mismatch is not None:
            return {
                "status": "failed",
                "layer_id": key[0],
                "occurrence": key[1],
                "mismatch": mismatch,
            }
    return {
        "status": "passed",
        "records_compared": len(native),
        "atol": float(atol),
        "rtol": float(rtol),
    }


def load_router_artifact(path: str | Path, *, role: str | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid router artifact JSON at {path}:{line_number}") from exc
        if record.get("kind") != "latchmoe_router_parity_v1":
            continue
        if role is None or record.get("role") == role:
            records.append(record)
    return records


def _ordered_records(records: list[dict[str, Any]]) -> list[tuple[tuple[int, int], dict[str, Any]]]:
    occurrences: dict[int, int] = {}
    ordered: list[tuple[tuple[int, int], dict[str, Any]]] = []
    for record in records:
        layer_id = int(record["layer_id"])
        occurrence = occurrences.get(layer_id, 0)
        occurrences[layer_id] = occurrence + 1
        ordered.append(((layer_id, occurrence), record))
    return ordered


def _compare_record(
    native: dict[str, Any],
    seam: dict[str, Any],
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any] | None:
    for field in ("router_logits", "topk_ids", "topk_weights"):
        expected = _flatten(native.get(field))
        observed = _flatten(seam.get(field))
        if len(expected) != len(observed):
            return {
                "field": field,
                "reason": "shape_or_length_mismatch",
                "native_length": len(expected),
                "seam_length": len(observed),
            }
        for index, (left, right) in enumerate(zip(expected, observed, strict=True)):
            if field == "topk_ids":
                equal = int(left) == int(right)
            else:
                equal = math.isclose(float(left), float(right), abs_tol=atol, rel_tol=rtol)
            if not equal:
                mismatch: dict[str, Any] = {
                    "field": field,
                    "flat_index": index,
                    "native": left,
                    "seam": right,
                }
                if field.startswith("topk_"):
                    width = _width(native.get(field))
                    mismatch["token"] = index // max(1, width)
                    mismatch["expert_position"] = index % max(1, width)
                return mismatch
    return None


def _tensor_to_nested_list(value: Any) -> list[Any]:
    detached = value.detach()
    if getattr(detached.device, "type", "cpu") != "cpu":
        detached = detached.to("cpu")
    return detached.tolist()


def _is_bounded_tensor(value: Any) -> bool:
    """Keep eager diagnostic snapshots out of graph-mode and large workloads."""

    if not callable(getattr(value, "detach", None)):
        return False
    try:
        max_tokens = max(1, int(os.getenv(ROUTER_PARITY_MAX_TOKENS_ENV, "64")))
        return int(value.shape[0]) <= max_tokens
    except (AttributeError, IndexError, TypeError, ValueError):
        return False


def _flatten(value: Any) -> list[Any]:
    if isinstance(value, list):
        return [item for child in value for item in _flatten(child)]
    return [value]


def _width(value: Any) -> int:
    return len(value[0]) if isinstance(value, list) and value and isinstance(value[0], list) else 1
