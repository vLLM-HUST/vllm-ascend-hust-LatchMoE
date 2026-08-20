#!/usr/bin/env python3
"""Shared helpers for the Issue #28 matched campaign.

The campaign deliberately keeps the contract and the raw unit artifacts
separate.  A command may be replaced by a real NPU launcher without changing
the verifier, while every unit still carries the exact contract digest.
"""

from __future__ import annotations

import hashlib
import json
import os
import string
from pathlib import Path
from typing import Any


SCHEMA = "latchmoe.issue28.campaign/v1"
REPEATS = 3
SUCCESS_STATUSES = {"ok", "success", "passed"}
CAPACITY_STATUSES = {"capacity_failure", "oom", "memory_failure"}
CAPACITY_MARKERS = (
    "out of memory",
    "out-of-memory",
    "oom",
    "ACL_ERROR_RT_MEMORY_ALLOCATION",
    "ACL_ERROR_GE_MEMORY_ALLOCATION",
    "failed to allocate",
    "KV cache memory",
)
REQUIRED_SUCCESS_ARTIFACTS = (
    "metrics.json",
    "outputs.json",
    "runtime.json",
    "memory.json",
    "transfers.json",
)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def contract_digest(contract: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(contract)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def load_contract(path: Path) -> dict[str, Any]:
    contract = read_json(path)
    if contract.get("schema") != SCHEMA:
        raise ValueError(f"unsupported Issue #28 contract schema: {contract.get('schema')!r}")
    if not isinstance(contract.get("campaign_id"), str) or not contract["campaign_id"]:
        raise ValueError("contract campaign_id must be a non-empty string")
    arms = contract.get("arms")
    if not isinstance(arms, list) or not arms:
        raise ValueError("contract arms must be a non-empty list")
    arm_names: set[str] = set()
    for arm in arms:
        if not isinstance(arm, dict) or not isinstance(arm.get("name"), str):
            raise ValueError("each arm must be an object with a name")
        name = str(arm["name"])
        if name in arm_names:
            raise ValueError(f"duplicate arm: {name}")
        arm_names.add(name)
        command = arm.get("command")
        if not isinstance(command, list) or not command or not all(
            isinstance(item, str) and item for item in command
        ):
            raise ValueError(f"arm {name} must declare a non-empty argv command")
        expected = str(arm.get("expected_status") or "")
        if expected not in {"success", "success_or_capacity_failure", "capacity_failure"}:
            raise ValueError(f"arm {name} has invalid expected_status: {expected!r}")
        if int(arm.get("repeats", REPEATS)) != REPEATS:
            raise ValueError(f"arm {name} must have exactly {REPEATS} repeats")

    order = contract.get("order")
    if not isinstance(order, list) or not order:
        raise ValueError("contract order must be a non-empty list")
    counts = {name: 0 for name in arm_names}
    for index, item in enumerate(order):
        if not isinstance(item, dict):
            raise ValueError(f"order entry {index} is not an object")
        arm = str(item.get("arm") or "")
        repeat = int(item.get("repeat", 0))
        if arm not in arm_names:
            raise ValueError(f"order entry {index} references unknown arm {arm!r}")
        if repeat not in range(1, REPEATS + 1):
            raise ValueError(f"order entry {index} has invalid repeat {repeat}")
        counts[arm] += 1
    for arm, count in counts.items():
        if count != REPEATS:
            raise ValueError(f"arm {arm} appears {count} times; expected {REPEATS}")
    return contract


def expected_units(contract: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "order_index": index,
            "arm": str(item["arm"]),
            "repeat": int(item["repeat"]),
            "unit_id": f"{item['arm']}-r{int(item['repeat'])}",
        }
        for index, item in enumerate(contract["order"])
    ]


def arm_map(contract: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(arm["name"]): arm for arm in contract["arms"]}


def expand_tokens(tokens: list[str], values: dict[str, str]) -> list[str]:
    result: list[str] = []
    for token in tokens:
        formatter = string.Formatter()
        fields = [field for _, field, _, _ in formatter.parse(token) if field]
        unknown = [field for field in fields if field not in values]
        if unknown:
            raise ValueError(f"command token {token!r} uses unknown placeholders {unknown}")
        result.append(token.format_map(values))
    return result


def selected_environment(environment: dict[str, str]) -> dict[str, str]:
    """Keep provenance useful without copying secrets from the parent shell."""

    prefixes = ("VLLM_", "ASCEND_", "PYTORCH_", "CUDA_")
    return {
        key: value
        for key, value in sorted(environment.items())
        if key.startswith(prefixes)
    }


def contains_capacity_marker(text: str) -> bool:
    lowered = text.lower()
    return any(marker.lower() in lowered for marker in CAPACITY_MARKERS)


def percentile(values: list[float], percent: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    offset = (len(ordered) - 1) * float(percent) / 100.0
    low = int(offset)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (offset - low)


def metric_values(metrics: dict[str, Any], key: str) -> list[float]:
    value = metrics.get(key)
    if isinstance(value, list):
        return [float(item) for item in value]
    if isinstance(value, (int, float)):
        return [float(value)]
    return []


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p95": 0.0}
    ordered = sorted(values)
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "median": median,
        "p95": percentile(values, 95),
    }


def inherited_environment() -> dict[str, str]:
    return dict(os.environ)
