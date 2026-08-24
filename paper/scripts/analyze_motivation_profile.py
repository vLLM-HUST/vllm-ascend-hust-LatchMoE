#!/usr/bin/env python3
"""Summarize routing pressure and cache churn from an MoE profile JSONL."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any


PREFILL_EVENT = "b2_work_conserving_prefill"
DECODE_EVENT = "decode_fixed_slot_stage"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--managed-layers", type=int, default=12)
    parser.add_argument("--skip-prefill-requests", type=int, default=1)
    parser.add_argument("--expected-requests", type=int, default=200)
    parser.add_argument("--cache-capacity", type=int, default=32)
    parser.add_argument(
        "--capacities",
        default="",
        help="Comma-separated offline slot capacities; defaults to --cache-capacity.",
    )
    parser.add_argument(
        "--expert-bytes",
        type=int,
        default=0,
        help="Optional per-expert bytes for HBM footprint reporting.",
    )
    return parser.parse_args()


def nearest_rank(values: list[int], percentile: float) -> int:
    if not values:
        raise ValueError("cannot compute a percentile of an empty sample")
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def percentage(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(100.0 * numerator / denominator, 4)


def parse_capacities(raw: str, fallback: int) -> list[int]:
    if not raw.strip():
        return [int(fallback)]
    values = sorted({int(value.strip()) for value in raw.split(",") if value.strip()})
    if not values or any(value <= 0 for value in values):
        raise ValueError("capacities must contain positive integers")
    return values


def nearest_rank_float(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of an empty sample")
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return round(float(ordered[rank - 1]), 6)


def _active_expert_set(event: dict[str, Any]) -> frozenset[int] | None:
    value = event.get("active_experts")
    if not isinstance(value, list):
        return None
    try:
        return frozenset(int(expert_id) for expert_id in value)
    except (TypeError, ValueError):
        return None


def summarize_temporal_dynamics(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize adjacent routing-set changes within each MoE layer.

    Events from different layers are interleaved in the JSONL stream. Comparing
    neighboring lines directly would therefore measure layer identity changes,
    not routing dynamics. We retain the stream order separately for every layer
    and compare only consecutive invocations of that layer.
    """

    by_layer: dict[int, list[frozenset[int]]] = {}
    events_with_ids = 0
    for event in events:
        active = _active_expert_set(event)
        if active is None:
            continue
        layer_id = int(event.get("_layer_id", event.get("layer_id", -1)))
        by_layer.setdefault(layer_id, []).append(active)
        events_with_ids += 1

    jaccard: list[float] = []
    new_expert_ratio: list[float] = []
    changed_expert_count: list[int] = []
    adjacent_pairs = 0
    for histories in by_layer.values():
        for previous, current in zip(histories, histories[1:]):
            adjacent_pairs += 1
            union = previous | current
            intersection = previous & current
            jaccard.append(len(intersection) / len(union) if union else 1.0)
            new_expert_ratio.append(
                len(current - previous) / len(current) if current else 0.0
            )
            changed_expert_count.append(len(current - previous))

    result: dict[str, Any] = {
        "events_with_active_expert_ids": events_with_ids,
        "layers_with_active_expert_ids": len(by_layer),
        "adjacent_same_layer_pairs": adjacent_pairs,
    }
    if not adjacent_pairs:
        result["available"] = False
        return result

    result["available"] = True
    result["jaccard"] = {
        "min": round(min(jaccard), 6),
        "p50_nearest_rank": nearest_rank_float(jaccard, 0.50),
        "p95_nearest_rank": nearest_rank_float(jaccard, 0.95),
        "max": round(max(jaccard), 6),
    }
    result["new_expert_ratio"] = {
        "min": round(min(new_expert_ratio), 6),
        "p50_nearest_rank": nearest_rank_float(new_expert_ratio, 0.50),
        "p95_nearest_rank": nearest_rank_float(new_expert_ratio, 0.95),
        "max": round(max(new_expert_ratio), 6),
    }
    result["changed_expert_count"] = {
        "min": min(changed_expert_count),
        "p50_nearest_rank": nearest_rank(changed_expert_count, 0.50),
        "p95_nearest_rank": nearest_rank(changed_expert_count, 0.95),
        "max": max(changed_expert_count),
    }
    return result


def summarize_capacity_sweep(
    events: list[dict[str, Any]],
    *,
    capacities: list[int],
    expert_bytes: int = 0,
) -> list[dict[str, Any]]:
    """Estimate prefill overflow and wave pressure for offline capacities."""

    active = [int(event["n_active"]) for event in events]
    rows: list[dict[str, Any]] = []
    for capacity in capacities:
        waves = [math.ceil(value / capacity) for value in active]
        overflow = sum(value > capacity for value in active)
        row: dict[str, Any] = {
            "capacity_experts": int(capacity),
            "capacity_fraction_of_max": round(
                float(capacity) / float(max(active)) if active else 0.0,
                6,
            ),
            "overflow_invocations": int(overflow),
            "overflow_rate_pct": percentage(overflow, len(active)),
            "required_waves": {
                "min": min(waves) if waves else 0,
                "p50_nearest_rank": nearest_rank(waves, 0.50) if waves else 0,
                "p95_nearest_rank": nearest_rank(waves, 0.95) if waves else 0,
                "max": max(waves) if waves else 0,
            },
        }
        if expert_bytes > 0:
            active_bytes = [value * int(expert_bytes) for value in active]
            row["expert_bytes"] = int(expert_bytes)
            row["active_hbm_bytes"] = {
                "p50_nearest_rank": nearest_rank(active_bytes, 0.50),
                "p95_nearest_rank": nearest_rank(active_bytes, 0.95),
                "max": max(active_bytes) if active_bytes else 0,
            }
            row["slot_budget_bytes"] = int(capacity) * int(expert_bytes)
        rows.append(row)
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_scoped_events(
    profile: Path,
    *,
    skip_prefill_invocations: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prefill_seen = 0
    in_scope = False
    prefill: list[dict[str, Any]] = []
    decode: list[dict[str, Any]] = []

    with profile.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {profile}:{line_number}") from exc

            name = event.get("name")
            if name == PREFILL_EVENT:
                prefill_seen += 1
                if prefill_seen > skip_prefill_invocations:
                    in_scope = True
                    payload = dict(event["payload"])
                    payload["_layer_id"] = int(event["layer_id"])
                    prefill.append(payload)
                continue

            if in_scope and name == DECODE_EVENT:
                payload = dict(event["payload"])
                payload["_layer_id"] = int(event["layer_id"])
                decode.append(payload)

    return prefill, decode


def summarize_prefill(
    events: list[dict[str, Any]],
    *,
    cache_capacity: int,
) -> dict[str, Any]:
    active = [int(event["n_active"]) for event in events]
    waves = [math.ceil(value / cache_capacity) for value in active]
    overflow = sum(value > cache_capacity for value in active)
    three_or_more = sum(value >= 3 for value in waves)

    summary = {
        "layer_invocations": len(events),
        "active_experts": {
            "min": min(active),
            "p50_nearest_rank": nearest_rank(active, 0.50),
            "p95_nearest_rank": nearest_rank(active, 0.95),
            "max": max(active),
        },
        "capacity_overflow": {
            "capacity_experts": cache_capacity,
            "overflow_invocations": overflow,
            "overflow_rate_pct": percentage(overflow, len(events)),
        },
        "required_capacity_groups": {
            "counts": {str(key): value for key, value in sorted(Counter(waves).items())},
            "three_or_more_invocations": three_or_more,
            "three_or_more_rate_pct": percentage(three_or_more, len(events)),
        },
    }
    summary["temporal_dynamics"] = summarize_temporal_dynamics(events)
    return summary


def summarize_decode(events: list[dict[str, Any]]) -> dict[str, Any]:
    active = [int(event["n_active"]) for event in events]
    hits = sum(int(event.get("n_hits", 0)) for event in events)
    misses = sum(int(event.get("n_misses", 0)) for event in events)
    mapping_updates = sum(int(event.get("log2phy_update_count", 0)) for event in events)
    miss_events = sum(int(event.get("n_misses", 0)) > 0 for event in events)
    update_events = sum(int(event.get("log2phy_update_count", 0)) > 0 for event in events)
    sample_rates = sorted({int(event.get("profile_sample_rate", 1)) for event in events})
    layer_invocations = Counter(int(event["_layer_id"]) for event in events)

    summary = {
        "sampled_layer_invocations": len(events),
        "profile_sample_rates": sample_rates,
        "layer_invocations": {
            str(layer_id): count
            for layer_id, count in sorted(layer_invocations.items())
        },
        "layer_invocation_imbalance": {
            "min": min(layer_invocations.values()),
            "max": max(layer_invocations.values()),
            "max_minus_min": (
                max(layer_invocations.values()) - min(layer_invocations.values())
            ),
        },
        "active_experts": {
            "min": min(active),
            "p50_nearest_rank": nearest_rank(active, 0.50),
            "p95_nearest_rank": nearest_rank(active, 0.95),
            "max": max(active),
        },
        "expert_cache_accesses": {
            "hits": hits,
            "misses": misses,
            "miss_rate_pct": percentage(misses, hits + misses),
        },
        "miss_bearing_invocations": {
            "count": miss_events,
            "rate_pct": percentage(miss_events, len(events)),
        },
        "mapping_updates": {
            "count": mapping_updates,
            "update_bearing_invocations": update_events,
            "update_bearing_rate_pct": percentage(update_events, len(events)),
        },
    }
    summary["temporal_dynamics"] = summarize_temporal_dynamics(events)
    return summary


def main() -> None:
    args = parse_args()
    if args.managed_layers <= 0 or args.expected_requests <= 0:
        raise ValueError("managed layers and expected requests must be positive")
    if args.cache_capacity <= 0 or args.skip_prefill_requests < 0:
        raise ValueError("cache capacity must be positive and skipped requests non-negative")
    capacities = parse_capacities(args.capacities, args.cache_capacity)
    if args.expert_bytes < 0:
        raise ValueError("expert bytes must be non-negative")

    skipped_invocations = args.skip_prefill_requests * args.managed_layers
    prefill, decode = load_scoped_events(
        args.profile,
        skip_prefill_invocations=skipped_invocations,
    )
    expected_prefill = args.expected_requests * args.managed_layers
    if len(prefill) != expected_prefill:
        raise ValueError(
            f"expected {expected_prefill} scoped prefill invocations, found {len(prefill)}"
        )
    if not decode:
        raise ValueError("no scoped decode events found")

    decode_sample_rates = {
        int(event.get("profile_sample_rate", 1)) for event in decode
    }
    decode_layer_counts = Counter(int(event["_layer_id"]) for event in decode)
    if decode_sample_rates == {1}:
        if len(decode_layer_counts) != args.managed_layers:
            raise ValueError(
                f"expected {args.managed_layers} fully profiled decode layers, "
                f"found {len(decode_layer_counts)}"
            )
        if len(set(decode_layer_counts.values())) != 1:
            raise ValueError(
                "full-rate decode profiling is unbalanced across managed layers: "
                f"{dict(sorted(decode_layer_counts.items()))}"
            )

    summary = {
        "schema_version": 1,
        "source": {
            "profile": str(args.profile),
            "sha256": sha256(args.profile),
        },
        "scope": {
            "managed_layers": args.managed_layers,
            "skipped_prefill_requests": args.skip_prefill_requests,
            "analyzed_requests": args.expected_requests,
            "cache_capacity_experts_per_layer": args.cache_capacity,
            "filter": (
                "Start at the first prefill event after the configured skipped "
                "request groups; retain all later prefill and decode events."
            ),
        },
        "prefill": summarize_prefill(prefill, cache_capacity=args.cache_capacity),
        "decode": summarize_decode(decode),
    }
    summary["prefill"]["capacity_sweep"] = summarize_capacity_sweep(
        prefill,
        capacities=capacities,
        expert_bytes=args.expert_bytes,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
