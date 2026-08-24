#!/usr/bin/env python3
"""Independently re-audit the three-model Motivation characterization.

This script intentionally does not import ``analyze_motivation_profile.py``.
It re-parses raw JSONL, recomputes the paper-facing statistics, and checks run
custody/provenance so a shared implementation bug cannot make the audit pass.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


PREFILL = "b2_work_conserving_prefill"
DECODE = "decode_fixed_slot_stage"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def nearest(values: Iterable[int | float], fraction: float) -> int | float:
    ordered = sorted(values)
    if not ordered:
        raise AssertionError("empty sample")
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def pct(numerator: int, denominator: int) -> float:
    return round(100.0 * numerator / denominator, 4) if denominator else 0.0


def stats(values: list[int]) -> dict[str, int]:
    return {
        "min": min(values),
        "p50_nearest_rank": int(nearest(values, 0.50)),
        "p95_nearest_rank": int(nearest(values, 0.95)),
        "max": max(values),
    }


def temporal(events: list[dict[str, Any]]) -> dict[str, Any]:
    histories: dict[int, list[frozenset[int]]] = defaultdict(list)
    for event in events:
        active = event.get("active_experts")
        if isinstance(active, list):
            histories[event["_layer_id"]].append(frozenset(map(int, active)))
    jaccard: list[float] = []
    new_ratio: list[float] = []
    changed: list[int] = []
    for layer_history in histories.values():
        for before, after in zip(layer_history, layer_history[1:]):
            union = before | after
            jaccard.append(len(before & after) / len(union) if union else 1.0)
            new_ratio.append(len(after - before) / len(after) if after else 0.0)
            changed.append(len(after - before))
    result: dict[str, Any] = {
        "events_with_active_expert_ids": sum(map(len, histories.values())),
        "layers_with_active_expert_ids": len(histories),
        "adjacent_same_layer_pairs": len(jaccard),
        "available": bool(jaccard),
    }
    if jaccard:
        def float_stats(values: list[float]) -> dict[str, float]:
            return {
                "min": round(min(values), 6),
                "p50_nearest_rank": round(float(nearest(values, 0.50)), 6),
                "p95_nearest_rank": round(float(nearest(values, 0.95)), 6),
                "max": round(max(values), 6),
            }
        result["jaccard"] = float_stats(jaccard)
        result["new_expert_ratio"] = float_stats(new_ratio)
        result["changed_expert_count"] = stats(changed)
    return result


def scan_profile(
    path: Path, *, skip_prefill_invocations: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    prefill_seen = 0
    active_scope = False
    prefill: list[dict[str, Any]] = []
    decode: list[dict[str, Any]] = []
    event_counts: Counter[str] = Counter()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                wrapper = json.loads(line)
            except json.JSONDecodeError as error:
                raise AssertionError(f"invalid JSON at {path}:{line_number}") from error
            name = str(wrapper.get("name", "<missing>"))
            event_counts[name] += 1
            if name == PREFILL:
                prefill_seen += 1
                if prefill_seen > skip_prefill_invocations:
                    active_scope = True
                    payload = dict(wrapper["payload"])
                    payload["_layer_id"] = int(wrapper["layer_id"])
                    prefill.append(payload)
            elif active_scope and name == DECODE:
                payload = dict(wrapper["payload"])
                payload["_layer_id"] = int(wrapper["layer_id"])
                decode.append(payload)
    return prefill, decode, dict(sorted(event_counts.items()))


def rebuild_summary(
    prefill: list[dict[str, Any]],
    decode: list[dict[str, Any]],
    *,
    capacities: list[int],
    expert_bytes: int,
    primary_capacity: int,
) -> dict[str, Any]:
    active_prefill = [int(event["n_active"]) for event in prefill]
    primary_waves = [math.ceil(value / primary_capacity) for value in active_prefill]
    overflow = sum(value > primary_capacity for value in active_prefill)
    sweep = []
    for capacity in capacities:
        waves = [math.ceil(value / capacity) for value in active_prefill]
        active_bytes = [value * expert_bytes for value in active_prefill]
        sweep.append({
            "capacity_experts": capacity,
            "capacity_fraction_of_max": round(capacity / max(active_prefill), 6),
            "overflow_invocations": sum(value > capacity for value in active_prefill),
            "overflow_rate_pct": pct(sum(value > capacity for value in active_prefill), len(prefill)),
            "required_waves": stats(waves),
            "expert_bytes": expert_bytes,
            "active_hbm_bytes": {
                "p50_nearest_rank": int(nearest(active_bytes, 0.50)),
                "p95_nearest_rank": int(nearest(active_bytes, 0.95)),
                "max": max(active_bytes),
            },
            "slot_budget_bytes": capacity * expert_bytes,
        })

    decode_layers = Counter(int(event["_layer_id"]) for event in decode)
    decode_active = [int(event["n_active"]) for event in decode]
    hits = sum(int(event.get("n_hits", 0)) for event in decode)
    misses = sum(int(event.get("n_misses", 0)) for event in decode)
    miss_invocations = sum(int(event.get("n_misses", 0)) > 0 for event in decode)
    updates = sum(int(event.get("log2phy_update_count", 0)) for event in decode)
    update_invocations = sum(int(event.get("log2phy_update_count", 0)) > 0 for event in decode)
    return {
        "prefill": {
            "layer_invocations": len(prefill),
            "active_experts": stats(active_prefill),
            "capacity_overflow": {
                "capacity_experts": primary_capacity,
                "overflow_invocations": overflow,
                "overflow_rate_pct": pct(overflow, len(prefill)),
            },
            "required_capacity_groups": {
                "counts": {str(key): value for key, value in sorted(Counter(primary_waves).items())},
                "three_or_more_invocations": sum(value >= 3 for value in primary_waves),
                "three_or_more_rate_pct": pct(sum(value >= 3 for value in primary_waves), len(prefill)),
            },
            "temporal_dynamics": temporal(prefill),
            "capacity_sweep": sweep,
        },
        "decode": {
            "sampled_layer_invocations": len(decode),
            "profile_sample_rates": sorted({int(event.get("profile_sample_rate", 1)) for event in decode}),
            "layer_invocations": {str(key): value for key, value in sorted(decode_layers.items())},
            "layer_invocation_imbalance": {
                "min": min(decode_layers.values()),
                "max": max(decode_layers.values()),
                "max_minus_min": max(decode_layers.values()) - min(decode_layers.values()),
            },
            "active_experts": stats(decode_active),
            "expert_cache_accesses": {
                "hits": hits,
                "misses": misses,
                "miss_rate_pct": pct(misses, hits + misses),
            },
            "miss_bearing_invocations": {
                "count": miss_invocations,
                "rate_pct": pct(miss_invocations, len(decode)),
            },
            "mapping_updates": {
                "count": updates,
                "update_bearing_invocations": update_invocations,
                "update_bearing_rate_pct": pct(update_invocations, len(decode)),
            },
            "temporal_dynamics": temporal(decode),
        },
    }


def contiguous_subsequence(haystack: list[str], needle: list[str]) -> bool:
    return any(haystack[index:index + len(needle)] == needle for index in range(len(haystack) - len(needle) + 1))


def audit_one(root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    summary_path = root / spec["summary"]
    artifact_dir = Path(spec["artifact_dir"])
    profile = artifact_dir / "moe_profile.jsonl"
    summary = load_json(summary_path)
    actual_hash = digest(profile)
    assert actual_hash == summary["source"]["sha256"]
    assert str(profile) == summary["source"]["profile"]
    scope = summary["scope"]
    assert scope["managed_layers"] == spec["managed_layers"]
    assert scope["analyzed_requests"] == spec["requests"]
    assert scope["skipped_prefill_requests"] == spec["skipped_requests"]

    prefill, decode, event_counts = scan_profile(
        profile,
        skip_prefill_invocations=spec["managed_layers"] * spec["skipped_requests"],
    )
    assert len(prefill) == spec["managed_layers"] * spec["requests"]
    assert decode
    capacities = [int(row["capacity_experts"]) for row in summary["prefill"]["capacity_sweep"]]
    rebuilt = rebuild_summary(
        prefill,
        decode,
        capacities=capacities,
        expert_bytes=spec["expert_bytes"],
        primary_capacity=scope["cache_capacity_experts_per_layer"],
    )
    assert rebuilt == {"prefill": summary["prefill"], "decode": summary["decode"]}

    manifest = load_json(artifact_dir / "unit_manifest.json")
    benchmark = load_json(artifact_dir / "benchmark.json")
    unit = load_json(artifact_dir / "unit_result.json")
    release = load_json(artifact_dir / "release_ack.json")
    argv = [str(value) for value in manifest["provenance"]["argv"]]
    assert contiguous_subsequence(argv, [str(value) for value in spec["required_argv"]])
    assert benchmark["status"] == "ok"
    assert benchmark["successful_requests"] == spec["requests"]
    assert benchmark["failed_requests"] == 0
    assert unit["status"] == "ok" and unit["stage"] == "completed"
    assert unit["release_status"] == "released" and release["status"] == "released"
    assert manifest["selected_env"]["VLLM_ASCEND_MOE_DECODE_PROFILE_SAMPLE_RATE"] == "1"
    assert manifest["selected_env"]["VLLM_ASCEND_MOE_PROFILE_EXPERT_LISTS"] == "0"

    wave_summaries = [event["wave_summary"] for event in prefill]
    wave_count = sum(int(item["wave_count"]) for item in wave_summaries)
    issued_before = sum(int(item["issued_before_compute_waves"]) for item in wave_summaries)
    after_compute = sum(int(item["prefetch_after_compute_issues"]) for item in wave_summaries)
    assert issued_before == wave_count
    assert after_compute == 0
    decode_ready_false = sum(not bool(event.get("mapping_published_after_ready")) for event in decode)
    decode_dependency_false = sum(not bool(event.get("consumer_dependency_installed")) for event in decode)
    assert decode_ready_false == 0 and decode_dependency_false == 0

    return {
        "model": spec["name"],
        "status": "verified",
        "raw_profile": str(profile),
        "raw_profile_sha256": actual_hash,
        "summary_sha256": digest(summary_path),
        "manifest_sha256": digest(artifact_dir / "unit_manifest.json"),
        "benchmark_sha256": digest(artifact_dir / "benchmark.json"),
        "event_counts": event_counts,
        "scope": {
            "requests": spec["requests"],
            "managed_layers": spec["managed_layers"],
            "prefill_layer_invocations": len(prefill),
            "decode_layer_invocations": len(decode),
        },
        "paper_facing": {
            "decode_miss_rate_pct": rebuilt["decode"]["expert_cache_accesses"]["miss_rate_pct"],
            "decode_miss_bearing_rate_pct": rebuilt["decode"]["miss_bearing_invocations"]["rate_pct"],
            "decode_update_bearing_rate_pct": rebuilt["decode"]["mapping_updates"]["update_bearing_rate_pct"],
            "decode_active_set_jaccard": rebuilt["decode"]["temporal_dynamics"]["jaccard"],
            "prefill_active_experts": rebuilt["prefill"]["active_experts"],
            "capacity_32": next(row for row in rebuilt["prefill"]["capacity_sweep"] if row["capacity_experts"] == 32),
        },
        "overlap_and_lifecycle": {
            "wave_count": wave_count,
            "issued_before_compute_waves": issued_before,
            "prefetch_before_compute_issues": sum(int(item["prefetch_before_compute_issues"]) for item in wave_summaries),
            "prefetch_after_compute_issues": after_compute,
            "h2d_bytes": sum(int(item["h2d_bytes"]) for item in wave_summaries),
            "stage_wait_ms": round(sum(float(item["stage_wait_ms"]) for item in wave_summaries), 6),
            "mlp_ms": round(sum(float(item["mlp_ms"]) for item in wave_summaries), 6),
            "stage_wait_over_mlp": round(
                sum(float(item["stage_wait_ms"]) for item in wave_summaries)
                / sum(float(item["mlp_ms"]) for item in wave_summaries), 6
            ),
            "decode_mapping_published_before_ready_violations": decode_ready_false,
            "decode_missing_consumer_dependency_violations": decode_dependency_false,
        },
        "custody": {
            "benchmark_status": benchmark["status"],
            "successful_requests": benchmark["successful_requests"],
            "failed_requests": benchmark["failed_requests"],
            "unit_status": unit["status"],
            "release_status": release["status"],
            "host_python": argv[argv.index("--host-python") + 1],
            "physical_device": int(argv[argv.index("--device") + 1]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("paper/data/motivation_sources.json"))
    parser.add_argument("--output", type=Path, default=Path("paper/data/audits/motivation_reaudit.json"))
    args = parser.parse_args()
    root = Path.cwd()
    config = load_json(root / args.config)
    rows = [audit_one(root, spec) for spec in config["datasets"]]
    result = {
        "schema_version": 1,
        "audit": "independent raw-event recomputation plus custody/provenance validation",
        "status": "verified",
        "models": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"VERIFIED {len(rows)} models -> {args.output}")


if __name__ == "__main__":
    main()
