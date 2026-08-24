#!/usr/bin/env python3
"""Verify the portable Issue-17 bundle and recompute paired uncertainty."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import random
import statistics
import tarfile
from pathlib import Path
from typing import Any


REPLICATES = 10_000
SEED = 20_260_823


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _percentile(values: list[float], probability: float) -> float:
    values = sorted(values)
    offset = (len(values) - 1) * probability
    low = int(offset)
    high = min(low + 1, len(values) - 1)
    weight = offset - low
    return values[low] * (1 - weight) + values[high] * weight


def _requests(payload: bytes) -> dict[str, dict[str, Any]]:
    benchmark = json.loads(payload)
    rows = {}
    for row in benchmark["per_request"]:
        request_id = row["request_id"]
        if request_id in rows:
            raise ValueError(f"duplicate request ID: {request_id}")
        rows[request_id] = row
    return rows


def _latency(row: dict[str, Any], metric: str) -> float:
    ttft = float(row["ttft_s"]) * 1000.0
    if metric == "ttft":
        return ttft
    return (float(row["total_s"]) - float(row["ttft_s"])) / max(1, int(row["output_tokens"])) * 1000.0


def _bootstrap(pair_logs: list[list[float]], confidence: float) -> dict[str, Any]:
    rng = random.Random(SEED)
    samples = []
    for _ in range(REPLICATES):
        medians = []
        for _ in range(len(pair_logs)):
            values = pair_logs[rng.randrange(len(pair_logs))]
            selected = [values[rng.randrange(len(values))] for _ in range(len(values))]
            medians.append(statistics.median(selected))
        samples.append(math.exp(statistics.fmean(medians)) - 1.0)
    alpha = (1.0 - confidence) / 2.0
    return {
        "replicates": REPLICATES,
        "seed": SEED,
        "confidence": confidence,
        "lower": _percentile(samples, alpha),
        "upper": _percentile(samples, 1 - alpha),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    bundle_path = Path(args.bundle).resolve()
    bundle_bytes = bundle_path.read_bytes()
    failures = []
    if _sha(bundle_bytes) != args.expected_sha256:
        failures.append("outer bundle digest mismatch")
    members: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(bundle_bytes), mode="r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            handle = archive.extractfile(member)
            assert handle is not None
            relative = member.name.removeprefix("issue-17-matched-ttft/")
            members[relative] = handle.read()
    checksum_lines = members["SHA256SUMS"].decode("utf-8").splitlines()
    for line in checksum_lines:
        expected, relative = line.split("  ", 1)
        payload = members.get(relative)
        if payload is None:
            failures.append(f"missing bundle member: {relative}")
        elif _sha(payload) != expected:
            failures.append(f"member digest mismatch: {relative}")
    summary = json.loads(members["matched_summary.original.json"])
    if summary.get("status") != "passed" or summary.get("failures"):
        failures.append("packaged campaign summary did not pass")
    pair_reports = []
    metrics: dict[str, list[list[float]]] = {"ttft": [], "tpot": []}
    oracle_tokens = None
    for pair in range(1, 4):
        full = _requests(members[f"units/pair-{pair}-full_layer/benchmark.json"])
        wave = _requests(members[f"units/pair-{pair}-multi_wave/benchmark.json"])
        if set(full) != set(wave) or len(full) != 200:
            failures.append(f"pair {pair}: request IDs or count differ")
            continue
        for rows in (full, wave):
            tokens = {key: value["output_token_ids"] for key, value in rows.items()}
            if oracle_tokens is None:
                oracle_tokens = tokens
            elif tokens != oracle_tokens:
                failures.append(f"pair {pair}: output tokens differ from oracle")
        item = {"pair": pair, "requests": len(full)}
        for metric in ("ttft", "tpot"):
            logs = [
                math.log(_latency(wave[key], metric) / _latency(full[key], metric))
                for key in sorted(full)
            ]
            metrics[metric].append(logs)
            item[f"{metric}_effect"] = math.exp(statistics.median(logs)) - 1.0
        pair_reports.append(item)
    statistics_report = {}
    if not failures:
        for metric, logs in metrics.items():
            estimate = math.exp(statistics.fmean(statistics.median(values) for values in logs)) - 1.0
            confidence = 0.95 if metric == "ttft" else 0.90
            interval = _bootstrap(logs, confidence)
            statistics_report[metric] = {
                "estimate": estimate,
                "interval": interval,
                "equivalent_within_5_percent": (
                    metric == "tpot" and interval["lower"] >= -0.05 and interval["upper"] <= 0.05
                ),
            }
    output = {
        "schema_version": "latchmoe-issue17-paper-audit-v1",
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "bundle": str(bundle_path),
        "bundle_sha256": _sha(bundle_bytes),
        "verified_members": len(checksum_lines),
        "packaged_comparison": summary.get("comparison"),
        "packaged_arms": summary.get("arms"),
        "paired_effects": pair_reports,
        "paired_statistics": statistics_report,
        "estimator": "exp(mean_start_pair(median_request(log(multi_wave/full_layer))))-1",
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": output["status"], "output": str(output_path), "failures": failures}))
    return 0 if output["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
