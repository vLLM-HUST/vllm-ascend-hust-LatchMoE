# SPDX-License-Identifier: Apache-2.0
"""Build SEW-Offload benchmark manifests from the ShareGPT dataset.

Per benchmark/configs/sew_offload_v1.yaml, every benchmark run (including smoke
and debugging) must draw prompts from the real ShareGPT dataset. Random,
synthetic, or seed-text-repeated prompts are not allowed for any run.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


def assert_no_random_dataset(config: dict[str, Any]) -> None:
    """Fail closed if the config does not mandate the real ShareGPT dataset."""
    dataset = config.get("dataset", {})
    source = str(dataset.get("source", "")).lower()
    if source != "sharegpt":
        raise ValueError(
            "benchmark dataset.source must be 'sharegpt'; random/synthetic "
            f"datasets are not allowed, got {dataset.get('source')!r}"
        )
    if dataset.get("random_dataset_allowed", False):
        raise ValueError("random_dataset_allowed must be false for SEW-Offload benchmarks")
    if dataset.get("synthetic_smoke_allowed", False):
        raise ValueError("synthetic_smoke_allowed must be false for SEW-Offload benchmarks")


def _bucket_token_window(bucket: dict[str, Any]) -> tuple[int, int]:
    """Return the (low, high) prompt-token window for a workload bucket."""
    prompt_tokens = bucket["prompt_tokens"]
    if prompt_tokens == "mixed":
        # Mixed bucket spans the full short..long range observed in the suite.
        return 128, 4096
    low, high = prompt_tokens
    return int(low), int(high)


def _first_human_prompt(record: dict[str, Any]) -> str | None:
    conversations = record.get("conversations")
    if not conversations:
        return None
    first = conversations[0]
    if first.get("from") != "human":
        return None
    value = first.get("value")
    if not isinstance(value, str) or not value.strip():
        return None
    return value


def build_sharegpt_manifest(
    *,
    config: dict[str, Any],
    manifest_path: Path,
    model_path: str | None = None,
    requests_per_bucket: int | None = None,
    buckets: set[str] | None = None,
) -> int:
    """Sample real ShareGPT prompts into a reproducible benchmark manifest.

    Returns the number of written requests.
    """
    assert_no_random_dataset(config)

    dataset_cfg = config["dataset"]
    local_path = Path(dataset_cfg["local_path"])
    if not local_path.exists():
        raise FileNotFoundError(f"ShareGPT dataset not found: {local_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path or config["model"]["path"],
        trust_remote_code=False,
    )

    seed = int(dataset_cfg["seed"])
    rng = random.Random(seed)

    with local_path.open(encoding="utf-8") as f:
        dataset = json.load(f)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with manifest_path.open("w", encoding="utf-8") as out:
        for bucket in config["workload_buckets"]:
            name = bucket["name"]
            if buckets is not None and name not in buckets:
                continue
            target = int(bucket["num_requests"])
            if requests_per_bucket is not None:
                target = min(target, requests_per_bucket)
            low, high = _bucket_token_window(bucket)
            output_tokens = int(bucket["output_tokens"])

            picked = 0
            for idx in indices:
                if picked >= target:
                    break
                prompt = _first_human_prompt(dataset[idx])
                if prompt is None:
                    continue
                token_len = len(tokenizer.encode(prompt, add_special_tokens=False))
                if token_len < low or token_len > high:
                    continue
                record = {
                    "request_id": f"{name}_{picked:04d}",
                    "bucket": name,
                    "prompt": prompt,
                    "prompt_tokens": token_len,
                    "max_output_tokens": output_tokens,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "seed": seed,
                    "dataset": "sharegpt",
                    "source_id": dataset[idx].get("id"),
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                picked += 1
                written += 1

    if written == 0:
        raise ValueError(
            f"no ShareGPT prompts matched the selected buckets: {sorted(buckets) if buckets else 'all'}"
        )
    return written
