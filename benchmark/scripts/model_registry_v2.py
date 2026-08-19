#!/usr/bin/env python3
"""Build the Issue #27 model inventory and qualification matrix.

This tool is intentionally read-only with respect to checkpoints.  It records
the immutable config/index digests and reports model-class resolution separately
from a real NPU construction or graph qualification, so a blocked run cannot be
mistaken for a pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vllm_moe_offload_ascend.moe_offload.capabilities import (
    describe_checkpoint_config,
)


BYTES_PER_GIB = 1024**3
DEFAULT_HBM_GIB = 64.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_model_argument(value: str) -> tuple[str, Path]:
    name, marker, location = value.partition("=")
    if not marker or not name or not location:
        raise argparse.ArgumentTypeError("--model must use ID=/absolute/checkpoint/path")
    return name, Path(location).resolve()


def _int(config: dict[str, Any], *names: str, default: int = 0) -> int:
    for name in names:
        value = config.get(name)
        if value is not None:
            return int(value)
    return int(default)


def _dtype_bytes(config: dict[str, Any]) -> int:
    dtype = str(config.get("torch_dtype") or "bfloat16").lower()
    if "float32" in dtype:
        return 4
    if "int8" in dtype or "fp8" in dtype:
        return 1
    return 2


def _weight_files(model_path: Path) -> list[Path]:
    return sorted(
        file for file in model_path.glob("*.safetensors") if file.is_file()
    )


def estimate_memory(config: dict[str, Any], *, slot_count: int, hbm_gib: float) -> dict[str, int | float]:
    hidden = _int(config, "hidden_size")
    routed_intermediate = _int(config, "moe_intermediate_size", "intermediate_size")
    routed_experts = _int(config, "n_routed_experts", "num_experts")
    shared_experts = _int(config, "n_shared_experts")
    if shared_experts == 0 and _int(config, "shared_expert_intermediate_size") > 0:
        shared_experts = 1
    shared_intermediate = _int(
        config,
        "shared_expert_intermediate_size",
        default=routed_intermediate if shared_experts else 0,
    )
    layers = _int(config, "num_hidden_layers")
    moe_layers = max(0, layers - _int(config, "first_k_dense_replace"))
    dtype_bytes = _dtype_bytes(config)
    bytes_per_routed_expert = 3 * hidden * routed_intermediate * dtype_bytes
    routed_full_bytes = moe_layers * routed_experts * bytes_per_routed_expert
    shared_resident_bytes = moe_layers * shared_experts * 3 * hidden * shared_intermediate * dtype_bytes
    shared_gate_bytes = (
        moe_layers * hidden * dtype_bytes
        if _int(config, "shared_expert_intermediate_size") > 0
        else 0
    )
    slot_bytes = max(0, int(slot_count)) * bytes_per_routed_expert
    stage_bytes = slot_bytes * 2
    estimated_available_bytes = int(float(hbm_gib) * BYTES_PER_GIB)
    return {
        "dtype_bytes": dtype_bytes,
        "moe_layers": moe_layers,
        "routed_expert_bytes": bytes_per_routed_expert,
        "routed_full_bytes": routed_full_bytes,
        "resident_shared_bytes": shared_resident_bytes,
        "shared_gate_bytes": shared_gate_bytes,
        "slot_bytes": slot_bytes,
        "stage_buffer_bytes": stage_bytes,
        "hbm_budget_bytes": estimated_available_bytes,
        "hbm_after_routed_slots_shared_stage_bytes": estimated_available_bytes
        - slot_bytes
        - stage_bytes
        - shared_resident_bytes
        - shared_gate_bytes,
    }


def native_model_class_preflight(model_path: Path) -> dict[str, object]:
    """Resolve the native vLLM class without loading checkpoint weights or an NPU."""

    try:
        from vllm.config import ModelConfig
        from vllm.model_executor.models import ModelRegistry

        model_config = ModelConfig(
            model=str(model_path),
            tokenizer=str(model_path),
            trust_remote_code=True,
            dtype="bfloat16",
            enforce_eager=False,
        )
        model_class, architecture = ModelRegistry.resolve_model_cls(
            model_config.architectures,
            model_config,
        )
    except Exception as exc:
        return {
            "status": "failed",
            "scope": "ModelConfig_and_ModelRegistry",
            "error": f"{type(exc).__name__}: {exc}",
            "full_native_construction": "not_run",
        }
    return {
        "status": "passed",
        "scope": "ModelConfig_and_ModelRegistry",
        "architecture": str(architecture),
        "model_class": f"{model_class.__module__}.{model_class.__qualname__}",
        "full_native_construction": "not_run",
        "blockers": ["requires separate NPU weight-loading preflight"],
    }


def inventory_model(
    model_id: str,
    model_path: Path,
    *,
    slot_count: int,
    hbm_gib: float,
    run_native_preflight: bool,
) -> dict[str, object]:
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise ValueError(f"missing config.json: {model_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    index_path = model_path / "model.safetensors.index.json"
    weights = _weight_files(model_path)
    descriptor = describe_checkpoint_config(config)
    checkpoint_digest = sha256(index_path) if index_path.is_file() else None
    record: dict[str, object] = {
        "id": model_id,
        "checkpoint_path": str(model_path),
        "architecture": list(config.get("architectures") or []),
        "model_type": config.get("model_type"),
        "config_sha256": sha256(config_path),
        "checkpoint_index_sha256": checkpoint_digest,
        "checkpoint_weight_files": len(weights),
        "checkpoint_weight_bytes": sum(path.stat().st_size for path in weights),
        "capability_config": descriptor.to_jsonable(),
        "memory_estimate": estimate_memory(config, slot_count=slot_count, hbm_gib=hbm_gib),
        "qualification_status": "not_run",
        "native_model_class_preflight": (
            native_model_class_preflight(model_path)
            if run_native_preflight
            else {"status": "not_run", "scope": "ModelConfig_and_ModelRegistry"}
        ),
    }
    return record


def build_registry(
    models: list[tuple[str, Path]],
    *,
    slot_count: int,
    hbm_gib: float,
    run_native_preflight: bool,
) -> dict[str, object]:
    return {
        "schema_version": "latchmoe-model-registry-v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "slot_count_for_estimate": int(slot_count),
        "hbm_gib_for_estimate": float(hbm_gib),
        "models": [
            inventory_model(
                model_id,
                model_path,
                slot_count=slot_count,
                hbm_gib=hbm_gib,
                run_native_preflight=run_native_preflight,
            )
            for model_id, model_path in models
        ],
    }


def build_qualification_matrix(registry: dict[str, object]) -> dict[str, object]:
    """Create the Phase-B gate matrix without erasing blocked or unrun cells."""

    rows = []
    for record in registry.get("models", []):
        if not isinstance(record, dict):
            continue
        rows.append(
            {
                "id": record.get("id"),
                "config_sha256": record.get("config_sha256"),
                "checkpoint_index_sha256": record.get("checkpoint_index_sha256"),
                "capability_config": record.get("capability_config"),
                "status": "not_run",
                "gates": {
                    "native_oracle": "not_run",
                    "latchmoe_eager_diagnostic": "not_run",
                    "router_parity": "not_run",
                    "layer_boundary_parity": "not_run",
                    "token_exactness": "not_run",
                    "piecewise_capture": "not_run",
                    "piecewise_replay": "not_run",
                    "zero_eager_fallback": "not_run",
                    "h2d_lease_release": "not_run",
                    "prefill_overflow": "not_run",
                    "decode_cache_churn": "not_run",
                },
            }
        )
    return {
        "schema_version": "latchmoe-qualification-matrix-v2",
        "registry_schema_version": registry.get("schema_version"),
        "generated_at": registry.get("generated_at"),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the LatchMoE model registry v2.")
    parser.add_argument("--model", action="append", type=parse_model_argument, required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--matrix-output", type=Path)
    parser.add_argument("--slot-count", type=int, default=32)
    parser.add_argument("--hbm-gib", type=float, default=DEFAULT_HBM_GIB)
    parser.add_argument("--native-model-class-preflight", action="store_true")
    args = parser.parse_args()
    payload = build_registry(
        args.model,
        slot_count=args.slot_count,
        hbm_gib=args.hbm_gib,
        run_native_preflight=args.native_model_class_preflight,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.matrix_output is not None:
        args.matrix_output.parent.mkdir(parents=True, exist_ok=True)
        args.matrix_output.write_text(
            json.dumps(build_qualification_matrix(payload), indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
