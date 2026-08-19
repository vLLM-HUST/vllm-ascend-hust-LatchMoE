from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "benchmark" / "scripts" / "model_registry_v2.py"


def _module():
    spec = importlib.util.spec_from_file_location("model_registry_v2_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_qwen_next_shared_expert_is_in_the_static_hbm_ledger():
    registry = _module()
    estimate = registry.estimate_memory(
        {
            "hidden_size": 2048,
            "moe_intermediate_size": 512,
            "shared_expert_intermediate_size": 512,
            "num_experts": 512,
            "num_hidden_layers": 48,
            "num_experts_per_tok": 10,
            "torch_dtype": "bfloat16",
        },
        slot_count=32,
        hbm_gib=64,
    )

    assert estimate["resident_shared_bytes"] > 0
    assert estimate["shared_gate_bytes"] > 0


def test_qualification_matrix_preserves_all_unrun_gates():
    registry = _module()
    matrix = registry.build_qualification_matrix(
        {
            "schema_version": "latchmoe-model-registry-v2",
            "generated_at": "2026-08-19T00:00:00+00:00",
            "models": [
                {
                    "id": "external-shared",
                    "config_sha256": "config",
                    "checkpoint_index_sha256": "index",
                    "capability_config": {"shared_mode": "external_resident"},
                }
            ],
        }
    )

    row = matrix["rows"][0]
    assert row["status"] == "not_run"
    assert set(row["gates"].values()) == {"not_run"}
