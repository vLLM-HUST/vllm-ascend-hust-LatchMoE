from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_analysis_module():
    path = Path(__file__).parents[1] / "paper" / "scripts" / "analyze_motivation_profile.py"
    spec = importlib.util.spec_from_file_location("motivation_profile_analysis", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_temporal_dynamics_compares_only_consecutive_invocations_of_each_layer():
    module = _load_analysis_module()
    result = module.summarize_temporal_dynamics(
        [
            {"_layer_id": 3, "active_experts": [0, 1]},
            {"_layer_id": 7, "active_experts": [4, 5]},
            {"_layer_id": 3, "active_experts": [1, 2]},
            {"_layer_id": 7, "active_experts": [4, 5]},
        ]
    )

    assert result["events_with_active_expert_ids"] == 4
    assert result["layers_with_active_expert_ids"] == 2
    assert result["adjacent_same_layer_pairs"] == 2
    assert result["jaccard"]["p50_nearest_rank"] == 0.333333
    assert result["new_expert_ratio"]["p50_nearest_rank"] == 0.0
    assert result["changed_expert_count"]["max"] == 1


def test_temporal_dynamics_is_explicitly_unavailable_without_active_ids():
    module = _load_analysis_module()
    result = module.summarize_temporal_dynamics(
        [{"_layer_id": 3, "n_active": 8}]
    )

    assert result["available"] is False
    assert result["events_with_active_expert_ids"] == 0


def test_capacity_sweep_reports_overflow_waves_and_footprint():
    module = _load_analysis_module()
    rows = module.summarize_capacity_sweep(
        [{"n_active": 8}, {"n_active": 17}, {"n_active": 33}],
        capacities=[8, 16, 32],
        expert_bytes=100,
    )

    assert [row["capacity_experts"] for row in rows] == [8, 16, 32]
    assert rows[0]["overflow_rate_pct"] == 66.6667
    assert rows[1]["required_waves"]["max"] == 3
    assert rows[2]["required_waves"]["p95_nearest_rank"] == 2
    assert rows[2]["slot_budget_bytes"] == 3200
    assert rows[2]["active_hbm_bytes"]["max"] == 3300
