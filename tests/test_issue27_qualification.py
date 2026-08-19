from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
VERIFY_PATH = REPO_ROOT / "benchmark" / "scripts" / "verify_issue27_qualification.py"
SMOKE_PATH = REPO_ROOT / "benchmark" / "scripts" / "run_fixed_slot_smoke.py"
RUNNER_PATH = REPO_ROOT / "benchmark" / "scripts" / "run_issue27_qualification.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_output_comparator_reports_token_mismatch_coordinates():
    verifier = _load(VERIFY_PATH, "issue27_verify_output_test")

    report = verifier._same_outputs(
        {"request-0": [1, 2, 3]},
        {"request-0": [1, 9, 3]},
    )

    assert report["passed"] is False
    assert report["mismatched"]["request-0"] == {
        "expected": [1, 2, 3],
        "observed": [1, 9, 3],
    }


def test_layer_boundary_verifier_requires_shared_tuple_parity(tmp_path: Path):
    verifier = _load(VERIFY_PATH, "issue27_verify_layer_test")
    log = tmp_path / "run.log"
    log.write_text(
        "(EngineCore pid=123) SEW_LAYER_BOUNDARY_COMPARE layer=1 tokens=4 "
        "output_equal=True output_max_abs=0.0 output_mean_abs=0.0 "
        "shared_equal=True shared_max_abs=0.0 shared_mean_abs=0.0\n",
        encoding="utf-8",
    )

    passed, detail = verifier._layer_boundary_gate(
        log,
        shared_output_required=True,
    )

    assert passed is True
    assert detail["records"] == 1

    log.write_text(
        "(EngineCore pid=123) SEW_LAYER_BOUNDARY_COMPARE layer=1 tokens=4 "
        "output_equal=True output_max_abs=0.0 output_mean_abs=0.0 "
        "shared_contract_equal=False\n",
        encoding="utf-8",
    )
    passed, _ = verifier._layer_boundary_gate(log, shared_output_required=True)
    assert passed is False


def test_graph_verifier_requires_h2d_lease_and_release_evidence(tmp_path: Path):
    verifier = _load(VERIFY_PATH, "issue27_verify_graph_test")
    records = [
        {
            "name": "graph_slot_address_lock",
            "payload": {"w13_data_ptr": 1, "w2_data_ptr": 2, "log2phy_data_ptr": 3},
        },
        {
            "name": "graph_slot_address_validate",
            "payload": {
                "w13_data_ptr": 1,
                "w2_data_ptr": 2,
                "log2phy_data_ptr": 3,
                "matches_capture_fingerprint": True,
            },
        },
        {
            "name": "graph_replay_issue",
            "payload": {"synchronizes_npu": False},
        },
        {
            "name": "decode_fixed_slot_stage",
            "payload": {
                "step_id": 0,
                "h2d_bytes": 10,
                "consumer_dependency_installed": True,
                "mapping_published_after_ready": True,
            },
        },
        {
            "name": "decode_fixed_slot_stage",
            "payload": {
                "step_id": 1,
                "h2d_bytes": 10,
                "consumer_dependency_installed": True,
                "mapping_published_after_ready": True,
            },
        },
        {
            "name": "slot_generation_protected_until_compute_complete",
            "payload": {"leases_still_match": True},
        },
        {"name": "release_original_expert_weights", "payload": {}},
    ]
    (tmp_path / "moe_offload_profile.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "run.log").write_text(
        "LATCHMOE_GRAPH_CONFIG cudagraph_mode=PIECEWISE\n"
        "Graph capturing finished\n"
        "Replaying aclgraph\n",
        encoding="utf-8",
    )

    result = verifier._graph_gates(tmp_path)

    assert result["h2d_lease_release"][0] is True
    assert result["decode_cache_churn"][0] is True
    assert result["piecewise_replay"][0] is True


def test_failed_qualification_cannot_update_matrix(tmp_path: Path):
    verifier = _load(VERIFY_PATH, "issue27_verify_matrix_test")
    matrix = tmp_path / "matrix.json"
    matrix.write_text(
        json.dumps({"rows": [{"id": "glm-4.7-flash", "status": "not_run", "gates": {}}]}),
        encoding="utf-8",
    )
    report_path = tmp_path / "report.json"
    report_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="failed report"):
        verifier.update_matrix(
            matrix,
            report_path,
            {"status": "failed", "model_id": "glm-4.7-flash"},
        )

    assert json.loads(matrix.read_text(encoding="utf-8"))["rows"][0]["status"] == "not_run"


def test_wave_prefill_configures_profile_shape_hint(monkeypatch):
    smoke = _load(SMOKE_PATH, "issue27_smoke_env_test")
    monkeypatch.delenv("VLLM_ASCEND_MOE_OFFLOAD_MAX_NUM_SEQS_HINT", raising=False)

    smoke.configure_sew_offload_env(
        "fixed_slot_sync",
        num_slots=4,
        max_num_seqs_hint=1,
    )

    assert os.environ["VLLM_ASCEND_MOE_OFFLOAD_MAX_NUM_SEQS_HINT"] == "1"


def test_qualification_runner_uses_capacity_bounded_default_slots(
    monkeypatch,
):
    runner = _load(RUNNER_PATH, "issue27_runner_slots_test")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_issue27_qualification.py",
            "--model-id",
            "qwen3-next-80b-a3b-instruct",
            "--model-path",
            "/tmp/model",
            "--output-root",
            "/tmp/output",
            "--device",
            "5",
        ],
    )

    assert runner._parse_args().num_slots == 32


def test_capacity_limited_native_oracle_is_explicitly_classified(tmp_path: Path):
    verifier = _load(VERIFY_PATH, "issue27_verify_capacity_test")
    unit = tmp_path / "native-eager"
    unit.mkdir()
    (unit / "summary.json").write_text(
        json.dumps({
            "status": "failed",
            "mode": "no_offload",
            "error": "Engine core initialization failed",
        }),
        encoding="utf-8",
    )
    (unit / "run.log").write_text(
        "torch.OutOfMemoryError: NPU out of memory. Tried to allocate 2.00 GiB\n",
        encoding="utf-8",
    )

    passed, detail = verifier._capacity_failure(unit)

    assert passed is True
    assert detail["status"] == "capacity_limited"
