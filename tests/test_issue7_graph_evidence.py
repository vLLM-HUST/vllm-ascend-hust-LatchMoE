from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VERIFY_PATH = REPO_ROOT / "benchmark" / "scripts" / "verify_issue7_graph_unit.py"


def _load_verifier():
    spec = importlib.util.spec_from_file_location("issue7_graph_verifier", VERIFY_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _valid_unit(tmp_path: Path) -> Path:
    unit = tmp_path / "unit"
    unit.mkdir()
    model_config = tmp_path / "config.json"
    dataset = tmp_path / "workload.jsonl"
    model_config.write_text("{}", encoding="utf-8")
    dataset.write_text("{}\n", encoding="utf-8")
    ack = tmp_path / "release.json"
    _write_json(ack, {"status": "released"})
    provenance = {
        "repository_head_sha": "a" * 40,
        "repository_parent_sha": "b" * 40,
        "compatibility_lock_sha256": "c" * 64,
        "model_config_sha256": "d" * 64,
        "dataset_manifest_sha256": "e" * 64,
        "device": 5,
        "runtime_bundle_sha256": "f" * 64,
        "runtime_paths_dirty": False,
    }
    _write_json(unit / "unit_manifest.json", {"provenance": provenance})
    _write_json(
        unit / "unit_result.json",
        {"status": "ok", "release_status": "released", "release_ack": str(ack)},
    )
    _write_json(
        unit / "benchmark.json",
        {
            "successful_requests": 1,
            "failed_requests": 0,
            "total_output_tokens": 2,
            "per_request": [
                {"request_id": "request-1", "output_token_ids": [101, 102]}
            ],
        },
    )
    (unit / "server.log").write_text(
        "\n".join(
            (
                "LATCHMOE_GRAPH_CONFIG cudagraph_mode=PIECEWISE splitting_op=vllm::moe_offload_stage status=enabled",
                "Graph capturing finished in 2 secs",
                "Replaying aclgraph",
            )
        ),
        encoding="utf-8",
    )
    (unit / "client.log").write_text("ok\n", encoding="utf-8")
    (unit / "launcher_lifecycle.log").write_text("released\n", encoding="utf-8")
    profile = [
        {
            "name": "graph_slot_address_lock",
            "layer_id": 1,
            "payload": {"w13_data_ptr": 11, "w2_data_ptr": 12, "log2phy_data_ptr": 13},
        },
        {
            "name": "graph_slot_address_validate",
            "layer_id": 1,
            "payload": {
                "w13_data_ptr": 11,
                "w2_data_ptr": 12,
                "log2phy_data_ptr": 13,
                "matches_capture_fingerprint": True,
            },
        },
        {
            "name": "slot_generation_protected_until_compute_complete",
            "layer_id": 1,
            "payload": {"leases_still_match": True},
        },
        {
            "name": "decode_fixed_slot_stage",
            "layer_id": 1,
            "payload": {
                "h2d_bytes": 1024,
                "consumer_dependency_installed": True,
                "mapping_published_after_ready": True,
            },
        },
        {
            "name": "b2_work_conserving_prefill",
            "layer_id": 1,
            "payload": {
                "num_slots": 2,
                "n_active": 3,
                "wave_plan": {"waves": [{"experts": [1, 2]}]},
                "wave_summary": {"wave_count": 1, "prefetch_after_compute_issues": 0},
            },
        },
        {
            "name": "graph_replay_issue",
            "layer_id": None,
            "seconds": 0.001,
            "payload": {"profile_sample_rate": 1, "synchronizes_npu": False},
        },
    ]
    (unit / "moe_profile.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in profile),
        encoding="utf-8",
    )
    (unit / "npu_samples.jsonl").write_text(
        json.dumps(
            {
                "timestamp_ns": 1,
                "device": 5,
                "hbm_usage_percent": 91,
                "npu_utilization_percent": 20,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return unit


def test_issue7_verifier_accepts_complete_graph_bundle(tmp_path: Path) -> None:
    report = _load_verifier().verify_unit(_valid_unit(tmp_path))

    assert report["status"] == "passed"
    assert report["failures"] == []


def test_issue7_verifier_fails_closed_on_changed_address(tmp_path: Path) -> None:
    unit = _valid_unit(tmp_path)
    records = [json.loads(line) for line in (unit / "moe_profile.jsonl").read_text().splitlines()]
    records[1]["payload"]["w13_data_ptr"] = 99
    (unit / "moe_profile.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    report = _load_verifier().verify_unit(unit)

    assert report["status"] == "failed"
    assert "layer 1 changed w13_data_ptr" in report["failures"]


def test_issue7_verifier_fails_closed_without_release_ack(tmp_path: Path) -> None:
    unit = _valid_unit(tmp_path)
    result = json.loads((unit / "unit_result.json").read_text())
    result["release_status"] = "release-failed"
    _write_json(unit / "unit_result.json", result)

    report = _load_verifier().verify_unit(unit)

    assert report["status"] == "failed"
    assert "release status is 'release-failed'" in report["failures"]


def test_issue7_verifier_prefers_portable_local_release_ack(tmp_path: Path) -> None:
    unit = _valid_unit(tmp_path)
    result = json.loads((unit / "unit_result.json").read_text())
    result["release_ack"] = "/missing/on/fresh/checkout.json"
    _write_json(unit / "unit_result.json", result)
    _write_json(unit / "release_ack.json", {"status": "released"})

    report = _load_verifier().verify_unit(unit)

    assert report["status"] == "passed"


def test_issue7_verifier_fails_closed_on_over_capacity_wave(tmp_path: Path) -> None:
    unit = _valid_unit(tmp_path)
    records = [json.loads(line) for line in (unit / "moe_profile.jsonl").read_text().splitlines()]
    records[-2]["payload"]["wave_plan"]["waves"][0]["experts"] = [1, 2, 3]
    (unit / "moe_profile.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    report = _load_verifier().verify_unit(unit)

    assert report["status"] == "failed"
    assert "wave active working set exceeds slot capacity" in report["failures"]


def test_issue7_verifier_requires_exact_oracle_tokens(tmp_path: Path) -> None:
    unit = _valid_unit(tmp_path)
    oracle = tmp_path / "oracle.json"
    _write_json(
        oracle,
        {
            "per_request": [
                {"request_id": "request-1", "output_token_ids": [101, 999]}
            ]
        },
    )

    report = _load_verifier().verify_unit(unit, oracle_benchmark=oracle)

    assert report["status"] == "failed"
    assert report["oracle"]["mismatched_request_ids"] == ["request-1"]
    assert "oracle token IDs differ: ['request-1']" in report["failures"]


def test_issue7_verifier_accepts_oracle_subset(tmp_path: Path) -> None:
    unit = _valid_unit(tmp_path)
    oracle = tmp_path / "oracle.json"
    _write_json(
        oracle,
        {
            "per_request": [
                {"request_id": "request-1", "output_token_ids": [101, 102]}
            ]
        },
    )

    report = _load_verifier().verify_unit(unit, oracle_benchmark=oracle)

    assert report["status"] == "passed"
    assert report["oracle"]["exact_requests"] == 1
    assert report["oracle"]["exact_tokens"] == 2
