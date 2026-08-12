from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VERIFY = ROOT / "benchmark" / "scripts" / "verify_issue17_matched_ttft.py"
PACKAGE = ROOT / "benchmark" / "scripts" / "package_issue17_evidence.py"


def _module():
    spec = importlib.util.spec_from_file_location("issue17_verify", VERIFY)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _package_module():
    spec = importlib.util.spec_from_file_location("issue17_package", PACKAGE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _unit(tmp_path: Path, arm: str) -> Path:
    unit = tmp_path / arm
    unit.mkdir()
    provenance = {
        "repository_head_sha": "a" * 40,
        "repository_parent_sha": "b" * 40,
        "compatibility_lock_sha256": "c" * 64,
        "model_config_sha256": "d" * 64,
        "dataset_manifest_sha256": "e" * 64,
        "device": 5,
        "runtime_bundle_sha256": "f" * 64,
        "vllm_root_sha": "1" * 40,
        "seam_root_sha": "2" * 40,
        "runtime_paths_dirty": False,
    }
    case = f"sew_14gb_{arm}_matched"
    _write(unit / "unit_manifest.json", {"case": {"name": case}, "selected_env": {"VLLM_ASCEND_MOE_OFFLOAD_B2_OVERFLOW_MODE": arm}, "provenance": provenance})
    _write(unit / "unit_result.json", {"status": "ok", "release_status": "released"})
    _write(unit / "release_ack.json", {"status": "released"})
    requests = [
        {"request_id": f"r{index}", "ttft_s": 0.1 + index * 0.01, "total_s": 1.1 + index * 0.01, "output_tokens": 2, "output_token_ids": [10, index]}
        for index in range(3)
    ]
    _write(unit / "benchmark.json", {"successful_requests": 3, "failed_requests": 0, "total_output_tokens": 6, "output_throughput": 5.0, "per_request": requests})
    (unit / "server.log").write_text("LATCHMOE_GRAPH_CONFIG cudagraph_mode=PIECEWISE splitting_op=vllm::moe_offload_stage status=enabled\nGraph capturing finished\nReplaying aclgraph\n", encoding="utf-8")
    (unit / "client.log").write_text("ok\n", encoding="utf-8")
    (unit / "launcher_lifecycle.log").write_text("released\n", encoding="utf-8")
    name = "b2_reference_full_layer_prefill" if arm == "full_layer" else "b2_work_conserving_prefill"
    payload = {"execution_mode": "reference_full_layer" if arm == "full_layer" else "pair_microbatch", "fallback_reason": None}
    (unit / "moe_profile.jsonl").write_text(json.dumps({"name": name, "payload": payload}) + "\n", encoding="utf-8")
    (unit / "npu_samples.jsonl").write_text(json.dumps({"hbm_usage_percent": 5}) + "\n", encoding="utf-8")
    return unit


def test_percentile_reports_p95() -> None:
    assert _module().percentile([1.0, 2.0, 3.0], 95) == 2.9


def test_verifier_accepts_matched_multi_wave(tmp_path: Path) -> None:
    full = _unit(tmp_path, "full_layer")
    wave = _unit(tmp_path, "multi_wave")
    report = _module().verify_unit(wave, arm="multi_wave", expected_requests=3, oracle_benchmark=full / "benchmark.json")
    assert report["status"] == "passed"
    assert report["oracle"]["exact_tokens"] == 6
    assert report["distribution"]["ttft_ms"]["p95"] == pytest.approx(119.0)


def test_verifier_rejects_multi_wave_full_layer_fallback(tmp_path: Path) -> None:
    wave = _unit(tmp_path, "multi_wave")
    profile = wave / "moe_profile.jsonl"
    profile.write_text(profile.read_text() + json.dumps({"name": "b2_reference_full_layer_prefill", "payload": {"fallback_reason": "preflight"}}) + "\n", encoding="utf-8")
    report = _module().verify_unit(wave, arm="multi_wave", expected_requests=3)
    assert report["status"] == "failed"
    assert "multi_wave arm fell back to or mixed with full_layer" in report["failures"]


def test_verifier_rejects_token_mismatch(tmp_path: Path) -> None:
    full = _unit(tmp_path, "full_layer")
    wave = _unit(tmp_path, "multi_wave")
    benchmark = json.loads((wave / "benchmark.json").read_text())
    benchmark["per_request"][1]["output_token_ids"] = [99]
    _write(wave / "benchmark.json", benchmark)
    report = _module().verify_unit(wave, arm="multi_wave", expected_requests=3, oracle_benchmark=full / "benchmark.json")
    assert report["status"] == "failed"
    assert report["oracle"]["mismatched_request_ids"] == ["r1"]


def test_packager_requires_passing_campaign(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    _write(output / "campaign.json", {"units": []})
    _write(output / "matched_summary.json", {"status": "failed"})
    with pytest.raises(ValueError, match="has not passed"):
        _package_module().package(output, tmp_path / "bundle.tar.gz")
