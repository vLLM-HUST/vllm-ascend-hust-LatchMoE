from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HELPERS = ROOT / "benchmark/scripts/issue28_campaign.py"
RUNNER = ROOT / "benchmark/scripts/run_issue28_campaign.py"
VERIFY = ROOT / "benchmark/scripts/verify_issue28_campaign.py"


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _contract() -> dict:
    order = []
    for repeat, names in enumerate((("a", "b"), ("b", "a"), ("a", "b")), 1):
        order.extend({"arm": name, "repeat": repeat} for name in names)
    return {
        "schema": "latchmoe.issue28.campaign/v1",
        "campaign_id": "test-campaign",
        "model": {"checkpoint": "/model", "config_sha256": "c"},
        "request_manifest": {"path": "/requests", "sha256": "r", "request_count": 1, "order_frozen": True},
        "serving": {"device": 5, "hbm_capacity_mb": 65536, "graph_mode": "PIECEWISE"},
        "arms": [
            {"name": "a", "expected_status": "success", "repeats": 3, "command": ["{python}", "-c", "pass"]},
            {"name": "b", "expected_status": "success_or_capacity_failure", "repeats": 3, "command": ["{python}", "-c", "pass"]},
        ],
        "order": order,
    }


def _identity(contract: dict, digest: str) -> dict:
    return {
        "campaign_id": contract["campaign_id"],
        "contract_sha256": digest,
        "identity": {
            "model": contract["model"],
            "request_manifest": contract["request_manifest"],
            "serving": contract["serving"],
        },
    }


def _success_unit(root: Path, contract: dict, digest: str, item: dict, tokens: list[int] | None = None) -> None:
    unit = root / item["unit_id"]
    unit.mkdir(parents=True)
    _write(unit / "unit_manifest.json", {
        "schema": "latchmoe.issue28.unit/v1",
        **_identity(contract, digest),
        **item,
        "selected_env": {},
    })
    _write(unit / "runner_result.json", {
        "schema": "latchmoe.issue28.runner-result/v1",
        **_identity(contract, digest),
        "unit_id": item["unit_id"],
        "returncode": 0,
        "status": "success",
    })
    names = ["metrics.json", "outputs.json", "runtime.json", "memory.json", "transfers.json"]
    for name in names:
        (unit / name).touch()
    identity = _identity(contract, digest)
    _write(unit / "metrics.json", {**identity, "ttft_ms": [10.0, 11.0], "tpot_ms": [2.0, 2.2], "throughput_tok_s": [100.0]})
    _write(unit / "outputs.json", {**identity, "request_outputs": {"r0": tokens or [1, 2, 3]}})
    _write(unit / "runtime.json", {**identity, "graph_mode": "PIECEWISE", "graph_capture": True, "graph_replay": True, "fallback_count": 0, "wave_count": 2})
    _write(unit / "memory.json", {**identity, "hbm_peak_mb": 1000.0})
    _write(unit / "transfers.json", {**identity, "h2d_count": 2, "h2d_dependency_ok": True})
    (unit / "stdout.log").write_text("ok\n", encoding="utf-8")
    (unit / "stderr.log").write_text("", encoding="utf-8")
    (unit / "raw_trace.jsonl").write_text("{}\n", encoding="utf-8")
    _write(unit / "unit_result.json", {"status": "success", "release_status": "released", "raw_artifacts": ["raw_trace.jsonl"]})


def _campaign(root: Path, contract: dict, reports: list[dict]) -> Path:
    digest = _module(HELPERS, "issue28_helpers_campaign").contract_digest(contract)
    path = root / "campaign.json"
    _write(path, {
        "schema": "latchmoe.issue28.campaign-result/v1",
        "campaign_id": contract["campaign_id"],
        "contract_sha256": digest,
        "order": _module(HELPERS, "issue28_helpers_order").expected_units(contract),
        "units": reports,
    })
    return path


def test_contract_requires_three_runs_in_fixed_order(tmp_path: Path) -> None:
    helpers = _module(HELPERS, "issue28_helpers_contract")
    contract = _contract()
    assert len(helpers.expected_units(contract)) == 6
    broken = json.loads(json.dumps(contract))
    broken["order"] = broken["order"][:-1]
    path = tmp_path / "broken.json"
    _write(path, broken)
    with pytest.raises(ValueError, match="appears"):
        helpers.load_contract(path)


def test_verifier_accepts_complete_matched_campaign(tmp_path: Path) -> None:
    helpers = _module(HELPERS, "issue28_helpers_complete")
    verifier = _module(VERIFY, "issue28_verify_complete")
    contract = _contract()
    digest = helpers.contract_digest(contract)
    reports = []
    for item in helpers.expected_units(contract):
        _success_unit(tmp_path, contract, digest, item)
        reports.append({**item, "unit_dir": str(tmp_path / item["unit_id"]), "status": "success", "returncode": 0})
    campaign = _campaign(tmp_path, contract, reports)
    contract_path = tmp_path / "contract.json"
    _write(contract_path, contract)
    report = verifier.verify_campaign(campaign, contract_path)
    assert report["status"] == "passed"
    assert not report["failures"]


def test_verifier_rejects_cross_arm_token_mismatch(tmp_path: Path) -> None:
    helpers = _module(HELPERS, "issue28_helpers_mismatch")
    verifier = _module(VERIFY, "issue28_verify_mismatch")
    contract = _contract()
    digest = helpers.contract_digest(contract)
    reports = []
    for item in helpers.expected_units(contract):
        tokens = [9, 9] if item["arm"] == "b" else [1, 2, 3]
        _success_unit(tmp_path, contract, digest, item, tokens=tokens)
        reports.append({**item, "unit_dir": str(tmp_path / item["unit_id"]), "status": "success", "returncode": 0})
    campaign = _campaign(tmp_path, contract, reports)
    contract_path = tmp_path / "contract.json"
    _write(contract_path, contract)
    report = verifier.verify_campaign(campaign, contract_path)
    assert report["status"] == "failed"
    assert any("different request output" in failure for failure in report["failures"])


def test_runner_dry_run_records_all_units_with_active_python(tmp_path: Path) -> None:
    helpers = _module(HELPERS, "issue28_helpers_runner")
    runner = _module(RUNNER, "issue28_runner")
    contract = _contract()
    contract_path = tmp_path / "contract.json"
    _write(contract_path, contract)
    output = tmp_path / "output"
    assert runner.run_campaign(contract_path, output, python=Path(sys.executable), dry_run=True) == 0
    campaign = helpers.read_json(output / "campaign.json")
    assert campaign["status"] == "planned"
    assert len(campaign["units"]) == 6
    assert all((output / item["unit_id"] / "unit_manifest.json").is_file() for item in helpers.expected_units(contract))
