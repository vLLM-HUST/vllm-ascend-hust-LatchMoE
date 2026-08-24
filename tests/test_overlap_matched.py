from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_verifier():
    path = REPO_ROOT / "benchmark" / "scripts" / "verify_overlap_matched.py"
    spec = importlib.util.spec_from_file_location("overlap_verifier_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_baseline_verifier():
    path = REPO_ROOT / "benchmark" / "scripts" / "verify_baseline_matched.py"
    spec = importlib.util.spec_from_file_location("baseline_verifier_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_matched_cases_differ_only_in_async_load() -> None:
    config = yaml.safe_load(
        (REPO_ROOT / "benchmark" / "configs" / "sew_offload_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    cases = {case["name"]: case for case in config["cases"]}
    serial = cases["sew_14gb_serial_stage_matched"]
    overlap = cases["sew_14gb_overlap_stage_matched"]
    assert serial.get("server_args", []) == overlap.get("server_args", [])
    serial_env = dict(serial["env"])
    overlap_env = dict(overlap["env"])
    assert serial_env.pop("VLLM_ASCEND_MOE_OFFLOAD_ASYNC_LOAD") == "0"
    assert overlap_env.pop("VLLM_ASCEND_MOE_OFFLOAD_ASYNC_LOAD") == "1"
    assert serial_env == overlap_env


def test_performance_baselines_share_kv_and_graph_contract() -> None:
    config = yaml.safe_load(
        (REPO_ROOT / "benchmark" / "configs" / "sew_offload_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    cases = {case["name"]: case for case in config["cases"]}
    names = (
        "no_offload_kv512m_aclgraph",
        "native_prefetch_14gb_kv512m",
        "legacy_layered_14gb_kv512m",
        "sew_14gb_autoslots_kv512m",
    )
    for name in names:
        args = cases[name].get("server_args", [])
        index = args.index("--kv-cache-memory-bytes")
        assert args[index + 1] == "536870912"
        assert "--enforce-eager" not in args


def test_preregistered_estimator_uses_pair_medians() -> None:
    verifier = _load_verifier()
    pair_logs = [
        [0.0, 0.0, -0.2],
        [-0.1, -0.1, 0.4],
        [-0.3, -0.3, 0.8],
    ]
    expected = verifier.math.exp((-0.0 - 0.1 - 0.3) / 3.0) - 1.0
    assert verifier._estimate(pair_logs) == expected


def test_bootstrap_is_deterministic_and_preregistered() -> None:
    verifier = _load_verifier()
    pair_logs = [[-0.1] * 8, [-0.2] * 8, [-0.3] * 8]
    first = verifier._bootstrap(pair_logs, 0.95)
    second = verifier._bootstrap(pair_logs, 0.95)
    assert first == second
    assert first["replicates"] == 10_000
    assert first["seed"] == 20_260_823
    assert first["upper"] < 0.0


def test_baseline_exactness_mismatch_is_retained_as_unsupported() -> None:
    verifier = _load_baseline_verifier()
    assert verifier._verification_status([], ["output tokens differ"]) == "unsupported"
    assert verifier._verification_status(["graph replay missing"], []) == "failed"
    assert verifier._verification_status([], []) == "passed"
