from __future__ import annotations

import importlib.util
from pathlib import Path
from threading import Thread
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


REPO_ROOT = Path(__file__).resolve().parents[1]
SEW_BENCH_PATH = REPO_ROOT / "benchmark" / "scripts" / "sew_bench.py"
RUN_SUITE_PATH = REPO_ROOT / "benchmark" / "scripts" / "run_suite.py"
COLLECT_EVIDENCE_PATH = REPO_ROOT / "benchmark" / "scripts" / "collect_evidence.py"


def load_sew_bench():
    spec = importlib.util.spec_from_file_location("sew_bench_test_module", SEW_BENCH_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_run_suite():
    spec = importlib.util.spec_from_file_location("run_suite_test_module", RUN_SUITE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_collect_evidence():
    spec = importlib.util.spec_from_file_location(
        "collect_evidence_test_module", COLLECT_EVIDENCE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module




def test_benchmark_config_validates(monkeypatch):
    sew_bench = load_sew_bench()
    config = sew_bench.load_config()
    monkeypatch.setattr(Path, "exists", lambda _path: True)

    assert sew_bench.validate_config(config, check_resource_paths=False) == []
    assert config["dataset"]["source"] == "sharegpt"
    assert config["dataset"]["random_dataset_allowed"] is False
    assert config["dataset"]["synthetic_smoke_allowed"] is False


def test_benchmark_case_names_are_unique_and_cover_required_roles():
    sew_bench = load_sew_bench()
    config = sew_bench.load_config()
    cases = config["cases"]
    names = [case["name"] for case in cases]
    roles = {case["role"] for case in cases}

    assert len(names) == len(set(names))
    assert {
        "no_offload_capacity_probe",
        "native_prefetch_14gb",
        "legacy_layered_14gb",
        "sew_14gb_autoslots",
        "sew_28gb_autoslots",
    }.issubset(set(names))
    assert {"baseline", "main", "ablation", "sensitivity"}.issubset(roles)


def test_end_to_end_experiment_is_graph_only():
    sew_bench = load_sew_bench()
    config = sew_bench.load_config()
    e1_cases = set(config["experiments"]["e1_end_to_end"]["cases"])

    assert {
        "native_prefetch_14gb",
        "legacy_layered_14gb",
        "sew_14gb_autoslots",
        "sew_28gb_slots32",
    }.issubset(e1_cases)
    assert all(
        "--enforce-eager" not in case.get("server_args", [])
        for case in config["cases"]
    )


def test_validator_reports_unreadable_paths_instead_of_crashing(monkeypatch):
    sew_bench = load_sew_bench()
    config = sew_bench.load_config()
    original_exists = Path.exists

    def guarded_exists(path):
        if str(path).endswith("Qwen3-30B-A3B"):
            raise PermissionError("blocked test path")
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", guarded_exists)
    issues = sew_bench.validate_config(config)
    assert any("PermissionError" in issue for issue in issues)


def test_sew_cases_do_not_mix_native_prefetch_flags():
    sew_bench = load_sew_bench()
    config = sew_bench.load_config()

    for case in config["cases"]:
        env = case.get("env", {})
        server_args = {str(item) for item in case.get("server_args", [])}
        if env.get("VLLM_ASCEND_MOE_OFFLOAD_SEW_DATAPLANE") == "1":
            assert server_args.isdisjoint(sew_bench.NATIVE_OFFLOAD_FLAGS), case["name"]


def test_render_plan_for_single_unit():
    sew_bench = load_sew_bench()
    config = sew_bench.load_config()
    plan = sew_bench.render_plan(
        config,
        case_names=["sew_14gb_autoslots"],
        workload_names=["smoke"],
        python_exe="python",
    )

    assert len(plan["units"]) == 1
    unit = plan["units"][0]
    assert unit["case"] == "sew_14gb_autoslots"
    assert unit["workload"] == "smoke"
    assert unit["server_command"][:2] == ["vllm", "serve"]
    assert unit["client_command"][0] == "python"
    assert "--bucket" in unit["client_command"]
    assert "smoke" in unit["client_command"]


def test_wait_for_server_bypasses_proxy(monkeypatch):
    run_suite = load_run_suite()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"object":"list","data":[]}')

        def log_message(self, *_args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")
    monkeypatch.delenv("NO_PROXY", raising=False)
    try:
        run_suite._wait_for_server(
            f"http://127.0.0.1:{server.server_port}/v1/models",
            proc=None,
            timeout_s=2,
        )
    finally:
        server.shutdown()
        server.server_close()


def test_run_suite_profile_env_defaults_are_low_overhead(tmp_path, monkeypatch):
    run_suite = load_run_suite()
    monkeypatch.delenv("VLLM_ASCEND_MOE_PROFILE_EXPERT_LISTS", raising=False)
    monkeypatch.delenv("VLLM_ASCEND_MOE_DECODE_PROFILE_SAMPLE_RATE", raising=False)
    monkeypatch.delenv("VLLM_ASCEND_MOE_B2_PROFILE_DETAILS", raising=False)

    env = run_suite._unit_env({"env": {}}, tmp_path)

    assert env["VLLM_ASCEND_MOE_PROFILE_EXPERT_LISTS"] == "0"
    assert env["VLLM_ASCEND_MOE_DECODE_PROFILE_SAMPLE_RATE"] == "8"
    assert env["VLLM_ASCEND_MOE_B2_PROFILE_DETAILS"] == "0"


def test_run_suite_profile_env_defaults_can_be_overridden(tmp_path, monkeypatch):
    run_suite = load_run_suite()
    monkeypatch.delenv("VLLM_ASCEND_MOE_PROFILE_EXPERT_LISTS", raising=False)
    monkeypatch.delenv("VLLM_ASCEND_MOE_DECODE_PROFILE_SAMPLE_RATE", raising=False)
    monkeypatch.delenv("VLLM_ASCEND_MOE_B2_PROFILE_DETAILS", raising=False)

    env = run_suite._unit_env(
        {
            "env": {
                "VLLM_ASCEND_MOE_PROFILE_EXPERT_LISTS": "1",
                "VLLM_ASCEND_MOE_DECODE_PROFILE_SAMPLE_RATE": "1",
                "VLLM_ASCEND_MOE_B2_PROFILE_DETAILS": "1",
            }
        },
        tmp_path,
    )

    assert env["VLLM_ASCEND_MOE_PROFILE_EXPERT_LISTS"] == "1"
    assert env["VLLM_ASCEND_MOE_DECODE_PROFILE_SAMPLE_RATE"] == "1"
    assert env["VLLM_ASCEND_MOE_B2_PROFILE_DETAILS"] == "1"


def test_collect_evidence_extracts_log_and_profile_fields(tmp_path):
    collect = load_collect_evidence()
    server_log = tmp_path / "server.log"
    benchmark_json = tmp_path / "benchmark.json"
    profile_jsonl = tmp_path / "moe_profile.jsonl"
    result_json = tmp_path / "unit_result.json"

    server_log.write_text(
        "\n".join(
            [
                "Loading model weights took 46.7751 GB",
                "Available KV cache memory: 1.81 GiB",
                "GPU KV cache size: 19,712 tokens",
                "Maximum concurrency for 4,096 tokens per request: 4.81x",
                "Graph capturing finished in 2 secs",
                "splitting_ops=['vllm::moe_offload_stage']",
            ]
        ),
        encoding="utf-8",
    )
    benchmark_json.write_text(
        '{"successful_requests":1,"failed_requests":0,'
        '"median_ttft_ms":10.0,"median_tpot_ms":2.0,'
        '"output_throughput":3.0}',
        encoding="utf-8",
    )
    profile_jsonl.write_text(
        '{"name":"decode_fixed_slot_stage","memory_ledger":'
        '{"host_store_bytes":1073741824,"slot_bank_bytes":2147483648,'
        '"total_managed_bytes":3221225472,"registered_layers":1},'
        '"payload":{"num_slots":32,"h2d_bytes":1024,"stage_ms":1.5,'
        '"n_active":4,"profile_sample_rate":2}}\n',
        encoding="utf-8",
    )
    result_json.write_text(
        "{"
        '"status":"ok","stage":"completed",'
        '"case":{"name":"sew_14gb_autoslots"},'
        '"workload":{"name":"smoke"},'
        f'"server_log":"{server_log}",'
        f'"benchmark_json":"{benchmark_json}",'
        f'"profile_jsonl":"{profile_jsonl}"'
        "}",
        encoding="utf-8",
    )

    row = collect.collect_unit(result_json)
    assert row["status"] == "ok"
    assert row["weights_gb"] == 46.7751
    assert row["available_kv_gib"] == 1.81
    assert row["kv_cache_tokens"] == 19712
    assert row["graph_capture_completed"] is True
    assert row["moe_offload_stage_seen"] is True
    assert row["num_slots"] == 32
    assert row["slot_bank_gib"] == 2.0
    assert row["h2d_gib_total"] == 2048 / collect.BYTES_PER_GIB
    assert row["stage_ms_total"] == 3.0
