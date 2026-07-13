from __future__ import annotations

import importlib.util
from pathlib import Path
from threading import Thread
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


REPO_ROOT = Path(__file__).resolve().parents[1]
SEW_BENCH_PATH = REPO_ROOT / "benchmark" / "scripts" / "sew_bench.py"
RUN_SUITE_PATH = REPO_ROOT / "benchmark" / "scripts" / "run_suite.py"
COLLECT_EVIDENCE_PATH = REPO_ROOT / "benchmark" / "scripts" / "collect_evidence.py"
COLLECT_UVA_PATH = REPO_ROOT / "benchmark" / "scripts" / "collect_uva_feasibility.py"
RUN_UVA_PATH = REPO_ROOT / "benchmark" / "scripts" / "run_uva_feasibility.py"
RENDER_UVA_PATH = REPO_ROOT / "benchmark" / "scripts" / "render_uva_feasibility_report.py"


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


def load_collect_uva():
    spec = importlib.util.spec_from_file_location(
        "collect_uva_test_module", COLLECT_UVA_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_run_uva():
    spec = importlib.util.spec_from_file_location("run_uva_test_module", RUN_UVA_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_render_uva():
    spec = importlib.util.spec_from_file_location("render_uva_test_module", RENDER_UVA_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_benchmark_config_validates():
    sew_bench = load_sew_bench()
    config = sew_bench.load_config()

    assert sew_bench.validate_config(config) == []
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
        "native_prefetch_14gb_eager",
        "legacy_layered_14gb",
        "legacy_layered_14gb_eager",
        "sew_14gb_autoslots",
        "sew_14gb_capture_disabled",
        "sew_28gb_autoslots",
        "sew_28gb_slots32_capture_disabled",
    }.issubset(set(names))
    assert {"baseline", "main", "ablation", "sensitivity"}.issubset(roles)


def test_end_to_end_experiment_includes_eager_and_capture_disabled_baselines():
    sew_bench = load_sew_bench()
    config = sew_bench.load_config()
    e1_cases = set(config["experiments"]["e1_end_to_end"]["cases"])

    assert {
        "native_prefetch_14gb",
        "native_prefetch_14gb_eager",
        "legacy_layered_14gb",
        "legacy_layered_14gb_eager",
        "sew_14gb_capture_disabled",
        "sew_14gb_autoslots",
        "sew_28gb_slots32_capture_disabled",
        "sew_28gb_slots32",
    }.issubset(e1_cases)


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


def test_uva_verdict_treats_host_matmul_failure_as_non_viable():
    collect_uva = load_collect_uva()

    rows = [
        {"gate": "U0_runtime_mapping", "operation": "aclrtHostRegister", "ok": True},
        {"gate": "U0_framework_wrapping", "operation": "torch_npu_private_tensor_wrap", "ok": True},
        {"gate": "U1_tensor_access_matrix", "operation": "add_float16_zero", "ok": True},
        {"gate": "U1_tensor_access_matrix", "operation": "copy_uint8", "ok": False},
        {"gate": "U1_hbm_reference", "operation": "hbm_add_64MiB", "ok": True},
        {"gate": "U2_npugraph_replay", "operation": "npugraph_replay_host_update", "ok": True},
        {
            "gate": "U3_moe_shaped_matmul",
            "artifact": "probe_device4_matmul_m16_k4096_n4096.json",
            "operation": "host_registered_weight_matmul_m16_k4096_n4096",
            "status": "host_registered_matmul_failed",
            "ok": False,
            "note": "507057",
        },
        {
            "gate": "U3_hbm_matmul_reference",
            "operation": "hbm_weight_matmul_m16_k4096_n4096",
            "ok": True,
        },
    ]

    verdict = collect_uva.derive_verdict(rows)

    assert verdict["verdict"] == "not_viable_as_sew_baseline"
    assert verdict["comparison_to_sew"] == "compatibility_failure_not_latency_throughput_comparison"
    assert verdict["primary_blocker"] == "host_registered_matmul_weight_path_fails_507057"
    assert verdict["passed_gates"]["simple_npugraph_replay"] is True
    assert verdict["failed_gates"]["host_registered_matmul_weight_path"] is True


def test_uva_runner_marks_expected_nonzero_commands():
    run_uva = load_run_uva()
    config = {
        "uva_like_feasibility": {
            "expected_nonzero_commands": ["tensor_access_matrix", "matmul_probe_2mib"],
            "commands": {
                "small_runtime_probe": ["python3", "probe.py"],
                "tensor_access_matrix": ["python3", "matrix.py"],
                "matmul_probe_2mib": ["python3", "matmul.py"],
            },
        }
    }

    plan = run_uva.build_plan(config)

    expected = {item["name"]: item["expected_nonzero"] for item in plan}
    assert expected == {
        "small_runtime_probe": False,
        "tensor_access_matrix": True,
        "matmul_probe_2mib": True,
    }


def test_uva_runner_summary_accepts_expected_verdict(tmp_path):
    run_uva = load_run_uva()
    verdict_path = tmp_path / "verdict.json"
    verdict_path.write_text('{"verdict":"not_viable_as_sew_baseline"}', encoding="utf-8")
    config = {"uva_like_feasibility": {"expected_verdict": "not_viable_as_sew_baseline"}}
    records = [
        {"name": "small_runtime_probe", "status": "ok"},
        {"name": "matmul_probe_2mib", "status": "expected_nonzero"},
    ]

    summary = run_uva.derive_runner_status(config, records, verdict_path)

    assert summary["status"] == "ok"
    assert summary["verdict"] == "not_viable_as_sew_baseline"
    assert summary["verdict_matches"] is True


def test_uva_report_keeps_compatibility_failure_boundary():
    render_uva = load_render_uva()
    rows = [
        {
            "gate": "U0_runtime_mapping",
            "operation": "aclrtHostRegister",
            "ok": "True",
            "status": "runtime_mapping_possible",
            "size_mib": "14336",
        },
        {
            "gate": "U1_elementwise_read_bandwidth",
            "operation": "host_registered_add_64MiB",
            "ok": "True",
            "status": "tensor_wrap_possible",
            "size_mib": "64",
            "approx_source_read_gib_s": "8.69",
        },
        {
            "gate": "U1_hbm_reference",
            "operation": "hbm_add_64MiB",
            "ok": "True",
            "status": "ok",
            "size_mib": "64",
            "approx_source_read_gib_s": "399.21",
        },
        {
            "gate": "U1_elementwise_read_bandwidth",
            "operation": "host_registered_add_256MiB",
            "ok": "True",
            "status": "tensor_wrap_possible",
            "size_mib": "256",
            "approx_source_read_gib_s": "9.06",
        },
        {
            "gate": "U1_hbm_reference",
            "operation": "hbm_add_256MiB",
            "ok": "True",
            "status": "ok",
            "size_mib": "256",
            "approx_source_read_gib_s": "339.31",
        },
        {
            "gate": "U3_moe_shaped_matmul",
            "operation": "host_registered_weight_matmul_m16_k1024_n1024",
            "ok": "False",
            "status": "host_registered_matmul_failed",
            "size_mib": "2",
            "note": "507057",
        },
        {
            "gate": "U3_moe_shaped_matmul",
            "operation": "host_registered_weight_matmul_m16_k4096_n4096",
            "ok": "False",
            "status": "host_registered_matmul_failed",
            "size_mib": "32",
            "note": "507057",
        },
    ]
    verdict = {
        "verdict": "not_viable_as_sew_baseline",
        "comparison_to_sew": "compatibility_failure_not_latency_throughput_comparison",
        "primary_blocker": "host_registered_matmul_weight_path_fails_507057",
        "offload_budget_gb": 14,
        "allowed_claim": "partial path, not drop-in",
    }
    runner = {
        "summary": {
            "status": "ok",
            "expected_verdict": "not_viable_as_sew_baseline",
            "verdict": "not_viable_as_sew_baseline",
            "verdict_matches": True,
        },
        "records": [],
    }

    report = render_uva.render(rows, verdict, runner)

    assert "compatibility-failure baseline" in report
    assert "direct UVA-like expert matmul is not runnable" in report
    assert "not_viable_as_sew_baseline" in report
