#!/usr/bin/env python3
"""Run the standard SEW-Offload benchmark suite."""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from sew_bench import (  # noqa: E402
    DEFAULT_CONFIG,
    build_client_command,
    build_server_command,
    load_config,
    repo_relative,
    select_cases,
    select_workloads,
    utc_stamp,
    validate_config,
    write_json,
)


DEFAULT_OUTPUT_ROOT = Path("benchmark/artifacts/runs")
DEFAULT_MANAGER = Path("third_party/vllm-hust-dev-hub/manage.sh")
IMAGE_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--workload", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-start-server", action="store_true")
    parser.add_argument("--server-manager", default=str(DEFAULT_MANAGER))
    parser.add_argument("--device", type=int, choices=(5, 6))
    parser.add_argument("--runtime-image")
    parser.add_argument("--runtime-image-digest")
    parser.add_argument("--container-name")
    parser.add_argument("--custody-unit-prefix", default="latchmoe-suite")
    parser.add_argument("--release-ack-dir")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--startup-timeout-s", type=float)
    parser.add_argument(
        "--max-requests",
        type=int,
        default=0,
        help="Override workload num_requests without resampling the manifest.",
    )
    parser.add_argument(
        "--client-concurrency",
        type=int,
        default=0,
        help="Override client concurrency for this suite run.",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=0,
        help="Override the vLLM serving max_num_seqs for this suite run.",
    )
    return parser.parse_args()


def _health_url(config: dict[str, Any]) -> str:
    server = config["server"]
    return f"http://{server['host']}:{int(server['port'])}/v1/models"


def _wait_for_server(
    url: str, proc: subprocess.Popen[Any] | None, timeout_s: float
) -> None:
    deadline = time.monotonic() + float(timeout_s)
    last_error = ""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(url, headers={"Connection": "close"})
    while time.monotonic() < deadline:
        try:
            with opener.open(request, timeout=5) as resp:
                if 200 <= int(resp.status) < 500:
                    return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = str(exc)
        time.sleep(5)
    raise TimeoutError(f"server did not become ready at {url}: {last_error}")


def _manager_path(value: str) -> Path:
    expected = (Path(__file__).resolve().parents[2] / DEFAULT_MANAGER).resolve()
    candidate = Path(value).resolve()
    if candidate != expected or not candidate.is_file() or not os.access(candidate, os.X_OK):
        raise ValueError("server manager must be the pinned repository dev-hub manage.sh")
    return candidate


def _assert_port_free(port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError as exc:
            raise RuntimeError(f"port {port} is occupied; refusing server reuse") from exc


def _assert_device_free(device: int) -> None:
    completed = subprocess.run(
        ["npu-smi", "info", "-t", "proc-mem", "-i", str(device)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if "No process in device." not in completed.stdout:
        raise RuntimeError(f"NPU{device} is occupied; refusing to continue")


def _managed_env(
    config: dict[str, Any],
    case: dict[str, Any],
    unit_dir: Path,
    args: argparse.Namespace,
) -> dict[str, str]:
    if args.device not in {5, 6}:
        raise ValueError("a physical --device 5 or 6 is required")
    if not args.runtime_image or not IMAGE_DIGEST.fullmatch(str(args.runtime_image_digest or "")):
        raise ValueError("an immutable runtime image and sha256 digest are required")
    if not args.container_name:
        raise ValueError("a unique --container-name is required")
    unit_name = f"{args.custody_unit_prefix}-{unit_dir.parent.name}-{unit_dir.name}.service"
    if len(unit_name) > 200 or not re.fullmatch(r"[A-Za-z0-9_.@-]+\.service", unit_name):
        raise ValueError("generated custody unit name is invalid")
    server = config["server"]
    shape = config["serving_shape"]
    model = config["model"]
    image = str(args.runtime_image).split("@", 1)[0]
    env = _unit_env(case, unit_dir)
    env.update(
        {
            "VLLM_ENGINE_SYSTEMD_UNIT": unit_name,
            "VLLM_ENGINE_CONTAINER": str(args.container_name),
            "VLLM_ENGINE_IMAGE": f"{image}@{args.runtime_image_digest}",
            "VLLM_ENGINE_AUTO_CREATE_CONTAINER": "true",
            "VLLM_ENGINE_RECREATE_CONTAINER": "false",
            "VLLM_ENGINE_REPLACE_EXISTING": "false",
            "VLLM_ENGINE_MODEL_PATH": str(model["path"]),
            "VLLM_ENGINE_SERVED_MODEL_NAME": str(model["served_model_name"]),
            "VLLM_ENGINE_PORT": str(server["port"]),
            "VLLM_ENGINE_TP_SIZE": str(model["tensor_parallel_size"]),
            "VLLM_ENGINE_NPU_DEVICES": str(args.device),
            "VLLM_ENGINE_MAX_MODEL_LEN": str(shape["max_model_len"]),
            "VLLM_ENGINE_MAX_NUM_BATCHED_TOKENS": str(shape["max_num_batched_tokens"]),
            "VLLM_ENGINE_MAX_NUM_SEQS": str(shape["max_num_seqs"]),
            "VLLM_ENGINE_GPU_MEM_UTIL": str(shape["gpu_memory_utilization"]),
            "VLLM_ENGINE_DTYPE": str(model["dtype"]),
            # Prefix-cache hits are outside LatchMoE's correctness contract.
            # Pin the managed launcher to the full-prefill path even when the
            # dev-hub default or the caller's shell enables cache reuse.
            "VLLM_ENGINE_ENABLE_PREFIX_CACHING": "0",
            "VLLM_ENGINE_ENFORCE_EAGER": "0",
            "VLLM_ENGINE_EXTRA_ARGS_JSON": json.dumps(
                ["--trust-remote-code", *[str(item) for item in case.get("server_args", [])]]
            ),
            "VLLM_OPTIMIZATION_REPO_CONTAINER": "/workspace/vllm-ascend-hust-LatchMoE",
            "VLLM_OPTIMIZATION_SRC_SUBDIR": "",
            "VLLM_ENGINE_EXTRA_ENV_PREFIXES": "VLLM_ASCEND_MOE_",
        }
    )
    forbidden = json.dumps({key: value for key, value in env.items() if key.startswith("VLLM_ENGINE")}).lower()
    if "--enforce-eager" in forbidden or '"vllm_engine_enforce_eager":"1"' in forbidden.replace(" ", ""):
        raise ValueError("forced eager is forbidden")
    return env


def _manager_call(manager: Path, action: str, env: dict[str, str], log_path: Path, *, check: bool) -> subprocess.CompletedProcess[str]:
    command = [str(manager), action]
    if action in {"status", "health"}:
        command.append("--json")
    completed = subprocess.run(
        command,
        env=env,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{time.time_ns()}] {action} rc={completed.returncode}\n{completed.stdout}")
        if completed.stdout and not completed.stdout.endswith("\n"):
            handle.write("\n")
    if check and completed.returncode != 0:
        raise RuntimeError(f"managed launcher {action} failed with code {completed.returncode}")
    return completed


def _unit_env(case: dict[str, Any], unit_dir: Path) -> dict[str, str]:
    env = dict(os.environ)
    profile_path = unit_dir / "moe_profile.jsonl"
    trace_path = unit_dir / "moe_trace.jsonl"
    env.setdefault("VLLM_ASCEND_MOE_GMM_PROFILE_PATH", str(profile_path))
    env.setdefault("VLLM_ASCEND_MOE_OFFLOAD_PROFILE_PATH", str(profile_path))
    env.setdefault("VLLM_ASCEND_MOE_OFFLOAD_TRACE_PATH", str(trace_path))
    env.setdefault("VLLM_ASCEND_MOE_PROFILE_EXPERT_LISTS", "0")
    env.setdefault("VLLM_ASCEND_MOE_DECODE_PROFILE_SAMPLE_RATE", "8")
    env.setdefault("VLLM_ASCEND_MOE_B2_PROFILE_DETAILS", "0")
    for key, value in (case.get("env") or {}).items():
        if value is None or value == "":
            env.pop(str(key), None)
        else:
            env[str(key)] = str(value)
    return env


def _selected_env(env: dict[str, str]) -> dict[str, str]:
    return {
        key: env.get(key, "")
        for key in sorted(env)
        if key.startswith("VLLM_ASCEND_MOE")
        or key in {"ASCEND_RT_VISIBLE_DEVICES", "PYTHONPATH"}
    }


def _write_unit_markdown(unit_dir: Path, payload: dict[str, Any]) -> None:
    bench = {}
    benchmark_path = Path(payload.get("benchmark_json", ""))
    if benchmark_path.exists():
        bench = json.loads(benchmark_path.read_text(encoding="utf-8"))
    lines = [
        f"# {payload['case']['name']} / {payload['workload']['name']}",
        "",
        f"- Status: {payload.get('status', '')}",
        f"- Stage: {payload.get('stage', '')}",
        f"- Server log: `{payload.get('server_log', '')}`",
        f"- Client log: `{payload.get('client_log', '')}`",
        f"- Benchmark JSON: `{payload.get('benchmark_json', '')}`",
        "",
        "## Metrics",
        "",
        f"- Successful requests: {bench.get('successful_requests', 0)}",
        f"- Failed requests: {bench.get('failed_requests', 0)}",
        f"- Median TTFT ms: {bench.get('median_ttft_ms', 0.0):.3f}",
        f"- Median TPOT ms: {bench.get('median_tpot_ms', 0.0):.3f}",
        f"- Output throughput tok/s: {bench.get('output_throughput', 0.0):.3f}",
        "",
    ]
    if payload.get("error"):
        lines.extend(["## Error", "", str(payload["error"]), ""])
    (unit_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def run_unit(
    *,
    config: dict[str, Any],
    case: dict[str, Any],
    workload: dict[str, Any],
    suite_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    unit_dir = suite_dir / str(case["name"]) / str(workload["name"])
    unit_dir.mkdir(parents=True, exist_ok=True)
    env = _unit_env(case, unit_dir)
    server_cmd = build_server_command(config, case)
    benchmark_json = unit_dir / "benchmark.json"
    client_cmd = build_client_command(
        config,
        workload,
        output_json=benchmark_json,
        python_exe=args.python,
    )
    manifest = {
        "case": case,
        "workload": workload,
        "server_command": server_cmd,
        "client_command": client_cmd,
        "selected_env": _selected_env(env),
        "profile_jsonl": str(unit_dir / "moe_profile.jsonl"),
        "trace_jsonl": str(unit_dir / "moe_trace.jsonl"),
    }
    write_json(unit_dir / "unit_manifest.json", manifest)
    if args.dry_run:
        result = {
            "case": case,
            "workload": workload,
            "status": "dry_run",
            "stage": "dry_run",
            "manifest": str(unit_dir / "unit_manifest.json"),
            "benchmark_json": str(benchmark_json),
            "profile_jsonl": str(unit_dir / "moe_profile.jsonl"),
            "trace_jsonl": str(unit_dir / "moe_trace.jsonl"),
        }
        write_json(unit_dir / "unit_result.json", result)
        _write_unit_markdown(unit_dir, result)
        return result

    server_log_path = unit_dir / "server.log"
    client_log_path = unit_dir / "client.log"
    manager: Path | None = None
    managed_env: dict[str, str] | None = None
    release_status = "not-started"
    try:
        if not args.no_start_server:
            manager = _manager_path(args.server_manager)
            _assert_port_free(int(config["server"]["port"]))
            _assert_device_free(int(args.device))
            managed_env = _managed_env(config, case, unit_dir, args)
            status = _manager_call(manager, "status", managed_env, server_log_path, check=False)
            if status.returncode == 0:
                payload = json.loads(status.stdout.strip().splitlines()[-1])
                if payload.get("active_state") == "active" or int(payload.get("main_pid", 0)) > 0:
                    raise RuntimeError("custody unit is active; refusing server reuse")
            _manager_call(manager, "start", managed_env, server_log_path, check=True)
            release_status = "started"
            timeout_s = (
                float(args.startup_timeout_s)
                if args.startup_timeout_s is not None
                else float(config["server"].get("startup_timeout_s", 1200))
            )
            _wait_for_server(_health_url(config), None, timeout_s)

        with client_log_path.open("w", encoding="utf-8") as client_log:
            completed = subprocess.run(
                client_cmd,
                cwd=Path.cwd(),
                env=env,
                stdout=client_log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError(f"client exited with code {completed.returncode}")

        result = {
            "case": case,
            "workload": workload,
            "status": "ok",
            "stage": "completed",
            "manifest": str(unit_dir / "unit_manifest.json"),
            "server_log": str(server_log_path),
            "client_log": str(client_log_path),
            "benchmark_json": str(benchmark_json),
            "profile_jsonl": str(unit_dir / "moe_profile.jsonl"),
            "trace_jsonl": str(unit_dir / "moe_trace.jsonl"),
        }
        write_json(unit_dir / "unit_result.json", result)
        _write_unit_markdown(unit_dir, result)
        return result
    except BaseException as exc:
        result = {
            "case": case,
            "workload": workload,
            "status": "failed",
            "stage": "server_or_client",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "manifest": str(unit_dir / "unit_manifest.json"),
            "server_log": str(server_log_path),
            "client_log": str(client_log_path),
            "benchmark_json": str(benchmark_json),
            "profile_jsonl": str(unit_dir / "moe_profile.jsonl"),
            "trace_jsonl": str(unit_dir / "moe_trace.jsonl"),
        }
        write_json(unit_dir / "unit_result.json", result)
        _write_unit_markdown(unit_dir, result)
        return result
    finally:
        if manager is not None and managed_env is not None:
            _manager_call(manager, "stop", managed_env, server_log_path, check=False)
            status = _manager_call(manager, "status", managed_env, server_log_path, check=False)
            if status.returncode == 0:
                payload = json.loads(status.stdout.strip().splitlines()[-1])
                if payload.get("active_state") == "active" or int(payload.get("main_pid", 0)) > 0:
                    release_status = "release-failed"
                else:
                    release_status = "released"
            if args.release_ack_dir:
                ack_dir = Path(args.release_ack_dir)
                ack_dir.mkdir(parents=True, exist_ok=True)
                write_json(
                    ack_dir / f"{unit_dir.parent.name}-{unit_dir.name}.json",
                    {
                        "custody_unit": managed_env["VLLM_ENGINE_SYSTEMD_UNIT"],
                        "device": args.device,
                        "released_at_ns": time.time_ns(),
                        "status": release_status,
                    },
                )


def _write_suite_summary(suite_dir: Path, results: list[dict[str, Any]]) -> None:
    lines = ["# SEW-Offload Benchmark Suite", ""]
    for result in results:
        case = result["case"]
        workload = result["workload"]
        bench = {}
        bench_path = Path(result.get("benchmark_json", ""))
        if bench_path.exists():
            bench = json.loads(bench_path.read_text(encoding="utf-8"))
        lines.extend(
            [
                f"## {case['name']} / {workload['name']}",
                "",
                f"- Status: {result.get('status', '')}",
                f"- Median TTFT ms: {bench.get('median_ttft_ms', 0.0):.3f}",
                f"- Median TPOT ms: {bench.get('median_tpot_ms', 0.0):.3f}",
                f"- Output throughput tok/s: {bench.get('output_throughput', 0.0):.3f}",
                "",
            ]
        )
    (suite_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if int(args.client_concurrency) > 0:
        config["client"]["concurrency"] = int(args.client_concurrency)
    if int(args.max_num_seqs) > 0:
        config["serving_shape"]["max_num_seqs"] = int(args.max_num_seqs)
    issues = validate_config(config)
    if issues:
        for issue in issues:
            print(f"ERROR {issue}", file=sys.stderr)
        return 1

    if not args.dry_run and not args.no_start_server:
        if not args.release_ack_dir:
            print("ERROR --release-ack-dir is required for managed runs", file=sys.stderr)
            return 1
        try:
            _manager_path(args.server_manager)
            _managed_env(config, select_cases(config, args.case or None)[0], Path("probe/cell"), args)
        except (ValueError, OSError) as exc:
            print(f"ERROR {exc}", file=sys.stderr)
            return 1

    cases = select_cases(config, args.case or None)
    workloads = select_workloads(config, args.workload or None)
    if int(args.max_requests) > 0:
        workloads = [
            {**workload, "num_requests": int(args.max_requests)}
            for workload in workloads
        ]
    suite_dir = repo_relative(args.output_root) / f"{config['benchmark']['suite']}-{utc_stamp()}"
    suite_dir.mkdir(parents=True, exist_ok=True)

    write_json(
        suite_dir / "suite_manifest.json",
        {
            "config_path": str(Path(args.config)),
            "benchmark": config["benchmark"],
            "model": config["model"],
            "dataset": config["dataset"],
            "cases": [case["name"] for case in cases],
            "workloads": [workload["name"] for workload in workloads],
            "argv": sys.argv,
            "dry_run": bool(args.dry_run),
        },
    )
    results = [
        run_unit(config=config, case=case, workload=workload, suite_dir=suite_dir, args=args)
        for case in cases
        for workload in workloads
    ]
    write_json(suite_dir / "suite_results.json", {"results": results})
    _write_suite_summary(suite_dir, results)
    print(f"Wrote benchmark artifacts to {suite_dir}")
    return 0 if all(result.get("status") in {"ok", "dry_run"} for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
