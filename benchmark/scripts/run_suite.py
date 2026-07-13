#!/usr/bin/env python3
"""Run the standard SEW-Offload benchmark suite."""

from __future__ import annotations

import argparse
import json
import os
import signal
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--workload", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-start-server", action="store_true")
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


def _wait_for_server(url: str, proc: subprocess.Popen[Any] | None, timeout_s: float) -> None:
    deadline = time.monotonic() + float(timeout_s)
    last_error = ""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(url, headers={"Connection": "close"})
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"server exited early with code {proc.returncode}")
        try:
            with opener.open(request, timeout=5) as resp:
                if 200 <= int(resp.status) < 500:
                    return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = str(exc)
        time.sleep(5)
    raise TimeoutError(f"server did not become ready at {url}: {last_error}")


def _terminate_process_group(proc: subprocess.Popen[Any], timeout_s: float) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(1)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


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
    server_proc: subprocess.Popen[Any] | None = None
    try:
        if not args.no_start_server:
            with server_log_path.open("w", encoding="utf-8") as server_log:
                server_proc = subprocess.Popen(
                    server_cmd,
                    cwd=Path.cwd(),
                    env=env,
                    stdout=server_log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
            timeout_s = (
                float(args.startup_timeout_s)
                if args.startup_timeout_s is not None
                else float(config["server"].get("startup_timeout_s", 1200))
            )
            _wait_for_server(_health_url(config), server_proc, timeout_s)

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
        if server_proc is not None:
            _terminate_process_group(
                server_proc,
                float(config["server"].get("shutdown_timeout_s", 30)),
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
