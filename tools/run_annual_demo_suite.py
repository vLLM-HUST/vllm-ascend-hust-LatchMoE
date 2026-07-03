#!/usr/bin/env python3
"""Run the annual-demo serving suite and preserve reproducibility artifacts."""

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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = "demo/annual_demo_config.json"
DEFAULT_OUTPUT_ROOT = "demo_runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-start-server", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--startup-timeout-s", type=float, default=None)
    return parser.parse_args()


def _load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        config = json.load(f)
    for key in ("suite", "model", "served_model_name", "host", "port", "cases"):
        if key not in config:
            raise ValueError(f"demo config missing required key: {key}")
    return config


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _format_arg(value: str, config: dict[str, Any]) -> str:
    return str(value).format(
        model=config["model"],
        served_model_name=config["served_model_name"],
        host=config["host"],
        port=int(config["port"]),
    )


def _server_command(config: dict[str, Any], case: dict[str, Any]) -> list[str]:
    server = config.get("server", {})
    command = [str(server.get("command", "vllm"))]
    command.extend(
        _format_arg(arg, config)
        for arg in server.get("common_args", [])
    )
    command.extend(str(arg) for arg in case.get("server_args", []))
    return command


def _case_env(config: dict[str, Any], case: dict[str, Any], case_dir: Path) -> dict[str, str]:
    env = dict(os.environ)
    profile_path = case_dir / "moe_profile.jsonl"
    env.setdefault("VLLM_ASCEND_MOE_GMM_PROFILE_PATH", str(profile_path))
    env.setdefault("VLLM_ASCEND_MOE_OFFLOAD_PROFILE_PATH", str(profile_path))
    for key, value in case.get("env", {}).items():
        if value is None or value == "":
            env.pop(str(key), None)
        else:
            env[str(key)] = str(value)
    return env


def _bench_command(
    *,
    config: dict[str, Any],
    case: dict[str, Any],
    output_json: Path,
    python_exe: str,
) -> list[str]:
    bench = config.get("bench", {})
    base_url = f"http://{config['host']}:{int(config['port'])}"
    command = [
        python_exe,
        "bench_sharegpt.py",
        "--base-url",
        base_url,
        "--model",
        str(config["served_model_name"]),
        "--dataset",
        str(config["dataset"]),
        "--n-prompts",
        str(bench.get("n_prompts", 50)),
        "--max-tokens",
        str(bench.get("max_tokens", 100)),
        "--max-prompt-chars",
        str(bench.get("max_prompt_chars", 800)),
        "--concurrency",
        str(bench.get("concurrency", 1)),
        "--seed",
        str(bench.get("seed", 42)),
        "--request-timeout-s",
        str(bench.get("request_timeout_s", 600)),
        "--label",
        str(case["name"]),
        "--output-json",
        str(output_json),
    ]
    tokenizer = str(config.get("tokenizer", "") or "")
    if tokenizer:
        command.extend(["--tokenizer", tokenizer])
    return command


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _git_metadata() -> dict[str, str]:
    def run_git(args: list[str]) -> str:
        try:
            return subprocess.check_output(["git", *args], text=True).strip()
        except Exception:
            return ""

    return {
        "commit": run_git(["rev-parse", "HEAD"]),
        "branch": run_git(["branch", "--show-current"]),
        "status_short": run_git(["status", "--short"]),
    }


def _health_url(config: dict[str, Any]) -> str:
    return f"http://{config['host']}:{int(config['port'])}/v1/models"


def _wait_for_server(url: str, proc: subprocess.Popen[Any] | None, timeout_s: float) -> None:
    deadline = time.monotonic() + float(timeout_s)
    last_error = ""
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"server exited early with code {proc.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
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


def _write_case_markdown(case_dir: Path, payload: dict[str, Any]) -> None:
    bench = payload.get("benchmark", {})
    lines = [
        f"# {payload['case']['name']}",
        "",
        f"- Role: {payload['case'].get('role', '')}",
        f"- Status: {payload.get('status', 'unknown')}",
        f"- Started: {payload.get('started_at', '')}",
        f"- Server log: `{payload.get('server_log', '')}`",
        f"- Benchmark JSON: `{payload.get('benchmark_json', '')}`",
        "",
        "## Metrics",
        "",
        f"- Successful requests: {bench.get('successful_requests', 0)}",
        f"- Failed requests: {bench.get('failed_requests', 0)}",
        f"- Median TTFT ms: {bench.get('median_ttft_ms', 0.0):.3f}",
        f"- Median TPOT ms: {bench.get('median_tpot_ms', 0.0):.3f}",
        f"- Output throughput tok/s: {bench.get('output_throughput_tok_s', 0.0):.3f}",
        "",
        "## Notes",
        "",
        payload["case"].get("description", ""),
        "",
    ]
    (case_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def run_case(
    *,
    config: dict[str, Any],
    case: dict[str, Any],
    suite_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    case_dir = suite_dir / str(case["name"])
    case_dir.mkdir(parents=True, exist_ok=True)
    env = _case_env(config, case, case_dir)
    server_cmd = _server_command(config, case)
    bench_json = case_dir / "benchmark.json"
    bench_cmd = _bench_command(
        config=config,
        case=case,
        output_json=bench_json,
        python_exe=args.python,
    )
    manifest = {
        "case": case,
        "server_command": server_cmd,
        "benchmark_command": bench_cmd,
        "selected_env": {
            key: env.get(key, "")
            for key in sorted(env)
            if key.startswith("VLLM_ASCEND_MOE")
            or key in ("ASCEND_RT_VISIBLE_DEVICES", "PYTHONPATH")
        },
    }
    _write_json(case_dir / "case_manifest.json", manifest)
    if args.dry_run:
        return {"case": case, "status": "dry_run", "manifest": manifest}

    server_log_path = case_dir / "server.log"
    server_proc = None
    started_at = _timestamp()
    try:
        if not args.no_start_server:
            server_log = server_log_path.open("w", encoding="utf-8")
            try:
                server_proc = subprocess.Popen(
                    server_cmd,
                    cwd=Path.cwd(),
                    env=env,
                    stdout=server_log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
            finally:
                server_log.close()
            timeout_s = (
                float(args.startup_timeout_s)
                if args.startup_timeout_s is not None
                else float(config.get("server", {}).get("startup_timeout_s", 900))
            )
            _wait_for_server(_health_url(config), server_proc, timeout_s)
        subprocess.run(
            bench_cmd,
            cwd=Path.cwd(),
            env=env,
            check=True,
        )
        benchmark = json.loads(bench_json.read_text(encoding="utf-8"))
        payload = {
            "case": case,
            "status": "ok",
            "started_at": started_at,
            "server_log": str(server_log_path),
            "benchmark_json": str(bench_json),
            "benchmark": benchmark,
        }
        _write_json(case_dir / "case_result.json", payload)
        _write_case_markdown(case_dir, payload)
        return payload
    finally:
        if server_proc is not None:
            _terminate_process_group(
                server_proc,
                float(config.get("server", {}).get("shutdown_timeout_s", 30)),
            )


def _write_suite_summary(suite_dir: Path, results: list[dict[str, Any]]) -> None:
    lines = ["# Annual Demo Suite", ""]
    for result in results:
        case = result["case"]
        bench = result.get("benchmark", {})
        lines.extend(
            [
                f"## {case['name']}",
                "",
                f"- Role: {case.get('role', '')}",
                f"- Status: {result.get('status', '')}",
                f"- Median TTFT ms: {bench.get('median_ttft_ms', 0.0):.3f}",
                f"- Median TPOT ms: {bench.get('median_tpot_ms', 0.0):.3f}",
                f"- Output throughput tok/s: {bench.get('output_throughput_tok_s', 0.0):.3f}",
                "",
            ]
        )
    (suite_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = _load_config(args.config)
    selected = set(args.case)
    cases = [
        case for case in config["cases"]
        if not selected or str(case["name"]) in selected
    ]
    if not cases:
        raise ValueError(f"no matching cases selected: {sorted(selected)}")

    suite_dir = Path(args.output_root) / f"{config['suite']}-{_timestamp()}"
    suite_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        suite_dir / "suite_manifest.json",
        {
            "config": config,
            "git": _git_metadata(),
            "argv": sys.argv,
            "created_at": _timestamp(),
        },
    )

    results = [
        run_case(config=config, case=case, suite_dir=suite_dir, args=args)
        for case in cases
    ]
    _write_json(suite_dir / "suite_results.json", {"results": results})
    _write_suite_summary(suite_dir, results)
    print(f"Wrote demo artifacts to {suite_dir}")


if __name__ == "__main__":
    main()
