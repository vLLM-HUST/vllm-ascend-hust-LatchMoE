#!/usr/bin/env python3
"""Utilities for the SEW-Offload Ascend benchmark config."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent
REPO_ROOT = BENCHMARK_DIR.parent
DEFAULT_CONFIG = BENCHMARK_DIR / "configs" / "sew_offload_v1.yaml"
DEFAULT_SCENARIOS = BENCHMARK_DIR / "scenarios" / "sew_offload_scenarios.json"

NATIVE_OFFLOAD_FLAGS = {
    "--offload-backend",
    "--cpu-offload-gb",
    "--offload-params",
    "--offload-group-size",
    "--offload-num-in-group",
    "--offload-prefetch-step",
}


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyYAML is required to read benchmark YAML configs. Run from the "
            "vLLM benchmark conda environment or install pyyaml."
        ) from exc
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return payload


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    return _load_yaml(Path(path))


def load_scenarios(path: str | Path = DEFAULT_SCENARIOS) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"scenario registry must be a JSON object: {path}")
    return payload


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def repo_relative(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def _names(items: list[dict[str, Any]], key: str = "name") -> list[str]:
    return [str(item.get(key, "")) for item in items]


def _duplicates(values: list[str]) -> list[str]:
    seen: set[str] = set()
    dupes: set[str] = set()
    for value in values:
        if value in seen:
            dupes.add(value)
        seen.add(value)
    return sorted(dupes)


def _is_unset(value: Any) -> bool:
    return value is None or value == ""


def validate_config(config: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    required = {
        "benchmark",
        "model",
        "dataset",
        "server",
        "client",
        "serving_shape",
        "workload_buckets",
        "cases",
        "experiments",
        "metrics",
        "artifact_contract",
    }
    missing = sorted(required - set(config))
    if missing:
        issues.append(f"missing top-level keys: {', '.join(missing)}")
        return issues

    dataset = config.get("dataset") or {}
    if dataset.get("source") != "sharegpt":
        issues.append("dataset.source must be 'sharegpt'")
    if dataset.get("random_dataset_allowed") is not False:
        issues.append("dataset.random_dataset_allowed must be false")
    if dataset.get("synthetic_smoke_allowed") is not False:
        issues.append("dataset.synthetic_smoke_allowed must be false")
    dataset_path = dataset.get("local_path")
    if dataset_path and not repo_relative(dataset_path).exists():
        issues.append(f"dataset.local_path does not exist: {dataset_path}")

    model = config.get("model") or {}
    for key in ("path", "tokenizer"):
        value = model.get(key)
        if value and not repo_relative(value).exists():
            issues.append(f"model.{key} does not exist: {value}")

    buckets = config.get("workload_buckets") or []
    bucket_names = _names(buckets)
    for duplicate in _duplicates(bucket_names):
        issues.append(f"duplicate workload bucket: {duplicate}")
    for bucket in buckets:
        name = str(bucket.get("name", ""))
        prompt_tokens = bucket.get("prompt_tokens")
        if prompt_tokens != "mixed":
            if (
                not isinstance(prompt_tokens, list)
                or len(prompt_tokens) != 2
                or int(prompt_tokens[0]) <= 0
                or int(prompt_tokens[1]) < int(prompt_tokens[0])
            ):
                issues.append(f"workload {name} has invalid prompt_tokens")
        if int(bucket.get("output_tokens", 0)) <= 0:
            issues.append(f"workload {name} output_tokens must be positive")
        if int(bucket.get("num_requests", 0)) <= 0:
            issues.append(f"workload {name} num_requests must be positive")

    cases = config.get("cases") or []
    case_names = _names(cases)
    for duplicate in _duplicates(case_names):
        issues.append(f"duplicate case: {duplicate}")

    experiments = config.get("experiments") or {}
    experiment_names = set(experiments)
    bucket_name_set = set(bucket_names)
    case_name_set = set(case_names)

    for case in cases:
        name = str(case.get("name", ""))
        env = case.get("env") or {}
        if not isinstance(env, dict):
            issues.append(f"case {name} env must be a mapping")
            continue
        server_args = [str(item) for item in case.get("server_args", [])]
        uses_native_flags = any(item in NATIVE_OFFLOAD_FLAGS for item in server_args)
        sew_enabled = str(env.get("VLLM_ASCEND_MOE_OFFLOAD_SEW_DATAPLANE", "")) == "1"
        if sew_enabled and uses_native_flags:
            issues.append(f"case {name} mixes SEW dataplane with native offload flags")
        if uses_native_flags and not _is_unset(env.get("VLLM_ASCEND_MOE_OFFLOAD_GB")):
            issues.append(f"case {name} mixes native offload flags with plugin offload GB")
        if case.get("role") in {"main", "ablation", "sensitivity"}:
            if _is_unset(env.get("VLLM_ASCEND_MOE_OFFLOAD_GB")):
                issues.append(f"case {name} must declare VLLM_ASCEND_MOE_OFFLOAD_GB")
            if not sew_enabled:
                issues.append(f"case {name} must enable SEW dataplane")
        for group in case.get("experiment_groups", []):
            if group not in experiment_names:
                issues.append(f"case {name} references unknown experiment group {group}")

    for exp_name, exp in experiments.items():
        for case_name in exp.get("cases", []):
            if case_name not in case_name_set:
                issues.append(f"experiment {exp_name} references unknown case {case_name}")
        for workload_name in exp.get("workloads", []):
            if workload_name not in bucket_name_set:
                issues.append(
                    f"experiment {exp_name} references unknown workload {workload_name}"
                )

    scenario_payload = load_scenarios()
    scenario_names = {str(item.get("name")) for item in scenario_payload.get("scenarios", [])}
    missing_scenarios = sorted(bucket_name_set - scenario_names)
    if missing_scenarios:
        issues.append(f"workload buckets missing from scenario registry: {', '.join(missing_scenarios)}")

    return issues


def select_cases(config: dict[str, Any], names: list[str] | None = None) -> list[dict[str, Any]]:
    cases = list(config.get("cases") or [])
    if names:
        wanted = set(names)
        selected = [case for case in cases if str(case.get("name")) in wanted]
        missing = sorted(wanted - {str(case.get("name")) for case in selected})
        if missing:
            raise KeyError(f"unknown cases: {', '.join(missing)}")
        return selected
    return [case for case in cases if bool(case.get("default_enabled", False))]


def select_workloads(
    config: dict[str, Any],
    names: list[str] | None = None,
) -> list[dict[str, Any]]:
    workloads = list(config.get("workload_buckets") or [])
    if names:
        wanted = set(names)
        selected = [item for item in workloads if str(item.get("name")) in wanted]
        missing = sorted(wanted - {str(item.get("name")) for item in selected})
        if missing:
            raise KeyError(f"unknown workloads: {', '.join(missing)}")
        return selected
    return [item for item in workloads if bool(item.get("default_enabled", False))]


def _format_arg(value: Any, config: dict[str, Any]) -> str:
    model = config["model"]
    server = config["server"]
    shape = config["serving_shape"]
    mapping = {
        "model_path": model["path"],
        "served_model_name": model["served_model_name"],
        "dtype": model["dtype"],
        "tensor_parallel_size": model["tensor_parallel_size"],
        "host": server["host"],
        "port": server["port"],
        "max_num_seqs": shape["max_num_seqs"],
        "max_model_len": shape["max_model_len"],
        "max_num_batched_tokens": shape["max_num_batched_tokens"],
        "gpu_memory_utilization": shape["gpu_memory_utilization"],
    }
    return str(value).format(**mapping)


def build_server_command(config: dict[str, Any], case: dict[str, Any]) -> list[str]:
    server = config["server"]
    command = [str(server.get("command", "vllm"))]
    command.extend(_format_arg(arg, config) for arg in server.get("common_args", []))
    command.extend(str(arg) for arg in case.get("server_args", []))
    return command


def build_client_command(
    config: dict[str, Any],
    workload: dict[str, Any],
    *,
    output_json: str | Path,
    python_exe: str,
    manifest_path: str | Path | None = None,
) -> list[str]:
    server = config["server"]
    client = config["client"]
    model = config["model"]
    manifest = manifest_path or config["dataset"]["manifest_path"]
    command = [
        python_exe,
        str(BENCHMARK_DIR / "scripts" / "run_openai_manifest.py"),
        "--base-url",
        f"http://{server['host']}:{int(server['port'])}",
        "--model",
        str(model["served_model_name"]),
        "--manifest",
        str(repo_relative(manifest)),
        "--bucket",
        str(workload["name"]),
        "--concurrency",
        str(client.get("concurrency", 1)),
        "--request-timeout-s",
        str(client.get("request_timeout_s", 900)),
        "--output-json",
        str(output_json),
    ]
    if int(workload.get("num_requests", 0)) > 0:
        command.extend(["--max-requests", str(int(workload["num_requests"]))])
    if client.get("tokenizer_count_output_tokens", True):
        command.extend(["--tokenizer", str(model["tokenizer"])])
    return command


def render_plan(
    config: dict[str, Any],
    *,
    case_names: list[str] | None = None,
    workload_names: list[str] | None = None,
    python_exe: str = sys.executable,
) -> dict[str, Any]:
    cases = select_cases(config, case_names)
    workloads = select_workloads(config, workload_names)
    units = []
    for case in cases:
        for workload in workloads:
            fake_output = (
                BENCHMARK_DIR
                / "artifacts"
                / "runs"
                / "<run-id>"
                / str(case["name"])
                / str(workload["name"])
                / "benchmark.json"
            )
            units.append(
                {
                    "case": case["name"],
                    "workload": workload["name"],
                    "server_command": build_server_command(config, case),
                    "client_command": build_client_command(
                        config,
                        workload,
                        output_json=fake_output,
                        python_exe=python_exe,
                    ),
                    "env": case.get("env", {}),
                }
            )
    return {
        "benchmark": config["benchmark"],
        "model": config["model"],
        "dataset": config["dataset"],
        "units": units,
    }


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def prepare_workloads(
    config: dict[str, Any],
    *,
    manifest_path: str | Path | None = None,
    requests_per_bucket: int | None = None,
    buckets: list[str] | None = None,
) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from sharegpt_manifest import build_sharegpt_manifest

    selected = set(buckets or []) or None
    target = repo_relative(manifest_path or config["dataset"]["manifest_path"])
    return build_sharegpt_manifest(
        config=config,
        manifest_path=target,
        model_path=config["model"]["tokenizer"],
        requests_per_bucket=requests_per_bucket,
        buckets=selected,
    )


def summarize_run(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    if not root.exists():
        raise FileNotFoundError(root)
    results = []
    for path in sorted(root.glob("*/*/unit_result.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        bench_path = Path(payload.get("benchmark_json", ""))
        benchmark: dict[str, Any] = {}
        if bench_path.exists():
            benchmark = json.loads(bench_path.read_text(encoding="utf-8"))
        profile_path = Path(payload.get("profile_jsonl", ""))
        profile_records = 0
        if profile_path.exists():
            with profile_path.open(encoding="utf-8") as handle:
                profile_records = sum(1 for line in handle if line.strip())
        results.append(
            {
                "case": payload.get("case", {}).get("name"),
                "workload": payload.get("workload", {}).get("name"),
                "status": payload.get("status"),
                "stage": payload.get("stage", ""),
                "successful_requests": benchmark.get("successful_requests", 0),
                "failed_requests": benchmark.get("failed_requests", 0),
                "median_ttft_ms": benchmark.get("median_ttft_ms", 0.0),
                "median_tpot_ms": benchmark.get("median_tpot_ms", 0.0),
                "output_throughput": benchmark.get("output_throughput", 0.0),
                "profile_records": profile_records,
                "result": str(path),
            }
        )
    return {
        "run_dir": str(root),
        "unit_count": len(results),
        "ok_count": sum(1 for item in results if item["status"] == "ok"),
        "failed_count": sum(1 for item in results if item["status"] == "failed"),
        "results": results,
    }


def _print_cases(config: dict[str, Any]) -> None:
    for case in config.get("cases", []):
        default = "default" if case.get("default_enabled") else "optional"
        print(f"{case['name']}\t{case.get('role', '')}\t{default}\t{case.get('title', '')}")


def _print_workloads(config: dict[str, Any]) -> None:
    for workload in config.get("workload_buckets", []):
        default = "default" if workload.get("default_enabled") else "optional"
        print(
            f"{workload['name']}\t{default}\t"
            f"n={workload.get('num_requests')}\tout={workload.get('output_tokens')}\t"
            f"{workload.get('title', '')}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("validate", help="Validate benchmark config invariants.")
    subparsers.add_parser("list-cases", help="List configured benchmark cases.")
    subparsers.add_parser("list-workloads", help="List configured workload buckets.")

    plan = subparsers.add_parser("render-plan", help="Render case x workload commands.")
    plan.add_argument("--case", action="append", default=[])
    plan.add_argument("--workload", action="append", default=[])
    plan.add_argument("--python", default=sys.executable)
    plan.add_argument("--output")

    prepare = subparsers.add_parser("prepare-workloads", help="Sample ShareGPT JSONL manifest.")
    prepare.add_argument("--manifest")
    prepare.add_argument("--requests-per-bucket", type=int)
    prepare.add_argument("--bucket", action="append", default=[])

    summary = subparsers.add_parser("summarize", help="Summarize a run directory.")
    summary.add_argument("run_dir")
    summary.add_argument("--output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    if args.command == "validate":
        issues = validate_config(config)
        if issues:
            for issue in issues:
                print(f"ERROR {issue}", file=sys.stderr)
            return 1
        print(f"OK {args.config}")
        return 0

    if args.command == "list-cases":
        _print_cases(config)
        return 0

    if args.command == "list-workloads":
        _print_workloads(config)
        return 0

    if args.command == "render-plan":
        issues = validate_config(config)
        if issues:
            raise SystemExit("\n".join(issues))
        payload = render_plan(
            config,
            case_names=args.case or None,
            workload_names=args.workload or None,
            python_exe=args.python,
        )
        if args.output:
            write_json(args.output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if args.command == "prepare-workloads":
        written = prepare_workloads(
            config,
            manifest_path=args.manifest,
            requests_per_bucket=args.requests_per_bucket,
            buckets=args.bucket or None,
        )
        print(f"MANIFEST_OK written={written}")
        return 0

    if args.command == "summarize":
        payload = summarize_run(args.run_dir)
        if args.output:
            write_json(args.output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
