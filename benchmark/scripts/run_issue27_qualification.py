#!/usr/bin/env python3
"""Run the native/eager/PIECEWISE Issue #27 qualification bundle."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE = REPO_ROOT / "benchmark" / "scripts" / "run_fixed_slot_smoke.py"

_PREFLIGHT_MODULES = ("acl", "torch", "torch_npu", "vllm", "vllm_ascend")
_PREFLIGHT_CODE = r'''
import importlib
import json
import os
import sys

result = {
    "python": sys.executable,
    "python_version": sys.version,
    "prefix": sys.prefix,
    "ascend_home_path": os.environ.get("ASCEND_HOME_PATH"),
    "pythonpath": os.environ.get("PYTHONPATH"),
    "modules": {},
}
for module_name in %r:
    try:
        module = importlib.import_module(module_name)
        result["modules"][module_name] = {
            "status": "ok",
            "path": getattr(module, "__file__", None),
            "version": getattr(module, "__version__", None),
        }
    except Exception as exc:
        result["modules"][module_name] = {
            "status": "failed",
            "type": type(exc).__name__,
            "error": str(exc),
        }
print(json.dumps(result, sort_keys=True))
'''


def _environment_preflight(python: str, env: dict[str, str]) -> dict[str, Any]:
    """Check the imports required by both the parent and EngineCore process."""
    code = _PREFLIGHT_CODE % (_PREFLIGHT_MODULES,)
    completed = subprocess.run(
        [python, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    stdout_lines = [line for line in completed.stdout.splitlines() if line.strip()]
    try:
        result = json.loads(stdout_lines[-1])
    except (IndexError, json.JSONDecodeError):
        result = {
            "modules": {},
            "stdout": completed.stdout[-4000:],
            "stderr": completed.stderr[-4000:],
        }
    modules = result.get("modules") or {}
    failures = {
        module: detail
        for module, detail in modules.items()
        if detail.get("status") != "ok"
    }
    result["returncode"] = int(completed.returncode)
    result["status"] = "passed" if completed.returncode == 0 and not failures else "failed"
    result["failed_modules"] = failures
    if completed.stderr.strip():
        result["stderr"] = completed.stderr[-4000:]
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", type=int, choices=(5, 6), required=True)
    # Profile/dummy forwards choose B2 automatically when their observed union
    # exceeds the fixed slot capacity. Keep this low enough to exercise actual
    # offload rather than masking the capacity mechanism with near-residency.
    parser.add_argument("--num-slots", type=int, default=32)
    parser.add_argument("--overflow-slots", type=int, default=4)
    parser.add_argument("--output-tokens", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=256)
    parser.add_argument("--max-num-batched-tokens", type=int, default=256)
    parser.add_argument("--kv-cache-memory-mb", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.96)
    parser.add_argument(
        "--ascend-additional-config-json",
        default="{}",
        help=(
            "JSON object passed to every smoke unit as vLLM-Ascend "
            "additional_config, for example '{\"mix_placement\": true}'."
        ),
    )
    parser.add_argument(
        "--prompt",
        default="Hi",
    )
    parser.add_argument(
        "--overflow-prompt",
        default=(
            "Read this repeated qualification context and answer with one concise sentence. "
            # Keep the default pressure prompt below the 256-token smoke
            # contract while still producing a multi-wave routed union.
            + "The routed expert workload must remain deterministic. " * 24
        ),
    )
    args = parser.parse_args()
    try:
        args.ascend_additional_config = json.loads(
            args.ascend_additional_config_json
        )
    except json.JSONDecodeError as exc:
        parser.error(
            "--ascend-additional-config-json is not valid JSON: "
            f"{exc.msg}"
        )
    if not isinstance(args.ascend_additional_config, dict):
        parser.error("--ascend-additional-config-json must decode to a JSON object")
    return args


def _base_command(args: argparse.Namespace, unit_dir: Path, *, mode: str, prompt: str, output_tokens: int, slots: int) -> list[str]:
    command = [
        str(args.python),
        str(SMOKE),
        "--mode",
        mode,
        "--model",
        str(args.model_path.resolve()),
        "--output-dir",
        str(unit_dir),
        "--inline-prompt",
        prompt,
        "--inline-max-output-tokens",
        str(int(output_tokens)),
        "--max-model-len",
        str(int(args.max_model_len)),
        "--max-num-batched-tokens",
        str(int(args.max_num_batched_tokens)),
        "--kv-cache-memory-mb",
        str(int(args.kv_cache_memory_mb)),
        "--gpu-memory-utilization",
        str(float(args.gpu_memory_utilization)),
        "--num-slots",
        str(int(slots)),
        "--ignore-eos",
        "--disable-ascend-norm-quant-fusion",
        "--ascend-additional-config-json",
        json.dumps(args.ascend_additional_config, sort_keys=True),
    ]
    return command


def _run_unit(
    args: argparse.Namespace,
    *,
    name: str,
    mode: str,
    prompt: str,
    output_tokens: int,
    slots: int,
    diagnostic_eager: bool = False,
    stage_seam: bool = False,
    cpu_first_load: bool = False,
    release_original_expert_weights: bool = False,
    router_parity: bool = False,
    layer_boundary_parity: bool = False,
    wave_prefill: bool = False,
) -> dict[str, Any]:
    unit_dir = args.output_root / name
    unit_dir.mkdir(parents=True, exist_ok=False)
    command = _base_command(
        args,
        unit_dir,
        mode=mode,
        prompt=prompt,
        output_tokens=output_tokens,
        slots=slots,
    )
    if diagnostic_eager:
        command.extend(["--enforce-eager", "--diagnostic-eager"])
    if stage_seam:
        command.append("--stage-seam")
        if cpu_first_load:
            command.append("--cpu-first-load")
        if release_original_expert_weights:
            command.append("--release-original-expert-weights")
        command.append("--layered-runtime")
    if router_parity:
        command.extend(["--router-parity", "--router-parity-max-tokens", "64"])
    if layer_boundary_parity:
        command.append("--layer-boundary-parity")
    if wave_prefill:
        command.append("--wave-prefill")

    env = dict(os.environ)
    env["ASCEND_RT_VISIBLE_DEVICES"] = str(args.device)
    env["ASCEND_VISIBLE_DEVICES"] = str(args.device)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    log_path = unit_dir / "run.log"
    (unit_dir / "unit_manifest.json").write_text(
        json.dumps(
            {
                "model_id": args.model_id,
                "model_path": str(args.model_path.resolve()),
                "unit": name,
                "command": command,
                "device": int(args.device),
                "qualification": {
                    "diagnostic_eager": diagnostic_eager,
                    "stage_seam": stage_seam,
                    "cpu_first_load": cpu_first_load,
                    "release_original_expert_weights": release_original_expert_weights,
                    "router_parity": router_parity,
                    "layer_boundary_parity": layer_boundary_parity,
                    "wave_prefill": wave_prefill,
                    "ascend_additional_config": args.ascend_additional_config,
                },
                "environment": {
                    key: env.get(key)
                    for key in (
                        "ASCEND_RT_VISIBLE_DEVICES",
                        "ASCEND_VISIBLE_DEVICES",
                        "ASCEND_HOME_PATH",
                        "ASCEND_TOOLKIT_HOME",
                        "PYTHONPATH",
                        "LD_LIBRARY_PATH",
                        "PYTORCH_NPU_ALLOC_CONF",
                        "VLLM_ASCEND_MOE_OFFLOAD_STAGE_SEAM",
                        "VLLM_ASCEND_MOE_OFFLOAD_CPU_FIRST_LOAD",
                    )
                    if env.get(key) is not None
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    return {
        "unit": name,
        "unit_dir": str(unit_dir),
        "returncode": int(completed.returncode),
        "status": "ok" if completed.returncode == 0 else "failed",
        "command": command,
    }


def main() -> int:
    args = _parse_args()
    args.model_path = args.model_path.resolve()
    args.output_root = args.output_root.resolve()
    if not args.model_path.is_dir():
        raise FileNotFoundError(args.model_path)
    if args.output_root.exists():
        raise FileExistsError(
            f"refusing to overwrite an existing qualification bundle: {args.output_root}"
        )
    args.output_root.mkdir(parents=True)

    env = dict(os.environ)
    env["ASCEND_RT_VISIBLE_DEVICES"] = str(args.device)
    env["ASCEND_VISIBLE_DEVICES"] = str(args.device)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    preflight = _environment_preflight(str(args.python), env)
    preflight["created_at"] = datetime.now(timezone.utc).isoformat()
    (args.output_root / "environment_preflight.json").write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if preflight["status"] != "passed":
        manifest = {
            "schema_version": "latchmoe-issue27-qualification-bundle-v1",
            "model_id": args.model_id,
            "model_path": str(args.model_path),
            "device": int(args.device),
            "status": "blocked",
            "blocked_reason": "qualification environment preflight failed",
            "environment_preflight": "environment_preflight.json",
            "units": [],
        }
        (args.output_root / "bundle_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 2

    units = [
        _run_unit(
            args,
            name="native-eager",
            mode="no_offload",
            prompt=args.prompt,
            output_tokens=args.output_tokens,
            slots=0,
            diagnostic_eager=True,
        ),
        _run_unit(
            args,
            name="latch-eager",
            mode="fixed_slot_sync",
            prompt=args.prompt,
            output_tokens=args.output_tokens,
            slots=args.num_slots,
            diagnostic_eager=True,
            stage_seam=True,
            cpu_first_load=True,
            release_original_expert_weights=True,
            router_parity=True,
            layer_boundary_parity=True,
        ),
        _run_unit(
            args,
            name="latch-graph",
            mode="fixed_slot_sync",
            prompt=args.prompt,
            output_tokens=args.output_tokens,
            slots=args.num_slots,
            stage_seam=True,
            cpu_first_load=True,
            release_original_expert_weights=True,
        ),
        _run_unit(
            args,
            name="overflow-graph",
            mode="fixed_slot_sync",
            prompt=args.overflow_prompt,
            output_tokens=1,
            slots=args.overflow_slots,
            stage_seam=True,
            cpu_first_load=True,
            release_original_expert_weights=True,
            wave_prefill=True,
        ),
    ]
    manifest = {
        "schema_version": "latchmoe-issue27-qualification-bundle-v1",
        "model_id": args.model_id,
        "model_path": str(args.model_path),
        "device": int(args.device),
        "units": units,
    }
    (args.output_root / "bundle_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if all(unit["status"] == "ok" for unit in units) else 1


if __name__ == "__main__":
    raise SystemExit(main())
