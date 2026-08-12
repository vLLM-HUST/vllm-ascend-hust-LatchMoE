#!/usr/bin/env python3
"""Custody manager for an immutable host LatchMoE runtime bundle.

This is the systemd-less counterpart to the pinned dev-hub manager. It owns a
single process group, records its identity in an atomic state file, and never
discovers or kills unrelated vLLM processes.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def _state_path() -> Path:
    value = os.environ.get("LATCHMOE_CUSTODY_STATE")
    if not value:
        raise ValueError("LATCHMOE_CUSTODY_STATE is required")
    return Path(value)


def _read_state(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
        return len(stat) > 2 and stat[2] != "Z"
    except (OSError, FileNotFoundError, PermissionError):
        return False


def _status(path: Path) -> dict:
    state = _read_state(path)
    pid = int(state.get("pid") or 0)
    alive = _pid_alive(pid)
    return {
        **state,
        "active_state": "active" if alive else "inactive",
        "main_pid": pid if alive else 0,
    }


def _required_env(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise ValueError(f"{name} is required")
    return value


def _start(path: Path) -> dict:
    current = _status(path)
    if current["active_state"] == "active":
        raise RuntimeError(f"custody already active with pid {current['main_pid']}")
    # Preserve a venv launcher path instead of resolving its symlink to the base
    # interpreter; Python uses argv[0] to recover the virtual-environment prefix.
    python = Path(_required_env("LATCHMOE_HOST_PYTHON")).absolute()
    runtime_root = Path(_required_env("LATCHMOE_RUNTIME_ROOT")).resolve()
    vllm_root = Path(_required_env("LATCHMOE_VLLM_ROOT")).resolve()
    seam_root = Path(_required_env("LATCHMOE_SEAM_ROOT")).resolve()
    log_path = Path(_required_env("VLLM_ENGINE_CONTAINER_LOG_FILE")).resolve()
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError(f"host runtime Python is not executable: {python}")
    for name, root in (("runtime", runtime_root), ("vLLM", vllm_root), ("seam", seam_root)):
        if not (root / ".git").exists():
            raise ValueError(f"{name} root is not a Git checkout: {root}")

    model = _required_env("VLLM_ENGINE_MODEL_PATH")
    command = [
        str(python),
        "-m",
        "vllm_moe_offload_ascend",
        "serve",
        model,
        "--served-model-name",
        _required_env("VLLM_ENGINE_SERVED_MODEL_NAME"),
        "--trust-remote-code",
        "--dtype",
        _required_env("VLLM_ENGINE_DTYPE"),
        "--tensor-parallel-size",
        _required_env("VLLM_ENGINE_TP_SIZE"),
        "--host",
        "127.0.0.1",
        "--port",
        _required_env("VLLM_ENGINE_PORT"),
        "--max-num-seqs",
        _required_env("VLLM_ENGINE_MAX_NUM_SEQS"),
        "--max-model-len",
        _required_env("VLLM_ENGINE_MAX_MODEL_LEN"),
        "--max-num-batched-tokens",
        _required_env("VLLM_ENGINE_MAX_NUM_BATCHED_TOKENS"),
        "--gpu-memory-utilization",
        _required_env("VLLM_ENGINE_GPU_MEM_UTIL"),
        "--no-enable-prefix-caching",
    ]
    extra_args = json.loads(os.environ.get("VLLM_ENGINE_EXTRA_ARGS_JSON", "[]"))
    if not isinstance(extra_args, list) or "--enforce-eager" in extra_args:
        raise ValueError("invalid or forced-eager VLLM_ENGINE_EXTRA_ARGS_JSON")
    command.extend(str(item) for item in extra_args)
    child_env = dict(os.environ)
    child_env["ASCEND_RT_VISIBLE_DEVICES"] = _required_env("VLLM_ENGINE_NPU_DEVICES")
    child_env["PYTHONPATH"] = ":".join(
        (str(runtime_root), str(vllm_root), str(seam_root))
    )
    child_env["VLLM_PLUGINS"] = "ascend"
    child_env.pop("VLLM_ENGINE_ENFORCE_EAGER", None)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab", buffering=0) as log_handle:
        process = subprocess.Popen(
            command,
            cwd=runtime_root,
            env=child_env,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    state = {
        "pid": int(process.pid),
        "pgid": int(os.getpgid(process.pid)),
        "started_at_ns": time.time_ns(),
        "command": command,
        "log_path": str(log_path),
        "runtime_root": str(runtime_root),
        "vllm_root": str(vllm_root),
        "seam_root": str(seam_root),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)
    time.sleep(0.25)
    return_code = process.poll()
    if return_code is not None:
        raise RuntimeError(
            f"managed runtime exited during startup with code {return_code}"
        )
    return _status(path)


def _stop(path: Path, timeout_s: float = 60.0) -> dict:
    state = _read_state(path)
    pid = int(state.get("pid") or 0)
    pgid = int(state.get("pgid") or 0)
    if _pid_alive(pid) and pgid > 0:
        os.killpg(pgid, signal.SIGTERM)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and _pid_alive(pid):
            time.sleep(0.25)
        if _pid_alive(pid):
            os.killpg(pgid, signal.SIGKILL)
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline and _pid_alive(pid):
                time.sleep(0.1)
    final = _status(path)
    final["stopped_at_ns"] = time.time_ns()
    path.write_text(json.dumps(final, indent=2, sort_keys=True), encoding="utf-8")
    return final


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "stop", "status"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        path = _state_path()
        if args.action == "start":
            result = _start(path)
        elif args.action == "stop":
            result = _stop(path)
        else:
            result = _status(path)
    except Exception as exc:
        print(json.dumps({"error": str(exc), "active_state": "unknown", "main_pid": 0}))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
