from __future__ import annotations

from argparse import Namespace
import json
import os
from pathlib import Path
import subprocess
import sys
import importlib.util

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_SUITE_PATH = REPO_ROOT / "benchmark" / "scripts" / "run_suite.py"


def _load_run_suite():
    spec = importlib.util.spec_from_file_location("managed_run_suite_test", RUN_SUITE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _args(**updates: object) -> Namespace:
    values = {
        "device": 5,
        "runtime_image": "registry.example/vllm-ascend:known",
        "runtime_image_digest": "sha256:" + "a" * 64,
        "container_name": "latchmoe-test",
        "custody_unit_prefix": "latchmoe",
        "managed_backend": "container",
        "host_python": None,
        "vllm_root": None,
        "seam_root": None,
    }
    values.update(updates)
    return Namespace(**values)


def _config() -> dict:
    return {
        "server": {"port": 8026},
        "model": {
            "path": "/data/model",
            "served_model_name": "model",
            "tensor_parallel_size": 1,
            "dtype": "bfloat16",
        },
        "serving_shape": {
            "max_model_len": 4096,
            "max_num_batched_tokens": 4096,
            "max_num_seqs": 1,
            "gpu_memory_utilization": 0.9,
        },
    }


def test_managed_env_pins_graph_device_and_digest(tmp_path: Path) -> None:
    env = _load_run_suite()._managed_env(
        _config(), {"server_args": [], "env": {}}, tmp_path / "case/work", _args()
    )
    assert env["VLLM_ENGINE_NPU_DEVICES"] == "5"
    assert env["VLLM_ENGINE_ENFORCE_EAGER"] == "0"
    assert env["VLLM_ENGINE_ENABLE_PREFIX_CACHING"] == "0"
    assert "@sha256:" in env["VLLM_ENGINE_IMAGE"]


@pytest.mark.parametrize(
    "updates",
    [
        {"device": 7},
        {"runtime_image_digest": "latest"},
        {"container_name": ""},
    ],
)
def test_managed_env_rejects_incomplete_custody(tmp_path: Path, updates: dict) -> None:
    with pytest.raises(ValueError):
        _load_run_suite()._managed_env(
            _config(),
            {"server_args": [], "env": {}},
            tmp_path / "case/work",
            _args(**updates),
        )


def test_managed_env_rejects_forced_eager(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="forced eager"):
        _load_run_suite()._managed_env(
            _config(),
            {"server_args": ["--enforce-eager"], "env": {}},
            tmp_path / "case/work",
            _args(),
        )


def test_locked_host_env_requires_pinned_runtime_coordinates(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="locked-host backend requires"):
        _load_run_suite()._managed_env(
            _config(),
            {"server_args": [], "env": {}},
            tmp_path / "case/work",
            _args(managed_backend="locked-host", runtime_image=None),
        )


def test_locked_host_env_pins_custody_and_piecewise_graph(tmp_path: Path) -> None:
    vllm_root = tmp_path / "vllm"
    seam_root = tmp_path / "seam"
    vllm_root.mkdir()
    seam_root.mkdir()
    env = _load_run_suite()._managed_env(
        _config(),
        {"server_args": [], "env": {}},
        tmp_path / "case/work",
        _args(
            managed_backend="locked-host",
            runtime_image=None,
            runtime_image_digest=None,
            host_python=sys.executable,
            vllm_root=str(vllm_root),
            seam_root=str(seam_root),
        ),
    )

    assert env["LATCHMOE_HOST_PYTHON"] == str(Path(sys.executable).absolute())
    assert env["LATCHMOE_CUSTODY_STATE"].endswith("custody_state.json")
    compilation = json.loads(env["VLLM_ENGINE_COMPILATION_CONFIG"])
    assert compilation["cudagraph_mode"] == "PIECEWISE"
    assert compilation["splitting_ops"] == ["vllm::moe_offload_stage"]


def test_locked_host_manager_stops_only_its_process_group(tmp_path: Path) -> None:
    manager = REPO_ROOT / "benchmark" / "scripts" / "manage_locked_host_runtime.py"
    state = tmp_path / "custody.json"
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
    )
    state.write_text(
        json.dumps({"pid": process.pid, "pgid": os.getpgid(process.pid)}),
        encoding="utf-8",
    )
    env = {**os.environ, "LATCHMOE_CUSTODY_STATE": str(state)}
    try:
        completed = subprocess.run(
            [sys.executable, str(manager), "stop", "--json"],
            env=env,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
        )
        assert completed.returncode == 0
        assert json.loads(completed.stdout)["active_state"] == "inactive"
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_wait_for_server_fails_fast_when_managed_process_dies() -> None:
    with pytest.raises(RuntimeError, match="managed server exited"):
        _load_run_suite()._wait_for_server(
            "http://127.0.0.1:9/v1/models",
            proc=None,
            timeout_s=60,
            is_alive=lambda: False,
        )


def test_npu_sampler_writes_machine_readable_usage(tmp_path: Path, monkeypatch) -> None:
    module = _load_run_suite()

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args[0],
            0,
            stdout="HBM Usage Rate(%) : 91\nNPU Utilization(%) : 27\n",
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    stop = module.threading.Event()
    stop.set()
    output = tmp_path / "npu_samples.jsonl"

    module._sample_npu_usage(5, output, stop, interval_s=0.0)

    sample = json.loads(output.read_text())
    assert sample["device"] == 5
    assert sample["hbm_usage_percent"] == 91
    assert sample["npu_utilization_percent"] == 27


def test_locked_host_manager_uses_package_entrypoint() -> None:
    manager_path = REPO_ROOT / "benchmark" / "scripts" / "manage_locked_host_runtime.py"
    source = manager_path.read_text(encoding="utf-8")

    assert '"vllm_moe_offload_ascend",' in source
    assert '"vllm_moe_offload_ascend.launcher",' not in source
    assert 'inherited_pythonpath = child_env.get("PYTHONPATH", "")' in source
    assert 'child_env.pop("VLLM_PLUGINS", None)' in source
    assert 'child_env["VLLM_PLUGINS"] = "ascend"' not in source


def test_runner_preserves_preflight_failure_before_service_start() -> None:
    runner_path = REPO_ROOT / "benchmark" / "scripts" / "run_suite.py"
    source = runner_path.read_text(encoding="utf-8")

    assert 'release_status not in {"released", "not-started"}' in source
