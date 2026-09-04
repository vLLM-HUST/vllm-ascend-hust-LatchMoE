from __future__ import annotations

from argparse import Namespace
import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import importlib.util

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_SUITE_PATH = REPO_ROOT / "benchmark" / "scripts" / "run_suite.py"
MANAGER_PATH = REPO_ROOT / "benchmark" / "scripts" / "manage_locked_host_runtime.py"


def _load_run_suite():
    spec = importlib.util.spec_from_file_location("managed_run_suite_test", RUN_SUITE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_manager():
    spec = importlib.util.spec_from_file_location("managed_host_manager_test", MANAGER_PATH)
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
    assert env["ASCEND_RT_VISIBLE_DEVICES"] == "5"
    compilation = json.loads(env["VLLM_ENGINE_COMPILATION_CONFIG"])
    assert compilation["cudagraph_mode"] == "PIECEWISE"
    assert "vllm::deepseek_v4_attention" in compilation["splitting_ops"]
    assert "vllm::unified_kv_cache_update" in compilation["splitting_ops"]
    assert compilation["splitting_ops"][-1] == "vllm::moe_offload_stage"


def test_locked_host_manager_consumes_piecewise_compilation_config(monkeypatch) -> None:
    monkeypatch.setenv(
        "VLLM_ENGINE_COMPILATION_CONFIG",
        '{"cudagraph_mode":"PIECEWISE","splitting_ops":["vllm::moe_offload_stage"]}',
    )
    manager = _load_manager()
    args = manager._compilation_config_args([])
    assert args[0] == "--compilation-config"
    assert json.loads(args[1]) == {
        "cudagraph_mode": "PIECEWISE",
        "splitting_ops": ["vllm::moe_offload_stage"],
    }


def test_locked_host_manager_rejects_compilation_config_override(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_ENGINE_COMPILATION_CONFIG", '{"cudagraph_mode":"PIECEWISE"}')
    manager = _load_manager()
    with pytest.raises(ValueError, match="must not override"):
        manager._compilation_config_args(["--compilation-config"])


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


def test_runtime_source_identity_covers_uncommitted_source_bytes(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    source_dir = tmp_path / "benchmark" / "scripts"
    source_dir.mkdir(parents=True)
    source = source_dir / "runner.py"
    source.write_text("value = 1\n", encoding="utf-8")
    module = _load_run_suite()
    first, first_count = module._runtime_source_identity(tmp_path)
    source.write_text("value = 2\n", encoding="utf-8")
    second, second_count = module._runtime_source_identity(tmp_path)
    assert first_count == second_count == 1
    assert first != second


def test_capability_identity_binds_checkpoint_and_registry_row(tmp_path: Path) -> None:
    module = _load_run_suite()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    config_path = checkpoint / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "num_experts": 8,
                "num_experts_per_tok": 2,
                "hidden_size": 32,
            }
        ),
        encoding="utf-8",
    )
    config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "id": "fixture-routed",
                        "checkpoint_path": str(checkpoint),
                        "config_sha256": config_sha,
                        "checkpoint_index_sha256": "index-digest",
                        "qualification_status": "not_run",
                        "capability_config": {"shared_mode": "none"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    identity = module._capability_identity(
        {"model": {"path": str(checkpoint)}},
        _args(capability_registry=str(registry_path)),
    )

    assert identity["registry_status"] == "matched"
    assert identity["registry_model_id"] == "fixture-routed"
    assert identity["descriptor_status"] == "recorded"
    assert len(identity["descriptor_sha256"]) == 64


def test_router_parity_env_is_opt_in_and_has_an_artifact_path(tmp_path: Path) -> None:
    env = _load_run_suite()._unit_env({}, tmp_path, router_parity=True)

    assert env["VLLM_ASCEND_MOE_OFFLOAD_COMPARE_ROUTER"] == "1"
    assert env["VLLM_ASCEND_MOE_ROUTER_PARITY_PATH"] == str(
        tmp_path / "moe_router_parity.jsonl"
    )
