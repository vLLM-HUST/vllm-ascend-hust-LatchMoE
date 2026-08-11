from __future__ import annotations

import importlib.util
from argparse import Namespace
from pathlib import Path

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
