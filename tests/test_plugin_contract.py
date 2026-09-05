from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

import tomllib

import vllm_moe_offload_ascend as plugin

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_general_plugin_entry_point_is_declared() -> None:
    config = (REPO_ROOT / "pyproject.toml").read_text()

    assert "[project.scripts]" in config
    assert 'latchmoe = "vllm_moe_offload_ascend.launcher:main"' in config
    assert '[project.entry-points."vllm.general_plugins"]' in config
    assert '[project.entry-points."vllm.platform_plugins"]' not in config
    assert 'moe_offload_ascend = "vllm_moe_offload_ascend:register"' in config


def test_extension_manager_registration_is_static_and_project_owned() -> None:
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())

    registrations = config["project"]["entry-points"]["vllm_hust.extension_bundles"]
    assert registrations == {"org.vllm-hust.latchmoe": "vllm_moe_offload_ascend"}


def test_extension_manager_manifest_preserves_runtime_boundary() -> None:
    path = resources.files("vllm_moe_offload_ascend").joinpath(
        "vllm-hust-extension-v0.2.json"
    )
    manifest = json.loads(path.read_text(encoding="utf-8"))

    assert manifest["schema_version"] == "0.2-experimental"
    assert manifest["extension_id"] == "org.vllm-hust.latchmoe"
    assert manifest["kind"] == "in_process_plugin"
    assert manifest["host"] == {
        "provider": "vllm",
        "name": "vllm",
        "version_range": ">=0.28.1rc1.dev319,<0.29",
    }
    assert manifest["runtime"]["process_scope"] == "vllm-ascend-worker"
    assert manifest["lifecycle_owner"] == "vllm"
    assert manifest["requires_services"] == []
    assert manifest["protocols"] == [
        {
            "name": "vllm.ascend.moe-offload-seam",
            "version_range": ">=2,<3",
        }
    ]


def test_vllm_hust_optimization_manifest_matches_entry_point() -> None:
    manifest = json.loads((REPO_ROOT / ".vllm-hust" / "optimization.json").read_text())

    assert manifest["schema_version"] == 1
    assert manifest["id"] == "latchmoe"
    assert manifest["entrypoint"] == {
        "group": "vllm.general_plugins",
        "name": "moe_offload_ascend",
    }
    assert manifest["parameters"]["offload_gb"]["default"] == "14"
    assert manifest["activation"]["vllm_plugins"] == [
        "ascend",
        "moe_offload_ascend",
    ]
    environment = manifest["activation"]["environment"]
    assert environment["VLLM_ASCEND_MOE_OFFLOAD_GB"] == "${offload_gb}"
    assert manifest["activation"]["extra_args"] == [
        "--tensor-parallel-size",
        "4",
    ]
    moe, dense = manifest["compatibility"]["model_qualifications"]
    assert moe["status"] == "compatible"
    assert moe["functional_compatibility"] == "passed"
    assert moe["effectiveness_qualification"]["status"] == (
        "not-beneficial-in-tested-cell"
    )
    assert moe["runtime_state_source"] == "live_instance_observation_only"
    assert dense["model"] == "Qwen3.8-27B"
    assert dense["status"] == "not_applicable"


def test_register_retries_idempotent_patch_path(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        "vllm_moe_offload_ascend.env_registry.register_environment_variables",
        lambda: calls.append("envs"),
    )
    monkeypatch.setattr(
        "vllm_moe_offload_ascend.patches.patch_fused_moe.apply_patches",
        lambda: calls.append("patched"),
    )

    plugin.register()
    plugin.register()

    assert calls == ["envs", "patched", "envs", "patched"]


def test_failed_registration_can_be_retried(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        "vllm_moe_offload_ascend.env_registry.register_environment_variables",
        lambda: calls.append("envs"),
    )

    def fail() -> None:
        calls.append("failed")
        raise RuntimeError("missing vllm-ascend hook")

    monkeypatch.setattr(
        "vllm_moe_offload_ascend.patches.patch_fused_moe.apply_patches",
        fail,
    )

    try:
        plugin.register()
    except RuntimeError as exc:
        assert str(exc) == "missing vllm-ascend hook"
    else:
        raise AssertionError("registration failure must propagate")

    monkeypatch.setattr(
        "vllm_moe_offload_ascend.patches.patch_fused_moe.apply_patches",
        lambda: calls.append("retried"),
    )
    plugin.register()

    assert calls == ["envs", "failed", "envs", "retried"]
