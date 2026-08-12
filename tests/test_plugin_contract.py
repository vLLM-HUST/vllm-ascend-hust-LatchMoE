from __future__ import annotations

import json
from pathlib import Path

import vllm_moe_offload_ascend as plugin


REPO_ROOT = Path(__file__).resolve().parents[1]

def test_general_plugin_entry_point_is_declared() -> None:
    config = (REPO_ROOT / "pyproject.toml").read_text()

    assert '[project.scripts]' in config
    assert 'latchmoe = "vllm_moe_offload_ascend.launcher:main"' in config
    assert '[project.entry-points."vllm.general_plugins"]' in config
    assert '[project.entry-points."vllm.platform_plugins"]' not in config
    assert (
        'moe_offload_ascend = "vllm_moe_offload_ascend:register"' in config
    )


def test_vllm_hust_optimization_manifest_matches_entry_point() -> None:
    manifest = json.loads(
        (REPO_ROOT / ".vllm-hust" / "optimization.json").read_text()
    )

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
