from __future__ import annotations

from pathlib import Path

import pytest

import vllm_moe_offload_ascend as plugin


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def reset_plugin_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(plugin, "_REGISTERED", False)


def test_platform_plugin_entry_point_is_declared() -> None:
    config = (REPO_ROOT / "pyproject.toml").read_text()

    assert '[project.entry-points."vllm.platform_plugins"]' in config
    assert (
        'moe_offload_ascend = "vllm_moe_offload_ascend:register"' in config
    )


def test_register_applies_patches_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        "vllm_moe_offload_ascend.patches.patch_fused_moe.apply_patches",
        lambda: calls.append("patched"),
    )

    plugin.register()
    plugin.register()

    assert calls == ["patched"]
    assert plugin._REGISTERED is True


def test_failed_registration_can_be_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail() -> None:
        raise RuntimeError("missing vllm-ascend hook")

    monkeypatch.setattr(
        "vllm_moe_offload_ascend.patches.patch_fused_moe.apply_patches",
        fail,
    )

    with pytest.raises(RuntimeError, match="missing vllm-ascend hook"):
        plugin.register()

    assert plugin._REGISTERED is False
