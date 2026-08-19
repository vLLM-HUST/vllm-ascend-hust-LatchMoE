from __future__ import annotations

import re
from pathlib import Path
from types import ModuleType

from vllm_moe_offload_ascend.env_registry import (
    ENVIRONMENT_VARIABLES,
    _register,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_registry_covers_every_plugin_owned_vllm_variable() -> None:
    names: set[str] = set()
    for path in (REPO_ROOT / "vllm_moe_offload_ascend").rglob("*.py"):
        names.update(
            re.findall(
                r"VLLM_ASCEND_MOE_[A-Z0-9_]+",
                path.read_text(encoding="utf-8"),
            )
        )

    assert names <= ENVIRONMENT_VARIABLES.keys()


def test_register_adds_missing_variables_without_overwriting_host_values() -> None:
    module = ModuleType("fake_envs")
    original = lambda: "host"  # noqa: E731
    module.environment_variables = {
        "VLLM_ASCEND_MOE_OFFLOAD_POLICY": original,
    }

    added = _register(module, "environment_variables")

    assert added == len(ENVIRONMENT_VARIABLES) - 1
    assert module.environment_variables["VLLM_ASCEND_MOE_OFFLOAD_POLICY"] is original
    assert "VLLM_ASCEND_MOE_OFFLOAD_GB" in module.environment_variables


def test_registered_values_are_typed(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_GB", "14.5")
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS", "32")
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_ENABLED", "true")
    monkeypatch.setenv("VLLM_ASCEND_MOE_ROUTER_PARITY_MAX_TOKENS", "32")

    assert ENVIRONMENT_VARIABLES["VLLM_ASCEND_MOE_OFFLOAD_GB"]() == 14.5
    assert ENVIRONMENT_VARIABLES["VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS"]() == 32
    assert ENVIRONMENT_VARIABLES["VLLM_ASCEND_MOE_OFFLOAD_ENABLED"]() is True
    assert ENVIRONMENT_VARIABLES["VLLM_ASCEND_MOE_ROUTER_PARITY_MAX_TOKENS"]() == 32
