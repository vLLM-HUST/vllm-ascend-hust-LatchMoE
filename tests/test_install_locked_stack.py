from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tools import install_locked_stack


def _lock() -> dict[str, str]:
    return {
        "vllm_repo": "https://example.test/vllm-hust.git",
        "vllm_commit": "vllm-commit",
        "vllm_tag": "v0.21.0",
        "seam_repo": "https://example.test/vllm-ascend-hust.git",
        "seam_branch": "feature/latchmoe-offload-seam-v1-v021",
        "seam_commit": "seam-commit",
    }


def test_read_lock_requires_install_coordinates(tmp_path):
    lock = tmp_path / "compatibility.lock"
    lock.write_text("vllm_repo=https://example.test/repo.git\n", encoding="utf-8")

    with pytest.raises(ValueError, match="compatibility lock is missing"):
        install_locked_stack.read_lock(lock)


def test_install_plan_uses_active_python_and_disables_unrelated_builds(tmp_path):
    plan = install_locked_stack.build_install_plan(
        workspace=tmp_path,
        plugin_root=Path("/plugin"),
        lock=_lock(),
    )

    assert len(plan) == 3
    assert all(command[:3] == [sys.executable, "-m", "pip"] for command, _ in plan)
    assert plan[0][1]["VLLM_TARGET_DEVICE"] == "empty"
    assert plan[1][1]["COMPILE_CUSTOM_KERNELS"] == "0"
    assert plan[0][0][-1] == str(tmp_path / "vllm-hust")
    assert plan[1][0][-1] == str(tmp_path / "vllm-ascend-hust")
    assert plan[2][0][-1] == "/plugin"


def test_base_runtime_check_reports_wrong_stack_and_missing_cann():
    lock = {
        **_lock(),
        "torch_version": "2.10.0",
        "torch_npu_version": "2.10.0.post2",
    }

    errors = install_locked_stack.base_runtime_errors(
        lock,
        versions={"torch": "2.10.0+cpu", "torch-npu": "2.9.0"},
        acl_origin=None,
    )

    assert "incompatible torch-npu" in errors[0]
    assert "CANN acl Python binding is not importable" in errors[1]


def test_dry_run_prints_locked_checkout_and_check_commands(tmp_path, capsys):
    lock_path = tmp_path / "compatibility.lock"
    lock_path.write_text(
        "\n".join(f"{key}={value}" for key, value in _lock().items()) + "\n",
        encoding="utf-8",
    )

    install_locked_stack.install_locked_stack(
        workspace=tmp_path / "stack",
        plugin_root=Path("/plugin"),
        lock_path=lock_path,
        dry_run=True,
    )

    output = capsys.readouterr().out
    assert "feature/latchmoe-offload-seam-v1-v021" in output
    assert "checkout --detach vllm-commit" in output
    assert "checkout --detach seam-commit" in output
    assert "VLLM_TARGET_DEVICE=empty" in output
    assert "COMPILE_CUSTOM_KERNELS=0" in output
    assert "vllm_moe_offload_ascend check" in output


def test_existing_checkout_with_unexpected_origin_is_rejected(tmp_path, monkeypatch):
    destination = tmp_path / "checkout"
    (destination / ".git").mkdir(parents=True)
    installer = install_locked_stack.Installer()
    outputs = iter(("", "https://example.test/wrong.git"))
    monkeypatch.setattr(installer, "output", lambda _command: next(outputs))

    with pytest.raises(RuntimeError, match="unexpected origin"):
        install_locked_stack._ensure_checkout(
            installer,
            repository="https://example.test/expected.git",
            destination=destination,
            commit="locked-commit",
        )


def test_git_remote_normalization_accepts_ssh_and_https_equivalence():
    assert install_locked_stack._normalized_git_remote(
        "git@github.com:vLLM-HUST/vllm-hust.git"
    ) == install_locked_stack._normalized_git_remote(
        "https://github.com/vLLM-HUST/vllm-hust.git"
    )
