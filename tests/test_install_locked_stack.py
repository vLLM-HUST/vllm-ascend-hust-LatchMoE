from __future__ import annotations

import sys
import subprocess
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


def test_installer_bootstraps_imports_from_a_fresh_checkout():
    result = subprocess.run(
        [sys.executable, "-S", str(Path(install_locked_stack.__file__)), "--help"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={"PATH": "/usr/bin:/bin"},
    )

    assert result.returncode == 0, result.stderr
    assert "Install the exact LatchMoE host commits" in result.stdout


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
        cann_version=None,
    )

    assert "CANN acl Python binding is not importable" in errors[0]


def test_base_runtime_profiles_are_matched_as_exact_tuples():
    lock = {
        **_lock(),
        "runtime_profile.issue7": (
            "2.10.0|2.10.0.post2|9.0.1|service_graph_qualified"
        ),
        "runtime_profile.cann9_legacy": (
            "2.10.0|2.10.0|9.0.0|issue4_graph_smoke_qualified"
        ),
    }

    assert install_locked_stack.base_runtime_errors(
        lock,
        versions={"torch": "2.10.0+cpu", "torch-npu": "2.10.0"},
        acl_origin="/cann/acl.so",
        cann_version="9.0.0",
    ) == ()
    errors = install_locked_stack.base_runtime_errors(
        lock,
        versions={"torch": "2.10.0+cpu", "torch-npu": "2.10.0"},
        acl_origin="/cann/acl.so",
        cann_version="9.0.1",
    )
    assert errors and "incompatible base runtime" in errors[0]


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
    assert "clone --filter=blob:none --no-checkout --depth 1" in output
    assert "fetch --filter=blob:none --depth 1 origin vllm-commit" in output
    assert "feature/latchmoe-offload-seam-v1-v021" in output
    assert "checkout --detach vllm-commit" in output
    assert "checkout --detach seam-commit" in output
    assert "VLLM_TARGET_DEVICE=empty" in output
    assert "COMPILE_CUSTOM_KERNELS=0" in output
    assert f"LATCHMOE_COMPATIBILITY_LOCK={lock_path}" in output
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


def test_dry_run_reuses_clean_checkout_after_read_only_validation(
    tmp_path, monkeypatch, capsys
):
    destination = tmp_path / "checkout"
    (destination / ".git").mkdir(parents=True)
    installer = install_locked_stack.Installer(dry_run=True)
    outputs = iter(("", "https://example.test/expected.git"))
    monkeypatch.setattr(installer, "output", lambda _command: next(outputs))

    install_locked_stack._ensure_checkout(
        installer,
        repository="https://example.test/expected.git",
        destination=destination,
        commit="locked-commit",
    )

    output = capsys.readouterr().out
    assert "fetch --filter=blob:none --depth 1 origin locked-commit" in output
    assert "checkout --detach locked-commit" in output


def test_git_remote_normalization_accepts_ssh_and_https_equivalence():
    assert install_locked_stack._normalized_git_remote(
        "git@github.com:vLLM-HUST/vllm-hust.git"
    ) == install_locked_stack._normalized_git_remote(
        "https://github.com/vLLM-HUST/vllm-hust.git"
    )
