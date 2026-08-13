#!/usr/bin/env python3
"""Install the locked LatchMoE host stack into the active Python.

The host repositories are deliberately installed without dependency resolution:
Ascend images already own the matching Torch, Torch-NPU, and CANN packages.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shlex
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from vllm_moe_offload_ascend.compatibility import (
    describe_runtime_profiles,
    match_runtime_profile,
)
from vllm_moe_offload_ascend.launcher import _detect_cann


DEFAULT_LOCK = REPOSITORY_ROOT / "vllm_moe_offload_ascend" / "compatibility.lock"
_INSTALL_ENV_KEYS = (
    "VLLM_TARGET_DEVICE",
    "COMPILE_CUSTOM_KERNELS",
    "LATCHMOE_COMPATIBILITY_LOCK",
)


def read_lock(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or not key.strip():
            raise ValueError(f"invalid compatibility lock line: {raw_line!r}")
        values[key.strip()] = value.strip()
    required = {
        "vllm_repo",
        "vllm_commit",
        "vllm_tag",
        "seam_repo",
        "seam_branch",
        "seam_commit",
    }
    missing = sorted(required - values.keys())
    if missing:
        raise ValueError("compatibility lock is missing: " + ", ".join(missing))
    return values


def _format_command(command: Sequence[str]) -> str:
    return shlex.join(str(item) for item in command)


def _normalized_git_remote(remote: str) -> str:
    value = str(remote).strip().removesuffix(".git").rstrip("/")
    if value.startswith("git@") and ":" in value:
        host, path = value[4:].split(":", 1)
        return f"{host}/{path}"
    for prefix in ("https://", "http://", "ssh://git@"):
        if value.startswith(prefix):
            return value[len(prefix):]
    return value


class Installer:
    def __init__(self, *, dry_run: bool = False) -> None:
        self.dry_run = bool(dry_run)

    def run(
        self,
        command: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
    ) -> None:
        prefixes = []
        if env is not None:
            prefixes = [
                f"{name}={shlex.quote(str(env[name]))}"
                for name in _INSTALL_ENV_KEYS
                if name in env and env.get(name) != os.environ.get(name)
            ]
        rendered = " ".join([*prefixes, _format_command(command)])
        print("+ " + rendered, flush=True)
        if self.dry_run:
            return
        subprocess.run(
            [str(item) for item in command],
            check=True,
            env=None if env is None else dict(env),
        )

    def output(self, command: Sequence[str]) -> str:
        return subprocess.run(
            [str(item) for item in command],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout.strip()


def _ensure_checkout(
    installer: Installer,
    *,
    repository: str,
    destination: Path,
    commit: str,
    branch: str | None = None,
    tag_repository: str | None = None,
    tag: str | None = None,
) -> None:
    if destination.exists():
        if not (destination / ".git").exists():
            raise RuntimeError(
                f"refusing to reuse non-Git destination: {destination}"
            )
        dirty = installer.output(
            ["git", "-C", str(destination), "status", "--porcelain"]
        )
        if dirty:
            raise RuntimeError(
                f"refusing to change dirty checkout: {destination}"
            )
        actual_origin = installer.output(
            ["git", "-C", str(destination), "remote", "get-url", "origin"]
        )
        normalized_expected = _normalized_git_remote(repository)
        normalized_actual = _normalized_git_remote(actual_origin)
        if normalized_actual != normalized_expected:
            raise RuntimeError(
                f"refusing to reuse checkout with unexpected origin: "
                f"expected {repository}, got {actual_origin}"
            )
    else:
        clone = [
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            "--depth",
            "1",
        ]
        if branch:
            clone.extend(["--branch", branch, "--single-branch"])
        clone.extend([repository, str(destination)])
        installer.run(clone)

    installer.run(
        [
            "git",
            "-C",
            str(destination),
            "fetch",
            "--filter=blob:none",
            "--depth",
            "1",
            "origin",
            commit,
        ]
    )
    if tag_repository and tag:
        installer.run(
            [
                "git",
                "-C",
                str(destination),
                "fetch",
                "--filter=blob:none",
                "--depth",
                "1",
                tag_repository,
                f"refs/tags/{tag}:refs/tags/{tag}",
            ]
        )
    installer.run(
        ["git", "-C", str(destination), "checkout", "--detach", commit]
    )


def build_install_plan(
    *,
    workspace: Path,
    plugin_root: Path,
    lock: Mapping[str, str],
) -> tuple[tuple[list[str], dict[str, str] | None], ...]:
    _ = lock
    pip_base = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-deps",
        "--no-build-isolation",
        "-e",
    ]
    base_env = dict(os.environ)
    vllm_env = dict(base_env)
    # vLLM-Ascend supplies the actual NPU platform. Building vLLM itself with
    # the empty target avoids pulling CUDA build requirements into the image.
    vllm_env["VLLM_TARGET_DEVICE"] = "empty"
    seam_env = dict(base_env)
    # The locked seam is a Python hook dependency for LatchMoE. Its full custom
    # kernel bundle contains unrelated operators and is not required here.
    seam_env["COMPILE_CUSTOM_KERNELS"] = "0"
    return (
        (pip_base + [str(workspace / "vllm-hust")], vllm_env),
        (pip_base + [str(workspace / "vllm-ascend-hust")], seam_env),
        (pip_base + [str(plugin_root)], base_env),
    )


def base_runtime_errors(
    lock: Mapping[str, str],
    *,
    versions: Mapping[str, str | None],
    acl_origin: str | None,
    cann_version: str | None,
) -> tuple[str, ...]:
    errors: list[str] = []
    for distribution in ("torch", "torch-npu"):
        if versions.get(distribution) is None:
            errors.append(f"required base package is missing: {distribution}")
    if acl_origin is None:
        errors.append(
            "CANN acl Python binding is not importable; source the CANN set_env.sh "
            "for this Python before installation"
        )
    if not errors and match_runtime_profile(
        lock,
        torch_version=versions.get("torch"),
        torch_npu_version=versions.get("torch-npu"),
        cann_version=cann_version,
    ) is None:
        errors.append(
            "incompatible base runtime: got "
            f"torch={versions.get('torch')}, "
            f"torch-npu={versions.get('torch-npu')}, "
            f"CANN={cann_version or '<missing>'}; supported profiles: "
            f"{describe_runtime_profiles(lock) or '<none>'}"
        )
    return tuple(errors)


def validate_base_runtime(lock: Mapping[str, str]) -> None:
    installed: dict[str, str | None] = {}
    for distribution in ("torch", "torch-npu"):
        try:
            installed[distribution] = str(version(distribution))
        except PackageNotFoundError:
            installed[distribution] = None
    acl_spec = importlib.util.find_spec("acl")
    acl_origin = None if acl_spec is None else acl_spec.origin
    cann_version, _cann_root = _detect_cann(os.environ)
    errors = base_runtime_errors(
        lock,
        versions=installed,
        acl_origin=acl_origin,
        cann_version=cann_version,
    )
    if errors:
        raise RuntimeError("; ".join(errors))


def install_locked_stack(
    *,
    workspace: Path,
    plugin_root: Path,
    lock_path: Path,
    dry_run: bool = False,
) -> None:
    lock = read_lock(lock_path)
    if not dry_run:
        validate_base_runtime(lock)
    if not dry_run:
        workspace.mkdir(parents=True, exist_ok=True)
    installer = Installer(dry_run=dry_run)
    _ensure_checkout(
        installer,
        repository=lock["vllm_repo"],
        destination=workspace / "vllm-hust",
        commit=lock["vllm_commit"],
        tag_repository="https://github.com/vllm-project/vllm.git",
        tag=lock["vllm_tag"],
    )
    _ensure_checkout(
        installer,
        repository=lock["seam_repo"],
        destination=workspace / "vllm-ascend-hust",
        commit=lock["seam_commit"],
        branch=lock["seam_branch"],
    )
    for command, environment in build_install_plan(
        workspace=workspace,
        plugin_root=plugin_root,
        lock=lock,
    ):
        installer.run(command, env=environment)
    check_env = dict(os.environ)
    check_env["LATCHMOE_COMPATIBILITY_LOCK"] = str(lock_path)
    installer.run(
        [sys.executable, "-m", "vllm_moe_offload_ascend", "check"],
        env=check_env,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Install the exact LatchMoE host commits into this Python."
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        required=True,
        help="Directory that will contain vllm-hust and vllm-ascend-hust.",
    )
    parser.add_argument(
        "--plugin-root",
        type=Path,
        default=REPOSITORY_ROOT,
        help="LatchMoE checkout to install (default: this repository).",
    )
    parser.add_argument(
        "--lock",
        type=Path,
        default=DEFAULT_LOCK,
        help="Compatibility lock to consume.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without cloning, checking out, or installing.",
    )
    arguments = parser.parse_args(argv)
    try:
        install_locked_stack(
            workspace=arguments.workspace.resolve(),
            plugin_root=arguments.plugin_root.resolve(),
            lock_path=arguments.lock.resolve(),
            dry_run=arguments.dry_run,
        )
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
