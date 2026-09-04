"""Launch vLLM through the interpreter that owns the LatchMoE plugin."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, entry_points, version
from pathlib import Path
from typing import Mapping, Sequence

from vllm_moe_offload_ascend.compatibility import (
    describe_runtime_profiles,
    match_runtime_profile,
    public_version,
)


PLATFORM_PLUGIN_GROUP = "vllm.platform_plugins"
GENERAL_PLUGIN_GROUP = "vllm.general_plugins"
REQUIRED_PLATFORM_PLUGINS = {
    "ascend": "vllm_ascend:register",
}
REQUIRED_GENERAL_PLUGINS = {
    "moe_offload_ascend": "vllm_moe_offload_ascend:register",
}
DEFAULT_COMPATIBILITY_LOCK = Path(__file__).with_name("compatibility.lock")
_PREFIX_CACHE_ENABLE_FLAGS = {
    "--enable-prefix-caching",
    "--enable_prefix_caching",
}
_PREFIX_CACHE_DISABLE_FLAGS = {
    "--no-enable-prefix-caching",
    "--no-enable_prefix_caching",
}


@dataclass(frozen=True)
class GitCheckout:
    root: str
    commit: str
    branch: str | None
    dirty_paths: tuple[str, ...]

    @property
    def dirty(self) -> bool:
        return bool(self.dirty_paths)


@dataclass(frozen=True)
class EnvironmentReport:
    python: str
    vllm: str | None
    vllm_ascend: str | None
    plugin: str | None
    acl: str | None
    platform_plugins: tuple[tuple[str, str], ...]
    general_plugins: tuple[tuple[str, str], ...]
    vllm_plugins_filter: str | None
    compatibility_lock: str | None
    versions: tuple[tuple[str, str | None], ...]
    cann_root: str | None
    runtime_profile: str | None
    qualification: str | None
    runtime_checkout: GitCheckout | None
    vllm_checkout: GitCheckout | None
    seam_checkout: GitCheckout | None
    seam_abi: str | None
    strict: bool
    warnings: tuple[str, ...]
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_jsonable(self) -> dict[str, object]:
        payload = asdict(self)
        payload["ok"] = self.ok
        return payload


def _module_origin(module_name: str) -> str | None:
    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, ModuleNotFoundError, ValueError):
        return None
    if spec is None:
        return None
    if spec.origin:
        return str(spec.origin)
    if spec.submodule_search_locations:
        return str(next(iter(spec.submodule_search_locations), "")) or None
    return None


def _discover_platform_plugins() -> tuple[tuple[str, str], ...]:
    discovered = (
        (str(item.name), str(item.value))
        for item in entry_points(group=PLATFORM_PLUGIN_GROUP)
    )
    return tuple(sorted(discovered))


def _discover_general_plugins() -> tuple[tuple[str, str], ...]:
    discovered = (
        (str(item.name), str(item.value))
        for item in entry_points(group=GENERAL_PLUGIN_GROUP)
    )
    return tuple(sorted(discovered))


def _read_key_value_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or not key.strip():
            raise ValueError(f"invalid compatibility lock line: {raw_line!r}")
        values[key.strip()] = value.strip()
    return values


def _load_compatibility_lock(
    environ: Mapping[str, str],
) -> tuple[str | None, dict[str, str]]:
    configured = environ.get("LATCHMOE_COMPATIBILITY_LOCK")
    path = Path(configured).expanduser() if configured else DEFAULT_COMPATIBILITY_LOCK
    try:
        return str(path.resolve()), _read_key_value_file(path)
    except OSError:
        return str(path), {}


def _distribution_version(distribution: str) -> str | None:
    try:
        return str(version(distribution))
    except PackageNotFoundError:
        return None


def _run_git(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.rstrip()


def _git_checkout_for_origin(
    origin: str | None,
    *,
    clean_paths: tuple[str, ...] = (),
) -> GitCheckout | None:
    if origin is None:
        return None
    start = Path(origin).resolve().parent
    root_text = _run_git(start, "rev-parse", "--show-toplevel")
    if not root_text:
        return None
    root = Path(root_text)
    commit = _run_git(root, "rev-parse", "HEAD")
    if not commit:
        return None
    branch = _run_git(root, "branch", "--show-current") or None
    status_args = ["status", "--porcelain", "--untracked-files=all"]
    if clean_paths:
        status_args.extend(("--", *clean_paths))
    status = _run_git(root, *status_args) or ""
    dirty_paths = tuple(
        line[3:].strip() if len(line) > 3 else line.strip()
        for line in status.splitlines()
        if line.strip()
    )
    return GitCheckout(
        root=str(root),
        commit=commit,
        branch=branch,
        dirty_paths=dirty_paths,
    )


def _git_is_ancestor(root: str, ancestor: str, descendant: str) -> bool:
    return _run_git(Path(root), "merge-base", "--is-ancestor", ancestor, descendant) == ""


def _detect_cann(environ: Mapping[str, str]) -> tuple[str | None, str | None]:
    candidates: list[Path] = []
    for name in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME"):
        raw = environ.get(name)
        if raw:
            candidates.append(Path(raw).expanduser())
    candidates.extend(
        (
            Path("/usr/local/Ascend/ascend-toolkit/latest"),
            Path("/usr/local/Ascend/cann"),
        )
    )
    candidates.extend(sorted(Path("/usr/local/Ascend").glob("cann-*"), reverse=True))
    seen: set[str] = set()
    for candidate in candidates:
        try:
            root = candidate.resolve()
        except OSError:
            root = candidate
        root_key = str(root)
        if root_key in seen:
            continue
        seen.add(root_key)
        for relative in (
            "aarch64-linux/ascend_toolkit_install.info",
            "arm64-linux/ascend_toolkit_install.info",
        ):
            info = root / relative
            try:
                values = _read_key_value_file(info)
            except OSError:
                continue
            detected = values.get("version")
            if detected:
                return detected, str(root)
    return None, None


def _validate_seam_abi(root: str, abi: str) -> tuple[str, ...]:
    if abi not in {"1", "2"}:
        return (f"unsupported LatchMoE seam ABI: {abi}",)
    requirements = {
        "vllm_ascend/ops/fused_moe/fused_moe.py": (
            "def get_moe_offload_runtime",
            "before_gmm2_evt",
            "swiglu_limit",
        ),
        "vllm_ascend/ops/fused_moe/moe_comm_method.py": (
            "class FusedExpertsResult",
            "def _maybe_apply_moe_offload_plan",
            "before_gmm2_evt",
            "swiglu_limit",
        ),
        "vllm_ascend/ops/fused_moe/moe_runtime_args.py": (
            "MoEOffloadParams",
            "physical_expert_count",
        ),
        "vllm_ascend/ops/fused_moe/token_dispatcher.py": (
            "physical_expert_count",
        ),
        "vllm_ascend/utils.py": ("def adapt_patch",),
    }
    if abi == "2":
        requirements = {
            "vllm_ascend/ops/fused_moe/dataclass/fused_experts.py": (
                "class MoEOffloadParams",
                "physical_expert_count",
                "offload: MoEOffloadParams",
            ),
            "vllm_ascend/ops/fused_moe/dataclass/router_input.py": (
                "physical_expert_count",
            ),
            "vllm_ascend/ops/fused_moe/moe_comm_method.py": (
                "def _maybe_apply_moe_offload_plan",
                "before_gmm2_evt",
                "swiglu_limit",
            ),
            "vllm_ascend/ops/fused_moe/routed_experts.py": (
                "def get_moe_offload_runtime",
                "offload_enabled=offload_enabled",
            ),
            "vllm_ascend/ops/fused_moe/token_dispatcher.py": (
                "physical_expert_count",
            ),
        }
    errors: list[str] = []
    checkout = Path(root)
    for relative, required_tokens in requirements.items():
        path = checkout / relative
        try:
            source = path.read_text(encoding="utf-8")
        except OSError:
            errors.append(f"seam ABI {abi} is missing required file: {relative}")
            continue
        missing = [token for token in required_tokens if token not in source]
        if missing:
            errors.append(
                f"seam ABI {abi} contract mismatch in {relative}: "
                + ", ".join(missing)
            )
    return tuple(errors)


def _to_bool(raw: str | None) -> bool:
    return str(raw or "").strip().lower() in {"1", "true", "yes", "on"}


def inspect_environment(
    environ: Mapping[str, str] | None = None,
    *,
    strict: bool | None = None,
) -> EnvironmentReport:
    """Inspect prerequisites without importing vLLM or vLLM-Ascend."""

    environment = os.environ if environ is None else environ
    strict_mode = _to_bool(environment.get("LATCHMOE_STRICT_ENV")) if strict is None else strict
    lock_path, compatibility = _load_compatibility_lock(environment)
    origins = {
        "vllm": _module_origin("vllm"),
        "vllm_ascend": _module_origin("vllm_ascend"),
        "plugin": _module_origin("vllm_moe_offload_ascend"),
        "acl": _module_origin("acl"),
    }
    platform_plugins = _discover_platform_plugins()
    general_plugins = _discover_general_plugins()
    errors: list[str] = []
    warnings: list[str] = []

    if not compatibility:
        errors.append(f"compatibility lock is missing or unreadable: {lock_path}")

    for label, origin in origins.items():
        if origin is None:
            errors.append(f"required module is not importable: {label}")

    required_groups = (
        (PLATFORM_PLUGIN_GROUP, REQUIRED_PLATFORM_PLUGINS, platform_plugins),
        (GENERAL_PLUGIN_GROUP, REQUIRED_GENERAL_PLUGINS, general_plugins),
    )
    for group, required, discovered in required_groups:
        plugin_values = dict(discovered)
        for name, expected_value in required.items():
            actual_value = plugin_values.get(name)
            if actual_value is None:
                errors.append(f"missing {group} entry point: {name}")
            elif actual_value != expected_value:
                errors.append(
                    f"incorrect {group} entry point {name}: "
                    f"expected {expected_value}, got {actual_value}"
                )

    raw_filter = environment.get("VLLM_PLUGINS")
    if raw_filter is not None:
        allowed = {item.strip() for item in raw_filter.split(",") if item.strip()}
        required_names = {*REQUIRED_PLATFORM_PLUGINS, *REQUIRED_GENERAL_PLUGINS}
        filtered = [name for name in required_names if name not in allowed]
        if filtered:
            errors.append("VLLM_PLUGINS filters required plugins: " + ", ".join(filtered))

    versions = {
        "vllm": _distribution_version("vllm"),
        "torch": _distribution_version("torch"),
        "torch_npu": _distribution_version("torch-npu"),
    }
    expected_vllm_version = compatibility.get("vllm_version")
    if expected_vllm_version and public_version(versions["vllm"]) != expected_vllm_version:
        errors.append(
            "incompatible vllm version: "
            f"expected {expected_vllm_version}, got {versions['vllm'] or '<missing>'}"
        )

    cann_version, cann_root = _detect_cann(environment)
    runtime_profile = match_runtime_profile(
        compatibility,
        torch_version=versions["torch"],
        torch_npu_version=versions["torch_npu"],
        cann_version=cann_version,
    )
    if compatibility and runtime_profile is None:
        errors.append(
            "incompatible base runtime: got "
            f"torch={versions['torch'] or '<missing>'}, "
            f"torch-npu={versions['torch_npu'] or '<missing>'}, "
            f"CANN={cann_version or '<missing>'}; supported profiles: "
            f"{describe_runtime_profiles(compatibility) or '<none>'}"
        )

    clean_paths = tuple(
        item.strip()
        for item in compatibility.get("runtime_clean_paths", "").split(",")
        if item.strip()
    )
    runtime_checkout = _git_checkout_for_origin(origins["plugin"], clean_paths=clean_paths)
    vllm_checkout = _git_checkout_for_origin(origins["vllm"])
    seam_checkout = _git_checkout_for_origin(origins["vllm_ascend"])
    expected_runtime = environment.get("LATCHMOE_EXPECTED_RUNTIME_COMMIT")
    minimum_runtime = compatibility.get("runtime_min_commit")
    if runtime_checkout is None:
        message = "LatchMoE runtime module is not backed by a Git checkout"
        (errors if strict_mode or expected_runtime else warnings).append(message)
    else:
        if expected_runtime and runtime_checkout.commit != expected_runtime:
            errors.append(
                "runtime commit mismatch: "
                f"expected {expected_runtime}, got {runtime_checkout.commit}"
            )
        if minimum_runtime and not _git_is_ancestor(
            runtime_checkout.root,
            minimum_runtime,
            runtime_checkout.commit,
        ):
            errors.append(
                "runtime checkout is not based on the locked minimum commit: "
                f"{minimum_runtime}"
            )
        if runtime_checkout.dirty:
            message = "runtime checkout has relevant dirty paths: " + ", ".join(
                runtime_checkout.dirty_paths
            )
            (errors if strict_mode else warnings).append(message)

    expected_vllm = compatibility.get("vllm_commit")
    if vllm_checkout is None:
        message = "vllm module is not backed by the locked Git checkout"
        (errors if strict_mode else warnings).append(message)
    elif expected_vllm and vllm_checkout.commit != expected_vllm:
        errors.append(
            f"vllm commit mismatch: expected {expected_vllm}, "
            f"got {vllm_checkout.commit}"
        )

    expected_seam = compatibility.get("seam_commit")
    seam_abi = compatibility.get("seam_abi")
    if seam_checkout is None:
        errors.append("vllm_ascend module is not backed by the locked seam Git checkout")
    else:
        if expected_seam and seam_checkout.commit != expected_seam:
            errors.append(
                f"seam commit mismatch: expected {expected_seam}, got {seam_checkout.commit}"
            )
        if seam_checkout.dirty:
            errors.append(
                "seam checkout is dirty: " + ", ".join(seam_checkout.dirty_paths)
            )
        if seam_abi:
            errors.extend(_validate_seam_abi(seam_checkout.root, seam_abi))

    return EnvironmentReport(
        python=sys.executable,
        vllm=origins["vllm"],
        vllm_ascend=origins["vllm_ascend"],
        plugin=origins["plugin"],
        acl=origins["acl"],
        platform_plugins=platform_plugins,
        general_plugins=general_plugins,
        vllm_plugins_filter=raw_filter,
        compatibility_lock=lock_path,
        versions=tuple(sorted(versions.items())),
        cann_root=cann_root,
        runtime_profile=None if runtime_profile is None else runtime_profile.name,
        qualification=None if runtime_profile is None else runtime_profile.qualification,
        runtime_checkout=runtime_checkout,
        vllm_checkout=vllm_checkout,
        seam_checkout=seam_checkout,
        seam_abi=seam_abi,
        strict=bool(strict_mode),
        warnings=tuple(warnings),
        errors=tuple(errors),
    )


def _print_report(report: EnvironmentReport, *, as_json: bool = False) -> None:
    if as_json:
        print(json.dumps(report.to_jsonable(), indent=2, sort_keys=True))
        return

    status = "PASS" if report.ok else "FAIL"
    platform_names = ", ".join(name for name, _value in report.platform_plugins)
    general_names = ", ".join(name for name, _value in report.general_plugins)
    print(f"LatchMoE environment check: {status}")
    print(f"python = {report.python}")
    print(f"vllm = {report.vllm or '<missing>'}")
    print(f"vllm_ascend = {report.vllm_ascend or '<missing>'}")
    print(f"plugin = {report.plugin or '<missing>'}")
    print(f"acl = {report.acl or '<missing>'}")
    print(f"platform_plugins = {platform_names or '<none>'}")
    print(f"general_plugins = {general_names or '<none>'}")
    print(f"VLLM_PLUGINS = {report.vllm_plugins_filter or '<unset>'}")
    print(f"compatibility_lock = {report.compatibility_lock or '<missing>'}")
    print("versions = " + ", ".join(f"{name}={value or '<missing>'}" for name, value in report.versions))
    print(f"CANN = {report.cann_root or '<missing>'}")
    print(f"runtime_profile = {report.runtime_profile or '<unmatched>'}")
    print(f"qualification = {report.qualification or '<none>'}")
    if report.runtime_checkout is not None:
        print(
            "runtime_git = "
            f"{report.runtime_checkout.commit} "
            f"root={report.runtime_checkout.root} dirty={report.runtime_checkout.dirty}"
        )
    if report.vllm_checkout is not None:
        print(
            "vllm_git = "
            f"{report.vllm_checkout.commit} "
            f"root={report.vllm_checkout.root} "
            f"dirty={report.vllm_checkout.dirty}"
        )
    if report.seam_checkout is not None:
        print(
            "seam_git = "
            f"{report.seam_checkout.commit} "
            f"root={report.seam_checkout.root} dirty={report.seam_checkout.dirty}"
        )
    print(f"seam_abi = {report.seam_abi or '<missing>'}")
    print(f"strict = {report.strict}")
    for warning in report.warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    for error in report.errors:
        print(f"ERROR: {error}", file=sys.stderr)


def _run_vllm_cli(argv: Sequence[str]) -> int:
    # General plugin discovery normally registers LatchMoE while importing
    # EngineArgs. Retry explicitly for programmatic launchers and idempotency.
    from vllm.engine.arg_utils import EngineArgs as _EngineArgs  # noqa: F401
    from vllm_moe_offload_ascend import register

    register()
    from vllm.entrypoints.cli.main import main as vllm_main

    original_argv = sys.argv
    sys.argv = [original_argv[0], *argv]
    try:
        result = vllm_main()
    finally:
        sys.argv = original_argv
    return int(result or 0)


def _force_prefix_cache_disabled(argv: Sequence[str]) -> list[str]:
    """Keep the supported LatchMoE serving path independent of KV reuse.

    Prefix-cache hits change the number of tokens that enter prefill and the
    corresponding KV/block metadata.  That is a separate vLLM/Ascend path and
    is intentionally outside LatchMoE's correctness contract.  Reject an
    explicit opt-in rather than silently changing a user's requested mode.
    """

    arguments = list(argv)
    if not arguments or arguments[0] != "serve":
        return arguments
    for argument in arguments:
        option = argument.split("=", 1)[0]
        if option in _PREFIX_CACHE_ENABLE_FLAGS:
            raise ValueError(
                "LatchMoE does not support prefix caching; remove "
                f"{argument!r} and use the fixed no-prefix-cache path"
            )
    if any(argument in _PREFIX_CACHE_DISABLE_FLAGS for argument in arguments):
        return arguments
    return [*arguments, "--no-enable-prefix-caching"]


def main(argv: Sequence[str] | None = None) -> int:
    """Check the current interpreter, then delegate to the vLLM CLI."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] in {"check", "doctor"}:
        extra = arguments[1:]
        allowed = {"--json", "--strict"}
        if any(item not in allowed for item in extra):
            print("usage: latchmoe check [--json] [--strict]", file=sys.stderr)
            return 2
        report = inspect_environment(strict="--strict" in extra or None)
        _print_report(report, as_json="--json" in extra)
        return 0 if report.ok else 1

    report = inspect_environment()
    if not report.ok:
        _print_report(report)
        print(
            "LatchMoE refused to start vLLM because the active Python "
            "environment is incompatible.",
            file=sys.stderr,
        )
        return 1

    try:
        arguments = _force_prefix_cache_disabled(arguments)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        f"LatchMoE launcher: using {report.python}; environment check passed",
        file=sys.stderr,
    )
    return _run_vllm_cli(arguments)
