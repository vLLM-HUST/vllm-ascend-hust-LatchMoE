from __future__ import annotations

import json
from pathlib import Path

from vllm_moe_offload_ascend import launcher


def _patch_complete_environment(monkeypatch):
    origins = {
        "vllm": "/env/site-packages/vllm/__init__.py",
        "vllm_ascend": "/env/site-packages/vllm_ascend/__init__.py",
        "vllm_moe_offload_ascend": "/src/plugin/__init__.py",
        "acl": "/usr/local/Ascend/cann/python/site-packages/acl.so",
    }
    monkeypatch.setattr(
        launcher,
        "_module_origin",
        lambda name: origins.get(name),
    )
    monkeypatch.setattr(
        launcher,
        "_discover_platform_plugins",
        lambda: (("ascend", "vllm_ascend:register"),),
    )
    monkeypatch.setattr(
        launcher,
        "_discover_general_plugins",
        lambda: (
            ("moe_offload_ascend", "vllm_moe_offload_ascend:register"),
        ),
    )
    monkeypatch.setattr(
        launcher,
        "_load_compatibility_lock",
        lambda _environ: (
            "/src/plugin/compatibility.lock",
            {
                "runtime_min_commit": "runtime-base",
                "runtime_clean_paths": "vllm_moe_offload_ascend,tests",
                "seam_abi": "1",
                "seam_commit": "seam-commit",
                "vllm_commit": "vllm-commit",
                "vllm_version": "0.21.0",
                "torch_version": "2.10.0",
                "torch_npu_version": "2.10.0",
                "cann_version": "9.0.0",
            },
        ),
    )
    installed_versions = {
        "vllm": "0.21.0+empty",
        "torch": "2.10.0+cpu",
        "torch-npu": "2.10.0",
    }
    monkeypatch.setattr(
        launcher,
        "_distribution_version",
        lambda name: installed_versions.get(name),
    )
    monkeypatch.setattr(
        launcher,
        "_detect_cann",
        lambda _environ: ("9.0.0", "/usr/local/Ascend/cann-9.0.0"),
    )

    def checkout(origin, *, clean_paths=()):
        if origin is None:
            return None
        if "vllm_ascend" in origin:
            return launcher.GitCheckout(
                root="/src/seam",
                commit="seam-commit",
                branch="hook-seam",
                dirty_paths=(),
            )
        if "plugin" in origin:
            return launcher.GitCheckout(
                root="/src/plugin",
                commit="runtime-head",
                branch="main",
                dirty_paths=(),
            )
        if "/vllm/" in origin:
            return launcher.GitCheckout(
                root="/src/vllm",
                commit="vllm-commit",
                branch="v0.21.0",
                dirty_paths=(),
            )
        return None

    monkeypatch.setattr(launcher, "_git_checkout_for_origin", checkout)
    monkeypatch.setattr(launcher, "_git_is_ancestor", lambda *_args: True)
    monkeypatch.setattr(launcher, "_validate_seam_abi", lambda *_args: ())


def test_environment_check_accepts_complete_overlay(monkeypatch):
    _patch_complete_environment(monkeypatch)

    report = launcher.inspect_environment({})

    assert report.ok
    assert report.errors == ()
    assert report.runtime_profile == "default"
    assert report.qualification == "legacy_lock"


def test_environment_check_rejects_filtered_plugin(monkeypatch):
    _patch_complete_environment(monkeypatch)

    report = launcher.inspect_environment({"VLLM_PLUGINS": "ascend"})

    assert not report.ok
    assert report.errors == (
        "VLLM_PLUGINS filters required plugins: moe_offload_ascend",
    )


def test_environment_check_rejects_missing_entry_point(monkeypatch):
    _patch_complete_environment(monkeypatch)
    monkeypatch.setattr(
        launcher,
        "_discover_general_plugins",
        lambda: (),
    )

    report = launcher.inspect_environment({})

    assert not report.ok
    assert "missing vllm.general_plugins entry point: moe_offload_ascend" in (
        report.errors
    )


def test_environment_check_rejects_missing_acl_python_binding(monkeypatch):
    _patch_complete_environment(monkeypatch)
    original = launcher._module_origin
    monkeypatch.setattr(
        launcher,
        "_module_origin",
        lambda name: None if name == "acl" else original(name),
    )

    report = launcher.inspect_environment({})

    assert not report.ok
    assert "required module is not importable: acl" in report.errors


def test_environment_check_rejects_incorrect_entry_point_value(monkeypatch):
    _patch_complete_environment(monkeypatch)
    monkeypatch.setattr(
        launcher,
        "_discover_platform_plugins",
        lambda: (("ascend", "wrong.module:register"),),
    )

    report = launcher.inspect_environment({})

    assert not report.ok
    assert any("incorrect vllm.platform_plugins entry point ascend" in error for error in report.errors)


def test_environment_check_rejects_wrong_seam_commit(monkeypatch):
    _patch_complete_environment(monkeypatch)
    original = launcher._git_checkout_for_origin

    def wrong_seam(origin, *, clean_paths=()):
        checkout = original(origin, clean_paths=clean_paths)
        if checkout is not None and checkout.root == "/src/seam":
            return launcher.GitCheckout(
                root=checkout.root,
                commit="wrong-seam",
                branch=checkout.branch,
                dirty_paths=(),
            )
        return checkout

    monkeypatch.setattr(launcher, "_git_checkout_for_origin", wrong_seam)

    report = launcher.inspect_environment({})

    assert not report.ok
    assert "seam commit mismatch: expected seam-commit, got wrong-seam" in report.errors


def test_environment_check_reports_dirty_runtime_and_strict_rejects_it(monkeypatch):
    _patch_complete_environment(monkeypatch)
    original = launcher._git_checkout_for_origin

    def dirty_runtime(origin, *, clean_paths=()):
        checkout = original(origin, clean_paths=clean_paths)
        if checkout is not None and checkout.root == "/src/plugin":
            return launcher.GitCheckout(
                root=checkout.root,
                commit=checkout.commit,
                branch=checkout.branch,
                dirty_paths=("vllm_moe_offload_ascend/runtime.py",),
            )
        return checkout

    monkeypatch.setattr(launcher, "_git_checkout_for_origin", dirty_runtime)

    development = launcher.inspect_environment({})
    strict = launcher.inspect_environment({}, strict=True)

    assert development.ok
    assert development.warnings
    assert not strict.ok
    assert any("runtime checkout has relevant dirty paths" in error for error in strict.errors)


def test_environment_check_rejects_incompatible_stack_version(monkeypatch):
    _patch_complete_environment(monkeypatch)
    monkeypatch.setattr(
        launcher,
        "_distribution_version",
        lambda name: "2.9.0" if name == "torch-npu" else {
            "vllm": "0.21.0+empty",
            "torch": "2.10.0+cpu",
        }.get(name),
    )

    report = launcher.inspect_environment({})

    assert not report.ok
    assert any("incompatible base runtime" in error for error in report.errors)


def test_bundled_compatibility_lock_declares_current_seam_contract():
    lock_path = Path(launcher.__file__).with_name("compatibility.lock")

    lock = launcher._read_key_value_file(lock_path)

    assert lock["seam_commit"] == "4806367eeeb7d62b32078ae90cd929cc06d825fe"
    assert lock["seam_abi"] == "1"
    assert lock["vllm_commit"] == "ad7125a431e176d4161099480a66f0169609a690"
    assert lock["vllm_tag"] == "v0.21.0"
    assert lock["vllm_version"] == "0.21.0"
    assert lock["runtime_profile.issue7"].startswith(
        "2.10.0|2.10.0.post2|9.0.1|"
    )
    assert lock["runtime_profile.cann9_legacy"].startswith(
        "2.10.0|2.10.0|9.0.0|"
    )


def test_check_json_reports_result(monkeypatch, capsys):
    _patch_complete_environment(monkeypatch)

    assert launcher.main(["check", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["general_plugins"][0][0] == "moe_offload_ascend"
    assert payload["seam_checkout"]["commit"] == "seam-commit"
    assert payload["runtime_profile"] == "default"


def test_launcher_delegates_to_vllm_after_check(monkeypatch):
    _patch_complete_environment(monkeypatch)
    delegated = []
    monkeypatch.setattr(
        launcher,
        "_run_vllm_cli",
        lambda argv: delegated.append(list(argv)) or 0,
    )

    assert launcher.main(["serve", "/models/qwen"]) == 0
    assert delegated == [["serve", "/models/qwen", "--no-enable-prefix-caching"]]


def test_launcher_rejects_prefix_cache_opt_in(monkeypatch, capsys):
    _patch_complete_environment(monkeypatch)
    delegated = []
    monkeypatch.setattr(
        launcher,
        "_run_vllm_cli",
        lambda argv: delegated.append(list(argv)) or 0,
    )

    assert launcher.main(["serve", "/models/qwen", "--enable-prefix-caching"]) == 2
    assert delegated == []
    assert "does not support prefix caching" in capsys.readouterr().err


def test_launcher_preserves_explicit_prefix_cache_disable(monkeypatch):
    _patch_complete_environment(monkeypatch)
    delegated = []
    monkeypatch.setattr(
        launcher,
        "_run_vllm_cli",
        lambda argv: delegated.append(list(argv)) or 0,
    )

    assert launcher.main(["serve", "/models/qwen", "--no-enable-prefix-caching"]) == 0
    assert delegated == [["serve", "/models/qwen", "--no-enable-prefix-caching"]]


def test_launcher_does_not_add_serve_flags_to_other_commands(monkeypatch):
    _patch_complete_environment(monkeypatch)
    delegated = []
    monkeypatch.setattr(
        launcher,
        "_run_vllm_cli",
        lambda argv: delegated.append(list(argv)) or 0,
    )

    assert launcher.main(["--help"]) == 0
    assert launcher.main(["bench", "latency", "--help"]) == 0
    assert delegated == [["--help"], ["bench", "latency", "--help"]]


def test_vllm_cli_retry_registers_after_engine_args_import(monkeypatch):
    import vllm_moe_offload_ascend as plugin

    events = []
    monkeypatch.setattr(plugin, "register", lambda: events.append("register"))
    monkeypatch.setattr(
        "vllm.entrypoints.cli.main.main",
        lambda: events.append("main") or 0,
    )

    assert launcher._run_vllm_cli(["serve", "/models/qwen"]) == 0
    assert events[-2:] == ["register", "main"]


def test_launcher_fails_closed_before_delegation(monkeypatch):
    _patch_complete_environment(monkeypatch)
    monkeypatch.setattr(launcher, "_module_origin", lambda _name: None)
    delegated = []
    monkeypatch.setattr(
        launcher,
        "_run_vllm_cli",
        lambda argv: delegated.append(list(argv)) or 0,
    )

    assert launcher.main(["serve", "/models/qwen"]) == 1
    assert delegated == []


def test_check_strict_forwards_strict_policy(monkeypatch, capsys):
    _patch_complete_environment(monkeypatch)

    assert launcher.main(["check", "--strict", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["strict"] is True
