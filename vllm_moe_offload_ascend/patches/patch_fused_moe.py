#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Install LatchMoE implementations on vLLM-Ascend's explicit hook seam."""

from __future__ import annotations

import ctypes
import importlib
import inspect
import os
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any


def _ascend_device_op_is_initializing() -> bool:
    """Return whether vLLM-Ascend is in its device-op import cycle."""

    module = sys.modules.get("vllm_ascend.device.device_op")
    return module is not None and not hasattr(module, "DeviceOperator")


def _register_plugin_ops() -> None:
    """Import the plugin-owned custom ops without aliasing host namespaces."""
    import importlib

    for name in (
        "moe_offload_stage_op",
        "moe_router_op",
        "moe_mlp_op",
        "moe_seam_inject",
    ):
        importlib.import_module(
            f"vllm_moe_offload_ascend.ops.fused_moe.{name}"
        )


# ---------------------------------------------------------------------------
# Step 2 – module-global rebind for already-imported modules
# ---------------------------------------------------------------------------

def _apply_env_defaults_from_gb() -> None:
    """Eagerly apply MoE offload env defaults in every process.

    With VLLM_WORKER_MULTIPROC_METHOD=spawn, worker subprocesses inherit
    os.environ at spawn time.  The _patch_engine_args_autoconfig path runs
    apply_moe_offload_defaults *after* workers are spawned, so the workers
    never see VLLM_ASCEND_MOE_OFFLOAD_ENABLED=1 etc.  Re-derive them here
    from VLLM_ASCEND_MOE_OFFLOAD_GB which *is* inherited (set during CLI
    arg parsing, before spawn).
    """
    from vllm_moe_offload_ascend.moe_offload.autoconfig import (
        get_moe_offload_gb,
        _DEFAULT_ENV_VARS,
        _FANOUT_THRESHOLD_ENV,
        _NUM_SLOTS_BOOTSTRAP_ENV,
        _PREFILL_RESIDENCY_PROFILE_ENV,
        _NUM_SLOTS_ENV,
        _RESIDENT_LAYER_IDS_ENV,
        _SEW_DATAPLANE_ENV_VARS,
        apply_profile_guided_residency,
        _sew_dataplane_selected,
        derive_num_slots_defaults,
        derive_prefetch_defaults,
    )
    import os
    gb = get_moe_offload_gb()
    if gb <= 0:
        return
    sew_dataplane = _sew_dataplane_selected()
    explicit_num_slots = _NUM_SLOTS_ENV in os.environ
    explicit_fanout_threshold = _FANOUT_THRESHOLD_ENV in os.environ
    for env_name, value in _DEFAULT_ENV_VARS.items():
        if sew_dataplane and env_name in _SEW_DATAPLANE_ENV_VARS:
            continue
        os.environ.setdefault(env_name, value)
    if sew_dataplane:
        for env_name, value in _SEW_DATAPLANE_ENV_VARS.items():
            os.environ.setdefault(env_name, value)
    try:
        plan = derive_prefetch_defaults(gb)
        if sew_dataplane:
            plan = apply_profile_guided_residency(
                plan,
                profile_path=os.getenv(_PREFILL_RESIDENCY_PROFILE_ENV),
            )
        if _RESIDENT_LAYER_IDS_ENV not in os.environ:
            os.environ[_RESIDENT_LAYER_IDS_ENV] = ",".join(
                str(lid) for lid in plan["resident_layer_ids"]
            )
        slot_defaults = derive_num_slots_defaults(gb, None, plan, None)
        if not explicit_num_slots:
            os.environ[_NUM_SLOTS_ENV] = str(slot_defaults["num_slots"])
            os.environ[_NUM_SLOTS_BOOTSTRAP_ENV] = "1"
        if not explicit_fanout_threshold:
            os.environ[_FANOUT_THRESHOLD_ENV] = os.environ.get(
                _NUM_SLOTS_ENV,
                str(slot_defaults["fanout_threshold"]),
            )
    except Exception:
        pass


def _current_forward_phase() -> str | None:
    """Return prefill/decode/mixed from authoritative vLLM metadata."""

    try:
        from vllm.forward_context import (
            get_forward_context,
            is_forward_context_available,
        )

        if not is_forward_context_available():
            return None
        attn_metadata = get_forward_context().attn_metadata
    except Exception:
        return None

    if isinstance(attn_metadata, list):
        metas = []
        for item in attn_metadata:
            if isinstance(item, dict):
                metas.extend(item.values())
            elif item is not None:
                metas.append(item)
    elif isinstance(attn_metadata, dict):
        metas = list(attn_metadata.values())
    elif attn_metadata is None:
        metas = []
    else:
        metas = [attn_metadata]

    has_prefill = False
    has_decode = False
    for meta in metas:
        try:
            if int(getattr(meta, "num_prefills", 0) or 0) > 0:
                has_prefill = True
        except Exception:
            continue
    for meta in metas:
        try:
            if int(getattr(meta, "num_decodes", 0) or 0) > 0:
                has_decode = True
        except Exception:
            continue
    if has_prefill and has_decode:
        return "mixed"
    if has_prefill:
        return "prefill"
    if has_decode:
        return "decode"
    return None


def _is_torch_compile_tracing() -> bool:
    """Return tracing state without creating a Dynamo-unsafe closure."""

    try:
        import torch

        compiler = getattr(torch, "compiler", None)
        if compiler is not None and hasattr(compiler, "is_compiling"):
            return bool(compiler.is_compiling())
    except Exception:
        pass
    try:
        import torch

        return bool(torch._dynamo.is_compiling())
    except Exception:
        return False


def _is_current_graph_capturing() -> bool:
    """Read Ascend graph-capture state without a Dynamo-local closure."""

    try:
        from vllm_moe_offload_ascend.moe_offload.runtime import (
            _is_current_graph_capturing as runtime_is_current_graph_capturing,
        )

        return bool(runtime_is_current_graph_capturing())
    except Exception:
        return False


def _is_moe_offload_profile_run_active() -> bool:
    """Return the explicit vLLM profile/dummy-run marker, if installed."""

    try:
        from vllm_moe_offload_ascend.ops.fused_moe.moe_offload_stage_op import (
            is_moe_offload_profile_run_active,
        )

        return bool(is_moe_offload_profile_run_active())
    except Exception:
        return False


def _is_moe_offload_graph_warmup_active() -> bool:
    """Return the scoped graph warm-up marker, if the runner patch is installed."""

    try:
        from vllm_moe_offload_ascend.ops.fused_moe.moe_offload_stage_op import (
            is_moe_offload_graph_warmup_active,
        )

        return bool(is_moe_offload_graph_warmup_active())
    except Exception:
        return False


def _current_forward_is_prefill() -> bool | None:
    """Backward-compatible bool view; mixed contains prefill work."""
    phase = _current_forward_phase()
    if phase in {"prefill", "mixed"}:
        return True
    if phase == "decode":
        return False
    return None


def _infer_forward_is_prefill_from_tokens(num_tokens: int | None) -> bool | None:
    """Fallback phase inference for vLLM profile/dummy runs.

    Real serving forwards usually expose num_prefills/num_decodes through
    attention metadata. vLLM's profile_run may skip attention metadata while
    still running a large prompt-shaped dummy batch. In that case, a batch with
    more tokens than max_num_seqs cannot be a pure one-token decode batch.
    """

    phase = _current_forward_is_prefill()
    if phase is not None:
        return bool(phase)
    fallback_tokens = num_tokens
    try:
        from vllm.forward_context import (
            get_forward_context,
            is_forward_context_available,
        )

        if is_forward_context_available():
            ctx = get_forward_context()
            ctx_tokens = getattr(ctx, "additional_kwargs", {}).get("num_tokens")
            if ctx_tokens is not None:
                fallback_tokens = int(ctx_tokens)
    except Exception:
        pass
    if fallback_tokens is None:
        return None
    try:
        from vllm.config import get_current_vllm_config_or_none

        vllm_config = get_current_vllm_config_or_none()
        if vllm_config is None:
            return None
        max_num_seqs = int(vllm_config.scheduler_config.max_num_seqs)
    except Exception:
        return None
    try:
        return bool(int(fallback_tokens) > max_num_seqs)
    except Exception:
        return None


def _infer_forward_phase_int(num_tokens: int | None) -> int:
    """Map the tri-state prefill inference to the moe_offload_stage phase int.

    Threads the full tri-state through to the staging op instead of collapsing
    ``None`` (phase unknown, e.g. profile/dummy run) down to ``False`` (decode).
    The previous ``bool(...)`` collapse made the op unable to distinguish a real
    decode from an unknown phase, so its overflow-deferral path matched genuine
    decode batches and left stale log2phy in place.  Returns one of the
    moe_offload_stage_op.PHASE_* constants.
    """
    from vllm_moe_offload_ascend.ops.fused_moe.moe_offload_stage_op import (
        PHASE_DECODE,
        PHASE_MIXED,
        PHASE_PREFILL,
        PHASE_UNKNOWN,
    )

    authoritative = _current_forward_phase()
    if authoritative == "mixed":
        return PHASE_MIXED
    if authoritative == "prefill":
        return PHASE_PREFILL
    if authoritative == "decode":
        return PHASE_DECODE
    phase = _infer_forward_is_prefill_from_tokens(num_tokens)
    if phase is True:
        return PHASE_PREFILL
    if phase is False:
        return PHASE_DECODE
    return PHASE_UNKNOWN


def _moe_offload_kv_backstop_active() -> bool:
    """Return true when KV-capacity errors should mention MoE offload slots."""

    try:
        return (
            _to_bool_env("VLLM_ASCEND_MOE_OFFLOAD_ENABLED", "0")
            or float(os.getenv("VLLM_ASCEND_MOE_OFFLOAD_GB", "0") or "0") > 0
            or int(os.getenv("VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS", "0") or "0") > 0
        )
    except (TypeError, ValueError):
        return _to_bool_env("VLLM_ASCEND_MOE_OFFLOAD_ENABLED", "0")


def _moe_offload_kv_backstop_hint(
    *,
    max_model_len: int | None = None,
    max_concurrency: float | None = None,
) -> str:
    parts = [
        "MoE offload KV-capacity backstop: the resolved KV cache cannot hold one full request.",
        "For vllm-moe-offload-ascend, reduce `VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS` or `VLLM_ASCEND_MOE_OFFLOAD_GB`,",
        "or reduce `max_model_len`/`max_num_seqs`; then rerun AutoConfig.",
    ]
    details: list[str] = []
    gb = os.getenv("VLLM_ASCEND_MOE_OFFLOAD_GB")
    slots = os.getenv("VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS")
    if gb:
        details.append(f"offload_gb={gb}")
    if slots:
        details.append(f"num_slots={slots}")
    if max_model_len is not None:
        details.append(f"max_model_len={int(max_model_len)}")
    if max_concurrency is not None:
        details.append(f"max_concurrency={float(max_concurrency):.3f}x")
    if details:
        parts.append("(" + ", ".join(details) + ".)")
    return " ".join(parts)


def _patch_kv_cache_capacity_backstop() -> None:
    """Add MoE-slot-aware fail-fast errors around vLLM's KV planning path."""

    try:
        import vllm.v1.core.kv_cache_utils as _kv_utils
    except Exception:
        return

    patch_tag = "vllm_moe_offload_ascend.kv_capacity_backstop"

    current_check = getattr(_kv_utils, "_check_enough_kv_cache_memory", None)
    if callable(current_check) and getattr(current_check, "_ascend_moe_offload_patch_tag", None) != patch_tag:
        original_check = current_check

        def _check_enough_kv_cache_memory(*args, **kwargs):
            try:
                return original_check(*args, **kwargs)
            except ValueError as exc:
                if not _moe_offload_kv_backstop_active():
                    raise
                max_model_len = kwargs.get("max_model_len")
                if max_model_len is None and len(args) >= 3:
                    max_model_len = args[2]
                hint = _moe_offload_kv_backstop_hint(max_model_len=max_model_len)
                raise ValueError(f"{exc}\n{hint}") from exc

        _check_enough_kv_cache_memory._ascend_moe_offload_patch_tag = patch_tag
        _check_enough_kv_cache_memory.__wrapped__ = original_check
        _kv_utils._check_enough_kv_cache_memory = _check_enough_kv_cache_memory

    current_report = getattr(_kv_utils, "_report_kv_cache_config", None)
    if callable(current_report) and getattr(current_report, "_ascend_moe_offload_patch_tag", None) != patch_tag:
        original_report = current_report

        def _report_kv_cache_config(vllm_config, kv_cache_config):
            if _moe_offload_kv_backstop_active():
                try:
                    max_model_len = int(vllm_config.model_config.max_model_len)
                    max_concurrency = float(
                        _kv_utils.get_max_concurrency_for_kv_cache_config(
                            vllm_config, kv_cache_config
                        )
                    )
                except Exception:
                    max_model_len = None
                    max_concurrency = None
                if max_concurrency is not None and max_concurrency < 1.0:
                    hint = _moe_offload_kv_backstop_hint(
                        max_model_len=max_model_len,
                        max_concurrency=max_concurrency,
                    )
                    raise ValueError(hint)
            return original_report(vllm_config, kv_cache_config)

        _report_kv_cache_config._ascend_moe_offload_patch_tag = patch_tag
        _report_kv_cache_config.__wrapped__ = original_report
        _kv_utils._report_kv_cache_config = _report_kv_cache_config


def _opapi_supports_add_rms_norm_bias() -> bool:
    """Return whether the active CANN runtime exports the fused RMSNorm ABI."""
    try:
        opapi = ctypes.CDLL("libopapi.so")
    except OSError:
        return False
    return all(
        hasattr(opapi, symbol)
        for symbol in (
            "aclnnAddRmsNormBias",
            "aclnnAddRmsNormBiasGetWorkspaceSize",
        )
    )


def _patch_rms_norm_quant_fusion_cann_compat() -> bool:
    """Skip only fusion patterns that require the unavailable custom op."""
    try:
        import importlib

        fusion_pass = importlib.import_module(
            "vllm_ascend.compilation.passes.norm_quant_fusion_pass"
        )
    except Exception:
        return False

    original = getattr(fusion_pass, "enable_custom_op", None)
    if not callable(original):
        return False
    if getattr(original, "_latchmoe_cann_rmsnorm_compat", False):
        return False

    def _disable_unsupported_rmsnorm_patterns() -> bool:
        return False

    _disable_unsupported_rmsnorm_patterns._latchmoe_cann_rmsnorm_compat = True
    _disable_unsupported_rmsnorm_patterns.__wrapped__ = original
    fusion_pass.enable_custom_op = _disable_unsupported_rmsnorm_patterns
    return True


def _patch_rms_norm_bias_cann_compat() -> None:
    """Use the supported torch_npu RMSNorm path on older CANN runtimes.

    This affects only vLLM-Ascend's residual RMSNorm custom op. It deliberately
    leaves the LatchMoE router -> stage -> MLP custom-op seam enabled.
    """
    if _opapi_supports_add_rms_norm_bias():
        return
    changed = _patch_rms_norm_quant_fusion_cann_compat()
    try:
        import importlib

        layernorm = importlib.import_module("vllm_ascend.ops.layernorm")
    except Exception:
        layernorm = None

    for class_name, is_gemma in (
        ("AscendRMSNorm", False),
        ("AscendGemmaRMSNorm", True),
    ):
        cls = getattr(layernorm, class_name, None)
        original = getattr(cls, "forward_oot", None)
        if (
            cls is None
            or not callable(original)
            or getattr(original, "_latchmoe_cann_rmsnorm_compat", False)
        ):
            continue

        def _forward_oot(
            self,
            x,
            residual=None,
            *,
            _original=original,
            _is_gemma=is_gemma,
        ):
            if residual is None:
                return _original(self, x, residual)

            import torch
            import torch_npu

            residual = torch.ops.vllm.maybe_chunk_residual(x, residual)
            weight = 1.0 + self.weight if _is_gemma else self.weight
            x, _, residual = torch_npu.npu_add_rms_norm(
                x,
                residual,
                weight,
                self.variance_epsilon,
            )
            if not _is_gemma and self.bias is not None:
                x.add_(self.bias)
            return x, residual

        _forward_oot._latchmoe_cann_rmsnorm_compat = True
        _forward_oot.__wrapped__ = original
        cls.forward_oot = _forward_oot
        changed = True

    if changed:
        print(
            "LATCHMOE_CANN_COMPAT rmsnorm_bias=fallback "
            "reason=missing_aclnnAddRmsNormBias",
            flush=True,
        )


def _install_cann_compat_when_ready() -> bool:
    """Install only backend compatibility fixes after Ascend op registration."""

    if _ascend_device_op_is_initializing():
        return False

    import importlib

    importlib.import_module("vllm_ascend.ops.register_custom_ops")
    _patch_rms_norm_bias_cann_compat()
    return True


def _patch_cann_compat_adapt_retry() -> None:
    """Retry compat-only installation at the worker's stable adapt boundary."""

    try:
        import vllm_ascend.utils as _utils
    except Exception:
        return

    current = getattr(_utils, "adapt_patch", None)
    if current is None or getattr(current, "_latchmoe_cann_compat_retry", False):
        return

    def adapt_patch(*args, **kwargs):
        result = current(*args, **kwargs)
        _install_cann_compat_when_ready()
        return result

    adapt_patch._latchmoe_cann_compat_retry = True
    adapt_patch.__wrapped__ = current
    _utils.adapt_patch = adapt_patch


def _patch_cann_compat_config_retry() -> None:
    """Retry compat-only installation at the final platform config boundary."""

    try:
        import vllm_ascend.platform as _platform
    except Exception:
        return

    current = _platform.NPUPlatform.check_and_update_config
    current_fn = getattr(current, "__func__", current)
    if getattr(current_fn, "_latchmoe_cann_compat_retry", False):
        return
    original_fn = current_fn

    @classmethod
    def _patched(cls, vllm_config):
        original_fn(cls, vllm_config)
        _install_cann_compat_when_ready()

    _patched.__func__._latchmoe_cann_compat_retry = True
    _patched.__func__.__wrapped__ = original_fn
    _platform.NPUPlatform.check_and_update_config = _patched


def apply_cann_compat_patches() -> None:
    """Install CANN compatibility only, without any MoE offload patches."""

    _patch_cann_compat_adapt_retry()
    _patch_cann_compat_config_retry()
    _install_cann_compat_when_ready()


def apply_patches() -> None:
    # 0. Eagerly write env defaults so spawned worker processes see them.
    _apply_env_defaults_from_gb()
    _bootstrap_gdn_custom_op_vendor_env()

    _patch_adapt_patch_reinstall()
    _install_runtime_patches_when_ready()

    # CLI arg registration and engine args autoconfig must always run.
    _patch_platform_autoconfig()
    _patch_platform_splitting_ops()
    _patch_engine_args_autoconfig()
    _patch_kv_cache_capacity_backstop()




def _install_runtime_patches_when_ready() -> bool:
    """Register plugin ops and runtime adapters after device-op initialization.

    General plugin discovery can call ``register()`` while
    ``vllm_ascend.device.device_op`` is still importing. Importing
    ``vllm_ascend.ops`` in that window leaves its package cached without the
    custom-op registrations that its own cycle guard skipped. NPUWorker calls
    ``adapt_patch()`` after the device module is complete, which retries this
    function at a stable point.
    """

    if _ascend_device_op_is_initializing():
        return False

    # Retry after the Ascend package has completed lazy extension discovery.
    # This is harmless when the platform bootstrap already configured it.
    _bootstrap_gdn_custom_op_vendor_env()

    # The parent ops package may already be cached from the guarded partial
    # import, so import the registration module explicitly before loading MoE
    # modules that reference torch.ops.vllm custom ops.
    import importlib

    importlib.import_module("vllm_ascend.ops.register_custom_ops")
    _patch_rms_norm_bias_cann_compat()
    _register_plugin_ops()
    _install_runtime_module_patches()
    return True


def _patch_adapt_patch_reinstall() -> None:
    """Reinstall plugin runtime hooks after vLLM-Ascend worker patches load.

    Under spawn, the EngineCore process imports plugins while vLLM modules may
    still be partially initialized. Some fused-MoE imports are therefore best
    effort during register(). NPUWorker later calls vllm_ascend.utils.adapt_patch()
    after those imports settle; use that stable point to retry the runtime hooks.
    """
    try:
        import vllm_ascend.utils as _utils
    except Exception:
        return

    current = getattr(_utils, "adapt_patch", None)
    if current is None or getattr(current, "_ascend_moe_offload_reinstall_patch", False):
        return

    def adapt_patch(*args, **kwargs):
        result = current(*args, **kwargs)
        try:
            _install_runtime_patches_when_ready()
            _patch_kv_cache_capacity_backstop()
        except Exception as exc:
            if _to_bool_env("SEW_PATCH_PROBE", "0"):
                print(f"SEW_PATCH adapt_reinstall_failed: {exc!r}", flush=True)
        return result

    adapt_patch._ascend_moe_offload_reinstall_patch = True
    adapt_patch.__wrapped__ = current
    _utils.adapt_patch = adapt_patch


def _patch_model_runner_profile_context() -> None:
    """Mark only vLLM's ``_dummy_run(is_profile=True)`` as profile execution.

    Profile forwards often omit serving attention metadata, so phase inference
    alone cannot distinguish them from a short decode. vLLM and vLLM-Ascend
    runners carry the explicit ``is_profile`` argument; wrap that narrow
    boundary and scope the marker to the dummy forward.
    """

    runner_modules = (
        ("vllm.v1.worker.gpu_model_runner", "GPUModelRunner"),
        ("vllm.v1.worker.gpu.model_runner", "GPUModelRunner"),
        # NPUModelRunner overrides _dummy_run, so wrapping only vLLM's base
        # class misses the actual Ascend profile forward.
        ("vllm_ascend.worker.model_runner_v1", "NPUModelRunner"),
        ("vllm_ascend.worker.v2.model_runner", "NPUModelRunner"),
        ("vllm_ascend._310p.model_runner_310p", "NPUModelRunner310"),
    )
    for module_name, class_name in runner_modules:
        try:
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
            current = getattr(cls, "_dummy_run")
        except Exception:
            continue
        if not callable(current) or getattr(
            current,
            "_latchmoe_profile_context_patch",
            False,
        ):
            continue
        try:
            signature = inspect.signature(current)
        except (TypeError, ValueError):
            signature = None

        def _dummy_run_with_profile_context(
            self,
            *args,
            _original=current,
            _signature=signature,
            **kwargs,
        ):
            is_profile = bool(kwargs.get("is_profile", False))
            if not is_profile and _signature is not None:
                try:
                    bound = _signature.bind_partial(self, *args, **kwargs)
                    is_profile = bool(bound.arguments.get("is_profile", False))
                except TypeError:
                    pass
            if not is_profile:
                return _original(self, *args, **kwargs)

            from vllm_moe_offload_ascend.ops.fused_moe.moe_offload_stage_op import (
                reset_moe_offload_profile_run_active,
                set_moe_offload_profile_run_active,
            )

            token = set_moe_offload_profile_run_active(True)
            try:
                return _original(self, *args, **kwargs)
            finally:
                reset_moe_offload_profile_run_active(token)

        _dummy_run_with_profile_context._latchmoe_profile_context_patch = True
        _dummy_run_with_profile_context.__wrapped__ = current
        cls._dummy_run = _dummy_run_with_profile_context


def _patch_model_runner_graph_warmup_context() -> None:
    """Mark graph warm-up without changing serving or profile-run semantics.

    vLLM invokes ``_dummy_run(is_profile=False)`` while ``capture_model`` builds
    PIECEWISE graphs. The stage seam has no serving phase metadata there, so an
    overflowing active union previously took the decode-safe fixed-slot path and
    failed before capture. The context is scoped to ``capture_model`` only; the
    stage op still sees actual stream capture and remains a no-op at that point.
    """

    runner_modules = (
        ("vllm.v1.worker.gpu_model_runner", "GPUModelRunner"),
        ("vllm.v1.worker.gpu.model_runner", "GPUModelRunner"),
        ("vllm_ascend.worker.model_runner_v1", "NPUModelRunner"),
        ("vllm_ascend.worker.v2.model_runner", "NPUModelRunner"),
        ("vllm_ascend._310p.model_runner_310p", "NPUModelRunner310"),
    )
    for module_name, class_name in runner_modules:
        try:
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
            current = getattr(cls, "capture_model")
        except Exception:
            continue
        if not callable(current) or getattr(
            current,
            "_latchmoe_graph_warmup_context_patch",
            False,
        ):
            continue

        def _capture_model_with_graph_warmup_context(
            self,
            *args,
            _original=current,
            **kwargs,
        ):
            from vllm_moe_offload_ascend.ops.fused_moe.moe_offload_stage_op import (
                reset_moe_offload_graph_warmup_active,
                set_moe_offload_graph_warmup_active,
            )

            token = set_moe_offload_graph_warmup_active(True)
            try:
                return _original(self, *args, **kwargs)
            finally:
                reset_moe_offload_graph_warmup_active(token)

        _capture_model_with_graph_warmup_context._latchmoe_graph_warmup_context_patch = True
        _capture_model_with_graph_warmup_context.__wrapped__ = current
        cls.capture_model = _capture_model_with_graph_warmup_context


_GDN_CAUSAL_CONV1D_OP_NAME = "npu_causal_conv1d_custom"
_GDN_CAUSAL_CONV1D_TENSOR_ARGS = (
    "query_start_loc_opt",
    "cache_indices_opt",
    "initial_state_mode_opt",
    "num_accepted_tokens_opt",
)
_GDN_CUSTOM_OP_VENDOR_ENV = "ASCEND_CUSTOM_OPP_PATH"
_GDN_CUSTOM_OP_VENDOR_RELATIVE_PATH = Path(
    "_cann_ops_custom/vendors/custom_transformer"
)
_GDN_CUSTOM_OP_VENDOR_LIBRARY = Path("op_api/lib/libcust_opapi.so")


def _gdn_extension_custom_op_vendor() -> Path | None:
    """Find custom-op assets adjacent to the registered Ascend extension.

    Some deployment overlays keep Python sources in one editable checkout and
    the compiled ``vllm_ascend_C`` extension in another. vLLM-Ascend's normal
    bootstrap derives the vendor directory only from the source module, which
    leaves its custom CausalConv1d symbols undiscoverable in that layout.
    The extension is the authoritative owner of the matching op binary.
    """

    try:
        spec = importlib.util.find_spec("vllm_ascend.vllm_ascend_C")
    except (ImportError, AttributeError, ValueError):
        return None
    origin = getattr(spec, "origin", None)
    if not origin or origin in {"built-in", "frozen"}:
        return None
    vendor = Path(origin).resolve().parent / _GDN_CUSTOM_OP_VENDOR_RELATIVE_PATH
    if (vendor / _GDN_CUSTOM_OP_VENDOR_LIBRARY).is_file():
        return vendor
    return None


def _bootstrap_gdn_custom_op_vendor_env() -> bool:
    """Expose extension-owned CausalConv1d assets to the Ascend op resolver.

    Only ``ASCEND_CUSTOM_OPP_PATH`` is required: the custom-op resolver opens
    its vendor library from that directory. Avoiding an ``LD_LIBRARY_PATH``
    mutation keeps the process-wide loader precedence unchanged.
    """

    vendor = _gdn_extension_custom_op_vendor()
    if vendor is None:
        return False
    value = str(vendor)
    entries = [entry for entry in os.environ.get(_GDN_CUSTOM_OP_VENDOR_ENV, "").split(":") if entry]
    if value in entries:
        return False
    os.environ[_GDN_CUSTOM_OP_VENDOR_ENV] = ":".join((value, *entries))
    print("LATCHMOE_BACKEND_COMPAT gdn_custom_opp=extension_vendor", flush=True)
    return True


def _gdn_causal_conv1d_schema_uses_tensor_args(operator: object) -> bool:
    """Return whether a registered GDN conv1d op uses the newer Tensor ABI.

    The pinned Python frontend represents the four dynamic GDN arguments as
    host tuples, while newer custom-op binaries accept optional device Tensors.
    The schema is the only safe source of truth because both binary versions
    can be present in supported environments.
    """

    schema = getattr(operator, "_schema", None)
    if schema is None:
        default = getattr(operator, "default", None)
        schema = getattr(default, "_schema", None)
    if schema is None:
        return False
    normalized = "".join(str(schema).split())
    return all(
        f"Tensor?{argument}" in normalized
        for argument in _GDN_CAUSAL_CONV1D_TENSOR_ARGS
    )


def _gdn_causal_conv1d_tensor_arg(value: object, reference: object, torch_module: object) -> object:
    """Convert legacy host arguments to the Tensor ABI without touching tuples.

    The original tuples remain available to vLLM-Ascend's graph-update code.
    Conversion occurs immediately before dispatching the custom op, where the
    newer ABI requires device tensors. Empty legacy lists mean "not supplied"
    and map to ``None``, matching the newer frontend.
    """

    if value is None:
        return None
    tensor_type = getattr(torch_module, "Tensor")
    if isinstance(value, tensor_type):
        if int(value.numel()) == 0:
            return None
        return value.to(device=reference.device, dtype=torch_module.int64)
    if isinstance(value, (tuple, list)) and len(value) == 0:
        return None
    return torch_module.tensor(value, device=reference.device, dtype=torch_module.int64)


def _patch_gdn_causal_conv1d_tensor_abi(torch_module: object | None = None) -> bool:
    """Adapt only a detected Tensor-ABI GDN op to the pinned tuple frontend.

    This is intentionally an operator boundary adapter rather than a rewrite of
    ``vllm_ascend.ops.gdn``: graph bookkeeping keeps its tuple arithmetic and
    only the final call receives the representation required by the loaded
    binary. The patch is a no-op for the older ``int[]`` schema.
    """

    if torch_module is None:
        import torch as torch_module

    try:
        namespace = torch_module.ops._C_ascend
        original = getattr(namespace, _GDN_CAUSAL_CONV1D_OP_NAME)
    except (AttributeError, RuntimeError):
        return False
    if getattr(original, "_latchmoe_gdn_tensor_abi_patch", False):
        return True
    if not _gdn_causal_conv1d_schema_uses_tensor_args(original):
        return False

    def _tensor_abi_adapter(output, x, weight, *args, **kwargs):
        positional = list(args)
        # The three explicit positional arguments are followed by conv_state,
        # bias_opt, then the four ABI-sensitive arguments.
        for index, argument in enumerate(_GDN_CAUSAL_CONV1D_TENSOR_ARGS, start=2):
            if index < len(positional):
                positional[index] = _gdn_causal_conv1d_tensor_arg(
                    positional[index], x, torch_module
                )
        converted_kwargs = dict(kwargs)
        for argument in _GDN_CAUSAL_CONV1D_TENSOR_ARGS:
            if argument in converted_kwargs:
                converted_kwargs[argument] = _gdn_causal_conv1d_tensor_arg(
                    converted_kwargs[argument], x, torch_module
                )
        return original(output, x, weight, *positional, **converted_kwargs)

    _tensor_abi_adapter._latchmoe_gdn_tensor_abi_patch = True
    _tensor_abi_adapter.__wrapped__ = original
    setattr(namespace, _GDN_CAUSAL_CONV1D_OP_NAME, _tensor_abi_adapter)
    print("LATCHMOE_BACKEND_COMPAT gdn_causal_conv1d=tensor_abi", flush=True)
    return True


def _patch_gdn_causal_conv1d_tensor_abi_when_loaded() -> None:
    """Retry the ABI patch at the first GDN core call after lazy op loading."""

    try:
        gdn_module = importlib.import_module("vllm_ascend.ops.gdn")
        attention_class = getattr(gdn_module, "AscendGatedDeltaNetAttention")
        current = getattr(attention_class, "_forward_core")
    except Exception:
        return
    if getattr(current, "_latchmoe_gdn_tensor_abi_retry_patch", False):
        return
    original = current

    def _forward_core_with_tensor_abi(self, *args, **kwargs):
        _patch_gdn_causal_conv1d_tensor_abi()
        # Custom-op registration is complete by the first forward. Avoid a
        # per-layer compatibility probe after that point.
        attention_class._forward_core = original
        return original(self, *args, **kwargs)

    _forward_core_with_tensor_abi._latchmoe_gdn_tensor_abi_retry_patch = True
    _forward_core_with_tensor_abi.__wrapped__ = original
    attention_class._forward_core = _forward_core_with_tensor_abi


def _install_runtime_module_patches() -> None:
    from vllm_moe_offload_ascend.moe_offload.runtime import get_moe_offload_runtime, MoeOffloadDecisionPath
    from vllm_moe_offload_ascend.moe_offload.pipeline import get_moe_pipeline_profiler

    _patch_model_runner_profile_context()
    _patch_model_runner_graph_warmup_context()
    _patch_gdn_causal_conv1d_tensor_abi()
    _patch_gdn_causal_conv1d_tensor_abi_when_loaded()

    try:
        import vllm_ascend.ops.fused_moe.moe_comm_method as _comm
        _comm.get_moe_offload_runtime = get_moe_offload_runtime
        _comm.MoeOffloadDecisionPath = MoeOffloadDecisionPath
        _comm.get_moe_pipeline_profiler = get_moe_pipeline_profiler
        _patch_moe_comm_method_runtime_hooks(_comm)
    except Exception as exc:
        if _to_bool_env("SEW_PATCH_PROBE", "0"):
            print(f"SEW_PATCH comm_hook_failed: {exc!r}", flush=True)

    try:
        import vllm_ascend.ops.fused_moe.fused_moe as _fused_moe
        _fused_moe.get_moe_offload_runtime = get_moe_offload_runtime
        _patch_fused_moe_runtime_hooks(_fused_moe)
        if "vllm_ascend.ops.fused_moe.moe_comm_method" in sys.modules:
            _comm = sys.modules["vllm_ascend.ops.fused_moe.moe_comm_method"]
            if hasattr(_comm, "setup_moe_comm_method"):
                _fused_moe.setup_moe_comm_method = _comm.setup_moe_comm_method
    except Exception as exc:
        if _to_bool_env("SEW_PATCH_PROBE", "0"):
            print(f"SEW_PATCH fused_hook_failed: {exc!r}", flush=True)

    try:
        import vllm_ascend.ops.fused_moe.token_dispatcher as _td
        _td.get_moe_pipeline_profiler = get_moe_pipeline_profiler
    except Exception as exc:
        if _to_bool_env("SEW_PATCH_PROBE", "0"):
            print(f"SEW_PATCH token_dispatcher_hook_failed: {exc!r}", flush=True)

    _patch_kv_cache_capacity_backstop()
    _patch_aclgraph_replay_profile()


def _patch_aclgraph_replay_profile() -> None:
    """Sample graph replay issue latency without synchronizing the NPU stream."""

    try:
        from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
        from vllm_ascend.ascend_forward_context import get_forward_context
    except Exception:
        return
    current = ACLGraphWrapper.__call__
    if getattr(current, "_latchmoe_replay_profile_patch", False):
        return
    original = current

    def _profiled_call(self, *args, **kwargs):
        context = get_forward_context()
        runtime_mode = getattr(context, "cudagraph_runtime_mode", None)
        batch_descriptor = getattr(context, "batch_descriptor", None)
        entry = getattr(self, "concrete_aclgraph_entries", {}).get(batch_descriptor)
        is_replay = bool(
            runtime_mode is not None
            and runtime_mode == getattr(self, "runtime_mode", None)
            and entry is not None
            and getattr(entry, "aclgraph", None) is not None
        )
        if not is_replay:
            return original(self, *args, **kwargs)

        counter = int(getattr(self, "_latchmoe_replay_calls", 0)) + 1
        self._latchmoe_replay_calls = counter
        try:
            sample_rate = max(
                1,
                int(
                    os.getenv(
                        "VLLM_ASCEND_MOE_GRAPH_REPLAY_PROFILE_SAMPLE_RATE",
                        "1024",
                    )
                ),
            )
        except ValueError:
            sample_rate = 1024
        sampled = counter == 1 or counter % sample_rate == 0
        start = perf_counter() if sampled else 0.0
        output = original(self, *args, **kwargs)
        if sampled:
            from vllm_moe_offload_ascend.moe_offload.runtime import (
                get_moe_offload_runtime,
            )

            get_moe_offload_runtime()._record_profile_event(
                "graph_replay_issue",
                layer_id=None,
                start=start,
                payload={
                    "profile_sample_rate": 1 if counter == 1 else sample_rate,
                    "replay_call_index": counter,
                    "runtime_mode": str(runtime_mode),
                    "synchronizes_npu": False,
                },
            )
        return output

    _profiled_call._latchmoe_replay_profile_patch = True
    _profiled_call.__wrapped__ = original
    ACLGraphWrapper.__call__ = _profiled_call


def _to_bool_env(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


def _b2_reference_full_tokens_enabled(*, async_stage: bool) -> bool:
    enabled = _to_bool_env(
        "VLLM_ASCEND_MOE_OFFLOAD_B2_REFERENCE_FULL_TOKENS",
        "0",
    )
    if enabled and async_stage:
        raise RuntimeError(
            "VLLM_ASCEND_MOE_OFFLOAD_B2_REFERENCE_FULL_TOKENS=1 only "
            "supports synchronous B2 staging; set "
            "VLLM_ASCEND_MOE_OFFLOAD_ASYNC_LOAD=0"
        )
    return enabled


def _b2_reference_full_layer_enabled(*, async_stage: bool) -> bool:
    enabled = _to_bool_env(
        "VLLM_ASCEND_MOE_OFFLOAD_B2_REFERENCE_FULL_LAYER",
        "0",
    )
    # This path always performs its own blocking full-layer copy. The global
    # async setting still controls the ordinary fixed-slot decode path.
    _ = async_stage
    return enabled


def _b2_overflow_mode() -> str:
    """Return the execution mode for slot-capacity overflow.

    ``experimental_wave`` remains a compatibility alias for the now-qualified
    ``multi_wave`` mode.  ``full_layer`` stays available as an explicit mode
    and as the automatic fallback for recoverable wave qualification failures.
    """

    mode = os.getenv(
        "VLLM_ASCEND_MOE_OFFLOAD_B2_OVERFLOW_MODE",
        "multi_wave",
    ).strip().lower()
    if mode == "experimental_wave":
        mode = "multi_wave"
    if mode not in {"full_layer", "multi_wave"}:
        raise RuntimeError(
            "VLLM_ASCEND_MOE_OFFLOAD_B2_OVERFLOW_MODE must be "
            "'multi_wave' (legacy alias 'experimental_wave') or "
            "'full_layer', got "
            f"{mode!r}"
        )
    return mode


def _b2_multi_wave_preflight_failure(runtime: Any) -> str | None:
    """Return a safe fallback reason before any wave staging or compute."""

    if not _to_bool_env(
        "VLLM_ASCEND_MOE_OFFLOAD_B2_DIRECT_SCATTER",
        "1",
    ):
        return "native_recombine_disabled"
    if int(getattr(runtime.config, "num_slots", 0) or 0) <= 0:
        return "no_physical_slots"
    if (
        bool(getattr(runtime.config, "async_load", False))
        and int(getattr(runtime.config, "effective_prefill_prefetch_depth", 0)) > 0
        and int(getattr(runtime.config, "effective_prefill_buffer_count", 1)) < 2
    ):
        return "overlap_requires_two_buffers"
    return None




def _graph_capture_slot_weights(runtime: Any, *, layer_id: int):
    """Return graph-visible slot weights or reject an unsafe native fallback."""
    if not bool(getattr(runtime.config, "graph_compatible_offload", False)):
        return None
    if not runtime.should_use_fixed_slot_plan_for_layer(int(layer_id)):
        return None
    capture_weights = runtime.capture_safe_slot_weights(layer_id=int(layer_id))
    if capture_weights is None:
        raise RuntimeError(
            "graph-compatible MoE offload requires registered fixed-slot "
            f"weights for layer {int(layer_id)}; refusing to capture or replay "
            "with the native-weight fallback"
        )
    return capture_weights


def _unpack_mlp_apply_result(result: Any) -> tuple[Any, Any | None]:
    """Normalize old Tensor and new ``(Tensor, event)`` MoE MLP returns."""

    if not isinstance(result, tuple):
        return result, None
    if len(result) != 2:
        raise RuntimeError(
            "MoE _apply_mlp returned an unsupported tuple with "
            f"{len(result)} elements"
        )
    return result[0], result[1]


def _sum_profile_number(profile: dict[str, Any], key: str) -> float:
    try:
        return float(profile.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _sum_profile_int(profile: dict[str, Any], key: str) -> int:
    try:
        return int(float(profile.get(key, 0) or 0))
    except (TypeError, ValueError):
        return 0


def _estimate_b2_wave_h2d_bytes(
    runtime: Any,
    *,
    layer_id: int,
    wave: tuple[int, ...],
    readiness: dict[int, bool],
) -> int:
    miss_experts = tuple(
        int(expert_id)
        for expert_id in wave
        if not bool(readiness.get(int(expert_id), False))
    )
    cached_expert_bytes_fn = getattr(
        runtime,
        "cached_layer_expert_weight_bytes",
        None,
    )
    layer_expert_weight_bytes = (
        int(cached_expert_bytes_fn(layer_id=layer_id))
        if callable(cached_expert_bytes_fn)
        else 0
    )
    if layer_expert_weight_bytes > 0:
        return int(layer_expert_weight_bytes) * len(miss_experts)
    return int(
        sum(
            runtime.estimate_expert_weight_bytes(
                layer_id=layer_id,
                expert_id=int(expert_id),
            )
            for expert_id in miss_experts
        )
    )


def _b2_profile_details_enabled() -> bool:
    return _to_bool_env("VLLM_ASCEND_MOE_B2_PROFILE_DETAILS", "0")


def _attach_b2_profile_details(
    payload: dict[str, object],
    *,
    wave_plan,
    async_schedule,
    async_stage: bool,
    wave_profiles: list[dict[str, Any]],
) -> dict[str, object]:
    if _b2_profile_details_enabled():
        payload.update(
            {
                "wave_plan": wave_plan.to_jsonable(),
                "async_schedule": (
                    async_schedule.to_jsonable() if async_stage else None
                ),
                "waves": wave_profiles,
            }
        )
    else:
        payload["profile_details"] = "omitted"
    return payload


def _summarize_b2_wave_profiles(
    wave_profiles: list[dict[str, Any]],
    *,
    layer_scatter_ms: float = 0.0,
) -> dict[str, object]:
    """Layer-level B2 profile summary for Prefill bottleneck analysis."""

    summary: dict[str, object] = {
        "wave_count": int(len(wave_profiles)),
        "hit_only_waves": 0,
        "miss_only_waves": 0,
        "mixed_waves": 0,
        "main_slot_hit_waves": 0,
        "staged_waves": 0,
        "sync_slot_cache_waves": 0,
        "issued_before_compute_waves": 0,
        "issued_before_microbatch_materialize_waves": 0,
        "prefetch_before_compute_issues": 0,
        "prefetch_after_compute_issues": 0,
        "tokens": 0,
        "pairs": 0,
        "hits": 0,
        "misses": 0,
        "h2d_bytes": 0,
        "d2d_bytes": 0,
    }
    ms_keys = (
        "stage_ms",
        "stage_issue_ms",
        "stage_wait_ms",
        "mlp_ms",
        "pair_wave_ms",
        "microbatch_ms",
        "dispatch_ms",
        "build_mlp_input_ms",
        "gmm_ms",
        "combine_ms",
        "scatter_ms",
            "issue_start_to_compute_ms",
        "issue_end_to_compute_ms",
        "issue_end_to_microbatch_end_ms",
    )
    for key in ms_keys:
        summary[key] = 0.0
    max_issue_end_to_compute_ms = 0.0
    max_stage_wait_ms = 0.0

    for profile in wave_profiles:
        hits = _sum_profile_int(profile, "hits")
        misses = _sum_profile_int(profile, "misses")
        if misses == 0 and hits > 0:
            summary["hit_only_waves"] = int(summary["hit_only_waves"]) + 1
        elif hits == 0 and misses > 0:
            summary["miss_only_waves"] = int(summary["miss_only_waves"]) + 1
        elif hits > 0 and misses > 0:
            summary["mixed_waves"] = int(summary["mixed_waves"]) + 1

        stage_mode = str(profile.get("stage_mode", ""))
        if stage_mode == "main_slot_hit":
            summary["main_slot_hit_waves"] = int(summary["main_slot_hit_waves"]) + 1
        elif stage_mode == "sync_slot_cache":
            summary["sync_slot_cache_waves"] = int(summary["sync_slot_cache_waves"]) + 1
        elif stage_mode:
            summary["staged_waves"] = int(summary["staged_waves"]) + 1

        if bool(profile.get("issued_before_compute", False)):
            summary["issued_before_compute_waves"] = (
                int(summary["issued_before_compute_waves"]) + 1
            )
        if bool(profile.get("issued_before_microbatch_materialize", False)):
            summary["issued_before_microbatch_materialize_waves"] = (
                int(summary["issued_before_microbatch_materialize_waves"]) + 1
            )

        for key in (
            "tokens",
            "pairs",
            "hits",
            "misses",
            "h2d_bytes",
            "d2d_bytes",
        ):
            summary[key] = int(summary[key]) + _sum_profile_int(profile, key)
        summary["prefetch_before_compute_issues"] = (
            int(summary["prefetch_before_compute_issues"])
            + _sum_profile_int(profile, "prefetch_before_compute_count")
        )
        summary["prefetch_after_compute_issues"] = (
            int(summary["prefetch_after_compute_issues"])
            + _sum_profile_int(profile, "prefetch_after_compute_count")
        )
        for key in ms_keys:
            summary[key] = float(summary[key]) + _sum_profile_number(profile, key)
        max_issue_end_to_compute_ms = max(
            max_issue_end_to_compute_ms,
            _sum_profile_number(profile, "issue_end_to_compute_ms"),
        )
        max_stage_wait_ms = max(
            max_stage_wait_ms,
            _sum_profile_number(profile, "stage_wait_ms"),
        )

    for key in ms_keys:
        summary[key] = round(float(summary[key]), 3)
    summary["layer_scatter_ms"] = round(float(layer_scatter_ms), 3)
    summary["max_issue_end_to_compute_ms"] = round(max_issue_end_to_compute_ms, 3)
    summary["max_stage_wait_ms"] = round(max_stage_wait_ms, 3)
    return summary


def _patch_moe_comm_method_runtime_hooks(_comm: Any) -> None:
    cls = getattr(_comm, "MoECommMethod", None)
    if cls is None:
        return

    patch_tag = "vllm_moe_offload_ascend.moe_comm_method_runtime"
    current_maybe_apply = getattr(cls, "_maybe_apply_moe_offload_plan", None)
    current_fused_experts = getattr(cls, "fused_experts", None)
    current_patch_active = (
        getattr(current_maybe_apply, "_ascend_moe_offload_patch_tag", None)
        == patch_tag
        and getattr(current_fused_experts, "_ascend_moe_offload_patch_tag", None)
        == patch_tag
    )
    original_setup_moe_comm_method = getattr(_comm, "setup_moe_comm_method", None)
    if current_patch_active:
        if _to_bool_env("SEW_PATCH_PROBE", "0"):
            print("SEW_PATCH comm_hook_already_active", flush=True)
        if (
            callable(original_setup_moe_comm_method)
            and not getattr(
                original_setup_moe_comm_method,
                "_ascend_moe_offload_runtime_patch",
                False,
            )
        ):
            _wrap_setup_moe_comm_method(_comm, original_setup_moe_comm_method)
        return

    import os as _os

    import torch
    from time import perf_counter

    from vllm_moe_offload_ascend.moe_offload.runtime import (
        _is_current_graph_capturing,
        get_moe_offload_runtime,
    )

    def _resolve_runtime_contract(name: str) -> Any:
        contract = getattr(_comm, name, None)
        if contract is not None:
            return contract

        # Newer vLLM-Ascend keeps the typed payloads in moe_runtime_args and
        # no longer re-exports every contract from moe_comm_method.
        import importlib

        moe_runtime_args = importlib.import_module(
            "vllm_ascend.ops.fused_moe.moe_runtime_args"
        )

        return getattr(moe_runtime_args, name)

    MoEFusedExpertsInput = _resolve_runtime_contract("MoEFusedExpertsInput")
    MoEOffloadParams = _resolve_runtime_contract("MoEOffloadParams")
    MoERoutingParams = _resolve_runtime_contract("MoERoutingParams")
    MoEWeights = _resolve_runtime_contract("MoEWeights")
    FusedExpertsResult = _comm.FusedExpertsResult
    build_mlp_compute_input = _comm.build_mlp_compute_input
    build_token_dispatch_input = _comm.build_token_dispatch_input

    original_fused_experts = getattr(
        cls,
        "_ascend_moe_offload_original_fused_experts",
        None,
    )
    if original_fused_experts is None:
        original_fused_experts = cls.fused_experts
        cls._ascend_moe_offload_original_fused_experts = original_fused_experts

    original_maybe_apply = getattr(
        cls,
        "_ascend_moe_offload_original_maybe_apply",
        None,
    )
    if original_maybe_apply is None:
        original_maybe_apply = cls._maybe_apply_moe_offload_plan
        cls._ascend_moe_offload_original_maybe_apply = original_maybe_apply

    def _with_prepared_slot_weights(
        self,
        fused_experts_input,
        prepared_weights,
        *,
        offload_enabled=None,
        use_log2phy=True,
    ):
        compute_handle = get_moe_offload_runtime().begin_slot_compute(
            prepared_weights
        )
        setattr(self, "_ascend_moe_offload_slot_compute_handle", compute_handle)
        return MoEFusedExpertsInput(
            hidden_states=fused_experts_input.hidden_states,
            topk_weights=fused_experts_input.topk_weights,
            topk_ids=fused_experts_input.topk_ids,
            weights=MoEWeights(
                w1=prepared_weights.w1,
                w2=prepared_weights.w2,
                w1_bias=fused_experts_input.weights.w1_bias,
                w2_bias=fused_experts_input.weights.w2_bias,
                w1_scale=fused_experts_input.weights.w1_scale,
                w2_scale=fused_experts_input.weights.w2_scale,
                w1_scale_bias=fused_experts_input.weights.w1_scale_bias,
                w2_scale_bias=fused_experts_input.weights.w2_scale_bias,
                w1_offset=fused_experts_input.weights.w1_offset,
                w2_offset=fused_experts_input.weights.w2_offset,
            ),
            routing=MoERoutingParams(
                expert_map=fused_experts_input.routing.expert_map,
                global_redundant_expert_num=(
                    fused_experts_input.routing.global_redundant_expert_num
                ),
                mc2_mask=fused_experts_input.routing.mc2_mask,
                apply_router_weight_on_input=(
                    fused_experts_input.routing.apply_router_weight_on_input
                ),
                log2phy=(prepared_weights.log2phy if use_log2phy else None),
                physical_expert_count=prepared_weights.physical_expert_count,
                pertoken_scale=fused_experts_input.routing.pertoken_scale,
            ),
            quant=fused_experts_input.quant,
            activation=fused_experts_input.activation,
            need_trans=fused_experts_input.need_trans,
            dynamic_eplb=fused_experts_input.dynamic_eplb,
            swiglu_limit=fused_experts_input.swiglu_limit,
            offload=MoEOffloadParams(
                enabled=(
                    fused_experts_input.offload.enabled
                    if offload_enabled is None
                    else bool(offload_enabled)
                ),
                profile_only=fused_experts_input.offload.profile_only,
                layer_id=fused_experts_input.offload.layer_id,
                num_logical_experts=fused_experts_input.offload.num_logical_experts,
                expected_device_type=(
                    fused_experts_input.offload.expected_device_type
                ),
                step_id=fused_experts_input.offload.step_id,
            ),
        )

    def _maybe_apply_moe_offload_plan(self, fused_experts_input):
        offload = fused_experts_input.offload
        if offload is None or not offload.enabled:
            return fused_experts_input

        runtime = get_moe_offload_runtime()
        capture_weights = _graph_capture_slot_weights(
            runtime,
            layer_id=int(offload.layer_id),
        )
        if runtime.config.graph_compatible_offload:
            if _os.environ.get("SEW_OFFLOAD_PROBE"):
                buf = runtime.log2phy_buffer(offload.layer_id)
                print(
                    f"SEW_PROBE branch=GRAPH_COMPAT_SLOT layer={offload.layer_id} "
                    f"capturing={_is_current_graph_capturing()} "
                    f"capture_weights_none={capture_weights is None} "
                    f"buf_numel={None if buf is None else buf.numel()} "
                    f"buf_id={None if buf is None else id(buf)}",
                    flush=True,
                )
            if capture_weights is not None:
                return self._with_prepared_slot_weights(
                    fused_experts_input,
                    capture_weights,
                )

        return original_maybe_apply(self, fused_experts_input)

    def _maybe_run_b2_wave_prefill(self, fused_experts_input, before_dispatch_evt):
        from vllm_moe_offload_ascend.moe_offload.phase_split import (
            B2WaveFallback,
            count_routed_tokens_by_expert,
        )

        b2_control_start = perf_counter()
        offload = fused_experts_input.offload
        if offload is None or not offload.enabled:
            return None
        runtime = get_moe_offload_runtime()
        fused_layout_fn = getattr(runtime, "fused_shared_lane_layout", None)
        fused_shared_layout = (
            fused_layout_fn(int(offload.layer_id))
            if callable(fused_layout_fn)
            else None
        )
        profile_adaptive_wave = _is_moe_offload_profile_run_active()
        graph_warmup_adaptive_wave = _is_moe_offload_graph_warmup_active()
        adaptive_wave_context = profile_adaptive_wave or graph_warmup_adaptive_wave
        if not runtime.config.b2_wave_prefill and not adaptive_wave_context:
            return None
        if _is_current_graph_capturing():
            return None
        if self.token_dispatcher.__class__.__name__ != "TokenDispatcherWithAllGather":
            if _os.environ.get("SEW_B2_PROBE") or _os.environ.get("SEW_OFFLOAD_PROBE"):
                print(
                    f"SEW_B2 branch=SKIP reason=unsupported_dispatcher "
                    f"dispatcher={self.token_dispatcher.__class__.__name__}",
                    flush=True,
                )
            return None

        phase_is_prefill = _infer_forward_is_prefill_from_tokens(
            int(fused_experts_input.topk_ids.shape[0])
        )
        forward_phase = _current_forward_phase()
        if forward_phase is None:
            forward_phase = (
                "prefill" if phase_is_prefill is True
                else "decode" if phase_is_prefill is False
                else "unknown"
            )
        if runtime.is_resident_layer(int(offload.layer_id)):
            if _os.environ.get("SEW_B2_PROBE") or _os.environ.get("SEW_OFFLOAD_PROBE"):
                print(
                    f"SEW_B2 branch=SKIP reason=resident_prefill "
                    f"layer={offload.layer_id}",
                    flush=True,
                )
            return None
        if not runtime.should_use_fixed_slot_plan_for_layer(int(offload.layer_id)):
            return None
        route_stats_cache_enabled = (
            adaptive_wave_context
            or _to_bool_env("VLLM_ASCEND_MOE_OFFLOAD_ROUTE_STATS_CACHE", "0")
        )
        route_stats = (
            runtime.consume_prefill_route_stats_record(
                layer_id=int(offload.layer_id),
                topk_ids=fused_experts_input.topk_ids,
            )
            if route_stats_cache_enabled
            else None
        )
        route_stats_cache_hit = route_stats is not None
        max_num_seqs_hint = int(
            getattr(runtime.config, "max_num_seqs_hint", 0) or 0
        )
        unknown_prefill_candidate = (
            phase_is_prefill is None
            and max_num_seqs_hint > 0
            and int(fused_experts_input.topk_ids.shape[0]) > max_num_seqs_hint
        )
        if (
            phase_is_prefill is None
            and route_stats is None
            and not unknown_prefill_candidate
            and not adaptive_wave_context
        ):
            return None
        token_counts = (
            dict(route_stats.token_counts_by_expert)
            if route_stats is not None
            else None
        )
        token_count_start = perf_counter()
        if token_counts is None:
            token_counts = count_routed_tokens_by_expert(fused_experts_input.topk_ids)
            token_count_ms = (perf_counter() - token_count_start) * 1000.0
        else:
            token_count_ms = 0.0
        if fused_shared_layout is not None:
            # The backend appends always-active shared IDs to every token's
            # top-k. Capacity and B2 planning are routed-only: the shared suffix
            # has fixed storage and is executed exactly once in a separate step.
            routed_count = int(fused_shared_layout.routed_expert_count)
            token_counts = {
                int(expert_id): int(pair_count)
                for expert_id, pair_count in token_counts.items()
                if 0 <= int(expert_id) < routed_count
            }
        active_experts = tuple(sorted(token_counts))
        active_expert_count = len(active_experts)
        b2_phase_match = runtime.should_use_b2_pair_waves(
            layer_id=offload.layer_id,
            active_expert_count=active_expert_count,
        )
        b2_overflow_handoff = (
            (
                adaptive_wave_context
                or route_stats is not None
                or unknown_prefill_candidate
            )
            and active_expert_count > int(runtime.config.num_slots)
        )
        if not (b2_phase_match or b2_overflow_handoff):
            return None

        control_profile = {
            "b2_total_start": b2_control_start,
            "token_count_ms": token_count_ms,
            "route_stats_cache_hit": route_stats_cache_hit,
            "forward_phase": str(forward_phase),
            "profile_adaptive_wave": profile_adaptive_wave,
            "graph_warmup_adaptive_wave": graph_warmup_adaptive_wave,
            # Cached offsets refer to the full routed+shared matrix. The
            # routed B2 executor receives a narrower routed-only view, so
            # rebuilding offsets is required to preserve pair positions.
            "pair_offsets_by_expert": (
                None
                if fused_shared_layout is not None or route_stats is None
                else route_stats.pair_offsets_by_expert
            ),
        }
        try:
            if fused_shared_layout is not None:
                return self._run_b2_fused_shared_once(
                    fused_experts_input=fused_experts_input,
                    before_dispatch_evt=before_dispatch_evt,
                    token_counts=token_counts,
                    shared_layout=fused_shared_layout,
                    control_profile=control_profile,
                )
            return self._run_b2_wave_prefill(
                fused_experts_input=fused_experts_input,
                active_experts=active_experts,
                token_counts=token_counts,
                before_dispatch_evt=before_dispatch_evt,
                control_profile=control_profile,
            )
        except B2WaveFallback as exc:
            fallback_reason = f"{type(exc).__name__}: {exc}"
            if _os.environ.get("SEW_B2_PROBE") or _os.environ.get(
                "SEW_OFFLOAD_PROBE"
            ):
                print(
                    "SEW_B2 branch=FALLBACK_FULL_LAYER "
                    f"layer={offload.layer_id} reason={fallback_reason}",
                    flush=True,
                )
            return self._run_b2_full_layer_reference(
                fused_experts_input=fused_experts_input,
                before_dispatch_evt=before_dispatch_evt,
                b2_total_start=b2_control_start,
                forward_phase=str(forward_phase),
                route_stats_cache_hit=route_stats_cache_hit,
                fallback_reason=fallback_reason,
            )

    def _run_b2_fused_shared_once(
        self,
        *,
        fused_experts_input,
        before_dispatch_evt,
        token_counts,
        shared_layout,
        control_profile,
    ):
        """Run B2 for routed pairs and the fused shared suffix exactly once.

        ``mix_placement`` flattens routed and shared pairs into one backend
        input. Splitting that tensor naively would either count pinned shared
        IDs against dynamic slot capacity or repeat them in every routed wave.
        This method creates a routed-only B2 view and invokes the native fused
        kernel once for the immutable shared suffix, then adds the two outputs.
        """

        runtime = get_moe_offload_runtime()
        offload = fused_experts_input.offload
        shared_count = int(shared_layout.shared_expert_count)
        routed_count = int(shared_layout.routed_expert_count)
        total_count = int(shared_layout.total_logical_expert_count)
        topk_ids = fused_experts_input.topk_ids
        topk_weights = fused_experts_input.topk_weights
        if topk_ids.ndim != 2 or topk_weights.ndim != 2:
            raise RuntimeError(
                "fused shared B2 requires rank-2 top-k tensors; "
                f"ids={tuple(topk_ids.shape)}, weights={tuple(topk_weights.shape)}"
            )
        if int(topk_ids.shape[1]) <= shared_count:
            raise RuntimeError(
                "fused shared B2 top-k has no routed columns after removing "
                f"shared suffix: topk={int(topk_ids.shape[1])}, shared={shared_count}"
            )

        shared_ids = topk_ids[:, -shared_count:]
        shared_weights = topk_weights[:, -shared_count:]
        expected_shared_ids = torch.arange(
            routed_count,
            total_count,
            dtype=shared_ids.dtype,
            device=shared_ids.device,
        ).view(1, shared_count).expand_as(shared_ids)
        if not bool(torch.equal(shared_ids, expected_shared_ids)):
            raise RuntimeError(
                "fused shared B2 expected appended pinned shared IDs "
                f"[{routed_count}, {total_count}), but router suffix differs"
            )

        routed_input = replace(
            fused_experts_input,
            topk_ids=topk_ids[:, :-shared_count],
            topk_weights=topk_weights[:, :-shared_count],
        )
        routed_profile = dict(control_profile)
        routed_profile["pair_offsets_by_expert"] = None
        routed_result = self._run_b2_wave_prefill(
            fused_experts_input=routed_input,
            active_experts=tuple(sorted(int(expert_id) for expert_id in token_counts)),
            token_counts=token_counts,
            before_dispatch_evt=before_dispatch_evt,
            control_profile=routed_profile,
        )
        if routed_result is None:
            raise RuntimeError("fused shared B2 routed executor produced no result")

        pinned_weights = runtime.capture_safe_slot_weights(
            layer_id=int(offload.layer_id)
        )
        if pinned_weights is None:
            raise RuntimeError(
                "fused shared B2 requires the registered pinned shared lane; "
                f"layer={int(offload.layer_id)}"
            )
        pinned_weights.validate_backend_ready(
            expected_device_type=offload.expected_device_type
        )
        shared_input = replace(
            self._with_prepared_slot_weights(
                fused_experts_input,
                pinned_weights,
            ),
            topk_ids=shared_ids,
            topk_weights=shared_weights,
        )
        # Ascend's AllGather dispatcher derives its ``active_num`` from the
        # instance-level top_k rather than the input tensor width. The routed
        # path keeps that value at the model's routed top-k, while this native
        # call deliberately contains only the pinned shared suffix. Set it for
        # this call only, then restore it before the next routed wave.
        dispatcher = getattr(self, "token_dispatcher", None)
        if dispatcher is None or not hasattr(dispatcher, "top_k"):
            raise RuntimeError(
                "fused shared B2 requires a token dispatcher with a mutable top_k"
            )
        original_dispatch_top_k = int(dispatcher.top_k)
        dispatcher.top_k = shared_count
        try:
            shared_result = original_fused_experts(self, shared_input)
        finally:
            dispatcher.top_k = original_dispatch_top_k
        combined_out = routed_result.routed_out + shared_result.routed_out
        runtime._record_profile_event(
            "fused_shared_b2_once",
            layer_id=int(offload.layer_id),
            start=perf_counter(),
            payload={
                "routed_expert_count": routed_count,
                "pinned_shared_expert_count": shared_count,
                "shared_pairs": int(shared_ids.numel()),
                "shared_pair_execution_count": 1,
                "routed_pairs": int(routed_input.topk_ids.numel()),
            },
        )
        return replace(routed_result, routed_out=combined_out)

    def _run_b2_wave_prefill(
        self,
        *,
        fused_experts_input,
        active_experts,
        token_counts=None,
        before_dispatch_evt,
        control_profile=None,
    ):
        from vllm_moe_offload_ascend.moe_offload.phase_split import (
            B2WaveFallback,
            build_b2_native_combine_inputs,
            count_routed_tokens_by_expert,
            plan_b2_prefill_async_schedule,
            plan_balanced_b2_waves,
            scatter_add_b2_pair_outputs,
        )

        runtime = get_moe_offload_runtime()
        offload = fused_experts_input.offload
        routed_expert_count = runtime.routed_expert_count_for_layer(
            layer_id=int(offload.layer_id),
            fallback=int(offload.num_logical_experts),
        )
        num_slots = int(runtime.config.num_slots)
        device = fused_experts_input.topk_ids.device
        b2_total_start = (
            float(control_profile["b2_total_start"])
            if control_profile and "b2_total_start" in control_profile
            else perf_counter()
        )
        token_count_ms = (
            float(control_profile.get("token_count_ms", 0.0))
            if control_profile
            else 0.0
        )
        route_stats_cache_hit = (
            bool(control_profile.get("route_stats_cache_hit", False))
            if control_profile
            else False
        )
        forward_phase = (
            str(control_profile.get("forward_phase", "unknown"))
            if control_profile
            else "unknown"
        )
        pair_offsets_by_expert = (
            control_profile.get("pair_offsets_by_expert")
            if control_profile
            else None
        )
        overflow_mode = _b2_overflow_mode()
        if (
            _b2_reference_full_layer_enabled(
                async_stage=bool(runtime.config.async_load)
            )
            or overflow_mode == "full_layer"
        ):
            return self._run_b2_full_layer_reference(
                fused_experts_input=fused_experts_input,
                before_dispatch_evt=before_dispatch_evt,
                b2_total_start=b2_total_start,
                forward_phase=forward_phase,
                route_stats_cache_hit=route_stats_cache_hit,
            )
        preflight_failure = _b2_multi_wave_preflight_failure(runtime)
        if preflight_failure is not None:
            return self._run_b2_full_layer_reference(
                fused_experts_input=fused_experts_input,
                before_dispatch_evt=before_dispatch_evt,
                b2_total_start=b2_total_start,
                forward_phase=forward_phase,
                route_stats_cache_hit=route_stats_cache_hit,
                fallback_reason=preflight_failure,
            )
        if token_counts is None:
            token_count_start = perf_counter()
            token_counts = count_routed_tokens_by_expert(fused_experts_input.topk_ids)
            token_count_ms = (perf_counter() - token_count_start) * 1000.0
        unique_active = tuple(sorted(token_counts))
        readiness_start = perf_counter()
        readiness, _ = runtime.slot_readiness_and_ready_slot_ids_for_experts(
            layer_id=offload.layer_id,
            expert_ids=unique_active,
        )
        readiness_ms = (perf_counter() - readiness_start) * 1000.0
        reference_full_tokens_requested = _to_bool_env(
            "VLLM_ASCEND_MOE_OFFLOAD_B2_REFERENCE_FULL_TOKENS",
            "0",
        )
        wave_plan_start = perf_counter()
        try:
            wave_plan = plan_balanced_b2_waves(
                token_counts,
                num_slots,
                slot_readiness=(
                    {} if reference_full_tokens_requested else readiness
                ),
                fold_partial_hits_into_miss=(
                    True
                    if reference_full_tokens_requested
                    else not bool(runtime.config.b2_avoid_mixed_wave_d2d)
                ),
            )
        except ValueError as exc:
            raise B2WaveFallback(f"wave planning failed: {exc}") from exc
        wave_plan_ms = (perf_counter() - wave_plan_start) * 1000.0
        waves = wave_plan.waves
        wave_token_counts = tuple(int(wave_plan.wave_tokens(wave)) for wave in waves)
        max_wave_tokens = max(wave_token_counts, default=0)
        min_wave_tokens = min(wave_token_counts, default=0)
        mean_wave_tokens = (
            sum(wave_token_counts) / len(wave_token_counts)
            if wave_token_counts
            else 0.0
        )
        wave_token_imbalance = (
            float(max_wave_tokens) / float(mean_wave_tokens)
            if mean_wave_tokens > 0.0
            else 0.0
        )
        if _os.environ.get("SEW_B2_PROBE") or _os.environ.get("SEW_OFFLOAD_PROBE"):
            print(
                f"SEW_B2 branch=WAVE_RUN layer={offload.layer_id} "
                f"n_active={len(unique_active)} num_slots={num_slots} "
                f"n_waves={len(waves)} n_pairs={wave_plan.total_pairs} "
                f"wave_tokens_min={min_wave_tokens} "
                f"wave_tokens_max={max_wave_tokens}",
                flush=True,
            )

        profile_start = b2_total_start
        accumulated = torch.zeros_like(
            fused_experts_input.hidden_states,
            dtype=torch.float32,
        )
        reference_accumulated = (
            torch.zeros_like(
                fused_experts_input.hidden_states,
                dtype=torch.float32,
            )
            if reference_full_tokens_requested
            else None
        )
        last_group_list_type = None
        last_expert_tokens = None
        last_before_gmm2_evt = None
        before_combine_evt = before_dispatch_evt
        wave_profiles = []
        masked_full_prompt_pairs = int(
            fused_experts_input.topk_ids.numel() * len(waves)
        )
        async_stage = bool(runtime.config.async_load) and len(waves) > 1
        reference_full_tokens = _b2_reference_full_tokens_enabled(
            async_stage=async_stage
        )
        async_schedule = None
        schedule_ms = 0.0
        initial_issue_ms = 0.0
        initial_stage_target = 0
        initial_stage_issued = 0
        microbatch_materialize_ms = 0.0
        microbatch_materialize_end_time = None
        loop_start = None
        loop_ms = 0.0
        scatter_total_ms = 0.0
        pending_pair_outputs = {}
        pending_direct_scatter_payloads = {}
        pending_full_token_outputs = {}
        pending_restore_indices = {}
        pair_index_ms = 0.0
        wave_microbatch_plan_ms = 0.0
        pair_index_cache_hit = pair_offsets_by_expert is not None
        pair_index = None
        pair_index_start = perf_counter()
        from vllm_moe_offload_ascend.moe_offload.phase_split import (
            build_b2_device_wave_microbatch_plans,
            build_b2_routed_pair_index,
            build_b2_wave_microbatch_plans,
            materialize_b2_pair_microbatches_from_plans,
        )

        device_pair_planning = _to_bool_env(
            "VLLM_ASCEND_MOE_OFFLOAD_DEVICE_PAIR_PLANNING",
            "1",
        )
        if not reference_full_tokens and not device_pair_planning:
            pair_index = build_b2_routed_pair_index(
                fused_experts_input.topk_ids,
                fused_experts_input.topk_weights,
                pair_offsets_by_expert=pair_offsets_by_expert,
                validate_pair_offsets=(
                    not pair_index_cache_hit
                    or bool(_os.environ.get("SEW_B2_VALIDATE"))
                ),
            )
        pair_index_ms = (perf_counter() - pair_index_start) * 1000.0
        wave_microbatch_plan_start = perf_counter()
        try:
            if reference_full_tokens:
                wave_microbatch_plans = None
            elif device_pair_planning:
                wave_microbatch_plans = build_b2_device_wave_microbatch_plans(
                    fused_experts_input.topk_ids,
                    waves,
                    num_logical_experts=routed_expert_count,
                    active_experts=unique_active,
                    physical_slot_by_expert={},
                    wave_pair_counts=tuple(
                        int(wave_plan.wave_tokens(wave)) for wave in waves
                    ),
                    include_logical_expert_ids=False,
                )
            else:
                wave_microbatch_plans = build_b2_wave_microbatch_plans(
                    pair_index,
                    waves,
                    physical_slot_by_expert={},
                    preserve_pair_order=False,
                    include_logical_expert_ids=False,
                    include_topk_positions=False,
                    normalize_inputs=False,
                )
        except ValueError as exc:
            raise B2WaveFallback(
                f"wave microbatch planning failed: {exc}"
            ) from exc
        wave_microbatch_plan_ms = (
            perf_counter() - wave_microbatch_plan_start
        ) * 1000.0
        wave_microbatches = None
        wave_h2d_bytes_by_index = {}
        wave_compute_cost_by_index = {}
        if async_stage:
            schedule_start = perf_counter()
            for wave_index, wave in enumerate(waves):
                wave_h2d_bytes_by_index[int(wave_index)] = (
                    _estimate_b2_wave_h2d_bytes(
                        runtime,
                        layer_id=offload.layer_id,
                        wave=tuple(int(expert_id) for expert_id in wave),
                        readiness=readiness,
                    )
                )
                wave_compute_cost_by_index[int(wave_index)] = float(
                    wave_plan.wave_tokens(wave)
                )
            async_schedule = plan_b2_prefill_async_schedule(
                waves,
                # Every B2 prefill wave uses a canonical temporary bank. Actual
                # readiness still controls D2D versus H2D inside staging, but no
                # wave may bypass staging through readiness-dependent main slots.
                slot_readiness={},
                prefetch_depth=runtime.config.effective_prefill_prefetch_depth,
                buffer_count=runtime.config.effective_prefill_buffer_count,
                h2d_bytes_by_wave=wave_h2d_bytes_by_index,
                compute_cost_by_wave=wave_compute_cost_by_index,
                transfer_aware=runtime.config.transfer_aware_wave_schedule,
            )
            schedule_ms = (perf_counter() - schedule_start) * 1000.0
            stage_records = {}
            buffer_release_events = {}
            staged_issue_order = list(async_schedule.staged_issue_order)
            staged_wave_index_set = set(int(i) for i in async_schedule.staged_wave_indices)
            staged_issue_cursor = 0
            stage_issue_sequence = 0
            prefetch_depth = int(async_schedule.prefetch_depth)
            prefill_buffer_count = int(async_schedule.buffer_count)
            initial_stage_target = int(async_schedule.initial_stage_count)
            compute_position_by_wave = {
                int(wave_idx): int(pos)
                for pos, wave_idx in enumerate(async_schedule.compute_order)
            }

            def _issue_wave(
                wave_index,
                *,
                force_buffer_index=None,
                issue_reason="demand",
                current_compute_position=None,
            ):
                nonlocal stage_issue_sequence
                wave_index = int(wave_index)
                wave = waves[wave_index]
                hit_experts = tuple(
                    int(expert_id)
                    for expert_id in wave
                    if bool(readiness.get(int(expert_id), False))
                )
                miss_experts = tuple(
                    int(expert_id)
                    for expert_id in wave
                    if not bool(readiness.get(int(expert_id), False))
                )
                stage_start = perf_counter()
                buffer_index = (
                    int(force_buffer_index)
                    if force_buffer_index is not None
                    else int(wave_index % prefill_buffer_count)
                )
                prepared, ready_event, stage_payload = (
                    runtime.prepare_prefill_stage_plan(
                        layer_id=offload.layer_id,
                        active_experts=wave,
                        num_logical_experts=offload.num_logical_experts,
                        device=device,
                        buffer_index=buffer_index,
                        async_load=True,
                        wait_event=buffer_release_events.get(buffer_index),
                        build_log2phy=False,
                        known_miss=not bool(hit_experts),
                    )
                )
                stage_issue_ms = (perf_counter() - stage_start) * 1000.0
                stage_issue_sequence += 1
                issue_end_time = perf_counter()
                wave_compute_position = int(
                    compute_position_by_wave.get(int(wave_index), int(wave_index))
                )
                if current_compute_position is None:
                    issue_compute_distance = None
                else:
                    issue_compute_distance = int(wave_compute_position) - int(
                        current_compute_position
                    )
                stage_records[wave_index] = {
                    "buffer_index": stage_payload.get("buffer_index"),
                    "wave": wave,
                    "wave_tokens": int(wave_plan.wave_tokens(wave)),
                    "per_expert_tokens": {
                        int(e): int(token_counts.get(int(e), 0))
                        for e in wave
                    },
                    "prepared": prepared,
                    "ready_event": ready_event,
                    "hit_experts": hit_experts,
                    "miss_experts": miss_experts,
                    "h2d_bytes": int(stage_payload.get("h2d_bytes", 0)),
                    "d2d_bytes": int(stage_payload.get("d2d_bytes", 0)),
                    "stage_ms": stage_issue_ms,
                    "stage_mode": str(stage_payload.get("stage_mode", "async_double_buffer")),
                    "issue_sequence": int(stage_issue_sequence),
                    "issue_start_time": float(stage_start),
                    "issue_end_time": float(issue_end_time),
                    "issue_reason": str(issue_reason),
                    "issue_compute_distance": issue_compute_distance,
                    "issued_before_compute": (
                        bool(issue_compute_distance > 0)
                        if issue_compute_distance is not None
                        else False
                    ),
                    "stage_payload": stage_payload,
                    "stage_profile_ms": stage_payload.get("profile_ms", {}),
                    "use_wave_plan_physical_ids": not bool(
                        stage_payload.get("log2phy_built", True)
                    ),
                }

            def _inflight_stage_count():
                return sum(
                    1
                    for record in stage_records.values()
                    if record.get("buffer_index") is not None
                )

            def _next_free_buffer(*, current_buffer_index=None):
                busy_buffers = {
                    int(record["buffer_index"])
                    for record in stage_records.values()
                    if record.get("buffer_index") is not None
                }
                if current_buffer_index is not None:
                    busy_buffers.add(int(current_buffer_index))
                for buffer_index in range(prefill_buffer_count):
                    if int(buffer_index) not in busy_buffers:
                        return int(buffer_index)
                return None

            def _issue_next_staged_wave(
                *,
                current_buffer_index=None,
                issue_reason="prefetch",
                current_compute_position=None,
            ):
                nonlocal staged_issue_cursor
                if staged_issue_cursor >= len(staged_issue_order):
                    return False
                buffer_index = _next_free_buffer(
                    current_buffer_index=current_buffer_index
                )
                if buffer_index is None:
                    return False
                wave_index = int(staged_issue_order[staged_issue_cursor])
                staged_issue_cursor += 1
                _issue_wave(
                    wave_index,
                    force_buffer_index=buffer_index,
                    issue_reason=issue_reason,
                    current_compute_position=current_compute_position,
                )
                return True

            def _prefetch_ahead(
                *,
                current_buffer_index=None,
                issue_reason="prefetch",
                current_compute_position=None,
            ):
                if prefetch_depth <= 0:
                    return 0
                occupied_by_current = 1 if current_buffer_index is not None else 0
                target_future = min(
                    int(prefetch_depth),
                    max(prefill_buffer_count - occupied_by_current, 0),
                )
                issued_count = 0
                while _inflight_stage_count() < target_future:
                    if not _issue_next_staged_wave(
                        current_buffer_index=current_buffer_index,
                        issue_reason=issue_reason,
                        current_compute_position=current_compute_position,
                    ):
                        break
                    issued_count += 1
                return int(issued_count)

            initial_issue_start = perf_counter()
            while staged_issue_cursor < int(async_schedule.initial_stage_count):
                if not _issue_next_staged_wave(
                    issue_reason="initial_prefetch",
                    current_compute_position=-1,
                ):
                    break
            initial_issue_ms = (perf_counter() - initial_issue_start) * 1000.0
            initial_stage_issued = int(
                sum(
                    1
                    for record in stage_records.values()
                    if record.get("buffer_index") is not None
                )
            )
            microbatch_materialize_start = perf_counter()
            wave_microbatches = materialize_b2_pair_microbatches_from_plans(
                fused_experts_input.hidden_states,
                fused_experts_input.topk_weights,
                wave_microbatch_plans,
            )
            microbatch_materialize_ms = (
                perf_counter() - microbatch_materialize_start
            ) * 1000.0
            microbatch_materialize_end_time = perf_counter()

            loop_start = perf_counter()
            for compute_position, wave_index in enumerate(async_schedule.compute_order):
                if wave_index not in stage_records:
                    if int(wave_index) in staged_wave_index_set:
                        buffer_index = _next_free_buffer()
                        if buffer_index is None:
                            raise RuntimeError(
                                "B2 prefill cannot issue current staged wave "
                                f"{int(wave_index)} because all "
                                f"{prefill_buffer_count} stage buffers are busy"
                            )
                        _issue_wave(
                            wave_index,
                            force_buffer_index=buffer_index,
                            issue_reason="demand_stage",
                            current_compute_position=compute_position,
                        )
                    else:
                        _issue_wave(
                            wave_index,
                            issue_reason="hit_main_slot",
                            current_compute_position=compute_position,
                        )
                inflight_before_consume = _inflight_stage_count()
                record = stage_records.pop(wave_index)
                wait_start = perf_counter()
                runtime.wait_prefill_stage_plan(record["ready_event"])
                stage_wait_ms = (perf_counter() - wait_start) * 1000.0
                compute_ready_time = perf_counter()
                prepared = record["prepared"]
                prepared.validate_backend_ready(
                    expected_device_type=offload.expected_device_type
                )
                current_buffer_index = record["buffer_index"]
                inflight_after_wait = _inflight_stage_count()
                prefetch_before_compute_count = _prefetch_ahead(
                    current_buffer_index=(
                        int(current_buffer_index)
                        if current_buffer_index is not None
                        else None
                    ),
                    issue_reason="prefetch_before_compute",
                    current_compute_position=compute_position,
                )
                mlp_start = perf_counter()
                wave_output = self._run_b2_pair_wave(
                    fused_experts_input=fused_experts_input,
                    prepared=prepared,
                    wave=record["wave"],
                    microbatch_plan=wave_microbatch_plans[int(wave_index)],
                    microbatch=wave_microbatches[int(wave_index)],
                    use_wave_plan_physical_ids=bool(
                        record.get("use_wave_plan_physical_ids", False)
                    ),
                )
                mlp_ms = (perf_counter() - mlp_start) * 1000.0
                prefetch_after_compute_count = 0
                if current_buffer_index is not None:
                    release_event = torch.npu.current_stream().record_event()
                    buffer_release_events[int(current_buffer_index)] = release_event
                    runtime.remember_prefill_stage_buffer_release(
                        layer_id=offload.layer_id,
                        buffer_index=int(current_buffer_index),
                        event=release_event,
                    )
                    prefetch_after_compute_count = _prefetch_ahead(
                        issue_reason="prefetch_after_compute",
                        current_compute_position=compute_position,
                    )
                issue_start_to_compute_ms = (
                    (compute_ready_time - float(record["issue_start_time"])) * 1000.0
                    if record.get("issue_start_time") is not None
                    else 0.0
                )
                issue_end_to_compute_ms = (
                    (compute_ready_time - float(record["issue_end_time"])) * 1000.0
                    if record.get("issue_end_time") is not None
                    else 0.0
                )
                issued_before_microbatch_materialize = (
                    bool(
                        microbatch_materialize_end_time is not None
                        and float(record["issue_end_time"])
                        <= float(microbatch_materialize_end_time)
                    )
                    if record.get("issue_end_time") is not None
                    else False
                )
                issue_end_to_microbatch_end_ms = (
                    (
                        float(microbatch_materialize_end_time)
                        - float(record["issue_end_time"])
                    )
                    * 1000.0
                    if (
                        microbatch_materialize_end_time is not None
                        and record.get("issue_end_time") is not None
                    )
                    else 0.0
                )
                if wave_output is None:
                    wave_profiles.append(
                        {
                            "buffer_index": record["buffer_index"],
                            "experts": [int(e) for e in record["wave"]],
                            "tokens": int(record["wave_tokens"]),
                            "per_expert_tokens": {
                                str(int(e)): int(v)
                                for e, v in record["per_expert_tokens"].items()
                            },
                            "max_expert_tokens": int(max(record["per_expert_tokens"].values(), default=0)),
                            "min_expert_tokens": int(min(record["per_expert_tokens"].values(), default=0)),
                            "pairs": 0,
                            "hits": len(record["hit_experts"]),
                            "misses": len(record["miss_experts"]),
                            "h2d_bytes": int(record["h2d_bytes"]),
                            "d2d_bytes": int(record["d2d_bytes"]),
                            "stage_ms": round(record["stage_ms"] + stage_wait_ms, 3),
                            "stage_issue_ms": round(record["stage_ms"], 3),
                            "stage_wait_ms": round(stage_wait_ms, 3),
                            "stage_profile_ms": record["stage_profile_ms"],
                            "mlp_ms": round(mlp_ms, 3),
                            "stage_mode": record["stage_mode"],
                            "log2phy_built": bool(
                                (record.get("stage_payload") or {}).get(
                                    "log2phy_built",
                                    True,
                                )
                            ),
                            "log2phy_source": (
                                "wave_plan"
                                if record.get("use_wave_plan_physical_ids")
                                else "log2phy"
                            ),
                            "compute_index": int(compute_position),
                            "issue_sequence": int(record["issue_sequence"]),
                            "issue_reason": str(record["issue_reason"]),
                            "issue_compute_distance": record[
                                "issue_compute_distance"
                            ],
                            "issued_before_compute": bool(
                                record["issued_before_compute"]
                            ),
                            "issued_before_microbatch_materialize": bool(
                                issued_before_microbatch_materialize
                            ),
                            "issue_end_to_microbatch_end_ms": round(
                                issue_end_to_microbatch_end_ms,
                                3,
                            ),
                            "issue_start_to_compute_ms": round(
                                issue_start_to_compute_ms,
                                3,
                            ),
                            "issue_end_to_compute_ms": round(
                                issue_end_to_compute_ms,
                                3,
                            ),
                            "inflight_before_consume": int(
                                inflight_before_consume
                            ),
                            "inflight_after_wait": int(inflight_after_wait),
                            "prefetch_before_compute_count": int(
                                prefetch_before_compute_count
                            ),
                            "prefetch_after_compute_count": int(
                                prefetch_after_compute_count
                            ),
                            "prefetch_depth": int(prefetch_depth),
                            "prefill_buffer_count": int(prefill_buffer_count),
                        }
                    )
                    continue
                wave_result, restore_token_indices, pair_profile = wave_output
                before_combine_evt = wave_result.before_combine_evt
                last_before_gmm2_evt = wave_result.before_gmm2_evt
                last_group_list_type = wave_result.group_list_type
                last_expert_tokens = wave_result.expert_tokens
                direct_scatter_payload = pair_profile.get(
                    "direct_scatter_payload"
                )
                if direct_scatter_payload is None:
                    pending_pair_outputs[int(wave_index)] = wave_result.routed_out
                    pending_restore_indices[int(wave_index)] = restore_token_indices
                else:
                    pending_direct_scatter_payloads[int(wave_index)] = replace(
                        direct_scatter_payload,
                        wave_index=int(wave_index),
                    )
                wave_profiles.append(
                    {
                        "buffer_index": record["buffer_index"],
                        "experts": [int(e) for e in record["wave"]],
                        "pairs": int(restore_token_indices.numel()),
                        "tokens": int(wave_plan.wave_tokens(record["wave"])),
                        "per_expert_tokens": {
                            str(int(e)): int(v)
                            for e, v in record["per_expert_tokens"].items()
                        },
                        "max_expert_tokens": int(max(record["per_expert_tokens"].values(), default=0)),
                        "min_expert_tokens": int(min(record["per_expert_tokens"].values(), default=0)),
                        "hits": len(record["hit_experts"]),
                        "misses": len(record["miss_experts"]),
                        "h2d_bytes": int(record["h2d_bytes"]),
                        "d2d_bytes": int(record["d2d_bytes"]),
                        "stage_ms": round(record["stage_ms"] + stage_wait_ms, 3),
                        "stage_issue_ms": round(record["stage_ms"], 3),
                        "stage_wait_ms": round(stage_wait_ms, 3),
                        "stage_profile_ms": record["stage_profile_ms"],
                        "mlp_ms": round(mlp_ms, 3),
                        "pair_wave_ms": round(pair_profile.get("pair_wave_ms", mlp_ms), 3),
                        "microbatch_ms": round(pair_profile.get("microbatch_ms", 0.0), 3),
                        "microbatch_source": str(pair_profile.get("microbatch_source", "")),
                        "dispatch_ms": round(pair_profile.get("dispatch_ms", 0.0), 3),
                        "build_mlp_input_ms": round(pair_profile.get("build_mlp_input_ms", 0.0), 3),
                        "gmm_ms": round(pair_profile.get("gmm_ms", 0.0), 3),
                        "combine_ms": round(pair_profile.get("combine_ms", 0.0), 3),
                        "combine_mode": str(pair_profile.get("combine_mode", "token_combine")),
                        "scatter_ms": 0.0,
                        "scatter_mode": str(
                            pair_profile.get("scatter_mode", "layer_batch")
                        ),
                        "stage_mode": record["stage_mode"],
                        "log2phy_built": bool(
                            (record.get("stage_payload") or {}).get(
                                "log2phy_built",
                                True,
                            )
                        ),
                        "log2phy_source": (
                            "wave_plan"
                            if record.get("use_wave_plan_physical_ids")
                            else "log2phy"
                        ),
                        "compute_index": int(compute_position),
                        "issue_sequence": int(record["issue_sequence"]),
                        "issue_reason": str(record["issue_reason"]),
                        "issue_compute_distance": record[
                            "issue_compute_distance"
                        ],
                        "issued_before_compute": bool(
                            record["issued_before_compute"]
                        ),
                        "issued_before_microbatch_materialize": bool(
                            issued_before_microbatch_materialize
                        ),
                        "issue_end_to_microbatch_end_ms": round(
                            issue_end_to_microbatch_end_ms,
                            3,
                        ),
                        "issue_start_to_compute_ms": round(
                            issue_start_to_compute_ms,
                            3,
                        ),
                        "issue_end_to_compute_ms": round(
                            issue_end_to_compute_ms,
                            3,
                        ),
                        "inflight_before_consume": int(inflight_before_consume),
                        "inflight_after_wait": int(inflight_after_wait),
                        "prefetch_before_compute_count": int(
                            prefetch_before_compute_count
                        ),
                        "prefetch_after_compute_count": int(
                            prefetch_after_compute_count
                        ),
                        "prefetch_depth": int(prefetch_depth),
                        "prefill_buffer_count": int(prefill_buffer_count),
                    }
                )
            loop_ms = (perf_counter() - loop_start) * 1000.0
        else:
            if not reference_full_tokens:
                microbatch_materialize_start = perf_counter()
                wave_microbatches = materialize_b2_pair_microbatches_from_plans(
                    fused_experts_input.hidden_states,
                    fused_experts_input.topk_weights,
                    wave_microbatch_plans,
                )
                microbatch_materialize_ms = (
                    perf_counter() - microbatch_materialize_start
                ) * 1000.0
                microbatch_materialize_end_time = perf_counter()
            loop_start = perf_counter()
            for wave_index, wave in enumerate(waves):
                hit_experts = tuple(
                    int(expert_id)
                    for expert_id in wave
                    if bool(readiness.get(int(expert_id), False))
                )
                miss_experts = tuple(
                    int(expert_id)
                    for expert_id in wave
                    if not bool(readiness.get(int(expert_id), False))
                )
                h2d_bytes = sum(
                    runtime.estimate_expert_weight_bytes(
                        layer_id=offload.layer_id,
                        expert_id=expert_id,
                    )
                    for expert_id in miss_experts
                )
                stage_start = perf_counter()
                prepared, ready_event, stage_payload = (
                    runtime.prepare_prefill_stage_plan(
                        layer_id=offload.layer_id,
                        active_experts=wave,
                        num_logical_experts=offload.num_logical_experts,
                        device=device,
                        buffer_index=int(
                            wave_index
                            % runtime.config.effective_prefill_buffer_count
                        ),
                        async_load=False,
                        build_log2phy=bool(reference_full_tokens),
                        known_miss=not bool(hit_experts),
                    )
                )
                runtime.wait_prefill_stage_plan(ready_event)
                h2d_bytes = int(stage_payload.get("h2d_bytes", h2d_bytes))
                d2d_bytes = int(stage_payload.get("d2d_bytes", 0))
                stage_ms = (perf_counter() - stage_start) * 1000.0
                prepared.validate_backend_ready(
                    expected_device_type=offload.expected_device_type
                )
                mlp_start = perf_counter()
                if reference_full_tokens:
                    wave_output = self._run_b2_single_wave(
                        fused_experts_input=fused_experts_input,
                        prepared=prepared,
                    )
                else:
                    wave_output = self._run_b2_pair_wave(
                        fused_experts_input=fused_experts_input,
                        prepared=prepared,
                        wave=wave,
                        microbatch_plan=wave_microbatch_plans[int(wave_index)],
                        microbatch=wave_microbatches[int(wave_index)],
                        use_wave_plan_physical_ids=True,
                    )
                mlp_ms = (perf_counter() - mlp_start) * 1000.0
                if wave_output is None:
                    per_expert_tokens = {
                        int(e): int(token_counts.get(int(e), 0))
                        for e in wave
                    }
                    wave_profiles.append(
                        {
                            "experts": [int(e) for e in wave],
                            "tokens": int(wave_plan.wave_tokens(wave)),
                            "per_expert_tokens": {
                                str(int(e)): int(v)
                                for e, v in per_expert_tokens.items()
                            },
                            "max_expert_tokens": int(max(per_expert_tokens.values(), default=0)),
                            "min_expert_tokens": int(min(per_expert_tokens.values(), default=0)),
                            "pairs": 0,
                            "hits": len(hit_experts),
                            "misses": len(miss_experts),
                            "h2d_bytes": int(h2d_bytes),
                            "d2d_bytes": int(d2d_bytes),
                            "stage_ms": round(stage_ms, 3),
                            "mlp_ms": round(mlp_ms, 3),
                            "microbatch_source": "wave_plan",
                            "stage_mode": "sync_temp_bank",
                        }
                    )
                    continue
                if reference_full_tokens:
                    wave_result = wave_output
                    restore_token_indices = None
                    pair_profile = {
                        "pair_wave_ms": mlp_ms,
                        "microbatch_ms": 0.0,
                        "microbatch_source": "reference_full_tokens",
                        "combine_mode": "token_combine",
                        "scatter_mode": "full_token_add",
                    }
                else:
                    wave_result, restore_token_indices, pair_profile = wave_output
                per_expert_tokens = {
                    int(e): int(token_counts.get(int(e), 0))
                    for e in wave
                }
                before_combine_evt = wave_result.before_combine_evt
                last_before_gmm2_evt = wave_result.before_gmm2_evt
                last_group_list_type = wave_result.group_list_type
                last_expert_tokens = wave_result.expert_tokens
                direct_scatter_payload = pair_profile.get(
                    "direct_scatter_payload"
                )
                if reference_full_tokens:
                    pending_full_token_outputs[int(wave_index)] = wave_result.routed_out
                elif direct_scatter_payload is None:
                    pending_pair_outputs[int(wave_index)] = wave_result.routed_out
                    pending_restore_indices[int(wave_index)] = restore_token_indices
                else:
                    pending_direct_scatter_payloads[int(wave_index)] = replace(
                        direct_scatter_payload,
                        wave_index=int(wave_index),
                    )
                wave_profiles.append(
                    {
                        "experts": [int(e) for e in wave],
                        "pairs": (
                            int(fused_experts_input.topk_ids.numel())
                            if reference_full_tokens
                            else int(restore_token_indices.numel())
                        ),
                        "tokens": int(wave_plan.wave_tokens(wave)),
                        "per_expert_tokens": {
                            str(int(e)): int(v)
                            for e, v in per_expert_tokens.items()
                        },
                        "max_expert_tokens": int(max(per_expert_tokens.values(), default=0)),
                        "min_expert_tokens": int(min(per_expert_tokens.values(), default=0)),
                        "hits": len(hit_experts),
                        "misses": len(miss_experts),
                        "h2d_bytes": int(h2d_bytes),
                        "d2d_bytes": int(d2d_bytes),
                        "stage_ms": round(stage_ms, 3),
                        "mlp_ms": round(mlp_ms, 3),
                        "pair_wave_ms": round(pair_profile.get("pair_wave_ms", mlp_ms), 3),
                        "microbatch_ms": round(pair_profile.get("microbatch_ms", 0.0), 3),
                        "microbatch_source": str(pair_profile.get("microbatch_source", "")),
                        "dispatch_ms": round(pair_profile.get("dispatch_ms", 0.0), 3),
                        "build_mlp_input_ms": round(pair_profile.get("build_mlp_input_ms", 0.0), 3),
                        "gmm_ms": round(pair_profile.get("gmm_ms", 0.0), 3),
                        "combine_ms": round(pair_profile.get("combine_ms", 0.0), 3),
                        "combine_mode": str(pair_profile.get("combine_mode", "token_combine")),
                        "scatter_ms": 0.0,
                        "scatter_mode": str(
                            pair_profile.get("scatter_mode", "layer_batch")
                        ),
                        "stage_mode": "sync_temp_bank",
                    }
                )
            loop_ms = (perf_counter() - loop_start) * 1000.0

        if not waves:
            return None
        scatter_start = perf_counter()
        if pending_direct_scatter_payloads:
            native_permuted, native_expanded, combine_metadata = (
                build_b2_native_combine_inputs(
                    tuple(
                        payload
                        for _, payload in sorted(
                            pending_direct_scatter_payloads.items()
                        )
                    ),
                    total_pairs=int(fused_experts_input.topk_ids.numel()),
                )
            )
            native_metadata = replace(
                combine_metadata,
                topk_weights=fused_experts_input.topk_weights,
                expanded_row_idx=native_expanded,
                restore_shape=fused_experts_input.hidden_states.shape,
            )
            accumulated = self.token_dispatcher.token_combine(
                hidden_states=native_permuted,
                combine_metadata=native_metadata,
            )
        if pending_pair_outputs:
            pair_wave_indices = sorted(pending_pair_outputs)
            scatter_add_b2_pair_outputs(
                accumulated,
                tuple(pending_pair_outputs[index] for index in pair_wave_indices),
                tuple(pending_restore_indices[index] for index in pair_wave_indices),
            )
        for _, full_token_output in sorted(pending_full_token_outputs.items()):
            reference_accumulated.add_(full_token_output.to(torch.float32))
        if reference_accumulated is not None:
            accumulated.copy_(reference_accumulated.to(accumulated.dtype))
        wave_accumulation_dtype = str(accumulated.dtype)
        accumulated = accumulated.to(fused_experts_input.hidden_states.dtype)
        scatter_total_ms = (perf_counter() - scatter_start) * 1000.0
        end_to_end_ms = (perf_counter() - b2_total_start) * 1000.0
        wave_summary = _summarize_b2_wave_profiles(
            wave_profiles,
            layer_scatter_ms=scatter_total_ms,
        )
        if (
            runtime.config.gmm_profile_path
            or _os.getenv("VLLM_ASCEND_MOE_GMM_PROFILE_PATH")
            or _os.getenv("VLLM_ASCEND_MOE_OFFLOAD_PROFILE_PATH")
        ):
            payload = {
                "n_tokens": int(fused_experts_input.hidden_states.shape[0]),
                "n_active": int(len(unique_active)),
                "n_pairs": int(
                    masked_full_prompt_pairs
                    if reference_full_tokens
                    else wave_plan.total_pairs
                ),
                "logical_pairs": int(wave_plan.total_pairs),
                "num_slots": int(num_slots),
                "n_waves": int(len(waves)),
                "masked_full_prompt_pairs": int(masked_full_prompt_pairs),
                "work_conserving_saved_pairs": int(
                    0
                    if reference_full_tokens
                    else masked_full_prompt_pairs - wave_plan.total_pairs
                ),
                "execution_mode": (
                    "reference_full_tokens"
                    if reference_full_tokens
                    else "pair_microbatch"
                ),
                "wave_membership_uses_slot_readiness": False,
                "wave_accumulation_dtype": wave_accumulation_dtype,
                "wave_token_summary": {
                    "min": int(min_wave_tokens),
                    "max": int(max_wave_tokens),
                    "mean": round(float(mean_wave_tokens), 3),
                    "imbalance": round(float(wave_token_imbalance), 3),
                },
                "control_ms": {
                    "token_count": round(token_count_ms, 3),
                    "readiness": round(readiness_ms, 3),
                    "wave_plan": round(wave_plan_ms, 3),
                    "schedule": round(schedule_ms, 3),
                    "initial_issue": round(initial_issue_ms, 3),
                    "initial_stage_target": int(initial_stage_target),
                    "initial_stage_issued": int(initial_stage_issued),
                    "pair_index": round(pair_index_ms, 3),
                    "wave_microbatch_plan": round(wave_microbatch_plan_ms, 3),
                    "microbatch_materialize": round(
                        microbatch_materialize_ms,
                        3,
                    ),
                    "loop": round(loop_ms, 3),
                    "scatter_total": round(scatter_total_ms, 3),
                    "end_to_end": round(end_to_end_ms, 3),
                },
                "route_stats_cache_hit": bool(route_stats_cache_hit),
                "pair_index_cache_hit": bool(pair_index_cache_hit),
                "pair_planner_mode": (
                    "not_used_reference_full_tokens"
                    if reference_full_tokens
                    else "npu_device" if device_pair_planning else "host_offsets"
                ),
                "forward_phase": str(forward_phase),
                "wave_summary": wave_summary,
            }
            payload = _attach_b2_profile_details(
                payload,
                wave_plan=wave_plan,
                async_schedule=async_schedule,
                async_stage=async_stage,
                wave_profiles=wave_profiles,
            )
            runtime._record_profile_event(
                "b2_work_conserving_prefill",
                layer_id=offload.layer_id,
                start=profile_start,
                payload=payload,
            )
        return FusedExpertsResult(
            routed_out=accumulated,
            before_dispatch_evt=before_dispatch_evt,
            before_gmm2_evt=last_before_gmm2_evt,
            before_combine_evt=before_combine_evt,
            group_list_type=last_group_list_type,
            expert_tokens=last_expert_tokens,
            swiglu_limit=fused_experts_input.swiglu_limit,
        )

    def _run_b2_full_layer_reference(
        self,
        *,
        fused_experts_input,
        before_dispatch_evt,
        b2_total_start,
        forward_phase,
        route_stats_cache_hit,
        fallback_reason=None,
    ):
        runtime = get_moe_offload_runtime()
        offload = fused_experts_input.offload
        prepared, stage_payload = runtime.prepare_full_layer_prefill_plan(
            layer_id=offload.layer_id,
            num_logical_experts=offload.num_logical_experts,
            device=fused_experts_input.topk_ids.device,
            step_id=offload.step_id,
        )
        prepared.validate_backend_ready(
            expected_device_type=offload.expected_device_type
        )
        native_input = self._with_prepared_slot_weights(
            fused_experts_input,
            prepared,
            offload_enabled=False,
            use_log2phy=False,
        )
        compare_original = bool(
            _to_bool_env(
                "VLLM_ASCEND_MOE_OFFLOAD_COMPARE_ORIGINAL",
                "0",
            )
            and int(fused_experts_input.hidden_states.shape[0]) <= 64
        )
        original_result = None
        try:
            if compare_original:
                original_input = MoEFusedExpertsInput(
                    hidden_states=fused_experts_input.hidden_states,
                    topk_weights=fused_experts_input.topk_weights,
                    topk_ids=fused_experts_input.topk_ids,
                    weights=fused_experts_input.weights,
                    routing=fused_experts_input.routing,
                    quant=fused_experts_input.quant,
                    activation=fused_experts_input.activation,
                    need_trans=fused_experts_input.need_trans,
                    dynamic_eplb=fused_experts_input.dynamic_eplb,
                    swiglu_limit=fused_experts_input.swiglu_limit,
                    offload=MoEOffloadParams(
                        enabled=False,
                        profile_only=fused_experts_input.offload.profile_only,
                        layer_id=fused_experts_input.offload.layer_id,
                        num_logical_experts=(
                            fused_experts_input.offload.num_logical_experts
                        ),
                        expected_device_type=(
                            fused_experts_input.offload.expected_device_type
                        ),
                        step_id=fused_experts_input.offload.step_id,
                    ),
                )
                original_result = original_fused_experts(self, original_input)
                original_routed_out = original_result.routed_out.clone()
            result = original_fused_experts(self, native_input)
            torch.npu.synchronize()
            if compare_original:
                import torch_npu

                original_w1 = fused_experts_input.weights.w1
                original_w2 = fused_experts_input.weights.w2
                host_buffer = runtime._host_store.get_layer_buffer(
                    int(offload.layer_id)
                )
                sampled_experts = tuple(
                    expert_id
                    for expert_id in (0, 1, 63, 127)
                    if expert_id < int(offload.num_logical_experts)
                )
                original_bank_w1_equal = all(
                    bool(torch.equal(original_w1[expert_id], prepared.w1[expert_id]))
                    for expert_id in sampled_experts
                )
                original_bank_w2_equal = all(
                    bool(torch.equal(original_w2[expert_id], prepared.w2[expert_id]))
                    for expert_id in sampled_experts
                )
                host_bank_w1_equal = all(
                    bool(
                        torch.equal(
                            host_buffer.w13[expert_id],
                            prepared.w1[expert_id].cpu(),
                        )
                    )
                    for expert_id in sampled_experts
                )
                host_bank_w2_equal = all(
                    bool(
                        torch.equal(
                            host_buffer.w2[expert_id],
                            prepared.w2[expert_id].cpu(),
                        )
                    )
                    for expert_id in sampled_experts
                )
                output_diff = (
                    original_routed_out.float() - result.routed_out.float()
                ).abs()
                print(
                    "SEW_FULL_LAYER_COMPARE "
                    f"layer={int(offload.layer_id)} "
                    f"tokens={int(fused_experts_input.hidden_states.shape[0])} "
                    f"sampled_experts={sampled_experts} "
                    f"original_bank_w1_equal={original_bank_w1_equal} "
                    f"original_bank_w2_equal={original_bank_w2_equal} "
                    f"host_bank_w1_equal={host_bank_w1_equal} "
                    f"host_bank_w2_equal={host_bank_w2_equal} "
                    f"original_w1_format={torch_npu.get_npu_format(original_w1)} "
                    f"bank_w1_format={torch_npu.get_npu_format(prepared.w1)} "
                    f"original_w2_format={torch_npu.get_npu_format(original_w2)} "
                    f"bank_w2_format={torch_npu.get_npu_format(prepared.w2)} "
                    f"original_w1_stride={tuple(original_w1.stride())} "
                    f"bank_w1_stride={tuple(prepared.w1.stride())} "
                    f"original_w2_stride={tuple(original_w2.stride())} "
                    f"bank_w2_stride={tuple(prepared.w2.stride())} "
                    f"output_equal={bool(torch.equal(original_routed_out, result.routed_out))} "
                    f"output_max_abs={float(output_diff.max().item())} "
                    f"output_mean_abs={float(output_diff.mean().item())}",
                    flush=True,
                )
        finally:
            compute_handle = getattr(
                self,
                "_ascend_moe_offload_slot_compute_handle",
                None,
            )
            if hasattr(self, "_ascend_moe_offload_slot_compute_handle"):
                delattr(self, "_ascend_moe_offload_slot_compute_handle")
            runtime.end_slot_compute(compute_handle)
        end_to_end_ms = (perf_counter() - b2_total_start) * 1000.0
        if _os.environ.get("SEW_B2_PROBE") or _os.environ.get("SEW_OFFLOAD_PROBE"):
            print(
                f"SEW_B2 branch=REFERENCE_FULL_LAYER layer={offload.layer_id} "
                f"n_tokens={int(fused_experts_input.hidden_states.shape[0])} "
                f"physical_experts={prepared.physical_expert_count} "
                f"h2d_bytes={stage_payload['h2d_bytes']} "
                f"fallback_reason={fallback_reason}",
                flush=True,
            )
        if (
            runtime.config.gmm_profile_path
            or _os.getenv("VLLM_ASCEND_MOE_GMM_PROFILE_PATH")
            or _os.getenv("VLLM_ASCEND_MOE_OFFLOAD_PROFILE_PATH")
        ):
            runtime._record_profile_event(
                "b2_reference_full_layer_prefill",
                layer_id=offload.layer_id,
                start=b2_total_start,
                payload={
                    **stage_payload,
                    "n_tokens": int(fused_experts_input.hidden_states.shape[0]),
                    "top_k": int(fused_experts_input.topk_ids.shape[-1]),
                    "forward_phase": str(forward_phase),
                    "route_stats_cache_hit": bool(route_stats_cache_hit),
                    "fallback_reason": fallback_reason,
                    "end_to_end_ms": round(end_to_end_ms, 3),
                },
            )
        return FusedExpertsResult(
            routed_out=result.routed_out,
            before_dispatch_evt=before_dispatch_evt,
            before_gmm2_evt=result.before_gmm2_evt,
            before_combine_evt=result.before_combine_evt,
            group_list_type=result.group_list_type,
            expert_tokens=result.expert_tokens,
            swiglu_limit=getattr(result, "swiglu_limit", 0.0),
        )

    def _run_b2_pair_wave(
        self,
        *,
        fused_experts_input,
        prepared,
        wave,
        pair_index=None,
        microbatch_plan=None,
        microbatch=None,
        use_wave_plan_physical_ids=False,
    ):
        from vllm_moe_offload_ascend.moe_offload.phase_split import (
            B2DirectScatterPayload,
            build_b2_pair_microbatch,
            build_b2_pair_microbatch_from_index,
            build_b2_pair_microbatch_from_plan,
        )

        pair_wave_start = perf_counter()
        microbatch_start = perf_counter()
        if microbatch is not None:
            microbatch_source = "materialized_wave_plan"
        elif microbatch_plan is not None:
            microbatch = build_b2_pair_microbatch_from_plan(
                fused_experts_input.hidden_states,
                fused_experts_input.topk_weights,
                None if use_wave_plan_physical_ids else prepared.log2phy,
                microbatch_plan,
            )
            microbatch_source = (
                "wave_plan_slots"
                if use_wave_plan_physical_ids
                else "wave_plan"
            )
        elif pair_index is None:
            microbatch = build_b2_pair_microbatch(
                fused_experts_input.hidden_states,
                fused_experts_input.topk_ids,
                fused_experts_input.topk_weights,
                prepared.log2phy,
                tuple(int(e) for e in wave),
            )
            microbatch_source = "scan"
        else:
            microbatch = build_b2_pair_microbatch_from_index(
                fused_experts_input.hidden_states,
                pair_index,
                prepared.log2phy,
                tuple(int(e) for e in wave),
            )
            microbatch_source = "pair_index"
        microbatch_ms = (perf_counter() - microbatch_start) * 1000.0
        if microbatch.num_pairs == 0:
            return None

        pertoken_start = perf_counter()
        pertoken_scale = fused_experts_input.routing.pertoken_scale
        if pertoken_scale is not None and pertoken_scale.shape[0] == fused_experts_input.hidden_states.shape[0]:
            pertoken_scale = pertoken_scale.index_select(0, microbatch.restore_token_indices)
        pertoken_scale_ms = (perf_counter() - pertoken_start) * 1000.0

        build_input_start = perf_counter()
        wave_input = MoEFusedExpertsInput(
            hidden_states=microbatch.hidden_states,
            topk_weights=microbatch.topk_weights,
            topk_ids=microbatch.topk_ids,
            weights=MoEWeights(
                w1=prepared.w1,
                w2=prepared.w2,
                w1_bias=fused_experts_input.weights.w1_bias,
                w2_bias=fused_experts_input.weights.w2_bias,
                w1_scale=fused_experts_input.weights.w1_scale,
                w2_scale=fused_experts_input.weights.w2_scale,
                w1_scale_bias=fused_experts_input.weights.w1_scale_bias,
                w2_scale_bias=fused_experts_input.weights.w2_scale_bias,
                w1_offset=fused_experts_input.weights.w1_offset,
                w2_offset=fused_experts_input.weights.w2_offset,
            ),
            routing=MoERoutingParams(
                expert_map=None,
                global_redundant_expert_num=(
                    fused_experts_input.routing.global_redundant_expert_num
                ),
                mc2_mask=fused_experts_input.routing.mc2_mask,
                apply_router_weight_on_input=(
                    fused_experts_input.routing.apply_router_weight_on_input
                ),
                log2phy=None,
                physical_expert_count=prepared.physical_expert_count,
                pertoken_scale=pertoken_scale,
            ),
            quant=fused_experts_input.quant,
            activation=fused_experts_input.activation,
            need_trans=fused_experts_input.need_trans,
            dynamic_eplb=fused_experts_input.dynamic_eplb,
            swiglu_limit=fused_experts_input.swiglu_limit,
            offload=fused_experts_input.offload,
        )
        build_input_ms = (perf_counter() - build_input_start) * 1000.0

        old_top_k = getattr(self.token_dispatcher, "top_k", None)
        runtime = get_moe_offload_runtime()
        compute_handle = runtime.begin_slot_compute(prepared)
        self.token_dispatcher.top_k = 1
        try:
            dispatch_start = perf_counter()
            token_dispatch_output = self.token_dispatcher.token_dispatch(
                token_dispatch_input=build_token_dispatch_input(
                    fused_experts_input=wave_input
                )
            )
            dispatch_ms = (perf_counter() - dispatch_start) * 1000.0
            build_mlp_input_start = perf_counter()
            mlp_compute_input = build_mlp_compute_input(
                fused_experts_input=wave_input,
                token_dispatch_output=token_dispatch_output,
                use_fusion_ops=self.use_fusion_ops,
            )
            build_mlp_input_ms = (perf_counter() - build_mlp_input_start) * 1000.0
            gmm_start = perf_counter()
            mlp_output, before_gmm2_evt = _unpack_mlp_apply_result(
                self._apply_mlp(mlp_compute_input)
            )
            gmm_ms = (perf_counter() - gmm_start) * 1000.0
            before_combine_evt = torch.npu.current_stream().record_event()
            direct_scatter_payload = None
            direct_scatter_enabled = _to_bool_env(
                "VLLM_ASCEND_MOE_OFFLOAD_B2_DIRECT_SCATTER",
                "1",
            )
            if not direct_scatter_enabled:
                raise RuntimeError(
                    "experimental B2 wave execution requires native layer-level "
                    "top-k recombine; refusing the non-equivalent wave-local "
                    "combine fallback"
                )
            if direct_scatter_enabled:
                combine_start = perf_counter()
                direct_scatter_payload = B2DirectScatterPayload(
                    permuted_tokens=mlp_output,
                    expanded_row_idx=(
                        token_dispatch_output.combine_metadata.expanded_row_idx
                    ),
                    topk_weights=microbatch.topk_weights,
                    restore_token_indices=microbatch.restore_token_indices,
                    pair_offsets=(
                        microbatch_plan.pair_offsets
                        if microbatch_plan is not None
                        else None
                    ),
                    combine_metadata=token_dispatch_output.combine_metadata,
                )
                routed_out = torch.empty(
                    0,
                    fused_experts_input.hidden_states.shape[-1],
                    dtype=mlp_output.dtype,
                    device=mlp_output.device,
                )
                combine_ms = (perf_counter() - combine_start) * 1000.0
                combine_mode = "native_recombine_payload"
                scatter_mode = "layer_native_topk"
            else:
                combine_start = perf_counter()
                routed_out = self.token_dispatcher.token_combine(
                    hidden_states=mlp_output,
                    combine_metadata=token_dispatch_output.combine_metadata,
                )
                combine_ms = (perf_counter() - combine_start) * 1000.0
                combine_mode = "token_combine"
                scatter_mode = "layer_batch"
        finally:
            runtime.end_slot_compute(compute_handle)
            if old_top_k is None:
                try:
                    delattr(self.token_dispatcher, "top_k")
                except AttributeError:
                    pass
            else:
                self.token_dispatcher.top_k = old_top_k

        pair_wave_ms = (perf_counter() - pair_wave_start) * 1000.0
        return (
            FusedExpertsResult(
                routed_out=routed_out,
                before_dispatch_evt=before_combine_evt,
                before_gmm2_evt=before_gmm2_evt,
                before_combine_evt=before_combine_evt,
                group_list_type=token_dispatch_output.group_list_type,
                expert_tokens=token_dispatch_output.group_list,
                swiglu_limit=fused_experts_input.swiglu_limit,
            ),
            microbatch.restore_token_indices,
            {
                "pair_wave_ms": pair_wave_ms,
                "microbatch_ms": microbatch_ms,
                "microbatch_source": microbatch_source,
                "pertoken_scale_ms": pertoken_scale_ms,
                "build_input_ms": build_input_ms,
                "dispatch_ms": dispatch_ms,
                "build_mlp_input_ms": build_mlp_input_ms,
                "gmm_ms": gmm_ms,
                "combine_ms": combine_ms,
                "combine_mode": combine_mode,
                "scatter_mode": scatter_mode,
                "scatter_descriptor_mode": (
                    "native_topk"
                    if direct_scatter_payload is not None
                    else "deferred"
                ),
                "direct_scatter_payload": direct_scatter_payload,
            },
        )

    def _run_b2_single_wave(self, *, fused_experts_input, prepared):
        from vllm_moe_offload_ascend.moe_offload.phase_split import (
            build_b2_wave_routing,
        )

        runtime = get_moe_offload_runtime()
        physical_topk_ids = prepared.log2phy[fused_experts_input.topk_ids]
        safe_ids, masked_weights = build_b2_wave_routing(
            physical_topk_ids,
            fused_experts_input.topk_weights,
        )

        wave_input = MoEFusedExpertsInput(
            hidden_states=fused_experts_input.hidden_states,
            topk_weights=masked_weights,
            topk_ids=safe_ids,
            weights=MoEWeights(
                w1=prepared.w1,
                w2=prepared.w2,
                w1_bias=fused_experts_input.weights.w1_bias,
                w2_bias=fused_experts_input.weights.w2_bias,
                w1_scale=fused_experts_input.weights.w1_scale,
                w2_scale=fused_experts_input.weights.w2_scale,
                w1_scale_bias=fused_experts_input.weights.w1_scale_bias,
                w2_scale_bias=fused_experts_input.weights.w2_scale_bias,
                w1_offset=fused_experts_input.weights.w1_offset,
                w2_offset=fused_experts_input.weights.w2_offset,
            ),
            routing=MoERoutingParams(
                expert_map=None,
                global_redundant_expert_num=(
                    fused_experts_input.routing.global_redundant_expert_num
                ),
                mc2_mask=fused_experts_input.routing.mc2_mask,
                apply_router_weight_on_input=(
                    fused_experts_input.routing.apply_router_weight_on_input
                ),
                log2phy=None,
                physical_expert_count=prepared.physical_expert_count,
                pertoken_scale=fused_experts_input.routing.pertoken_scale,
            ),
            quant=fused_experts_input.quant,
            activation=fused_experts_input.activation,
            need_trans=fused_experts_input.need_trans,
            dynamic_eplb=fused_experts_input.dynamic_eplb,
            swiglu_limit=fused_experts_input.swiglu_limit,
            offload=fused_experts_input.offload,
        )

        compute_handle = runtime.begin_slot_compute(prepared)
        try:
            token_dispatch_output = self.token_dispatcher.token_dispatch(
                token_dispatch_input=build_token_dispatch_input(
                    fused_experts_input=wave_input
                )
            )
            mlp_compute_input = build_mlp_compute_input(
                fused_experts_input=wave_input,
                token_dispatch_output=token_dispatch_output,
                use_fusion_ops=self.use_fusion_ops,
            )
            mlp_output, before_gmm2_evt = _unpack_mlp_apply_result(
                self._apply_mlp(mlp_compute_input)
            )
            before_combine_evt = torch.npu.current_stream().record_event()
            routed_out = self.token_dispatcher.token_combine(
                hidden_states=mlp_output,
                combine_metadata=token_dispatch_output.combine_metadata,
            )
        finally:
            runtime.end_slot_compute(compute_handle)
        return FusedExpertsResult(
            routed_out=routed_out,
            before_dispatch_evt=before_combine_evt,
            before_gmm2_evt=before_gmm2_evt,
            before_combine_evt=before_combine_evt,
            group_list_type=token_dispatch_output.group_list_type,
            expert_tokens=token_dispatch_output.group_list,
            swiglu_limit=fused_experts_input.swiglu_limit,
        )

    def fused_experts(self, fused_experts_input):
        offload = fused_experts_input.offload
        runtime = get_moe_offload_runtime()
        if (
            offload is not None
            and offload.enabled
            and bool(
                _infer_forward_is_prefill_from_tokens(
                    int(fused_experts_input.hidden_states.shape[0])
                )
            )
            and runtime.is_resident_layer(int(offload.layer_id))
        ):
            if _os.environ.get("SEW_B2_PROBE") or _os.environ.get("SEW_OFFLOAD_PROBE"):
                print(
                    f"SEW_B2 branch=SKIP reason=resident_prefill_native "
                    f"layer={offload.layer_id}",
                    flush=True,
                )
            profile_enabled = bool(
                getattr(runtime.config, "gmm_profile_path", None)
                or _os.getenv("VLLM_ASCEND_MOE_GMM_PROFILE_PATH")
                or _os.getenv("VLLM_ASCEND_MOE_OFFLOAD_PROFILE_PATH")
            )
            profile_start = perf_counter() if profile_enabled else 0.0
            out = original_fused_experts(self, fused_experts_input)
            if profile_enabled:
                runtime._record_profile_event(
                    "prefill_resident_native",
                    layer_id=int(offload.layer_id),
                    start=profile_start,
                    payload={
                        "n_tokens": int(fused_experts_input.hidden_states.shape[0]),
                        "path": "native_fused_moe",
                        "entry": "comm_method",
                    },
                )
            return out

        before_dispatch_evt = torch.npu.current_stream().record_event()
        b2_out = self._maybe_run_b2_wave_prefill(
            fused_experts_input,
            before_dispatch_evt,
        )
        if b2_out is not None:
            return b2_out
        try:
            return original_fused_experts(self, fused_experts_input)
        finally:
            compute_handle = getattr(
                self,
                "_ascend_moe_offload_slot_compute_handle",
                None,
            )
            if hasattr(self, "_ascend_moe_offload_slot_compute_handle"):
                delattr(self, "_ascend_moe_offload_slot_compute_handle")
            runtime.end_slot_compute(compute_handle)

    _maybe_apply_moe_offload_plan._ascend_moe_offload_patch_tag = patch_tag
    fused_experts._ascend_moe_offload_patch_tag = patch_tag
    _maybe_apply_moe_offload_plan.__wrapped__ = original_maybe_apply
    fused_experts.__wrapped__ = original_fused_experts
    cls._with_prepared_slot_weights = _with_prepared_slot_weights
    cls._maybe_apply_moe_offload_plan = _maybe_apply_moe_offload_plan
    cls._maybe_run_b2_wave_prefill = _maybe_run_b2_wave_prefill
    cls._run_b2_fused_shared_once = _run_b2_fused_shared_once
    cls._run_b2_wave_prefill = _run_b2_wave_prefill
    cls._run_b2_full_layer_reference = _run_b2_full_layer_reference
    cls._run_b2_pair_wave = _run_b2_pair_wave
    cls._run_b2_single_wave = _run_b2_single_wave
    cls.fused_experts = fused_experts
    cls._ascend_moe_offload_runtime_patch = True

    if _to_bool_env("SEW_PATCH_PROBE", "0"):
        print(
            "SEW_PATCH comm_hook_installed "
            f"maybe={cls._maybe_apply_moe_offload_plan.__module__}.",
            flush=True,
        )

    if (
        callable(original_setup_moe_comm_method)
        and not getattr(
            original_setup_moe_comm_method,
            "_ascend_moe_offload_runtime_patch",
            False,
        )
    ):
        _wrap_setup_moe_comm_method(_comm, original_setup_moe_comm_method)


def _wrap_setup_moe_comm_method(_comm: Any, original_setup_moe_comm_method: Callable[..., Any]) -> None:
    """Reinstall MoE runtime hooks after vLLM-Ascend creates comm instances."""

    def setup_moe_comm_method(*args, **kwargs):
        _patch_moe_comm_method_runtime_hooks(_comm)
        result = original_setup_moe_comm_method(*args, **kwargs)
        _patch_moe_comm_method_runtime_hooks(_comm)
        if _to_bool_env("SEW_PATCH_PROBE", "0"):
            methods = getattr(_comm, "_MoECommMethods", {})
            for key, method in getattr(methods, "items", lambda: ())():
                maybe_apply = getattr(type(method), "_maybe_apply_moe_offload_plan", None)
                print(
                    "SEW_PATCH comm_instance "
                    f"type={type(method).__name__} key={key} "
                    f"patched={getattr(maybe_apply, '_ascend_moe_offload_patch_tag', '')}",
                    flush=True,
                )
        return result

    setup_moe_comm_method._ascend_moe_offload_runtime_patch = True
    setup_moe_comm_method.__wrapped__ = original_setup_moe_comm_method
    _comm.setup_moe_comm_method = setup_moe_comm_method
    fused_moe_module = sys.modules.get("vllm_ascend.ops.fused_moe.fused_moe")
    if fused_moe_module is not None:
        try:
            fused_moe_module.setup_moe_comm_method = setup_moe_comm_method
        except Exception:
            pass


def _patch_fused_moe_runtime_hooks(_fused_moe: Any) -> None:
    _patch_unquantized_moe_method(_fused_moe)
    _patch_ascend_moe_runner(_fused_moe)


def _ondemand_stage_enabled() -> bool:
    """Naive per-layer on-demand expert staging (no fixed-slot dataplane).

    When on, the apply() wrapper stages a layer's CPU-resident expert weights to
    the compute device right before the grouped MLP and evicts them back right
    after. This is the honest baseline for "offload without fixed slots, eager":
    it produces tokens under a tight HBM budget by paying a full H2D copy of the
    active layer's experts on every forward, instead of a fixed-slot cache.
    """
    return _to_bool_env("VLLM_ASCEND_MOE_OFFLOAD_ONDEMAND_STAGE", "0")


def _ondemand_stage_device():
    """Current NPU device, or None when no NPU is available."""
    import torch

    npu = getattr(torch, "npu", None)
    if npu is None:
        return None
    try:
        if hasattr(npu, "is_available") and not npu.is_available():
            return None
        return torch.device("npu", npu.current_device())
    except Exception:
        return None


def _stage_layer_experts_to_device(layer):
    """Move a layer's CPU-resident expert weights to device for one forward.

    Returns (staged, restore_fn). ``restore_fn`` rebinds the original CPU tensors
    (zero-copy: the CPU tensor object is retained, never freed) and drops the
    device staging copy. Only ``w13_weight``/``w2_weight`` that are actually on
    CPU are touched, so resident layers and already-on-device layers are no-ops.
    """
    device = _ondemand_stage_device()
    if device is None:
        return False, None
    try:
        from vllm_ascend.utils import maybe_trans_nz
    except Exception:
        maybe_trans_nz = lambda w: w  # noqa: E731 - NZ recast is a no-op fallback

    moved = []
    for name in ("w13_weight", "w2_weight"):
        param = getattr(layer, name, None)
        data = getattr(param, "data", None)
        if data is None or data.device.type != "cpu":
            continue
        cpu_data = data
        # A CPU round-trip drops NZ back to ND; recast to match what the load
        # path produced on device (no-op when NZ is disabled for this dtype).
        param.data = maybe_trans_nz(cpu_data.to(device))
        moved.append((param, cpu_data))
    if not moved:
        return False, None

    def _restore() -> None:
        for param, cpu_data in moved:
            param.data = cpu_data
        _empty_npu_cache_if_available()

    return True, _restore


def _empty_npu_cache_if_available() -> None:
    import torch

    npu = getattr(torch, "npu", None)
    if npu is None or not hasattr(npu, "empty_cache"):
        return
    try:
        npu.empty_cache()
    except Exception:
        pass


def _patch_unquantized_moe_method(_fused_moe: Any) -> None:
    cls = getattr(_fused_moe, "AscendUnquantizedFusedMoEMethod", None)
    if cls is None or getattr(cls, "_ascend_moe_offload_runtime_patch", False):
        return

    import threading

    import torch

    from vllm_moe_offload_ascend.moe_offload.runtime import (
        get_moe_offload_runtime,
    )
    from vllm_moe_offload_ascend.moe_offload.cpu_first_loader import (
        ensure_moe_layer_id,
        maybe_create_unquantized_cpu_first_weights,
        maybe_process_unquantized_cpu_first_weights,
        maybe_register_unquantized_fixed_slot_weights,
    )
    from vllm_moe_offload_ascend.ops.fused_moe import moe_seam_inject

    base_unquantized_cls = getattr(_fused_moe, "UnquantizedFusedMoEMethod", None)
    if base_unquantized_cls is not None and not getattr(
        base_unquantized_cls,
        "_ascend_moe_offload_cpu_first_create_patch",
        False,
    ):
        original_create_weights = base_unquantized_cls.create_weights

        def create_weights(
            self,
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            **extra_weight_attrs,
        ):
            layer_id = ensure_moe_layer_id(layer)
            runtime = get_moe_offload_runtime()
            handled = maybe_create_unquantized_cpu_first_weights(
                self,
                layer,
                runtime=runtime,
                num_experts=num_experts,
                hidden_size=hidden_size,
                intermediate_size_per_partition=intermediate_size_per_partition,
                params_dtype=params_dtype,
                extra_weight_attrs=extra_weight_attrs,
            )
            if _to_bool_env("SEW_CPU_FIRST_PROBE", "0"):
                print(
                    "SEW_CPU_FIRST create "
                    f"method={type(self).__name__} layer_id={layer_id} "
                    f"moe_layer_id={getattr(layer, 'moe_layer_id', None)} "
                    f"cpu_first={runtime.config.cpu_first_load} handled={handled}",
                    flush=True,
                )
            if handled:
                return None
            return original_create_weights(
                self,
                layer,
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                params_dtype,
                **extra_weight_attrs,
            )

        create_weights.__wrapped__ = original_create_weights
        base_unquantized_cls.create_weights = create_weights
        base_unquantized_cls._ascend_moe_offload_cpu_first_create_patch = True

    original_process_weights = cls.process_weights_after_loading
    original_apply = cls.apply
    original_select_experts = _fused_moe.select_experts
    layer_context = threading.local()

    def select_experts(*args, **kwargs):
        layer_id = getattr(layer_context, "layer_id", None)
        if layer_id is not None:
            injected = moe_seam_inject.peek_injected_topk(int(layer_id))
            if injected is not None:
                hidden_states = kwargs.get("hidden_states")
                if hidden_states is None and args:
                    hidden_states = args[0]
                if (
                    _to_bool_env(
                        "VLLM_ASCEND_MOE_OFFLOAD_COMPARE_ROUTER",
                        "0",
                    )
                    and hidden_states is not None
                    and int(hidden_states.shape[0]) <= 64
                    and not _is_torch_compile_tracing()
                    and not _is_current_graph_capturing()
                ):
                    native_weights, native_ids = original_select_experts(
                        *args,
                        **kwargs,
                    )
                    injected_weights, injected_ids = injected
                    id_mismatch_count = int(
                        torch.count_nonzero(native_ids != injected_ids).item()
                    )
                    weight_diff = (
                        native_weights.float() - injected_weights.float()
                    ).abs()
                    print(
                        "SEW_ROUTER_COMPARE "
                        f"layer={int(layer_id)} "
                        f"tokens={int(hidden_states.shape[0])} "
                        f"ids_equal={bool(torch.equal(native_ids, injected_ids))} "
                        f"id_mismatch_count={id_mismatch_count} "
                        f"weights_equal={bool(torch.equal(native_weights, injected_weights))} "
                        f"weight_max_abs={float(weight_diff.max().item())} "
                        f"weight_mean_abs={float(weight_diff.mean().item())}",
                        flush=True,
                    )
                    from vllm_moe_offload_ascend.moe_offload.router_parity import (
                        record_router_snapshot,
                    )

                    native_logits = kwargs.get("router_logits")
                    if native_logits is None and len(args) > 1:
                        native_logits = args[1]
                    if native_logits is not None:
                        record_router_snapshot(
                            role="native",
                            layer_id=int(layer_id),
                            router_logits=native_logits,
                            topk_ids=native_ids,
                            topk_weights=native_weights,
                        )
                return injected
        topk_weights, topk_ids = original_select_experts(*args, **kwargs)
        if layer_id is not None:
            runtime = get_moe_offload_runtime()
            num_logical_experts = int(kwargs.get("num_experts", -1))
            if (
                runtime.config.offload_stage_seam
                and runtime.should_use_fixed_slot_plan_for_layer(int(layer_id))
                and num_logical_experts > 0
            ):
                torch.ops.vllm.moe_offload_stage(
                    topk_ids,
                    int(layer_id),
                    num_logical_experts,
                    _infer_forward_phase_int(int(topk_ids.shape[0])),
                )
        return topk_weights, topk_ids

    _fused_moe.select_experts = select_experts

    def process_weights_after_loading(self, layer):
        layer_id = ensure_moe_layer_id(layer)
        runtime = get_moe_offload_runtime()
        handled = maybe_process_unquantized_cpu_first_weights(
            self,
            layer,
            runtime=runtime,
        )
        if _to_bool_env("SEW_CPU_FIRST_PROBE", "0"):
            print(
                "SEW_CPU_FIRST process "
                f"method={type(self).__name__} layer_id={layer_id} "
                f"marker={getattr(layer, '_ascend_moe_cpu_first_load', False)} "
                f"cpu_first={runtime.config.cpu_first_load} handled={handled} "
                f"registered={runtime.is_layer_registered(layer_id) if layer_id >= 0 else False}",
                flush=True,
            )
        if handled:
            return None
        result = original_process_weights(self, layer)
        registered = maybe_register_unquantized_fixed_slot_weights(
            layer,
            runtime=runtime,
        )
        layer_id = int(getattr(layer, "layer_id", -1))
        if layer_id < 0:
            return result  # no layer_id → cannot identify slot bank; skip staging
        if (
            runtime.should_use_fixed_slot_plan_for_layer(layer_id)
            and registered
        ):
            buf = runtime.log2phy_buffer(layer_id)
            total_logical_experts = (
                int(buf.numel()) if buf is not None else int(getattr(layer, "w13_weight").shape[0])
            )
            routed_expert_count = runtime.routed_expert_count_for_layer(
                layer_id=layer_id,
                fallback=total_logical_experts,
            )
            if runtime.is_static_residency_regime(routed_expert_count):
                runtime.stage_full_residency_slot_plan(layer_id=layer_id)
        return result

    def apply(self, *args, **kwargs):
        layer = kwargs.get("layer")
        if layer is None and args:
            layer = args[0]
        old_layer_id = getattr(layer_context, "layer_id", None)
        if layer is not None:
            lid = int(getattr(layer, "layer_id", -1))
            if lid >= 0:
                layer_context.layer_id = lid
            # lid == -1: layer has no id; skip context so the seam does not
            # collide with other unidentified layers sharing key -1.
        # Naive on-demand staging (no fixed slots): if this layer's experts were
        # offloaded to CPU at load time, stage them to device for this forward
        # and evict right after, so the plain grouped matmul sees device weights.
        restore_staged = None
        if layer is not None and _ondemand_stage_enabled():
            _staged, restore_staged = _stage_layer_experts_to_device(layer)
        try:
            return original_apply(self, *args, **kwargs)
        finally:
            if restore_staged is not None:
                restore_staged()
            if old_layer_id is None and hasattr(layer_context, "layer_id"):
                delattr(layer_context, "layer_id")
            else:
                layer_context.layer_id = old_layer_id

    cls.process_weights_after_loading = process_weights_after_loading
    cls.apply = apply
    cls._ascend_moe_offload_runtime_patch = True


def _patch_ascend_moe_runner(_fused_moe: Any) -> None:
    cls = getattr(_fused_moe, "AscendMoERunner", None)
    if cls is None or getattr(cls, "_ascend_moe_offload_seam_patch", False):
        return

    import os as _os

    import torch
    from time import perf_counter

    original_select_forward = cls._select_forward

    def _seam_config_guards_pass(self) -> bool:
        from vllm_moe_offload_ascend.moe_offload.runtime import get_moe_offload_runtime

        def _probe(reason: str) -> None:
            if _os.environ.get("SEW_SEAM_PROBE"):
                print(
                    f"SEW_SEAM_SELECT layer={getattr(self, 'layer_name', '?')} "
                    f"config_guard={reason}",
                    flush=True,
                )

        runtime = get_moe_offload_runtime()
        if not runtime.config.offload_stage_seam:
            _probe("FAIL:offload_stage_seam_off")
            return False
        moe_config = self.moe_config
        if (
            getattr(moe_config, "dp_size", 1) > 1
            or getattr(moe_config, "ep_size", 1) > 1
            or getattr(moe_config, "tp_size", 1) > 1
            or getattr(moe_config, "pcp_size", 1) > 1
        ):
            _probe(
                f"FAIL:multicard dp={getattr(moe_config, 'dp_size', 1)} "
                f"ep={getattr(moe_config, 'ep_size', 1)} "
                f"tp={getattr(moe_config, 'tp_size', 1)} "
                f"pcp={getattr(moe_config, 'pcp_size', 1)}"
            )
            return False
        _probe("PASS")
        return True

    def _select_forward(self):
        from vllm_moe_offload_ascend.moe_offload.runtime import get_moe_offload_runtime

        runtime = get_moe_offload_runtime()
        if not runtime.config.offload_stage_seam:
            return original_select_forward(self)
        if self._seam_config_guards_pass():
            return self._seam_forward_entry
        raise RuntimeError(
            "LatchMoE offload-stage seam was requested with an unsupported "
            "global configuration; refusing native/eager fallback"
        )

    def _seam_forward_entry(
        self,
        hidden_states,
        router_logits,
        shared_experts_input,
        input_ids,
        layer_name,
        hidden_dim_unpadded=None,
    ):
        def _native_moe_forward(
            *,
            native_hidden_states=None,
            native_router_logits=None,
            native_layer_name=None,
        ):
            args = [
                (
                    hidden_states
                    if native_hidden_states is None
                    else native_hidden_states
                ),
                (
                    router_logits
                    if native_router_logits is None
                    else native_router_logits
                ),
                shared_experts_input,
                input_ids,
                layer_name if native_layer_name is None else native_layer_name,
            ]
            if hidden_dim_unpadded is not None:
                args.append(hidden_dim_unpadded)
            capability = getattr(self, "_seam_capability", None)
            if getattr(capability, "output_contract", "routed_tensor") == "shared_routed_tuple":
                return torch.ops.vllm.moe_forward_shared(*args)
            return torch.ops.vllm.moe_forward(*args)

        decision = getattr(self, "_seam_active", None)
        if decision is None:
            decision = self._resolve_seam_per_layer_guards()
            self._seam_active = decision

        if not decision:
            from vllm_moe_offload_ascend.moe_offload.capabilities import (
                CapabilitySupport,
                LatchMoECapabilityError,
                MoeCapabilityDescriptor,
            )

            descriptor = getattr(self, "_seam_capability", None)
            support = getattr(self, "_seam_capability_support", None)
            if isinstance(descriptor, MoeCapabilityDescriptor) and isinstance(
                support, CapabilitySupport
            ):
                raise LatchMoECapabilityError(descriptor, support)
            raise RuntimeError(
                "LatchMoE seam was selected but the layer capability could not "
                "be resolved; refusing native/eager fallback"
            )

        from vllm_moe_offload_ascend.moe_offload.runtime import (
            get_moe_offload_runtime,
        )
        runtime = get_moe_offload_runtime()
        phase_int = _infer_forward_phase_int(int(hidden_states.shape[0]))
        from vllm_moe_offload_ascend.ops.fused_moe.moe_offload_stage_op import (
            PHASE_PREFILL,
        )
        # Resident-layer native fast path only fires on a CONFIRMED prefill; an
        # unknown phase must not take the resident-native branch (preserves the
        # prior None->False behaviour for this guard).
        is_prefill = phase_int == PHASE_PREFILL
        if is_prefill and runtime.is_resident_layer(int(self._seam_layer_id)):
            is_compiling = _is_torch_compile_tracing()
            if _os.environ.get("SEW_SEAM_PROBE") and not is_compiling:
                print(
                    f"SEW_SEAM branch=PREFILL_RESIDENT_NATIVE "
                    f"layer={int(self._seam_layer_id)}",
                    flush=True,
                )
            profile_enabled = (
                not is_compiling
                and (
                    runtime.config.gmm_profile_path
                    or _os.getenv("VLLM_ASCEND_MOE_GMM_PROFILE_PATH")
                    or _os.getenv("VLLM_ASCEND_MOE_OFFLOAD_PROFILE_PATH")
                )
            )
            profile_start = perf_counter() if profile_enabled else 0.0
            out = _native_moe_forward()
            if profile_enabled:
                runtime._record_profile_event(
                    "prefill_resident_native",
                    layer_id=int(self._seam_layer_id),
                    start=profile_start,
                    payload={
                        "n_tokens": int(hidden_states.shape[0]),
                        "path": "native_fused_moe",
                    },
                )
            return out

        if layer_name == "from_forward_context":
            from vllm.model_executor.layers.fused_moe.runner.moe_runner import (
                get_layer_from_name,
            )

            indexed_layer = get_layer_from_name(layer_name)
            expected_layer = get_layer_from_name(self.layer_name)
            if indexed_layer is not expected_layer:
                raise RuntimeError(
                    "MoE seam forward-context order mismatch: "
                    f"expected {self.layer_name!r}, got "
                    f"layer_id={getattr(indexed_layer, 'layer_id', '?')}"
                )

        compare_layer_boundary = bool(
            _to_bool_env(
                "VLLM_ASCEND_MOE_OFFLOAD_COMPARE_LAYER_BOUNDARY",
                "0",
            )
            and int(hidden_states.shape[0]) <= 64
            and runtime.should_use_fixed_slot_plan_for_layer(
                int(self._seam_layer_id)
            )
        )
        native_routed_out = None
        real_name = self.layer_name
        original_weights_unavailable = bool(
            getattr(runtime.config, "cpu_first_load", False)
            or getattr(runtime.config, "release_original_expert_weights", False)
        )
        if compare_layer_boundary:
            if original_weights_unavailable:
                # CPU-first/release qualification deliberately removes the full
                # routed expert tensor. Compare the native runner and seam using
                # the same staged fixed-slot values instead of reading CPU or
                # zero-element placeholders through grouped matmul.
                topk_weights, topk_ids = torch.ops.vllm.moe_router_indirect(
                    hidden_states,
                    router_logits,
                    real_name,
                )
                torch.ops.vllm.moe_offload_stage(
                    topk_ids,
                    self._seam_layer_id,
                    self._seam_num_logical_experts,
                    phase_int,
                )
                native_result = _native_moe_forward(
                    native_hidden_states=hidden_states.clone(),
                    native_router_logits=router_logits.clone(),
                    native_layer_name=self.layer_name,
                )
            else:
                with runtime.suspend_fixed_slot_execution():
                    native_result = _native_moe_forward(
                        native_hidden_states=hidden_states.clone(),
                        native_router_logits=router_logits.clone(),
                        native_layer_name=self.layer_name,
                    )
            if isinstance(native_result, tuple):
                native_routed_out = (
                    native_result[0].clone(),
                    native_result[1].clone(),
                )
            else:
                native_routed_out = native_result.clone()

        if not compare_layer_boundary or not original_weights_unavailable:
            topk_weights, topk_ids = torch.ops.vllm.moe_router_indirect(
                hidden_states,
                router_logits,
                real_name,
            )

        torch.ops.vllm.moe_offload_stage(
            topk_ids,
            self._seam_layer_id,
            self._seam_num_logical_experts,
            phase_int,
        )
        # Normal selection always resolves a descriptor before reaching this
        # point.  Keep the historic routed-only default for test harnesses and
        # explicit integrations that set ``_seam_active=True`` themselves.
        capability = getattr(self, "_seam_capability", None)
        if getattr(capability, "output_contract", "routed_tensor") == "shared_routed_tuple":
            routed_out = torch.ops.vllm.moe_mlp_shared(
                hidden_states,
                router_logits,
                topk_weights,
                topk_ids,
                shared_experts_input,
                input_ids,
                real_name,
            )
        else:
            routed_out = torch.ops.vllm.moe_mlp(
                hidden_states,
                router_logits,
                topk_weights,
                topk_ids,
                shared_experts_input,
                input_ids,
                real_name,
            )
        # Tensor-value comparisons are diagnostics only.  Dynamo cannot
        # capture data-dependent operators such as ``torch.equal`` or scalar
        # reductions, so defer the whole comparison until the eager boundary.
        if (
            native_routed_out is not None
            and not _is_torch_compile_tracing()
            and not _is_current_graph_capturing()
        ):
            native_has_shared = isinstance(native_routed_out, tuple)
            seam_has_shared = isinstance(routed_out, tuple)
            native_compare = (
                native_routed_out[1]
                if native_has_shared
                else native_routed_out
            )
            routed_compare = routed_out[1] if seam_has_shared else routed_out
            output_diff = (native_compare.float() - routed_compare.float()).abs()
            shared_diagnostic = "shared_contract_equal=True"
            if native_has_shared and seam_has_shared:
                native_shared = native_routed_out[0]
                seam_shared = routed_out[0]
                shared_diff = (native_shared.float() - seam_shared.float()).abs()
                shared_diagnostic = (
                    f"shared_equal={bool(torch.equal(native_shared, seam_shared))} "
                    f"shared_max_abs={float(shared_diff.max().item())} "
                    f"shared_mean_abs={float(shared_diff.mean().item())}"
                )
            elif native_has_shared or seam_has_shared:
                shared_diagnostic = "shared_contract_equal=False"
            print(
                "SEW_LAYER_BOUNDARY_COMPARE "
                f"layer={int(self._seam_layer_id)} "
                f"tokens={int(hidden_states.shape[0])} "
                f"output_equal={bool(torch.equal(native_compare, routed_compare))} "
                f"output_max_abs={float(output_diff.max().item())} "
                f"output_mean_abs={float(output_diff.mean().item())} "
                f"{shared_diagnostic}",
                flush=True,
            )
        return routed_out

    def _resolve_seam_per_layer_guards(self) -> bool:
        from vllm.model_executor.layers.fused_moe.runner.moe_runner import (
            get_layer_from_name,
        )
        from vllm_moe_offload_ascend.moe_offload.capabilities import (
            CapabilitySupport,
            describe_layer_capability,
            evaluate_support,
        )

        try:
            layer = get_layer_from_name(self.layer_name)
        except Exception:
            return False
        self._seam_layer_id = int(getattr(layer, "layer_id", -1))
        if self._seam_layer_id < 0:
            # Layer lacks a layer_id attribute; the seam cannot safely key its
            # injection/log2phy registry by -1 (all such layers would collide).
            return False
        descriptor = describe_layer_capability(layer, self)
        support = evaluate_support(descriptor)
        extra_blockers: list[str] = list(support.blockers)
        if getattr(layer, "enable_npugraph_ex_static_kernel", False):
            extra_blockers.append("unsupported_static_expert_kernel")
        if (
            getattr(layer, "zero_expert_num", 0)
            and getattr(layer, "zero_expert_type", None) is not None
        ):
            extra_blockers.append("unsupported_zero_expert_path")
        if extra_blockers != list(support.blockers):
            support = CapabilitySupport(state="unsupported", blockers=tuple(extra_blockers))
        self._seam_capability = descriptor
        self._seam_capability_support = support
        from vllm_moe_offload_ascend.moe_offload.runtime import (
            get_moe_offload_runtime,
        )
        runtime = get_moe_offload_runtime()
        if descriptor.shared_mode == "fused_mix_placement":
            # A fused router emits appended shared IDs in the same top-k tensor
            # as routed IDs. The seam must not proceed until the runtime has
            # registered the corresponding immutable physical suffix; otherwise
            # a later stage call could treat shared IDs as evictable routed rows.
            layout_fn = getattr(runtime, "fused_shared_lane_layout", None)
            layout = (
                layout_fn(self._seam_layer_id)
                if callable(layout_fn)
                else None
            )
            expected_pinned_ids = tuple(
                range(
                    int(descriptor.routed_expert_count),
                    int(descriptor.routed_expert_count)
                    + int(descriptor.shared_expert_count),
                )
            )
            try:
                layout_matches = (
                    layout is not None
                    and int(layout.routed_expert_count)
                    == int(descriptor.routed_expert_count)
                    and int(layout.shared_expert_count)
                    == int(descriptor.shared_expert_count)
                    and int(layout.total_logical_expert_count)
                    == int(descriptor.routed_expert_count)
                    + int(descriptor.shared_expert_count)
                    and tuple(int(expert_id) for expert_id in layout.pinned_logical_ids)
                    == expected_pinned_ids
                )
            except (AttributeError, TypeError, ValueError):
                layout_matches = False
            if not layout_matches:
                blocker = (
                    "fused_shared_lane_unregistered"
                    if layout is None
                    else "fused_shared_lane_layout_mismatch"
                )
                support = CapabilitySupport(
                    state="unsupported",
                    blockers=tuple((*support.blockers, blocker)),
                )
                self._seam_capability_support = support
        if not support.enabled:
            return False
        self._seam_num_logical_experts = descriptor.routed_expert_count
        if descriptor.shared_mode == "external_resident":
            shared_experts = getattr(self, "_shared_experts", None)
            shared_gate = getattr(self, "shared_expert_gate", None)
            if shared_gate is None:
                shared_gate = getattr(shared_experts, "expert_gate", None)
            if shared_gate is None:
                shared_gate = getattr(
                    getattr(shared_experts, "_experts", None),
                    "expert_gate",
                    None,
                )
            runtime.register_resident_shared_weights(
                layer_id=self._seam_layer_id,
                shared_experts=shared_experts,
                shared_gate=shared_gate,
            )
        return True

    cls._seam_config_guards_pass = _seam_config_guards_pass
    cls._select_forward = _select_forward
    cls._seam_forward_entry = _seam_forward_entry
    cls._resolve_seam_per_layer_guards = _resolve_seam_per_layer_guards
    cls._ascend_moe_offload_seam_patch = True


def _patch_platform_autoconfig() -> None:
    try:
        import vllm_ascend.platform as _platform
        from vllm_moe_offload_ascend.moe_offload.autoconfig import register_moe_offload_cli_arg

        current = _platform.NPUPlatform.pre_register_and_update
        current_fn = getattr(current, "__func__", current)
        if getattr(current_fn, "_ascend_moe_offload_autoconfig_patch", False):
            return
        _original_fn = current_fn

        @classmethod  # type: ignore[misc]
        def _patched(cls, parser=None):
            _original_fn(cls, parser)
            _apply_env_defaults_from_gb()
            if parser is not None:
                register_moe_offload_cli_arg(parser)

        _patched.__func__._ascend_moe_offload_autoconfig_patch = True
        _platform.NPUPlatform.pre_register_and_update = _patched
    except Exception:
        pass  # best-effort: CLI arg registration is non-critical


def _patch_platform_splitting_ops() -> None:
    """Ensure PIECEWISE ACLGraph splits at the MoE offload staging seam.

    Newer hook branches carry this in platform.py.  Keep the same behavior as a
    plugin-side fallback for older hooks: after the platform has finished its
    normal config update, add vllm::moe_offload_stage when the default-off SEW
    seam switch is enabled.
    """
    try:
        import vllm_ascend.platform as _platform

        current = _platform.NPUPlatform.check_and_update_config
        current_fn = getattr(current, "__func__", current)
        if getattr(current_fn, "_ascend_moe_offload_splitting_ops_patch", False):
            return
        original_fn = current_fn

        @classmethod  # type: ignore[misc]
        def _patched(cls, vllm_config):
            original_fn(cls, vllm_config)
            _install_runtime_patches_when_ready()
            _ensure_moe_offload_splitting_op(vllm_config)

        _patched.__func__._ascend_moe_offload_splitting_ops_patch = True
        _platform.NPUPlatform.check_and_update_config = _patched
    except Exception:
        pass


def _ensure_moe_offload_splitting_op(
    vllm_config: Any,
    *,
    fail_closed: bool = False,
) -> bool:
    """Put the eager staging seam in the final piecewise graph config.

    General-plugin discovery can invoke ``register()`` while
    ``vllm_ascend.platform`` is only partially initialized. In that import
    order, installing the NPUPlatform wrapper is best-effort and may not take
    effect. This helper is also called at EngineArgs' final config boundary,
    after all platform defaults have settled, so a graph-enabled SEW launch
    cannot silently capture the dynamic staging logic.
    """

    if not _to_bool_env("VLLM_ASCEND_MOE_OFFLOAD_STAGE_SEAM", "0"):
        return False

    compilation_config = getattr(vllm_config, "compilation_config", None)
    if compilation_config is None:
        if fail_closed:
            raise RuntimeError(
                "LatchMoE SEW graph mode requires vLLM compilation_config"
            )
        return False

    cudagraph_mode = getattr(compilation_config, "cudagraph_mode", None)
    try:
        requires_piecewise = bool(
            cudagraph_mode is not None
            and cudagraph_mode.requires_piecewise_compilation()
        )
        has_full_graph = bool(
            cudagraph_mode is not None and cudagraph_mode.has_full_cudagraphs()
        )
    except Exception as exc:
        if fail_closed:
            raise RuntimeError(
                "LatchMoE could not validate the final ACLGraph mode"
            ) from exc
        return False

    if has_full_graph:
        if fail_closed:
            raise RuntimeError(
                "LatchMoE SEW offload requires PIECEWISE-only ACLGraph so "
                "vllm::moe_offload_stage executes eagerly during replay; "
                f"got cudagraph_mode={cudagraph_mode}"
            )
        return False

    if not requires_piecewise:
        return False

    splitting_ops = getattr(compilation_config, "splitting_ops", None)
    if splitting_ops is None:
        compilation_config.splitting_ops = []
        splitting_ops = compilation_config.splitting_ops
    if "vllm::moe_offload_stage" not in splitting_ops:
        splitting_ops.append("vllm::moe_offload_stage")

    present = "vllm::moe_offload_stage" in splitting_ops
    if fail_closed and not present:
        raise RuntimeError(
            "LatchMoE refused Graph startup because the final splitting_ops "
            "does not contain vllm::moe_offload_stage"
        )
    return present


def _patch_engine_args_autoconfig() -> None:
    """Wrap EngineArgs.create_engine_config to apply MoE offload defaults."""
    try:
        from vllm.engine.arg_utils import EngineArgs
        from vllm_moe_offload_ascend.moe_offload.autoconfig import apply_moe_offload_defaults

        current = EngineArgs.create_engine_config
        if getattr(current, "_ascend_moe_offload_autoconfig_patch", False):
            return
        _original = current

        def _patched(self, *args, **kwargs):
            apply_moe_offload_defaults(self)
            # Retry after EngineArgs and the active platform have finished
            # importing. This is the real vllm serve import order that the early
            # platform-plugin hook could previously miss.
            _patch_platform_splitting_ops()
            vllm_config = _original(self, *args, **kwargs)
            stage_present = _ensure_moe_offload_splitting_op(
                vllm_config,
                fail_closed=True,
            )
            if stage_present:
                print(
                    "LATCHMOE_GRAPH_CONFIG "
                    "cudagraph_mode=PIECEWISE "
                    "splitting_op=vllm::moe_offload_stage status=enabled",
                    flush=True,
                )
            return vllm_config

        _patched._ascend_moe_offload_autoconfig_patch = True
        EngineArgs.create_engine_config = _patched
    except Exception:
        pass  # best-effort
