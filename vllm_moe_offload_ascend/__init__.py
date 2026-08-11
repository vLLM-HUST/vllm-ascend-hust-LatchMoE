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
"""vllm-moe-offload-ascend: MoE Expert Offloading plugin for vllm-ascend.

Registers via vllm's platform_plugins entry point. Calling register()
monkey-patches the hook points in vllm_ascend so the real MoE offload
logic is active instead of the null stubs.
"""

import os


_COMPAT_ONLY_ENV = "VLLM_ASCEND_MOE_OFFLOAD_COMPAT_ONLY"


def _compat_only_enabled() -> bool:
    return os.getenv(_COMPAT_ONLY_ENV, "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def register() -> None:
    """Entry point called by vllm's platform plugin system at startup."""
    if _compat_only_enabled():
        from vllm_moe_offload_ascend.patches.patch_fused_moe import (
            apply_cann_compat_patches,
        )

        apply_cann_compat_patches()
        return

    from vllm_moe_offload_ascend.patches.patch_fused_moe import apply_patches

    apply_patches()
