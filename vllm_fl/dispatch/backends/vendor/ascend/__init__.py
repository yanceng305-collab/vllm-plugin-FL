# Copyright (c) 2026 BAAI. All rights reserved.
#
# Ascend vendor backend for vllm-plugin-FL.
#
# This package provides Ascend NPU support with two cooperating parts:
#   * ``AscendBackend`` - FlagOS dispatch operator implementations
#     (silu / rms_norm / rotary / attention) used by the plugin's shared models.
#   * ``register`` / ``register_model`` / ``_ensure_global_patch`` - the full
#     Ascend platform + DeepSeek-V4 model + DSA sparse attention + msmodelslim
#     w8a8 quantization stack (vendored and adapted), used to run DeepSeek-V4
#     on Ascend NPU.

from .ascend import AscendBackend

# NOTE: ``patch_mamba_config()`` is intentionally NOT invoked at import time.
# Importing it eagerly pulls in vLLM model modules during platform resolution
# and causes a circular import. Mamba / Qwen3-Next config patching is applied
# lazily through the global patch (``adapt_patch``) via ``_ensure_global_patch``.
from .fl_patch import patch_mamba_config  # noqa: F401  (exposed; applied lazily)

_GLOBAL_PATCH_APPLIED = False


def _ensure_global_patch():
    """Apply process-wide vLLM patches before engine-core initialization.

    vLLM loads general plugins in engine-core subprocesses. Global patches that
    affect scheduler and engine code must also be applied through these plugin
    entry points.
    """
    global _GLOBAL_PATCH_APPLIED
    if _GLOBAL_PATCH_APPLIED:
        return

    from vllm_fl.dispatch.backends.vendor.ascend.utils import adapt_patch

    adapt_patch(is_global_patch=True)
    _GLOBAL_PATCH_APPLIED = True


def register():
    """Register the Ascend NPU platform (DeepSeek-V4 stack)."""
    return "vllm_fl.dispatch.backends.vendor.ascend.platform.NPUPlatform"


def register_connector():
    _ensure_global_patch()

    from vllm_fl.dispatch.backends.vendor.ascend.distributed.kv_transfer import (
        register_connector,
    )

    register_connector()


def register_model_loader():
    _ensure_global_patch()

    from .model_loader.netloader import register_netloader
    from .model_loader.rfork import register_rforkloader

    register_netloader()
    register_rforkloader()


def register_service_profiling():
    _ensure_global_patch()

    from .profiling_config import generate_service_profiling_config

    generate_service_profiling_config()


def register_model():
    from .models import register_model

    register_model()


__all__ = [
    "AscendBackend",
    "register",
    "register_model",
    "_ensure_global_patch",
]
