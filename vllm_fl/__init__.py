# Copyright (c) 2025 BAAI. All rights reserved.

import os
import logging
import sys

# torch.float4_e2m1fn_x2 exists only in CUDA builds of PyTorch 2.7+.
# vllm.ir.tolerances references it at module level, so we inject a sentinel
# before any vllm.ir import can happen.
if "torch" in sys.modules:
    _torch = sys.modules["torch"]
    if not hasattr(_torch, "float4_e2m1fn_x2"):
        _torch.float4_e2m1fn_x2 = _torch.uint8
else:
    import torch as _torch
    if not hasattr(_torch, "float4_e2m1fn_x2"):
        _torch.float4_e2m1fn_x2 = _torch.uint8
del _torch

from vllm_fl.utils import get_op_config as _get_op_config

from . import version as version  # PyTorch-style: vllm_fl.version.git_version


logger = logging.getLogger(__name__)


def __getattr__(name):
    if name == "distributed":
        import importlib
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _patch_transformers_compat():
    """Patch transformers compatibility for ALLOWED_LAYER_TYPES and tokenizer."""
    import transformers.configuration_utils as cfg
    if not hasattr(cfg, "ALLOWED_LAYER_TYPES"):
        cfg.ALLOWED_LAYER_TYPES = getattr(
            cfg, "ALLOWED_ATTENTION_LAYER_TYPES", ()
        )


def _register_flagcx_connector():
    from vllm.distributed.kv_transfer.kv_connector.factory import (
        KVConnectorFactory,
    )

    for _alias in ("FlagCXConnector", "FlagcxConnector"):
        if _alias not in KVConnectorFactory._registry:
            KVConnectorFactory.register_connector(
                _alias,
                "vllm_fl.distributed.kv_transfer.flagcx_connector",
                "FlagCXConnector",
            )


def _patch_flash_attn_import():
    """Stub vllm.vllm_flash_attn if CUDA flash attention C extensions are missing."""
    import sys
    if "vllm.vllm_flash_attn" in sys.modules:
        return
    try:
        import vllm.vllm_flash_attn  # noqa: F401
    except ImportError:
        import types
        stub = types.ModuleType("vllm.vllm_flash_attn")
        stub.FA2_AVAILABLE = False
        stub.FA3_AVAILABLE = False
        stub.fa_version_unsupported_reason = lambda *a, **kw: "flash_attn C extensions not available"
        stub.flash_attn_varlen_func = None
        stub.get_scheduler_metadata = None
        stub.is_fa_version_supported = lambda *a, **kw: False
        sys.modules["vllm.vllm_flash_attn"] = stub


def _patch_custom_ops():
    """Register torch.ops._C op schemas when vllm._C is unavailable."""
    try:
        import vllm._C  # noqa: F401
        return
    except (ImportError, OSError):
        pass

    try:
        import vllm_fl._C  # noqa: F401
    except (ImportError, OSError) as e:
        logger.debug("Failed to import vllm_fl._C: %s", e)

    from vllm_fl.ops._C_ops_registry import register_op_schemas
    register_op_schemas()


def _is_ascend_environment():
    """Return True only on an Ascend NPU host with the vendored Ascend stack.

    The FL plugin vendors the Ascend DeepSeek-V4 stack under
    ``vllm_fl.dispatch.backends.vendor.ascend`` (NPU platform, DeepSeek-V4 model, DSA sparse
    attention, msmodelslim w8a8 quantization). Preconditions are checked with
    ``importlib.util.find_spec()`` so we never import torch_npu merely to make
    the decision, and we do NOT depend on the external Ascend plugin package.
    The final confirmation (``torch.npu.is_available()``) is guarded: any
    failure (no NPU driver, import error, missing packages, ...) makes this
    return False so the normal FL code path is used unchanged.
    """
    try:
        import importlib.util
        if importlib.util.find_spec("torch_npu") is None:
            return False
        if importlib.util.find_spec("vllm_fl.dispatch.backends.vendor.ascend") is None:
            return False
        import torch
        import torch_npu  # noqa: F401
        return bool(torch.npu.is_available())
    except Exception:
        return False


def register():
    """Register the FL platform."""
    if _is_ascend_environment():
        # Ascend host with the vendored Ascend stack present: activate the
        # vendored NPU platform so the model runs exactly like the working
        # baseline (NPUPlatform + ascend quant configs + Ascend DeepSeek-V4/MTP
        # models, all applied via
        # NPUPlatform.pre_register_and_update()/check_and_update_config()).
        # No dependency on the external Ascend plugin package.
        multiproc_method = os.environ.get("VLLM_WORKER_MULTIPROC_METHOD")
        if multiproc_method is None:
            os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        try:
            from vllm_fl.dispatch.backends.vendor.ascend import register as _ascend_register
            return _ascend_register()
        except Exception as e:
            logger.warning(
                "Vendored Ascend register() failed on Ascend host (%s); "
                "falling back to FL platform registration.", e
            )

    _patch_custom_ops()
    _patch_flash_attn_import()
    _patch_transformers_compat()

    # Model-specific platform patches
    from vllm_fl.patches.glm_moe_dsa import apply_platform_patches as glm5_platform
    glm5_platform()

    # Note: FlagCX connector registration is deferred to register_model()
    # to avoid circular imports during VllmConfig.__post_init__ in spawned
    # subprocesses.

    multiproc_method = os.environ.get("VLLM_WORKER_MULTIPROC_METHOD")
    if multiproc_method is None:
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    _get_op_config()

    return "vllm_fl.platform.PlatformFL"

def register_quant_linear():
    from vllm.platforms import current_platform
    # vllm.model_executor.kernels.linear triggers cutlass_scaled_mm_supports_fp8
    # at module level, which requires torch.ops._C — not available on MUSA.
    if current_platform.device_type == "musa":
        return
    from vllm_fl.quantization.quant_linear import add_oot_quant_kernel
    add_oot_quant_kernel()

def register_router():
    from vllm.platforms import current_platform
    # fused_moe import chain triggers cutlass_scaled_mm_supports_fp8 on MUSA
    if current_platform.device_type == "musa":
        return
    from vllm_fl.utils import is_oot_enabled
    if not is_oot_enabled():
        return
    from vllm_fl.ops.fused_moe.router import replace_router_with_fl
    replace_router_with_fl()

def register_model():
    """Register FL-specific models not yet upstream."""
    if _is_ascend_environment():
        # Delegate model registration to the vendored Ascend stack (Ascend
        # DeepSeek-V4 / DeepSeekV4MTP models, hunyuan compat shim, etc.) on
        # Ascend hosts. No dependency on the external Ascend plugin package.
        try:
            # Replicate the vendored Ascend stack's general-plugin behavior.
            # Upstream registers FOUR general plugins (ascend_kv_connector,
            # ascend_model_loader, ascend_service_profiling, ascend_model); the
            # first three each call _ensure_global_patch() ->
            # adapt_patch(is_global_patch=True), which installs the process-wide
            # vLLM monkey-patches (speculative-config rewrite, fused_moe,
            # kv-cache coordinator, distributed, structured output, ...).
            # load_general_plugins() runs in EVERY process (process0, engine
            # core, workers). Under VLLM_PLUGINS=fl those ascend_* entries are
            # filtered out, leaving this as the only general plugin, so we must
            # trigger the global patch here or workers/engine-core would run
            # WITHOUT the runtime patches the baseline relies on.
            # _ensure_global_patch() is guarded by _GLOBAL_PATCH_APPLIED and
            # adapt_patch is an idempotent module import, so calling it in a
            # process where NPUPlatform.pre_register_and_update() already ran
            # (process0) is a harmless no-op.
            from vllm_fl.dispatch.backends.vendor.ascend import _ensure_global_patch
            _ensure_global_patch()
            from vllm_fl.dispatch.backends.vendor.ascend import register_model as _ascend_register_model
            _ascend_register_model()
            return
        except Exception as e:
            logger.warning(
                "Vendored Ascend register_model() failed on Ascend host (%s); "
                "falling back to FL model registration.", e
            )

    from vllm import ModelRegistry

    _register_flagcx_connector()

    # Register OOT quant kernels so kernel selection can find them
    register_quant_linear()
    register_router()

    # Register GLM-5 (GlmMoeDsa) — config not yet upstream
    try:
        from vllm.transformers_utils.config import _CONFIG_REGISTRY
        from vllm_fl.configs.glm_moe_dsa import GlmMoeDsaConfig
        _CONFIG_REGISTRY["glm_moe_dsa"] = GlmMoeDsaConfig

        #from vllm_fl.patches.glm_moe_dsa import apply_model_patches as glm5_model
        #glm5_model()
    except Exception as e:
        logger.error(f"Register GlmMoeDsa model error: {str(e)}")

    # Register DeepseekV4 model
    try:
        ModelRegistry.register_model(
            "DeepseekV4ForCausalLM",
            "vllm_fl.models.deepseek_v4:DeepseekV4ForCausalLM"
        )
    except Exception as e:
        logger.error(f"Register DeepseekV4 model error: {str(e)}")


    # Register DeepseekV4 model
    try:
        ModelRegistry.register_model(
            "DeepSeekV4MTPModel",
            "vllm_fl.models.deepseek_v4_mtp:DeepSeekV4MTP"
        )
    except Exception as e:
        logger.error(f"Register DeepseekV4 model error: {str(e)}")
