# Copyright (c) 2026 BAAI. All rights reserved.
"""Ascend NPU quantization configs (msmodelslim w8a8 / compressed-tensors).

The implementation lives in the Ascend vendor backend
(``vllm_fl.dispatch.backends.vendor.ascend.quantization``); this module exposes
the quantization configs from ``vllm_fl.quantization`` to match the plugin's
layout. Importing it registers the ``ascend`` quantization method (via the
``register_quantization_config`` decorator on the config classes).
"""
from vllm_fl.dispatch.backends.vendor.ascend.quantization import (
    AscendCompressedTensorsConfig,
    AscendModelSlimConfig,
)

__all__ = ["AscendModelSlimConfig", "AscendCompressedTensorsConfig"]
