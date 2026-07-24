# Copyright (c) 2026 BAAI. All rights reserved.
"""Ascend NPU implementation of DeepSeek-V4 (w8a8 + DSA sparse attention + MTP).

The Ascend-specific implementation (NPU platform, DSA sparse attention,
msmodelslim w8a8 quantization and the compiled Ascend kernels) lives in the
Ascend vendor backend (``vllm_fl.dispatch.backends.vendor.ascend``). This
module exposes the model class so it is registered from ``vllm_fl.models``,
matching the plugin's model layout.
"""
from vllm_fl.dispatch.backends.vendor.ascend.models.deepseek_v4 import (
    AscendDeepseekV4ForCausalLM,
)

__all__ = ["AscendDeepseekV4ForCausalLM"]
