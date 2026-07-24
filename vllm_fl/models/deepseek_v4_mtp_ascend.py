# Copyright (c) 2026 BAAI. All rights reserved.
"""Ascend NPU DeepSeek-V4 MTP (multi-token prediction) draft model.

See ``vllm_fl.models.deepseek_v4_ascend`` for notes; the implementation lives
in the Ascend vendor backend.
"""
from vllm_fl.dispatch.backends.vendor.ascend.models.deepseek_v4_mtp import (
    DeepSeekV4MTP,
)

__all__ = ["DeepSeekV4MTP"]
