#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-plugin-FL project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Eagerly load Ascend custom kernels during GLM-5 (GlmMoeDsa) model build.

GLM-5 (``model_type=glm_moe_dsa``) reuses vLLM core's DeepSeek-V2 model
(``GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM)`` in ``deepseek_v2.py``), so its
layers are ``DeepseekV2DecoderLayer``. Under graph mode
(``cudagraph_mode=FULL_DECODE_ONLY``) the first ``enable_custom_op()`` call would
otherwise happen *inside* the Dynamo-traced region during graph capture and hit
the ``@torch._dynamo.disable``'d ``load_ascend_kernels()``::

    torch._dynamo.exc.Unsupported: Skip calling `torch.compiler.disable()`d
    function <function load_ascend_kernels ...>

This mirrors the DeepSeek-V4 (``models/deepseek_v4.py`` ``__init__``) and
Qwen3.5 (``patch_qwen3_5.py``) fix: eagerly call ``enable_custom_op()`` at
decoder-layer construction time (before compilation), so ``_CUSTOM_OP_ENABLED``
is already set and the loader guard short-circuits with zero graph-break during
tracing. ``enable_custom_op()`` is idempotent, so wrapping the DeepSeek-V2
decoder layer (shared by GLM-5 and other DeepSeek-V2 models) is harmless.
"""
import logging

from vllm_fl.dispatch.backends.vendor.ascend.utils import enable_custom_op

logger = logging.getLogger(__name__)

try:
    from vllm.model_executor.models.deepseek_v2 import DeepseekV2DecoderLayer

    if not getattr(DeepseekV2DecoderLayer, "_fl_eager_custom_op_patched", False):
        _orig_deepseek_v2_decoder_init = DeepseekV2DecoderLayer.__init__

        def _ascend_deepseek_v2_decoder_init(self, *args, **kwargs):
            _orig_deepseek_v2_decoder_init(self, *args, **kwargs)
            enable_custom_op()

        DeepseekV2DecoderLayer.__init__ = _ascend_deepseek_v2_decoder_init
        DeepseekV2DecoderLayer._fl_eager_custom_op_patched = True
except Exception as e:  # pragma: no cover
    logger.warning("patch_glm_moe_dsa (eager enable_custom_op) skipped: %s", e)
