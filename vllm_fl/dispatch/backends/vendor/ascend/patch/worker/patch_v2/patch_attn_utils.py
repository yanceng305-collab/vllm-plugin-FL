import vllm

from vllm_fl.dispatch.backends.vendor.ascend.worker.v2.attn_utils import _allocate_kv_cache, _reshape_kv_cache

vllm.v1.worker.gpu.attn_utils._allocate_kv_cache = _allocate_kv_cache
vllm.v1.worker.gpu.attn_utils._reshape_kv_cache = _reshape_kv_cache
