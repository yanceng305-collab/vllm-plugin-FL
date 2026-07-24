# Copyright (c) 2026 BAAI. All rights reserved.
"""Load the native Ascend operator kernels.

The compiled kernels are shipped as ``libfl_ascend_ops.so`` (with its
dependency ``libfl_ascend_kernels.so``) and register their operators into the
``torch.ops._C_ascend`` namespace when loaded. They are loaded explicitly by
file path so no importable native module name is required.
"""
import os

import torch

# Resolve the native kernel library path ONCE, at module import time. It must
# never be computed inside a torch.compile-traced region: ``os.path.abspath``
# lowers to ``posix._path_normpath``, a C builtin that Dynamo refuses to trace
# (hard skip), which crashes graph capture (e.g. FULL_DECODE_ONLY) if the load
# is first triggered from within the traced forward.
_LIB_DIR = os.path.dirname(os.path.abspath(__file__))
_SO_PATH = os.path.join(_LIB_DIR, "libfl_ascend_ops.so")

_loaded = False


@torch._dynamo.disable
def load_ascend_kernels() -> None:
    """Load (register) the Ascend kernels. Idempotent and Dynamo-safe.

    Decorated with ``torch._dynamo.disable`` so Dynamo executes it eagerly
    instead of tracing into ``torch.ops.load_library``. Combined with an eager
    call during model initialization, the ``_loaded`` guard short-circuits by
    the time any graph is captured.
    """
    global _loaded
    if _loaded:
        return
    torch.ops.load_library(_SO_PATH)
    _loaded = True
