# Copyright Mario "Neo" Sieg 2024. All rights reserved. mario.sieg.64@gmail.com

from enum import Enum
from ctypes import *

# Import generated RTML runtime bindings
from rtml_runtime import *

# Check if the dynamic library is loaded by testing for: rtml_global_init
assert rtml_global_init is not None, "Failed to load the RTML dynamic library"


def global_init() -> bool:
    return rtml_global_init()


def global_shutdown():
    rtml_global_shutdown()


class ComputeDevice(Enum):
    AUTO = 0
    CPU = 1
    GPU = 2
    TPU = 3


class Context:
    DEFAULT_POOL_SIZE = 2 << 30  # 2 GiB
    ACTIVE = None
    _INITIALIZED = False

    def __init__(self, name: str, device: ComputeDevice, mem_budget: int = DEFAULT_POOL_SIZE):
        self._lazy_init()
        mem_budget = max(mem_budget, self.DEFAULT_POOL_SIZE)
        rtml_context_create(name, device.value, mem_budget)
        self.name = name
        self.device = device
        self.mem_budget = mem_budget
        if self.ACTIVE is None:
            self.ACTIVE = self

    def _lazy_init(self):
        if not self._INITIALIZED:
            rtml_global_init()
            self._INITIALIZED = True

    def exists(name: str) -> bool:
        return rtml_context_exists(name)

    def active(self) -> 'Context':
        return self.ACTIVE


class Tensor:
    MAX_DIMS = 4

    class DType(Enum):
        F32 = 0

    def __init__(self, ctx: Context, shape: list[int], dtype: DType = DType.F32):
        assert ctx is not None, 'Invalid context'
        assert all([0 < x <= self.MAX_DIMS for x in shape]), 'Invalid tensor shape'
        d1 = shape[0]
        d2 = shape[1] if len(shape) > 1 else 1
        d3 = shape[2] if len(shape) > 2 else 1
        d4 = shape[3] if len(shape) > 3 else 1
        self._handle = rtml_context_create_tensor(ctx.name, dtype.value, d1, d2, d3, d4, len(shape), rtml_tensor_id_t(0), c_size_t(0))
        self._shape = shape
        self._dtype = dtype

    def shape(self) -> list[int]:
        return self._shape

    def dtype(self) -> DType:
        return self._dtype
