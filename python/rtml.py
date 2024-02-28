# Copyright Mario "Neo" Sieg 2024. All rights reserved. mario.sieg.64@gmail.com

from ctypes import *
from enum import Enum

from rtml_ffi import rtml_load_ffi

rtml = rtml_load_ffi()

def global_init() -> bool:
    return rtml.rtml_global_init()


def global_shutdown():
    rtml.rtml_global_shutdown()


class ComputeDevice(Enum):
    AUTO = 0
    CPU = 1
    GPU = 2
    TPU = 3


class Context:
    DEFAULT_POOL_SIZE = 2 << 30  # 2 GiB

    def __init__(self, name: bytes, device: ComputeDevice, mem_budget: int = DEFAULT_POOL_SIZE):
        mem_budget = max(mem_budget, self.DEFAULT_POOL_SIZE)
        rtml.rtml_context_create(c_char_p(name), c_int(device.value), c_size_t(mem_budget))
        self.name = name
        self.device = device
        self.mem_budget = mem_budget

    def exists(name: bytes) -> bool:
        return rtml.rtml_context_exists(c_char_p(name))

