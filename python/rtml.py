# Copyright Mario "Neo" Sieg 2024. All rights reserved. mario.sieg.64@gmail.com

import os
import sys
from enum import Enum

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
    __ACTIVE = None
    __INITIALIZED = False

    def __init__(self, name: str, device: ComputeDevice, mem_budget: int = DEFAULT_POOL_SIZE):
        self._lazy_init()
        mem_budget = max(mem_budget, self.DEFAULT_POOL_SIZE)
        rtml_context_create(name, device.value, mem_budget)
        self.name = name
        self.device = device
        self.mem_budget = mem_budget
        if self.__ACTIVE is None:
            self.__ACTIVE = self

    def _lazy_init(self):
        if not self.__INITIALIZED:
            rtml_global_init()
            self.__INITIALIZED = True

    def exists(name: str) -> bool:
        return rtml_context_exists(name)
