# Copyright Mario "Neo" Sieg 2024. All rights reserved. mario.sieg.64@gmail.com

from ctypes import *


def rtml_load_ffi() -> CDLL:
    rtml = CDLL('../bin/debug/librtml.dylib')

    rtml.rtml_global_init.argtypes = ()
    rtml.rtml_global_init.restype = c_bool

    rtml.rtml_global_shutdown.argtypes = ()
    rtml.rtml_global_shutdown.restype = None

    rtml.rtml_context_create.argtypes = (c_char_p, c_int, c_size_t)
    rtml.rtml_context_create.restype = None

    rtml.rtml_context_exists.argtypes = (c_char_p,)
    rtml.rtml_context_exists.restype = c_bool

    return rtml
