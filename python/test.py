import sys
import os

import rtml

rtml.global_init()

context = rtml.Context('Test', rtml.ComputeDevice.CPU)

print(rtml.Context.exists('Test'))

rtml.global_shutdown()
