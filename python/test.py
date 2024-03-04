import rtml

context = rtml.Context('Test', rtml.ComputeDevice.CPU)
assert rtml.Context.exists('Test')

rtml.global_shutdown()
