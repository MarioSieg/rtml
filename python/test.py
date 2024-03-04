import rtml

ctx = rtml.Context('Test', rtml.ComputeDevice.CPU)

a = rtml.Tensor(ctx, [2, 3, 4])

rtml.global_shutdown()
