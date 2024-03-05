import rtml

ctx = rtml.Isolate('Test', rtml.ComputeDevice.CPU)

a = rtml.Tensor(ctx, [8, 16, 16, 16])
print(a)

rtml.global_shutdown()
