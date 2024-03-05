import rtml

iso = rtml.Isolate('Test', rtml.ComputeDevice.CPU)

a = rtml.Tensor(iso, [8, 16, 16, 16])
print(a)

rtml.global_shutdown()
