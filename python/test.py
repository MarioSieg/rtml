import rtml

rtml.global_init()

context = rtml.Context(b'Test', rtml.ComputeDevice.CPU, 0)

print(rtml.Context.exists(b'Test'))

rtml.global_shutdown()
