import rtml

a = rtml.Tensor([2, 2, 1])
b = rtml.Tensor([2, 2, 1])
r = a.matmul(b)

print(r)
