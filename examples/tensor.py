import rtml

a = rtml.Tensor('a', [2, 2, 1])
b = rtml.Tensor('b', [2, 2, 1])
r = a.matmul(b)

print(r)
