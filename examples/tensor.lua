-- Copyright (c) 2024 Mario "Neo" Sieg. All Rights Reserved.

local rtml = require 'rtml'

local a = rtml.Tensor:new(2, 2, 1)
local b = rtml.Tensor:new(2, 2, 1)

local r = a:matmul(b)
print(r)
