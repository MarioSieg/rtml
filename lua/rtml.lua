-- Copyright (c) 2024 Mario "Neo" Sieg. All Rights Reserved.

local ffi = require 'ffi'
local C = ffi.C

ffi.cdef[[
    typedef int32_t rtml_tensor_id;
    typedef double rtml_dim;
    rtml_tensor_id rtml_tensor_new_1d(rtml_dim d1);
    rtml_tensor_id rtml_tensor_new_2d(rtml_dim d1, rtml_dim d2);
    rtml_tensor_id rtml_tensor_new_3d(rtml_dim d1, rtml_dim d2, rtml_dim d3);
    rtml_tensor_id rtml_tensor_new_4d(rtml_dim d1, rtml_dim d2, rtml_dim d3, rtml_dim d4);
]]

ffi.load('../bin/librtml.dylib', true)

local rtml = {
    MAX_DIMS = 4, -- Max number of dimensions
    MAX_ELEMENTS_PER_DIM = 2^53-1 -- Max number of elements per dimension (IEEE 754 double precision max integer)
}

local TENSOR_TABLE = {
    [1] = C.rtml_tensor_new_1d,
    [2] = C.rtml_tensor_new_2d,
    [3] = C.rtml_tensor_new_3d,
    [4] = C.rtml_tensor_new_4d
}
assert(#TENSOR_TABLE == rtml.MAX_DIMS)

rtml.Tensor = {
    id = 0,
    name = nil
}

function rtml.Tensor:new(dims)
    local N = #dims
    if N < 1 or N > self.MAX_DIMS then
        error(string.format('Dimensions must be within [1, %d]', self.MAX_DIMS))
    end
    local id = TENSOR_TABLE[N](unpack(dims))
    local t = {}
    setmetatable(t, {__index = self})   
    t.id = id
    return t
end

return rtml
