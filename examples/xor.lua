-- Copyright (c) 2024 Mario "Neo" Sieg. All Rights Reserved.

local rtml = require 'rtml'

local xor = rtml.FeedForward:new(2, 2, 1)

-- XOR training data
local inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}}
local targets = {{0}, {1}, {1}, {0}}

-- Training parameters
local epochs = 10000
local rate = 0.1

-- Train the network
for epoch = 1, epochs do
    local err = 0
    for i = 1, #inputs do
        err = err + xor:train(inputs[i], targets[i], rate)
    end
    if epoch % 1000 == 0 then
        print(string.format('Epoch %d, error: %.4f', epoch, err))
    end
end

-- Test the network with XOR inputs
for i = 1, #inputs do
    local output = xor:forward(inputs[i])
    print(string.format('Input: {%d, %d}, Output: %.4f', inputs[i][1], inputs[i][2], output[1]))
end
