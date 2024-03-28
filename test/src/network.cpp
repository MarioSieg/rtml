// Copyright Mario "Neo" Sieg 2024. All rights reserved. mario.sieg.64@gmail.com

#include <gtest/gtest.h>

#include <isolate.hpp>
#include <tensor.hpp>
#include <graph.hpp>
#include <net.hpp>

using namespace rtml;

TEST(net, with_ass) {
    std::shared_ptr ctx {isolate::create("alex", isolate::compute_device::cpu, 0x1000000)};
    net xor_network {*ctx, {2, 3, 1}};
    constexpr std::array<dtypes::f32, 2> inputs_data {
        0.0f, 1.0f
    };
    constexpr std::array<dtypes::f32, 4> targets_data {
        0.0f,
        1.0f,
        0.0f,
        0.0f
    };

    for (auto&& x : xor_network.forward_propagate(inputs_data)) {
        rtml_log_info("{}", x);
    }

    //xor_network.train(inputs, targets, 100000, 1.0f);
}
