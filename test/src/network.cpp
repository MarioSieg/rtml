// Copyright Mario "Neo" Sieg 2024. All rights reserved. mario.sieg.64@gmail.com

#include <gtest/gtest.h>

#include <isolate.hpp>
#include <tensor.hpp>
#include <graph.hpp>
#include <net.hpp>

using namespace rtml;

TEST(net, with_ass) {
    std::shared_ptr ctx {isolate::create("alex", isolate::compute_device::cpu, 0x100000)};
    net xor_network {*ctx, {2, 3, 1}};
    constexpr std::array<std::array<dtypes::f32, 2>, 4> inputs_data {
        std::array<dtypes::f32, 2>{ 0.0f, 0.0f },
        std::array<dtypes::f32, 2>{ 0.0f, 1.0f },
        std::array<dtypes::f32, 2>{ 1.0f, 0.0f },
        std::array<dtypes::f32, 2>{ 1.0f, 1.0f },
    };
    constexpr std::array<dtypes::f32, 4> targets_data {
        0.0f,
        1.0f,
        0.0f,
        0.0f
    };
    tensor_ref inputs {ctx->new_tensor<dtypes::f32>({2, 4})};
    std::memcpy(inputs->ptr(), inputs_data.data(), inputs->size());

    tensor_ref targets {ctx->new_tensor<dtypes::f32>({2, 4})};
    std::memcpy(targets->ptr(), targets_data.data(), targets->size());

    xor_network.train(inputs, targets, 100000, 1.0f);
}
