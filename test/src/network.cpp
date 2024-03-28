// Copyright Mario "Neo" Sieg 2024. All rights reserved. mario.sieg.64@gmail.com

#include <gtest/gtest.h>

#include <isolate.hpp>
#include <tensor.hpp>
#include <graph.hpp>
#include <net.hpp>

using namespace rtml;

TEST(net, with_ass) {
    std::shared_ptr ctx {isolate::create("alex", isolate::compute_device::cpu, 16_gib)};
    net xor_network {*ctx, {2, 3, 1}};
    std::array<tensor<>* const, 4> inputs_data {
        ctx->new_tensor({2}, {0.0f, 0.0f}),
        ctx->new_tensor({2}, {0.0f, 1.0f}),
        ctx->new_tensor({2}, {1.0f, 0.0f}),
        ctx->new_tensor({2}, {1.0f, 1.0f}),
    };
    std::array<tensor<>* const, 4> targets_data {
        ctx->new_tensor({1}, {0.0f}),
        ctx->new_tensor({1}, {1.0f}),
        ctx->new_tensor({1}, {1.0f}),
        ctx->new_tensor({1}, {0.0f}),
    };
    static_assert(inputs_data.size() == targets_data.size());

    xor_network.train(inputs_data, targets_data, 10000, 0.1f);

    for (std::size_t i {}; i < inputs_data.size(); ++i) {
        rtml_log_info("[{} ^ {}] = {}", inputs_data[i]->data()[0], inputs_data[i]->data()[1], xor_network.forward_propagate(inputs_data[i])->data()[0]);
    }
}
