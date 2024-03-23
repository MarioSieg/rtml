// Copyright Mario "Neo" Sieg 2024. All rights reserved. mario.sieg.64@gmail.com

#include <gtest/gtest.h>

#include <isolate.hpp>
#include <tensor.hpp>
#include <graph.hpp>

using namespace rtml;

TEST(graph, eval) {
    auto ctx = isolate::create("test", isolate::compute_device::cpu, 0x1000);
    tensor<>* a {ctx->new_tensor<dtypes::f32>({4, 4})};
    tensor<>* b {ctx->new_tensor<dtypes::f32>({4, 4})};
    tensor<>* c {ctx->new_tensor<dtypes::f32>({4, 4})};

    a->splat_one();
    b->splat_one();
    c->splat_zero();

    a->set_name("a");
    b->set_name("b");
    c->set_name("c");

    c->operation(opcode::add, a, b);
    graph::compute(c);

    c->print();
}
