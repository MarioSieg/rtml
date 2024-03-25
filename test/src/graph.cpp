// Copyright Mario "Neo" Sieg 2024. All rights reserved. mario.sieg.64@gmail.com

#include <gtest/gtest.h>

#include <fstream>
#include <isolate.hpp>
#include <tensor.hpp>
#include <graph.hpp>

using namespace rtml;

TEST(graph, eval) {
    auto ctx = isolate::create("test", isolate::compute_device::cpu, 0x1000);
    tensor_ref a {ctx->new_tensor<dtypes::f32>({4, 4})};
    tensor_ref b {ctx->new_tensor<dtypes::f32>({4, 4})};

    a->fill_one();
    b->fill_one();

    a->set_name("a");
    b->set_name("b");

    auto c {a + b};
    auto e {c * c};
    auto f {e - c};
    auto g {f * c};
    graph::compute(&*g);
    std::ofstream out {"graph.dot"};
    std::stringstream ss {};
    graph::generate_graphviz_dot_code(ss, &*g);
    out << ss.str();
    out.close();

    for (auto&& x : g->data()) {
        ASSERT_FLOAT_EQ(x, 2.0f*((std::pow(2.0f, 2.0f))-2.0f));
    }
}
