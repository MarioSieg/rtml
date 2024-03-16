#pragma once

#include <benchmark/benchmark.h>

#include <isolate.hpp>
#include <tensor.hpp>
#include <blas.hpp>

using namespace rtml;

class rtml_fixture : public benchmark::Fixture {
public:
    static constexpr dim k_shape {128};
    static constexpr std::array<dim, tensor<>::k_max_dims> k_shapes {k_shape, k_shape, k_shape, k_shape};
    std::shared_ptr<isolate> ctx {};
    tensor<>* a {};
    tensor<>* b {};
    tensor<>* c {};
    blas::compute_ctx cctx {};

    auto SetUp(benchmark::State& state) -> void override {
        constexpr float x {1.0f};
        constexpr float y {2.0f};
        ctx = isolate::create("test", isolate::compute_device::cpu, 4_gib);
        a = ctx->new_tensor<float>(k_shapes);
        b = ctx->new_tensor<float>(k_shapes);
        c = ctx->new_tensor<float>(k_shapes);
        a->splat(x);
        b->splat(y);
        c->splat_zero();
    }

    auto TearDown(benchmark::State& state) -> void override {
    }
};
