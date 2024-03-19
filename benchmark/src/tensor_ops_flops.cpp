// Copyright Mario "Neo" Sieg 2024. All rights reserved. mario.sieg.64@gmail.com

#include "fixture.hpp"

BENCHMARK_F(rtml_fixture, tensor_add)(benchmark::State& st) {
    for (auto _ : st) {
        blas::t_f32_add(cctx, *c, *a, *b);
     }
}

BENCHMARK_F(rtml_fixture, tensor_sub)(benchmark::State& st) {
    for (auto _ : st) {
        blas::t_f32_sub(cctx, *c, *a, *b);
    }
}

BENCHMARK_F(rtml_fixture, tensor_mul)(benchmark::State& st) {
    for (auto _ : st) {
        blas::t_f32_mul(cctx, *c, *a, *b);
    }
}

BENCHMARK_F(rtml_fixture, tensor_div)(benchmark::State& st) {
    for (auto _ : st) {
        blas::t_f32_div(cctx, *c, *a, *b);
    }
}

BENCHMARK_F(rtml_fixture, tensor_matmul)(benchmark::State& st) {
    for (auto _ : st) {
        blas::t_f32_matmul(cctx, *c, *a, *b);
    }
}
