// Copyright Mario "Neo" Sieg 2024. All rights reserved. mario.sieg.64@gmail.com

#include "fixture.hpp"

BENCHMARK_F(rtml_fixture, tensor_add)(benchmark::State& st) {
    for (auto _ : st) {
        blas::t_f32_add(cctx, *c, *a, *b);
     }
}

