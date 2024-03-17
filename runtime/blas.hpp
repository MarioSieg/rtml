// Copyright Mario "Neo" Sieg 2024. All rights reserved. mario.sieg.64@gmail.com
// BLAS (basic linear algebra subprograms) for RTML (runtime machine learning) library
// Implements core tensor operations (which are not strictly BLAS routines) and some basic linear algebra operations

#pragma once

#include "tensor_base.hpp"

namespace rtml::blas {
    struct compute_ctx {
        const dim thread_idx;     // Must be >= 0
        const dim num_threads;    // Must be > 0

        constexpr explicit compute_ctx(const dim thread_idx = 0, const dim num_threads = 1) noexcept
            : thread_idx{std::max<dim>(0, thread_idx)},
                num_threads{std::max<dim>(1, num_threads)} {}
    };

    extern auto t_f32_add(const compute_ctx& ctx, tensor<dtypes::f32>& r, const tensor<dtypes::f32>& x, const tensor<dtypes::f32>& y) noexcept -> void;
    extern auto t_f32_sub(const compute_ctx& ctx, tensor<dtypes::f32>& r, const tensor<dtypes::f32>& x, const tensor<dtypes::f32>& y) noexcept -> void;
    extern auto t_f32_mul(const compute_ctx& ctx, tensor<dtypes::f32>& r, const tensor<dtypes::f32>& x, const tensor<dtypes::f32>& y) noexcept -> void;
    extern auto t_f32_div(const compute_ctx& ctx, tensor<dtypes::f32>& r, const tensor<dtypes::f32>& x, const tensor<dtypes::f32>& y) noexcept -> void;
    extern auto t_f32_matmul(const compute_ctx& ctx, tensor<dtypes::f32>& r, const tensor<dtypes::f32>& x, const tensor<dtypes::f32>& y) noexcept -> void;
}
