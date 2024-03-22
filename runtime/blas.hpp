// Copyright Mario "Neo" Sieg 2024. All rights reserved. mario.sieg.64@gmail.com
// CPU backend only!
// BLAS (basic linear algebra subprograms) for RTML (runtime machine learning) library for the CPU backend
// Implements core tensor operations (which are not strictly BLAS routines) and some basic linear algebra operations

#pragma once

#include "tensor_base.hpp"

namespace rtml::blas {
    // Context for compute operations
    struct compute_ctx {
        const dim thread_idx;     // Current thread index - Must be >= 0
        const dim num_threads;    // Total number of threads Must be > 0

        constexpr explicit compute_ctx(const dim thread_idx = 0, const dim num_threads = 1) noexcept
            : thread_idx{std::max<dim>(0, thread_idx)},
                num_threads{std::max<dim>(1, num_threads)} {}
    };

    // unary ops

    extern auto softmax(const compute_ctx& ctx, tensor<dtypes::f32>& r, const tensor<dtypes::f32>& x) noexcept -> void; // r = softmax(x)
    extern auto sigmoid(const compute_ctx& ctx, tensor<dtypes::f32>& r, const tensor<dtypes::f32>& x) noexcept -> void; // r = sigmoid(x)
    extern auto tanh(const compute_ctx& ctx, tensor<dtypes::f32>& r, const tensor<dtypes::f32>& x) noexcept -> void;    // r = tanh(x)
    extern auto relu(const compute_ctx& ctx, tensor<dtypes::f32>& r, const tensor<dtypes::f32>& x) noexcept -> void;    // r = relu(x)
    extern auto gelu(const compute_ctx& ctx, tensor<dtypes::f32>& r, const tensor<dtypes::f32>& x) noexcept -> void;    // r = gelu(x)
    extern auto silu(const compute_ctx& ctx, tensor<dtypes::f32>& r, const tensor<dtypes::f32>& x) noexcept -> void;    // r = silu(x)

    // binary ops

    extern auto add(const compute_ctx& ctx, tensor<dtypes::f32>& r, const tensor<dtypes::f32>& x, const tensor<dtypes::f32>& y) noexcept -> void;       // r = x + y
    extern auto sub(const compute_ctx& ctx, tensor<dtypes::f32>& r, const tensor<dtypes::f32>& x, const tensor<dtypes::f32>& y) noexcept -> void;       // r = x - y
    extern auto mul(const compute_ctx& ctx, tensor<dtypes::f32>& r, const tensor<dtypes::f32>& x, const tensor<dtypes::f32>& y) noexcept -> void;       // r = x * y
    extern auto div(const compute_ctx& ctx, tensor<dtypes::f32>& r, const tensor<dtypes::f32>& x, const tensor<dtypes::f32>& y) noexcept -> void;       // r = x / y
    extern auto matmul(const compute_ctx& ctx, tensor<dtypes::f32>& r, const tensor<dtypes::f32>& x, const tensor<dtypes::f32>& y) noexcept -> void;    // r = x @ y
}
