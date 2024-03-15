// Copyright Mario "Neo" Sieg 2024. All rights reserved. mario.sieg.64@gmail.com

#pragma once

#include "tensor_base.hpp"

namespace rtml::blas {
    struct compute_ctx {
        dim thread_idx {0};     // Must be >= 0
        dim num_threads {1};    // Must be >= 1
    };

    extern auto t_f32_add(const compute_ctx& ctx, tensor<dtypes::f32>& r, const tensor<dtypes::f32>& x, const tensor<dtypes::f32>& y) noexcept -> void;
    extern auto t_f32_sub(const compute_ctx& ctx, tensor<dtypes::f32>& r, const tensor<dtypes::f32>& x, const tensor<dtypes::f32>& y) noexcept -> void;
    extern auto t_f32_mul(const compute_ctx& ctx, tensor<dtypes::f32>& r, const tensor<dtypes::f32>& x, const tensor<dtypes::f32>& y) noexcept -> void;
    extern auto t_f32_div(const compute_ctx& ctx, tensor<dtypes::f32>& r, const tensor<dtypes::f32>& x, const tensor<dtypes::f32>& y) noexcept -> void;
    extern auto t_f32_matmul(const compute_ctx& ctx, tensor<dtypes::f32>& r, const tensor<dtypes::f32>& x, const tensor<dtypes::f32>& y) noexcept -> void;
}
