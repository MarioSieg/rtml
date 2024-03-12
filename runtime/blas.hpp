// Copyright Mario "Neo" Sieg 2024. All rights reserved. mario.sieg.64@gmail.com

#pragma once

namespace rtml::blas {
    extern auto v_softmax(std::size_t n, float* ov, const float* x) noexcept -> void;
    extern auto v_sigmoid(std::size_t n, float* ov, const float* x) noexcept -> void;
    extern auto v_tanh(std::size_t n, float* ov, const float* x) noexcept -> void;
    extern auto v_relu(std::size_t n, float* ov, const float* x) noexcept -> void;
    extern auto v_gelu(std::size_t n, float* ov, const float* x) noexcept -> void;
    extern auto v_silu(std::size_t n, float* ov, const float* x) noexcept -> void;
    extern auto v_add(std::size_t n, float* ov, const float* x, const float* y) noexcept -> void;
    extern auto v_sub(std::size_t n, float* ov, const float* x, const float* y) noexcept -> void;
    extern auto v_mul(std::size_t n, float* ov, const float* x, const float* y) noexcept -> void;
    extern auto v_div(std::size_t n, float* ov, const float* x, const float* y) noexcept -> void;
    extern auto v_dot(std::size_t n, float* os, const float* x, const float* y) noexcept -> void;
}
