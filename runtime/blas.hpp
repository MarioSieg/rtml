// Copyright Mario "Neo" Sieg 2024. All rights reserved. mario.sieg.64@gmail.com

#pragma once
#include "tensor.hpp"

namespace rtml {
    class tensor;
}

namespace rtml::blas {
    // v = vector
    // t = tensor

    extern auto v_f32_softmax(std::size_t n, float* ov, const float* x) noexcept -> void;
    extern auto v_f32_sigmoid(std::size_t n, float* ov, const float* x) noexcept -> void;
    extern auto v_f32_tanh(std::size_t n, float* ov, const float* x) noexcept -> void;
    extern auto v_f32_relu(std::size_t n, float* ov, const float* x) noexcept -> void;
    extern auto v_f32_gelu(std::size_t n, float* ov, const float* x) noexcept -> void;
    extern auto v_f32_silu(std::size_t n, float* ov, const float* x) noexcept -> void;

    extern auto v_f32_add(std::size_t n, float* ov, const float* x, const float* y) noexcept -> void;
    extern auto v_f32_sub(std::size_t n, float* ov, const float* x, const float* y) noexcept -> void;
    extern auto v_f32_mul(std::size_t n, float* ov, const float* x, const float* y) noexcept -> void;
    extern auto v_f32_div(std::size_t n, float* ov, const float* x, const float* y) noexcept -> void;
    extern auto v_f32_dot(std::size_t n, float* os, const float* x, const float* y) noexcept -> void;

    extern auto t_f32_add(tensor& r, const tensor& x, const tensor& y) noexcept -> void;
    extern auto t_f32_sub(tensor& r, const tensor& x, const tensor& y) noexcept -> void;
    extern auto t_f32_mul(tensor& r, const tensor& x, const tensor& y) noexcept -> void;
    extern auto t_f32_div(tensor& r, const tensor& x, const tensor& y) noexcept -> void;
    extern auto t_f32_matmul(tensor& r, const tensor& x, const tensor& y) noexcept -> void;
}
