// Copyright Mario "Neo" Sieg 2024. All rights reserved. mario.sieg.64@gmail.com

#include <algorithm>
#include <cmath>

#include "blas.hpp"

namespace rtml::blas {
    static constexpr float k_rtml_sqrt2pi {0.79788456080286535587989211986876f};
    static constexpr float k_rtml_gelu_coeff {0.044715f};

    [[nodiscard]] static constexpr auto softmax(const float x) noexcept -> float {
        return std::expf(x);
    }
    [[nodiscard]] static constexpr auto sigmoid(const float x) noexcept -> float {
        return 1.0f / (1.0f + std::expf(-x));
    }
    [[nodiscard]] static constexpr auto tanh(const float x) noexcept -> float {
        return std::tanh(x);
    }
    [[nodiscard]] static constexpr auto relu(const float x) noexcept -> float {
        return std::max(x, 0.0f);
    }
    [[nodiscard]] static constexpr auto gelu(const float x) noexcept -> float {
        return 0.5f * x * (1.0f + std::tanh(k_rtml_sqrt2pi * x * (1.0f + k_rtml_gelu_coeff * x * x)));
    }
    [[nodiscard]] static constexpr auto silu(const float x) noexcept -> float {
        return x / (1.0f + std::exp(-x));
    }

    auto v_softmax(const std::size_t n, float* const ov, const float* const x) noexcept -> void {
        for (std::size_t i = 0; i < n; ++i) ov[i] = softmax(x[i]);
    }

    auto v_sigmoid(const std::size_t n, float* const ov, const float* const x) noexcept -> void {
        for (std::size_t i = 0; i < n; ++i) ov[i] = sigmoid(x[i]);
    }

    auto v_tanh(const std::size_t n, float* const ov, const float* const x) noexcept -> void {
        for (std::size_t i = 0; i < n; ++i) ov[i] = tanh(x[i]);
    }

    auto v_relu(const std::size_t n, float* const ov, const float* const x) noexcept -> void {
        for (std::size_t i = 0; i < n; ++i) ov[i] = relu(x[i]);
    }

    auto v_gelu(const std::size_t n, float* const ov, const float* const x) noexcept -> void {
        for (std::size_t i = 0; i < n; ++i) ov[i] = gelu(x[i]);
    }

    auto v_silu(const std::size_t n, float* const ov, const float* const x) noexcept -> void {
        for (std::size_t i = 0; i < n; ++i) ov[i] = silu(x[i]);
    }

    auto v_add(const std::size_t n, float* const ov, const float* const x, const float* const y) noexcept -> void {
        for (std::size_t i = 0; i < n; ++i) ov[i] = x[i] + y[i];
    }

    auto v_sub(const std::size_t n, float* const ov, const float* const x, const float* const y) noexcept -> void {
        for (std::size_t i = 0; i < n; ++i) ov[i] = x[i] - y[i];
    }

    auto v_mul(const std::size_t n, float* const ov, const float* const x, const float* const y) noexcept -> void {
        for (std::size_t i = 0; i < n; ++i) ov[i] = x[i] * y[i];
    }

    auto v_div(const std::size_t n, float* const ov, const float* const x, const float* const y) noexcept -> void {
        for (std::size_t i = 0; i < n; ++i) ov[i] = x[i] / y[i];
    }

    auto v_dot(const std::size_t n, float* const os, const float* const x, const float* const y) noexcept -> void {
        double sum = 0.0;
        for (std::size_t i = 0; i < n; ++i)
            sum += static_cast<double>(x[i] * y[i]);
        *os = static_cast<float>(sum);
    }
}
