// Copyright Mario "Neo" Sieg 2024. All rights reserved. mario.sieg.64@gmail.com

#include <algorithm>
#include <cmath>

#include "blas.hpp"
#include "isolate.hpp"
#include "tensor.hpp"

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

    auto v_f32_softmax(const std::size_t n, float* const ov, const float* const x) noexcept -> void {
        for (std::size_t i = 0; i < n; ++i) ov[i] = softmax(x[i]);
    }

    auto v_sigmoid_f3(const std::size_t n, float* const ov, const float* const x) noexcept -> void {
        for (std::size_t i = 0; i < n; ++i) ov[i] = sigmoid(x[i]);
    }

    auto v_f32_tanh(const std::size_t n, float* const ov, const float* const x) noexcept -> void {
        for (std::size_t i = 0; i < n; ++i) ov[i] = tanh(x[i]);
    }

    auto v_f32_relu(const std::size_t n, float* const ov, const float* const x) noexcept -> void {
        for (std::size_t i = 0; i < n; ++i) ov[i] = relu(x[i]);
    }

    auto v_f32_gelu(const std::size_t n, float* const ov, const float* const x) noexcept -> void {
        for (std::size_t i = 0; i < n; ++i) ov[i] = gelu(x[i]);
    }

    auto v_f32_silu(const std::size_t n, float* const ov, const float* const x) noexcept -> void {
        for (std::size_t i = 0; i < n; ++i) ov[i] = silu(x[i]);
    }

    auto v_f32_add(const std::size_t n, float* const ov, const float* const x, const float* const y) noexcept -> void {
        for (std::size_t i = 0; i < n; ++i) ov[i] = x[i] + y[i];
    }

    auto v_f32_sub(const std::size_t n, float* const ov, const float* const x, const float* const y) noexcept -> void {
        for (std::size_t i = 0; i < n; ++i) ov[i] = x[i] - y[i];
    }

    auto v_f32_mul(const std::size_t n, float* const ov, const float* const x, const float* const y) noexcept -> void {
        for (std::size_t i = 0; i < n; ++i) ov[i] = x[i] * y[i];
    }

    auto v_f32_div(const std::size_t n, float* const ov, const float* const x, const float* const y) noexcept -> void {
        for (std::size_t i = 0; i < n; ++i) ov[i] = x[i] / y[i];
    }

    auto v_f32_dot(const std::size_t n, float* const os, const float* const x, const float* const y) noexcept -> void {
        double sum = 0.0;
        for (std::size_t i = 0; i < n; ++i)
            sum += static_cast<double>(x[i] * y[i]);
        *os = static_cast<float>(sum);
    }

    auto t_f32_add(tensor<float>& r, const tensor<float>& x, const tensor<float>& y) noexcept -> void {
        std::uint8_t* const pr {r.ptr()};
        const std::uint8_t* const px {x.ptr()};
        const std::uint8_t* const py {y.ptr()};
        const dim num_rows {r.row_count()};
        const auto [x_d0, x_d1, x_d2, x_d3] {x.dims()};
        const auto [x_s0, x_s1, x_s2, x_s3] {x.strides()};
        const auto [y_d0, y_d1, y_d2, y_d3] {y.dims()};
        const auto [y_s0, y_s1, y_s2, y_s3] {y.strides()};
        const auto [r_d0, r_d1, r_d2, r_d3] {r.dims()};
        const auto [r_s0, r_s1, r_s2, r_s3] {r.strides()};
        if (y_s0 == sizeof(float)) { /* Tensor x has a contigous memory layout */
            for (dim row_i {}; row_i < num_rows; ++row_i) {
                const dim x_i3 {row_i / (x_d2*x_d1)}; /* Unroll index into dimensions */
                const dim x_i2 {(row_i - x_i3*x_d2*x_d1) / x_d1};
                const dim x_i1 {row_i - x_i3*x_d2*x_d1 - x_i2*x_d1};
                const dim y_i3 {x_i3 % y_d3}; /* Broadcast */
                const dim y_i2 {x_i2 % y_d2}; /* Broadcast */
                const dim y_i1 {x_i1 % y_d1}; /* Broadcast */
                auto* const pdst {reinterpret_cast<float*>(pr + x_i3*r_s3 + x_i2*r_s2 + x_i1*r_s1)};
                const auto* const psrc0 {reinterpret_cast<const float*>(px + x_i3*x_s3 + x_i2*x_s2 + x_i1*x_s1)};
                const auto* const psrc1 {reinterpret_cast<const float*>(py + y_i3*y_s3 + y_i2*y_s2 + y_i1*y_s1)};
                for (dim v {}; v < x_d0 / y_d0; ++v) /* Vector operation between contiguous rows */
                    v_f32_add(y_d0, pdst + v*y_d0, psrc0 + v*y_d0, psrc1);
            }
        } else {
            for (dim row_i {}; row_i < num_rows; ++row_i) {
                const dim x_i3 {row_i/(x_d2*x_d1)}; /* Unroll index into dimensions */
                const dim x_i2 {(row_i - x_i3*x_d2*x_d1)/x_d1};
                const dim x_i1 {row_i - x_i3*x_d2*x_d1 - x_i2*x_d1};
                const dim y_i3 {x_i3 % y_d3}; /* Broadcast */
                const dim y_i2 {x_i2 % y_d2}; /* Broadcast */
                const dim y_i1 {x_i1 % y_d1}; /* Broadcast */
                auto* const pdst {reinterpret_cast<float*>(pr + x_i3*r_s3 + x_i2*r_s2 + x_i1*r_s1)};
                const auto* const psrc0 {reinterpret_cast<const float*>(px + x_i3*x_s3 + x_i2*x_s2 + x_i1*x_s1)};
                for (dim i {}; i < r_d0; ++i) { /* Scalar operation */
                    const dim i10 {i % y_d0};
                    const auto* psrc1 = reinterpret_cast<const float*>(py + y_i3*y_s3 + y_i2*y_s2 + y_i1*y_s1 + i10*y_s0);
                    pdst[i] = psrc0[i] + *psrc1;
                }
            }
        }
    }

    auto t_f32_sub(tensor<float>& r, const tensor<float>& x, const tensor<float>& y) noexcept -> void {
        std::uint8_t* const pr {r.ptr()};
        const std::uint8_t* const px {x.ptr()};
        const std::uint8_t* const py {y.ptr()};
        const dim num_rows {r.row_count()};
        const auto [x_d0, x_d1, x_d2, x_d3] {x.dims()};
        const auto [x_s0, x_s1, x_s2, x_s3] {x.strides()};
        const auto [y_d0, y_d1, y_d2, y_d3] {y.dims()};
        const auto [y_s0, y_s1, y_s2, y_s3] {y.strides()};
        const auto [r_d0, r_d1, r_d2, r_d3] {r.dims()};
        const auto [r_s0, r_s1, r_s2, r_s3] {r.strides()};
        if (y_s0 == sizeof(float)) { /* Tensor x has a contigous memory layout */
            for (dim row_i {}; row_i < num_rows; ++row_i) {
                const dim x_i3 {row_i / (x_d2*x_d1)}; /* Unroll index into dimensions */
                const dim x_i2 {(row_i - x_i3*x_d2*x_d1) / x_d1};
                const dim x_i1 {row_i - x_i3*x_d2*x_d1 - x_i2*x_d1};
                const dim y_i3 {x_i3 % y_d3}; /* Broadcast */
                const dim y_i2 {x_i2 % y_d2}; /* Broadcast */
                const dim y_i1 {x_i1 % y_d1}; /* Broadcast */
                auto* const pdst {reinterpret_cast<float*>(pr + x_i3*r_s3 + x_i2*r_s2 + x_i1*r_s1)};
                const auto* const psrc0 {reinterpret_cast<const float*>(px + x_i3*x_s3 + x_i2*x_s2 + x_i1*x_s1)};
                const auto* const psrc1 {reinterpret_cast<const float*>(py + y_i3*y_s3 + y_i2*y_s2 + y_i1*y_s1)};
                for (dim v {}; v < x_d0 / y_d0; ++v) /* Vector operation between contiguous rows */
                    v_f32_sub(y_d0, pdst + v*y_d0, psrc0 + v*y_d0, psrc1);
            }
        } else {
            for (dim row_i {}; row_i < num_rows; ++row_i) {
                const dim x_i3 {row_i/(x_d2*x_d1)}; /* Unroll index into dimensions */
                const dim x_i2 {(row_i - x_i3*x_d2*x_d1)/x_d1};
                const dim x_i1 {row_i - x_i3*x_d2*x_d1 - x_i2*x_d1};
                const dim y_i3 {x_i3 % y_d3}; /* Broadcast */
                const dim y_i2 {x_i2 % y_d2}; /* Broadcast */
                const dim y_i1 {x_i1 % y_d1}; /* Broadcast */
                auto* const pdst {reinterpret_cast<float*>(pr + x_i3*r_s3 + x_i2*r_s2 + x_i1*r_s1)};
                const auto* const psrc0 {reinterpret_cast<const float*>(px + x_i3*x_s3 + x_i2*x_s2 + x_i1*x_s1)};
                for (dim i {}; i < r_d0; ++i) { /* Scalar operation */
                    const dim i10 {i % y_d0};
                    const auto* psrc1 = reinterpret_cast<const float*>(py + y_i3*y_s3 + y_i2*y_s2 + y_i1*y_s1 + i10*y_s0);
                    pdst[i] = psrc0[i] - *psrc1;
                }
            }
        }
    }

    auto t_f32_mul(tensor<float>& r, const tensor<float>& x, const tensor<float>& y) noexcept -> void {
        std::uint8_t* const pr {r.ptr()};
        const std::uint8_t* const px {x.ptr()};
        const std::uint8_t* const py {y.ptr()};
        const dim num_rows {r.row_count()};
        const auto [x_d0, x_d1, x_d2, x_d3] {x.dims()};
        const auto [x_s0, x_s1, x_s2, x_s3] {x.strides()};
        const auto [y_d0, y_d1, y_d2, y_d3] {y.dims()};
        const auto [y_s0, y_s1, y_s2, y_s3] {y.strides()};
        const auto [r_d0, r_d1, r_d2, r_d3] {r.dims()};
        const auto [r_s0, r_s1, r_s2, r_s3] {r.strides()};
        if (y_s0 == sizeof(float)) { /* Tensor x has a contigous memory layout */
            for (dim row_i {}; row_i < num_rows; ++row_i) {
                const dim x_i3 {row_i / (x_d2*x_d1)}; /* Unroll index into dimensions */
                const dim x_i2 {(row_i - x_i3*x_d2*x_d1) / x_d1};
                const dim x_i1 {row_i - x_i3*x_d2*x_d1 - x_i2*x_d1};
                const dim y_i3 {x_i3 % y_d3}; /* Broadcast */
                const dim y_i2 {x_i2 % y_d2}; /* Broadcast */
                const dim y_i1 {x_i1 % y_d1}; /* Broadcast */
                auto* const pdst {reinterpret_cast<float*>(pr + x_i3*r_s3 + x_i2*r_s2 + x_i1*r_s1)};
                const auto* const psrc0 {reinterpret_cast<const float*>(px + x_i3*x_s3 + x_i2*x_s2 + x_i1*x_s1)};
                const auto* const psrc1 {reinterpret_cast<const float*>(py + y_i3*y_s3 + y_i2*y_s2 + y_i1*y_s1)};
                for (dim v {}; v < x_d0 / y_d0; ++v) /* Vector operation between contiguous rows */
                    v_f32_mul(y_d0, pdst + v*y_d0, psrc0 + v*y_d0, psrc1);
            }
        } else {
            for (dim row_i {}; row_i < num_rows; ++row_i) {
                const dim x_i3 {row_i/(x_d2*x_d1)}; /* Unroll index into dimensions */
                const dim x_i2 {(row_i - x_i3*x_d2*x_d1)/x_d1};
                const dim x_i1 {row_i - x_i3*x_d2*x_d1 - x_i2*x_d1};
                const dim y_i3 {x_i3 % y_d3}; /* Broadcast */
                const dim y_i2 {x_i2 % y_d2}; /* Broadcast */
                const dim y_i1 {x_i1 % y_d1}; /* Broadcast */
                auto* const pdst {reinterpret_cast<float*>(pr + x_i3*r_s3 + x_i2*r_s2 + x_i1*r_s1)};
                const auto* const psrc0 {reinterpret_cast<const float*>(px + x_i3*x_s3 + x_i2*x_s2 + x_i1*x_s1)};
                for (dim i {}; i < r_d0; ++i) { /* Scalar operation */
                    const dim i10 {i % y_d0};
                    const auto* psrc1 = reinterpret_cast<const float*>(py + y_i3*y_s3 + y_i2*y_s2 + y_i1*y_s1 + i10*y_s0);
                    pdst[i] = psrc0[i] * *psrc1;
                }
            }
        }
    }

    auto t_f32_div(tensor<float>& r, const tensor<float>& x, const tensor<float>& y) noexcept -> void {
        std::uint8_t* const pr {r.ptr()};
        const std::uint8_t* const px {x.ptr()};
        const std::uint8_t* const py {y.ptr()};
        const dim num_rows {r.row_count()};
        const auto [x_d0, x_d1, x_d2, x_d3] {x.dims()};
        const auto [x_s0, x_s1, x_s2, x_s3] {x.strides()};
        const auto [y_d0, y_d1, y_d2, y_d3] {y.dims()};
        const auto [y_s0, y_s1, y_s2, y_s3] {y.strides()};
        const auto [r_d0, r_d1, r_d2, r_d3] {r.dims()};
        const auto [r_s0, r_s1, r_s2, r_s3] {r.strides()};
        if (y_s0 == sizeof(float)) { /* Tensor x has a contigous memory layout */
            for (dim row_i {}; row_i < num_rows; ++row_i) {
                const dim x_i3 {row_i / (x_d2*x_d1)}; /* Unroll index into dimensions */
                const dim x_i2 {(row_i - x_i3*x_d2*x_d1) / x_d1};
                const dim x_i1 {row_i - x_i3*x_d2*x_d1 - x_i2*x_d1};
                const dim y_i3 {x_i3 % y_d3}; /* Broadcast */
                const dim y_i2 {x_i2 % y_d2}; /* Broadcast */
                const dim y_i1 {x_i1 % y_d1}; /* Broadcast */
                auto* const pdst {reinterpret_cast<float*>(pr + x_i3*r_s3 + x_i2*r_s2 + x_i1*r_s1)};
                const auto* const psrc0 {reinterpret_cast<const float*>(px + x_i3*x_s3 + x_i2*x_s2 + x_i1*x_s1)};
                const auto* const psrc1 {reinterpret_cast<const float*>(py + y_i3*y_s3 + y_i2*y_s2 + y_i1*y_s1)};
                for (dim v {}; v < x_d0 / y_d0; ++v) /* Vector operation between contiguous rows */
                    v_f32_div(y_d0, pdst + v*y_d0, psrc0 + v*y_d0, psrc1);
            }
        } else {
            for (dim row_i {}; row_i < num_rows; ++row_i) {
                const dim x_i3 {row_i/(x_d2*x_d1)}; /* Unroll index into dimensions */
                const dim x_i2 {(row_i - x_i3*x_d2*x_d1)/x_d1};
                const dim x_i1 {row_i - x_i3*x_d2*x_d1 - x_i2*x_d1};
                const dim y_i3 {x_i3 % y_d3}; /* Broadcast */
                const dim y_i2 {x_i2 % y_d2}; /* Broadcast */
                const dim y_i1 {x_i1 % y_d1}; /* Broadcast */
                auto* const pdst {reinterpret_cast<float*>(pr + x_i3*r_s3 + x_i2*r_s2 + x_i1*r_s1)};
                const auto* const psrc0 {reinterpret_cast<const float*>(px + x_i3*x_s3 + x_i2*x_s2 + x_i1*x_s1)};
                for (dim i {}; i < r_d0; ++i) { /* Scalar operation */
                    const dim i10 {i % y_d0};
                    const auto* psrc1 = reinterpret_cast<const float*>(py + y_i3*y_s3 + y_i2*y_s2 + y_i1*y_s1 + i10*y_s0);
                    pdst[i] = psrc0[i] / *psrc1;
                }
            }
        }
    }

    auto t_f32_matmul(tensor<float>& r, const tensor<float>& x, const tensor<float>& y) noexcept -> void {
        std::uint8_t* const pr {r.ptr()};
        const std::uint8_t* const px {x.ptr()};
        const std::uint8_t* const py {y.ptr()};
        const auto [x_d0, x_d1, x_d2, x_d3] {x.dims()};
        const auto [x_s0, x_s1, x_s2, x_s3] {x.strides()};
        const auto [y_d0, y_d1, y_d2, y_d3] {y.dims()};
        const auto [y_s0, y_s1, y_s2, y_s3] {y.strides()};
        const auto [r_d0, r_d1, r_d2, r_d3] {r.dims()};
        const auto [r_s0, r_s1, r_s2, r_s3] {r.strides()};

        const dim r2 {y_d2/x_d2};
        const dim r3 {y_d3/x_d3};
        const dim row_size {y_d0*static_cast<dim>(sizeof(float))};
        const dim nr0 {x_d1}; // x rows
        const dim nr1 {y_d1*y_d2*y_d3}; // y rows
        const dim dr0 {nr0};
        const dim dr1 {nr1};
        const dim ir010 {0};
        const dim ir011 {std::min(ir010+dr0, nr0)};
        const dim ir110 {0};
        const dim ir111 {std::min(ir110+dr1, nr1)};

        // block tiling
        static constexpr dim block_x {16};
        static constexpr dim block_y {16};
        for (dim iir1 {ir110}; iir1 < ir111; iir1 += block_y) { // outer kernel
            for (dim iir0 {ir010}; iir0 < ir011; iir0 += block_x) { // inner kernel
                for (dim ir1 {iir1}; ir1 < iir1 + block_y; ++ir1) { // block row kernel
                    if (ir1 >= ir111) break;
                    const dim i13 {ir1 / (y_d2*y_d1)}; // Unroll index into dimensions
                    const dim i12 {(ir1 - i13*y_d2*y_d1) / y_d1};
                    const dim i11 {ir1 - i13*y_d2*y_d1 - i12*y_d1};
                    const dim i03 {i13/r3}; // Broadcast x -> y
                    const dim i02 {i12/r2}; // Broadcast x -> y
                    const dim i1 {i11};
                    const dim i2 {i12};
                    const dim i3 {i13};
                    const std::uint8_t* const x_row {
                        px + (i02*x_s2 + i03*x_s3)
                    };
                    const std::uint8_t* const y_col {
                        py + row_size*(i11 + i12*y_d1 + i13*y_d2*y_d1)
                    };
                    auto* const r_col {
                        reinterpret_cast<float*>(pr + (i1*r_s1 + i2*r_s2 + i3*r_s3))
                    };
                    for (dim ir0 {iir0}; ir0 < iir0 + block_x && ir0 < ir011; ++ir0) { // blas kernel
                        v_f32_dot(
                            x_d0,
                            r_col+ir0,
                            reinterpret_cast<const float*>(x_row + ir0*x_s1),
                            reinterpret_cast<const float*>(y_col)
                        );
                    }
                }
            }
        }
    }
}
