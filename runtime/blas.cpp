// Copyright Mario "Neo" Sieg 2024. All rights reserved. mario.sieg.64@gmail.com

#include <algorithm>
#include <cmath>

#include "blas.hpp"
#include "isolate.hpp"
#include "tensor.hpp"

namespace rtml::blas {
#define RTML_AINLINE __attribute__((always_inline))
#define RTML_HOT __attribute__((hot))

    namespace scalar {
        static constexpr float k_rtml_sqrt2pi {0.79788456080286535587989211986876f};
        static constexpr float k_rtml_gelu_coeff {0.044715f};

        template <typename S> requires is_dtype<S>
        [[nodiscard]] static constexpr auto RTML_AINLINE RTML_HOT add(const S x, const S y) noexcept -> S { return x + y; }
        template <typename S> requires is_dtype<S>
        [[nodiscard]] static constexpr auto RTML_AINLINE RTML_HOT sub(const S x, const S y) noexcept -> S { return x - y; }
        template <typename S> requires is_dtype<S>
        [[nodiscard]] static constexpr auto RTML_AINLINE RTML_HOT mul(const S x, const S y) noexcept -> S { return x * y; }
        template <typename S> requires is_dtype<S>
        [[nodiscard]] static constexpr auto RTML_AINLINE RTML_HOT div(const S x, const S y) noexcept -> S { return x / y; }

        [[nodiscard]] static constexpr auto RTML_AINLINE RTML_HOT softmax(const dtypes::f32 x) noexcept -> float {
            return std::expf(x);
        }
        [[nodiscard]] static constexpr auto RTML_AINLINE RTML_HOT sigmoid(const dtypes::f32 x) noexcept -> float {
            return 1.0f / (1.0f + std::expf(-x));
        }
        [[nodiscard]] static constexpr auto RTML_AINLINE RTML_HOT tanh(const dtypes::f32 x) noexcept -> float {
            return std::tanh(x);
        }
        [[nodiscard]] static constexpr auto RTML_AINLINE RTML_HOT relu(const dtypes::f32 x) noexcept -> float {
            return std::max(x, 0.0f);
        }
        [[nodiscard]] static constexpr auto RTML_AINLINE RTML_HOT gelu(const dtypes::f32 x) noexcept -> float {
            return 0.5f * x * (1.0f + std::tanh(k_rtml_sqrt2pi * x * (1.0f + k_rtml_gelu_coeff * x * x)));
        }
        [[nodiscard]] static constexpr auto RTML_AINLINE RTML_HOT silu(const dtypes::f32 x) noexcept -> float {
            return x / (1.0f + std::exp(-x));
        }
    }

    namespace vec {
        template <typename S> requires is_dtype<S>
        static auto RTML_HOT softmax(const std::size_t n, S* const ov, const S* const x) noexcept -> void {
            for (std::size_t i = 0; i < n; ++i) ov[i] = scalar::softmax(x[i]);
        }
        template <typename S> requires is_dtype<S>
        static auto RTML_HOT sigmoid(const std::size_t n, S* const ov, const S* const x) noexcept -> void {
            for (std::size_t i = 0; i < n; ++i) ov[i] = scalar::sigmoid(x[i]);
        }
        template <typename S> requires is_dtype<S>
        static auto RTML_HOT tanh(const std::size_t n, S* const ov, const S* const x) noexcept -> void {
            for (std::size_t i = 0; i < n; ++i) ov[i] = scalar::tanh(x[i]);
        }
        template <typename S> requires is_dtype<S>
        static auto RTML_HOT relu(const std::size_t n, S* const ov, const S* const x) noexcept -> void {
            for (std::size_t i = 0; i < n; ++i) ov[i] = scalar::relu(x[i]);
        }
        template <typename S> requires is_dtype<S>
        static auto RTML_HOT gelu(const std::size_t n, S* const ov, const S* const x) noexcept -> void {
            for (std::size_t i = 0; i < n; ++i) ov[i] = scalar::gelu(x[i]);
        }
        template <typename S> requires is_dtype<S>
        static auto RTML_HOT silu(const std::size_t n, S* const ov, const S* const x) noexcept -> void {
            for (std::size_t i = 0; i < n; ++i) ov[i] = scalar::silu(x[i]);
        }
        template <typename S> requires is_dtype<S>
        static auto RTML_HOT add(const std::size_t n, S* const ov, const S* const x, const S* const y) noexcept -> void {
            for (std::size_t i = 0; i < n; ++i) ov[i] = scalar::add(x[i], y[i]);
        }
        template <typename S> requires is_dtype<S>
        static auto RTML_HOT sub(const std::size_t n, S* const ov, const S* const x, const S* const y) noexcept -> void {
            for (std::size_t i = 0; i < n; ++i) ov[i] = scalar::sub(x[i], y[i]);
        }
        template <typename S> requires is_dtype<S>
        static auto RTML_HOT mul(const std::size_t n, S* const ov, const S* const x, const S* const y) noexcept -> void {
            for (std::size_t i = 0; i < n; ++i) ov[i] = scalar::mul(x[i], y[i]);
        }
        template <typename S> requires is_dtype<S>
        static auto RTML_HOT div(const std::size_t n, S* const ov, const S* const x, const S* const y) noexcept -> void {
            for (std::size_t i = 0; i < n; ++i) ov[i] = scalar::div(x[i], y[i]);
        }
        template <typename S> requires is_dtype<S>
        static auto RTML_HOT dot(const std::size_t n, S* const os, const S* const x, const S* const y) noexcept -> void {
            double sum = 0.0;
            for (std::size_t i = 0; i < n; ++i)
                sum += static_cast<double>(x[i] * y[i]);
            *os = static_cast<float>(sum);
        }
    }

    template <typename F, typename S>
    concept is_vector_op = requires {
        is_dtype<S>;
        std::is_nothrow_invocable_r_v<void, F, S*, const S*, const S*>; // auto f(S* r, const S* x, const S* y) -> void
    };

    template <typename F, typename S>
    concept is_scalar_op = requires {
        is_dtype<S>;
        std::is_nothrow_invocable_r_v<S, F, S, S>; // auto f(S x, S y) -> S
    };

    enum class kernel_density {
        dense,
        sparse
    };

    template <const kernel_density density, typename S, typename V_OP, typename S_OP>
        requires is_dtype<S> && is_vector_op<V_OP, S> && is_scalar_op<S_OP, S>
    static auto RTML_AINLINE RTML_HOT blas_tensor_generic_op(
        tensor<S>& r,       // result
        const tensor<S>& x, // X = src 0
        const tensor<S>& y, // Y = src 1
        V_OP&& v_op,        // Vector OP
        S_OP&& s_op         // Scalar OP
    ) noexcept -> void {
        const auto [x_d0, x_d1, x_d2, x_d3] {x.dims()};
        const auto [x_s0, x_s1, x_s2, x_s3] {x.strides()};
        const auto [y_d0, y_d1, y_d2, y_d3] {y.dims()};
        const auto [y_s0, y_s1, y_s2, y_s3] {y.strides()};
        const auto [r_d0, r_d1, r_d2, r_d3] {r.dims()};
        const auto [r_s0, r_s1, r_s2, r_s3] {r.strides()};
        std::uint8_t* const b_r {r.ptr()};                              // Data base ptr
        const std::uint8_t* const b_x {x.ptr()};                        // Data base ptr
        const std::uint8_t* const b_y {y.ptr()};                        // Data base ptr
        const dim num_rows {r.row_count()};
        for (dim row_i {}; row_i < num_rows; ++row_i) {                 // For each row
            const dim x_i3 {row_i / (x_d2*x_d1)};                       // Dimension 3 - Linear index to 3D index
            const dim x_i2 {(row_i - x_i3*x_d2*x_d1) / x_d1};           // Dimension 2 - Linear index to 3D index
            const dim x_i1 {row_i - x_i3*x_d2*x_d1 - x_i2*x_d1};        // Dimension 1 - Linear index to 3D index
            const dim y_i3 {x_i3 % y_d3};                               // Dimension 3 Broadcast x -> y
            const dim y_i2 {x_i2 % y_d2};                               // Dimension 2 Broadcast x -> y
            const dim y_i1 {x_i1 % y_d1};                               // Dimension 1 Broadcast x -> y
            auto* const p_r {reinterpret_cast<S*>(                      // Result destination ptr
                b_r + x_i3*r_s3 + x_i2*r_s2 + x_i1*r_s1
            )};
            const auto* const p_x {reinterpret_cast<const S*>(          // X Source ptr
                b_x + x_i3*x_s3 + x_i2*x_s2 + x_i1*x_s1
            )};
            if constexpr (density == kernel_density::dense) {           // Dense kernel for contiguous layout
                const auto* const p_y {reinterpret_cast<const S*>(      // Y Source ptr
                    b_y + y_i3*y_s3 + y_i2*y_s2 + y_i1*y_s1
                )};
                for (dim i {}; i < x_d0 / y_d0; ++i) {                  // Macro Kernel
                    v_op(y_d0, p_r + i*y_d0, p_x + i*y_d0, p_y);        // Micro Kernel - Apply vector operation
                }
            } else {                                                    // Sparse kernel
                for (dim i {}; i < r_d0; ++i) {                         // Micro Kernel
                    const auto* const p_y {reinterpret_cast<const S*>(  // Y Source ptr
                        b_y + y_i3*y_s3 + y_i2*y_s2 + y_i1*y_s1 + i%y_d0*y_s0
                    )};
                    p_r[i] = s_op(p_x[i], *p_y);                        // Apply scalar operation
                }
            }
        }
    }

    template <typename S, typename V_OP, typename S_OP>
        requires is_dtype<S> && is_vector_op<V_OP, S> && is_scalar_op<S_OP, S>
    static auto RTML_AINLINE RTML_HOT tensor_base_op(
        tensor<S>& r,
        const tensor<S>& x,
        const tensor<S>& y,
        V_OP&& v_op,        // Vector OP
        S_OP&& s_op         // Scalar OP
    ) noexcept -> void {
        if (y.strides()[0] == dtype_traits<S>::k_size) { // Sparse or dense kernel?
            blas_tensor_generic_op<kernel_density::dense, S, V_OP, S_OP>(
                r,
                x,
                y,
                v_op,
                s_op
            );
        } else {
            blas_tensor_generic_op<kernel_density::sparse, S, V_OP, S_OP>(
                r,
                x,
                y,
                v_op,
                s_op
            );
        }
    }

    auto blas::t_f32_add(tensor<dtypes::f32>& r, const tensor<dtypes::f32>& x, const tensor<dtypes::f32>& y) noexcept -> void {
        tensor_base_op
        <
            std::decay_t<decltype(r)>::dtype,
            decltype(vec::add<std::decay_t<decltype(r)>::dtype>),
            decltype(scalar::add<std::decay_t<decltype(r)>::dtype>)
        >(r, x, y, vec::add, scalar::add);
    }

    auto blas::t_f32_sub(tensor<dtypes::f32>& r, const tensor<dtypes::f32>& x, const tensor<dtypes::f32>& y) noexcept -> void {
        tensor_base_op
        <
            std::decay_t<decltype(r)>::dtype,
            decltype(vec::sub<std::decay_t<decltype(r)>::dtype>),
            decltype(scalar::sub<std::decay_t<decltype(r)>::dtype>)
        >(r, x, y, vec::add, scalar::add);
    }

    auto blas::t_f32_mul(tensor<dtypes::f32>& r, const tensor<dtypes::f32>& x, const tensor<dtypes::f32>& y) noexcept -> void {
        tensor_base_op
        <
            std::decay_t<decltype(r)>::dtype,
            decltype(vec::mul<std::decay_t<decltype(r)>::dtype>),
            decltype(scalar::mul<std::decay_t<decltype(r)>::dtype>)
        >(r, x, y, vec::add, scalar::add);
    }

    auto blas::t_f32_div(tensor<dtypes::f32>& r, const tensor<dtypes::f32>& x, const tensor<dtypes::f32>& y) noexcept -> void {
        tensor_base_op
        <
            std::decay_t<decltype(r)>::dtype,
            decltype(vec::div<std::decay_t<decltype(r)>::dtype>),
            decltype(scalar::div<std::decay_t<decltype(r)>::dtype>)
        >(r, x, y, vec::add, scalar::add);
    }

    auto t_f32_matmul(tensor<dtypes::f32>& r, const tensor<dtypes::f32>& x, const tensor<dtypes::f32>& y) noexcept -> void {

    }
}
