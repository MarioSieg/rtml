// Copyright Mario "Neo" Sieg 2024. All rights reserved. mario.sieg.64@gmail.com
// CPU backend only!
// BLAS (basic linear algebra subprograms) for RTML (runtime machine learning) library
// Implements core tensor operations (which are not strictly BLAS routines) and some basic linear algebra operations
// Remember: Sparse means non-contiguous memory layout

#include <algorithm>
#include <cmath>

#include "blas.hpp"
#include "isolate.hpp"
#include "tensor.hpp"

namespace rtml::blas {
    namespace scalar { // These are needed to implement the generic tensor operations with vector (for dense) and scalar (for sparse) kernels
        // TODO: optimize with SIMD and dynamic CPU detection for x86-64: AVX512, AVX2, FMA, SSE and ARM: NEON (SVE in the future?)
        template <typename S> requires is_dtype<S>
        [[nodiscard]] static constexpr auto RTML_AINLINE RTML_HOT add(const S x, const S y) noexcept -> S { return x + y; }
        template <typename S> requires is_dtype<S>
        [[nodiscard]] static constexpr auto RTML_AINLINE RTML_HOT sub(const S x, const S y) noexcept -> S { return x - y; }
        template <typename S> requires is_dtype<S>
        [[nodiscard]] static constexpr auto RTML_AINLINE RTML_HOT mul(const S x, const S y) noexcept -> S { return x * y; }
        template <typename S> requires is_dtype<S>
        [[nodiscard]] static constexpr auto RTML_AINLINE RTML_HOT div(const S x, const S y) noexcept -> S { return x / y; }
    }

    namespace vec {
        // TODO: optimize with SIMD and dynamic CPU detection for x86-64: AVX512, AVX2, FMA, SSE and ARM: NEON (SVE in the future?)
        // TODO: optimize with polynomial approximation for tanh, sigmoid, relu, gelu, silu
        static constexpr float k_rtml_sqrt2pi {0.79788456080286535587989211986876f}; // sqrt(2/PI)
        static constexpr float k_rtml_gelu_coeff {0.044715f}; // GeLU coefficient

        template <typename S> requires is_dtype<S>
        static auto RTML_HOT softmax(const std::size_t n, S* const ov, const S* const x) noexcept -> void {
            for (std::size_t i = 0; i < n; ++i)
                ov[i] = std::expf(x[i]);
        }
        template <typename S> requires is_dtype<S>
        static auto RTML_HOT sigmoid(const std::size_t n, S* const ov, const S* const x) noexcept -> void {
            for (std::size_t i = 0; i < n; ++i)
                ov[i] = 1.0f / (1.0f + std::expf(-x[i]));
        }
        template <typename S> requires is_dtype<S>
        static auto RTML_HOT tanh(const std::size_t n, S* const ov, const S* const x) noexcept -> void {
            for (std::size_t i = 0; i < n; ++i)
                ov[i] = std::tanh(x[i]);
        }
        template <typename S> requires is_dtype<S>
        static auto RTML_HOT relu(const std::size_t n, S* const ov, const S* const x) noexcept -> void {
            for (std::size_t i = 0; i < n; ++i)
                ov[i] = std::max(x[i], 0.0f);
        }
        template <typename S> requires is_dtype<S>
        static auto RTML_HOT gelu(const std::size_t n, S* const ov, const S* const x) noexcept -> void {
            for (std::size_t i = 0; i < n; ++i)
                ov[i] = 0.5f * x[i] * (1.0f + std::tanh(k_rtml_sqrt2pi * x[i] * (1.0f + k_rtml_gelu_coeff * x[i] * x[i])));
        }
        template <typename S> requires is_dtype<S>
        static auto RTML_HOT silu(const std::size_t n, S* const ov, const S* const x) noexcept -> void {
            for (std::size_t i = 0; i < n; ++i)
                ov[i] = x[i] / (1.0f + std::exp(-x[i]));
        }
        template <typename S> requires is_dtype<S>
        static auto RTML_HOT add(const std::size_t n, S* const ov, const S* const x, const S* const y) noexcept -> void {
            for (std::size_t i = 0; i < n; ++i)
                ov[i] = scalar::add(x[i], y[i]);
        }
        template <typename S> requires is_dtype<S>
        static auto RTML_HOT sub(const std::size_t n, S* const ov, const S* const x, const S* const y) noexcept -> void {
            for (std::size_t i = 0; i < n; ++i)
                ov[i] = scalar::sub(x[i], y[i]);
        }
        template <typename S> requires is_dtype<S>
        static auto RTML_HOT mul(const std::size_t n, S* const ov, const S* const x, const S* const y) noexcept -> void {
            for (std::size_t i = 0; i < n; ++i)
                ov[i] = scalar::mul(x[i], y[i]);
        }
        template <typename S> requires is_dtype<S>
        static auto RTML_HOT div(const std::size_t n, S* const ov, const S* const x, const S* const y) noexcept -> void {
            for (std::size_t i = 0; i < n; ++i)
                ov[i] = scalar::div(x[i], y[i]);
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

    // Generic tensor binary operation like +, -, *, /
    template <const kernel_density density, typename S, typename V_OP, typename S_OP>
        requires is_dtype<S> && is_vector_op<V_OP, S> && is_scalar_op<S_OP, S>
    static auto RTML_AINLINE RTML_HOT blas_tensor_gen_op_binary_kernel(
        const compute_ctx& ctx,
        tensor<S>& r,       // result
        const tensor<S>& x, // X = src 0
        const tensor<S>& y, // Y = src 1
        V_OP&& v_op,        // Vector OP
        S_OP&& s_op         // Scalar OP
    ) noexcept -> void {
        std::uint8_t* const b_r {r.ptr()};                              // Data base ptr
        const std::uint8_t* const b_x {x.ptr()};                        // Data base ptr
        const std::uint8_t* const b_y {y.ptr()};                        // Data base ptr
        const auto [x_d0, x_d1, x_d2, x_d3] {x.dims()};                 // Dimensions of x
        const auto [x_s0, x_s1, x_s2, x_s3] {x.strides()};              // Strides of x
        const auto [y_d0, y_d1, y_d2, y_d3] {y.dims()};                 // Dimensions of y
        const auto [y_s0, y_s1, y_s2, y_s3] {y.strides()};              // Strides of y
        const auto [r_d0, r_d1, r_d2, r_d3] {r.dims()};                 // Dimensions of r
        const auto [r_s0, r_s1, r_s2, r_s3] {r.strides()};              // Strides of r
        const dim rc {r.row_count()};                                   // Row count (number of columns in first dim): r.dims()[0]
        const dim tidx {ctx.thread_idx};                                // Current thread index
        const dim tc {ctx.num_threads};                                 // Current thread count
        const dim rpt {(rc + tc - 1)/tc};                               // Rows per thread
        const dim row_start {rpt * tidx};                               // Current thread row interval start
        const dim row_end {std::min(row_start + rpt, rc)};          // Current thread row interval end
        for (dim row_i {row_start}; row_i < row_end; ++row_i) {         // For each row
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

    // Wrapper for generic tensor binary operation like +, -, *, / which dispatches to dense or sparse kernel
    template <typename S, typename V_OP, typename S_OP>
        requires is_dtype<S> && is_vector_op<V_OP, S> && is_scalar_op<S_OP, S>
    static auto RTML_AINLINE RTML_HOT blas_tensor_gen_op_binary(
        const compute_ctx& ctx,
        tensor<S>& r,
        const tensor<S>& x,
        const tensor<S>& y,
        V_OP&& v_op,        // Vector OP
        S_OP&& s_op         // Scalar OP
    ) noexcept -> void {
        if (y.strides()[0] == dtype_traits<S>::k_size) { // Dense or sparse kernel? Sparse means non-contiguous memory layout
            blas_tensor_gen_op_binary_kernel<kernel_density::dense, S, V_OP, S_OP>(
                ctx,
                r,
                x,
                y,
                v_op,
                s_op
            );
        } else {
            blas_tensor_gen_op_binary_kernel<kernel_density::sparse, S, V_OP, S_OP>(
                ctx,
                r,
                x,
                y,
                v_op,
                s_op
            );
        }
    }

    /*
     * BLAS SGEMM (Single precision General Matrix Multiply)
     * Compute the matrix product of two matrices X and Y: R = X @ Y
     * TODO: This is a naive implementation and not optimized.
     * TODO: Thread partitioning
     * TODO: optimize for cache efficiency and SIMD (use vec::dot)
     * TODO: Handle broadcasting
     */
    static auto RTML_AINLINE RTML_HOT blas_tensor_sgemm_naive(
        const compute_ctx& ctx,
        tensor<>& r,       // result
        const tensor<>& x, // X = src 0
        const tensor<>& y  // Y = src 1
    ) noexcept -> void {
        static_assert(std::is_same_v<std::decay_t<decltype(r)>::dtype, dtypes::f32>);
        static constexpr dim block_x {16};
        static constexpr dim block_y {16};
        static_assert(block_x == block_y);
        std::uint8_t* const b_r {r.ptr()};                              // Data base ptr
        const std::uint8_t* const b_x {x.ptr()};                        // Data base ptr
        const std::uint8_t* const b_y {y.ptr()};                        // Data base ptr
        const auto [x_d0, x_d1, x_d2, x_d3] {x.dims()};                 // Dimensions of x
        const auto [x_s0, x_s1, x_s2, x_s3] {x.strides()};              // Strides of x
        const auto [y_d0, y_d1, y_d2, y_d3] {y.dims()};                 // Dimensions of y
        const auto [y_s0, y_s1, y_s2, y_s3] {y.strides()};              // Strides of y
        const auto [r_d0, r_d1, r_d2, r_d3] {r.dims()};                 // Dimensions of r
        const auto [r_s0, r_s1, r_s2, r_s3] {r.strides()};              // Strides of r
        for (dim i3 {}; i3 < r_d3; ++i3) {
            for (dim i2 {}; i2 < r_d2; ++i2) {
                for (dim i1 {}; i1 < r_d1; ++i1) {
                    for (dim i0 {}; i0 < r_d1; ++i0) {
                        double sum = 0.0f; // TODO: optimize and use vec::dot
                        for (dim k {}; k < x_d0; ++k) {
                            const auto* p_x {reinterpret_cast<const dtypes::f32*>(
                                b_x + k*x_s0 + i0*x_s1 + i2*x_s2 + i3*x_s3
                            )};
                            const auto* p_y {reinterpret_cast<const dtypes::f32*>(
                                b_y + i1*y_s0 + k*y_s1 + i2*y_s2 + i3*y_s3
                            )};
                            sum += static_cast<double>(*p_x * *p_y);
                        }
                        auto* p_r {reinterpret_cast<dtypes::f32*>(
                            b_r + i1*r_s0 + i0*r_s1 + i2*r_s2 + i3*r_s3
                        )};
                        *p_r = static_cast<dtypes::f32>(sum);
                    }
                }
            }
        }
    }

    /*
     * BLAS SGEMM (Single precision General Matrix Multiply), but a modified version.
     * Compute the matrix product of two matrices X and Y: R_T = X @ Y_T
     * This version is optimized for cache efficiency and SIMD (use vec::dot) BUT it tranposes Y and the result.
     * This allows for row-by-row processing which is cache efficient and SIMD friendly but different from the original SGEMM.
     * TODO: Use this version if it makes sense (e.g. if Y is transposed and the result is transposed).
     * TODO: Comments
     */
    static auto RTML_AINLINE RTML_HOT blas_tensor_sgemm_tranposed(
        const compute_ctx& ctx,
        tensor<>& r,       // result
        const tensor<>& x, // X = src 0
        const tensor<>& y  // Y = src 1
    ) noexcept -> void {
        static_assert(std::is_same_v<std::decay_t<decltype(r)>::dtype, dtypes::f32>);
        static constexpr dim block_x {16};
        static constexpr dim block_y {16};
        static_assert(block_x == block_y);
        std::uint8_t* const b_r {r.ptr()};                              // Data base ptr
        const std::uint8_t* const b_x {x.ptr()};                        // Data base ptr
        const std::uint8_t* const b_y {y.ptr()};                        // Data base ptr
        const auto [x_d0, x_d1, x_d2, x_d3] {x.dims()};                 // Dimensions of x
        const auto [x_s0, x_s1, x_s2, x_s3] {x.strides()};              // Strides of x
        const auto [y_d0, y_d1, y_d2, y_d3] {y.dims()};                 // Dimensions of y
        const auto [y_s0, y_s1, y_s2, y_s3] {y.strides()};              // Strides of y
        const auto [r_d0, r_d1, r_d2, r_d3] {r.dims()};                 // Dimensions of r
        const auto [r_s0, r_s1, r_s2, r_s3] {r.strides()};              // Strides of r
        const dim tidx {ctx.thread_idx};                                // Current thread index
        const dim tc {ctx.num_threads};                                 // Current thread count
        const bool y_dense {y.is_dense()};
        const dim r2 {y_d2/x_d2};
        const dim r3 {y_d3/x_d3};
        const dim row_size {y_d0*static_cast<dim>(dtype_traits<dtypes::f32>::k_size)};
        const dim nr0 {x_d1};
        const dim nr1 {r_d1*y_d2*y_d3};
        const dim nth0 {nr0 > nr1 ? tc : 1};
        const dim nth1 {nr0 > nr1 ? 1 : tc};
        const dim ith0 {tidx % nth0};
        const dim ith1 {tidx / nth0};
        const dim dr0 {(nr0 + nth0 - 1)/nth0};
        const dim dr1 {(nr1 + nth1 - 1)/nth1};
        const dim ir010 {dr0*ith0};
        const dim ir011 {std::min(ir010 + dr0, nr0)};
        const dim ir110 {dr1*ith1};
        const dim ir111 {std::min(ir110 + dr1, nr1)};
        if (ir010 >= ir011 || ir110 >= ir111) {
            // TODO: no work to do - yield threads?
            return;
        }
        for (dim iir1 {ir110}; iir1 < ir111; iir1 += block_y)
        for (dim iir0 {ir010}; iir0 < ir011; iir0 += block_x)
        for (dim ir1 {iir1}; ir1 < iir1 + block_y && ir1 < ir111; ++ir1) {
            const dim i13 {ir1/(y_d2*r_d1)};
            const dim i12 {(ir1 - i13*y_d2*r_d1)/r_d1};
            const dim i11 {ir1 - i13*y_d2*r_d1 - i12*r_d1};
            const dim i03 {i13/r3};
            const dim i02 {i12/r2};
            const dim i1 {i11};
            const dim i2 {i12};
            const dim i3 {i13};
            const std::uint8_t* const p_x_row {
                b_x + i02*x_s2 + i03*x_s3
            };
            const auto* const p_y_col {reinterpret_cast<const dtypes::f32*>(
                b_y + (y_dense ?
                    (i11 + i12*y_d1 + i13*y_d2*y_d1) * row_size :
                    i11*y_s1 + i12*y_s2 + i13*y_s3)
            )};
            auto* const p_r_col {reinterpret_cast<dtypes::f32*>(
                b_r + i1*r_s1 + i2*r_s2 + i3*r_s3
            )};
            for (dim ir0 {iir0}; ir0 < iir0 + block_x && ir0 < ir011; ++ir0) { // Micro kernel
                vec::dot( // BLAS kernel
                    x_d0,
                    &p_r_col[ir0 - iir0],
                    reinterpret_cast<const dtypes::f32*>(p_x_row + ir0*x_s1),
                    p_y_col
                );
            }
        }
    }

    auto blas::add(const compute_ctx& ctx, tensor<dtypes::f32>& r, const tensor<dtypes::f32>& x, const tensor<dtypes::f32>& y) noexcept -> void {
        blas_tensor_gen_op_binary
        <
            std::decay_t<decltype(r)>::dtype,
            decltype(vec::add<std::decay_t<decltype(r)>::dtype>),
            decltype(scalar::add<std::decay_t<decltype(r)>::dtype>)
        >(ctx, r, x, y, vec::add, scalar::add);
    }

    auto blas::sub(const compute_ctx& ctx, tensor<dtypes::f32>& r, const tensor<dtypes::f32>& x, const tensor<dtypes::f32>& y) noexcept -> void {
        blas_tensor_gen_op_binary
        <
            std::decay_t<decltype(r)>::dtype,
            decltype(vec::sub<std::decay_t<decltype(r)>::dtype>),
            decltype(scalar::sub<std::decay_t<decltype(r)>::dtype>)
        >(ctx, r, x, y, vec::sub, scalar::sub);
    }

    auto blas::mul(const compute_ctx& ctx, tensor<dtypes::f32>& r, const tensor<dtypes::f32>& x, const tensor<dtypes::f32>& y) noexcept -> void {
        blas_tensor_gen_op_binary
        <
            std::decay_t<decltype(r)>::dtype,
            decltype(vec::mul<std::decay_t<decltype(r)>::dtype>),
            decltype(scalar::mul<std::decay_t<decltype(r)>::dtype>)
        >(ctx, r, x, y, vec::mul, scalar::mul);
    }

    auto blas::div(const compute_ctx& ctx, tensor<dtypes::f32>& r, const tensor<dtypes::f32>& x, const tensor<dtypes::f32>& y) noexcept -> void {
        blas_tensor_gen_op_binary
        <
            std::decay_t<decltype(r)>::dtype,
            decltype(vec::div<std::decay_t<decltype(r)>::dtype>),
            decltype(scalar::div<std::decay_t<decltype(r)>::dtype>)
        >(ctx, r, x, y, vec::div, scalar::div);
    }

    auto matmul(const compute_ctx& ctx, tensor<>& r, const tensor<>& x, const tensor<>& y) noexcept -> void {
        blas_tensor_sgemm_naive(ctx, r, x, y);
        // TODO - Use this version if it makes sense
        //blas_tensor_sgemm_tranposed(ctx, r, x, y);
    }
}
