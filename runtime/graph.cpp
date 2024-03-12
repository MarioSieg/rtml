// Copyright Mario "Neo" Sieg 2024. All rights reserved. mario.sieg.64@gmail.com

#include "graph.hpp"
#include "tensor.hpp"
#include "isolate.hpp"
#include "blas.hpp"

namespace rtml::graph {
    [[nodiscard]] static constexpr auto validate_unary_op(const tensor* const dst, const std::span<const tensor*> src) -> bool {
        const tensor* const r {dst};
        if (!r) [[unlikely]] {
            return false;
        }
        if (const auto num {static_cast<std::size_t>(r->get_op())}; k_operands[num] != src.size() && src.size() == 1) [[unlikely]] {
            rtml_log_error("Operand count mismatch, expected {}, got {}", num, src.size());
            return false;
        }
        const tensor* const x {src[0]};
        if (!x) [[unlikely]] {
            return false;
        }
        if (!x->is_contiguous_except_dim1()) [[unlikely]] {
            return false;
        }
        if (!r->is_contiguous_except_dim1()) [[unlikely]] {
            return false;
        }
        if (r->is_shape_eq(x)) [[unlikely]] {
            return false;
        }
        return true;
    }

    [[nodiscard]] static constexpr auto validate_binary_op(const tensor* const dst, const std::span<const tensor*> src) -> bool {
        const tensor* const r {dst};
        if (!r) [[unlikely]] {
            return false;
        }
        if (const auto num {static_cast<std::size_t>(r->get_op())}; k_operands[num] != src.size() && src.size() == 2) [[unlikely]] {
            rtml_log_error("Operand count mismatch, expected {}, got {}", num, src.size());
            return false;
        }
        const tensor* const x {src[0]};
        const tensor* const y {src[1]};
        if (!x || !y) [[unlikely]] {
            return false;
        }
        if (x->get_strides()[0] != sizeof(float)) [[unlikely]] {
            return false;
        }
        if (r->get_strides()[0] != sizeof(float)) [[unlikely]] {
            return false;
        }
        if (!y->can_repeat(x)) [[unlikely]] {
            return false;
        }
        if (!x->is_shape_eq(r)) [[unlikely]] {
            return false;
        }
        return true;
    }

    static constexpr validate_function* const k_validators[] {
        // nullary ops
        [nop] = [](const tensor* const dst, const std::span<const tensor*> src) -> bool {
            return true;
        },

        // unary ops
        [softmax] = &validate_unary_op,
        [sigmoid] = &validate_unary_op,
        [tanh] = &validate_unary_op,
        [relu] = &validate_unary_op,
        [gelu] = &validate_unary_op,
        [silu] = &validate_unary_op,

        // binary ops
        [add] = &validate_binary_op,
        [sub] = &validate_binary_op,
        [mul] = &validate_binary_op,
        [div] = &validate_binary_op,
        [matmul] = &validate_binary_op,
    };
    static_assert(std::size(k_validators) == $count);

    // Implementation of a simple unary tensor operation evaluator
    #define impl_eval_unary(blas_func) \
        tensor* const r {dst}; \
        const tensor* const x {src[0]}; \
        const dim num_rows {x->row_count()}; \
        const dim num_cols {x->col_count()}; \
        for (dim row {}; row < num_rows; ++row) { \
            blas::blas_func( \
                num_cols, \
                reinterpret_cast<float*>(r->get_data() + row * r->get_strides()[1]), \
                reinterpret_cast<const float*>(x->get_data() + row * x->get_strides()[1]) \
            ); \
        }

    template <typename T>
    concept tensor_scalar = requires {
        std::is_arithmetic_v<T>;
    };

    // Implementation of an optimized binary tensor operation evaluator
    template <typename T> requires tensor_scalar<T>
    auto eval_tensor_op(tensor* const dst, const std::span<const tensor*> src) noexcept -> void {
        tensor* const r {dst};
        const tensor* const x {src[0]};
        const tensor* const y {src[1]};
        std::uint8_t* const pr {r->get_data()};
        const std::uint8_t* const px {x->get_data()};
        const std::uint8_t* const py {y->get_data()};
        const dim num_rows {r->row_count()};
        const auto [x_d0, x_d1, x_d2, x_d3] {x->get_dims()};
        const auto [x_s0, x_s1, x_s2, x_s3] {x->get_strides()};
        const auto [y_d0, y_d1, y_d2, y_d3] {y->get_dims()};
        const auto [y_s0, y_s1, y_s2, y_s3] {y->get_strides()};
        const auto [r_d0, r_d1, r_d2, r_d3] {r->get_dims()};
        const auto [r_s0, r_s1, r_s2, r_s3] {r->get_strides()};
        if (y_s0 == sizeof(T)) { // Tensor x has a contigous memory layout
            for (dim row_i {}; row_i < num_rows; ++row_i) {
                const dim x_i3 {row_i / (x_d2*x_d1)}; // Unroll index into dimensions
                const dim x_i2 {(row_i - x_i3*x_d2*x_d1) / x_d1};
                const dim x_i1 {row_i - x_i3*x_d2*x_d1 - x_i2*x_d1};
                const dim y_i3 {x_i3 % y_d3}; // Broadcast
                const dim y_i2 {x_i2 % y_d2}; // Broadcast
                const dim y_i1 {x_i1 % y_d1}; // Broadcast
                T* const pdst {reinterpret_cast<T*>(pr + x_i3*r_s3 + x_i2*r_s2 + x_i1*r_s1)};
                const T* const psrc0 {reinterpret_cast<const T*>(px + x_i3*x_s3 + x_i2*x_s2 + x_i1*x_s1)};
                const T* const psrc1 {reinterpret_cast<const T*>(py + y_i3*y_s3 + y_i2*y_s2 + y_i1*y_s1)};
                for (dim v {}; v < x_d0 / y_d0; ++v) // Vector operation between contiguous rows
                    blas::v_add(y_d0, pdst + v*y_d0, psrc0 + v*y_d0, psrc1);
            }
        } else {
            for (dim row_i {}; row_i < num_rows; ++row_i) {
                const dim x_i3 {row_i/(x_d2*x_d1)}; // Unroll index into dimensions
                const dim x_i2 {(row_i - x_i3*x_d2*x_d1)/x_d1};
                const dim x_i1 {row_i - x_i3*x_d2*x_d1 - x_i2*x_d1};
                const dim y_i3 {x_i3 % y_d3}; // Broadcast
                const dim y_i2 {x_i2 % y_d2}; // Broadcast
                const dim y_i1 {x_i1 % y_d1}; // Broadcast
                T* const pdst {reinterpret_cast<T*>(pr + x_i3*r_s3 + x_i2*r_s2 + x_i1*r_s1)};
                const T* const psrc0 {reinterpret_cast<const T*>(px + x_i3*x_s3 + x_i2*x_s2 + x_i1*x_s1)};
                for (dim i {}; i < r_d0; ++i) { // Scalar operation
                    const dim i10 {i % y_d0};
                    const T *p1 = reinterpret_cast<const T*>(py + y_i3*y_s3 + y_i2*y_s2 + y_i1*y_s1 + i10*y_s0);
                    pdst[i] = psrc0[i] + *p1;
                }
            }
        }
    }

    static constexpr eval_function* const k_evaluators[] {
        [nop] = [](tensor* const dst, const std::span<const tensor*> src) noexcept -> void {
        },
        [softmax] = [](tensor* const dst, const std::span<const tensor*> src) noexcept -> void {
            impl_eval_unary(v_softmax)
        },
        [sigmoid] = [](tensor* const dst, const std::span<const tensor*> src) noexcept -> void {
            impl_eval_unary(v_sigmoid)
        },
        [tanh] = [](tensor* const dst, const std::span<const tensor*> src) noexcept -> void {
            impl_eval_unary(v_tanh)
        },
        [relu] = [](tensor* const dst, const std::span<const tensor*> src) noexcept -> void {
            impl_eval_unary(v_relu)
        },
        [gelu] = [](tensor* const dst, const std::span<const tensor*> src) noexcept -> void {
            impl_eval_unary(v_gelu)
        },
        [silu] = [](tensor* const dst, const std::span<const tensor*> src) noexcept -> void {
            impl_eval_unary(v_silu)
        },
        [add] = [](tensor* const dst, const std::span<const tensor*> src) noexcept -> void {
        },
        [sub] = [](tensor* const dst, const std::span<const tensor*> src) noexcept -> void {
        },
        [mul] = [](tensor* const dst, const std::span<const tensor*> src) noexcept -> void {
        },
        [div] = [](tensor* const dst, const std::span<const tensor*> src) noexcept -> void {
        },
        [matmul] = [](tensor* const dst, const std::span<const tensor*> src) noexcept -> void {
        },
    };
    static_assert(std::size(k_evaluators) == $count);

#undef impl_eval_unary

    auto graph_visit(
        const tensor* const root,
        const graph_eval_order order,
        const std::function<auto(const tensor* t) -> void>& callback
    ) -> void {
        const auto& operands {root->get_operands()};
        for (std::size_t i {}; i < operands.size(); ++i) {
            const std::size_t ii {
                order == graph_eval_order::left_to_right
                    ? i : operands.size() - i - 1
            };
            graph_visit(operands[ii], order, callback);
        }
        callback(root);
    }
}
