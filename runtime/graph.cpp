// Copyright Mario "Neo" Sieg 2024. All rights reserved. mario.sieg.64@gmail.com

#include "graph.hpp"
#include "isolate.hpp"
#include "tensor.hpp"
#include "blas.hpp"

#if 0

namespace rtml::graph {
    [[nodiscard]] static constexpr auto validate_unary_op(const tensor* const dst, const std::span<const tensor*> src) -> bool {
        const tensor* const r {dst};
        if (!r) [[unlikely]] {
            return false;
        }
        if (const auto num {static_cast<std::size_t>(r->opcode())}; k_operands[num] != src.size() && src.size() == 1) [[unlikely]] {
            rtml_log_error("Operand count mismatch, expected {}, got {}", num, src.size());
            return false;
        }
        const tensor* const x {src[0]};
        if (!x) [[unlikely]] {
            return false;
        }
        if (!x->is_dense_except_dim1()) [[unlikely]] {
            return false;
        }
        if (!r->is_dense_except_dim1()) [[unlikely]] {
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
        if (const auto num {static_cast<std::size_t>(r->opcode())}; k_operands[num] != src.size() && src.size() == 2) [[unlikely]] {
            rtml_log_error("Operand count mismatch, expected {}, got {}", num, src.size());
            return false;
        }
        const tensor* const x {src[0]};
        const tensor* const y {src[1]};
        if (!x || !y) [[unlikely]] {
            return false;
        }
        if (x->strides()[0] != sizeof(float)) [[unlikely]] {
            return false;
        }
        if (r->strides()[0] != sizeof(float)) [[unlikely]] {
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
                reinterpret_cast<float*>(r->ptr() + row * r->strides()[1]), \
                reinterpret_cast<const float*>(x->ptr() + row * x->strides()[1]) \
            ); \
        }

    static constexpr eval_function* const k_evaluators[] {
        [nop] = [](tensor* const dst, const std::span<const tensor*> src) noexcept -> void {
        },
        [softmax] = [](tensor* const dst, const std::span<const tensor*> src) noexcept -> void {
            impl_eval_unary(v_f32_softmax)
        },
        [sigmoid] = [](tensor* const dst, const std::span<const tensor*> src) noexcept -> void {
            impl_eval_unary(v_f32_sigmoid)
        },
        [tanh] = [](tensor* const dst, const std::span<const tensor*> src) noexcept -> void {
            impl_eval_unary(v_f32_tanh)
        },
        [relu] = [](tensor* const dst, const std::span<const tensor*> src) noexcept -> void {
            impl_eval_unary(v_f32_relu)
        },
        [gelu] = [](tensor* const dst, const std::span<const tensor*> src) noexcept -> void {
            impl_eval_unary(v_f32_gelu)
        },
        [silu] = [](tensor* const dst, const std::span<const tensor*> src) noexcept -> void {
            impl_eval_unary(v_f32_silu)
        },
        [add] = [](tensor* const dst, const std::span<const tensor*> src) noexcept -> void {
            blas::t_f32_add(*dst, *src[0], *src[1]);
        },
        [sub] = [](tensor* const dst, const std::span<const tensor*> src) noexcept -> void {
            blas::t_f32_sub(*dst, *src[0], *src[1]);
        },
        [mul] = [](tensor* const dst, const std::span<const tensor*> src) noexcept -> void {
            blas::t_f32_mul(*dst, *src[0], *src[1]);
        },
        [div] = [](tensor* const dst, const std::span<const tensor*> src) noexcept -> void {
            blas::t_f32_div(*dst, *src[0], *src[1]);
        },
        [matmul] = [](tensor* const dst, const std::span<const tensor*> src) noexcept -> void {
            blas::t_f32_matmul(*dst, *src[0], *src[1]);
        },
    };
    static_assert(std::size(k_evaluators) == $count);

#undef impl_eval_unary

    auto graph_visit(
        const tensor* const root,
        const graph_eval_order order,
        const std::function<auto(const tensor* t) -> void>& callback
    ) -> void {
        const auto& operands {root->operands()};
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

#endif
