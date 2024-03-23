// Copyright Mario "Neo" Sieg 2024. All rights reserved. mario.sieg.64@gmail.com

#pragma once

#include <functional>
#include <span>

#include "base.hpp"
#include "blas.hpp"
#include "tensor.hpp"

namespace rtml::graph {
    template <typename S> requires is_dtype<S>
    using validate_function = auto (const tensor<S>* dst, std::span<const tensor<S>* const> src) -> bool;
    template <typename S> requires is_dtype<S>
    using eval_function = auto (const blas::compute_ctx& ctx, tensor<S>* dst, std::span<const tensor<S>* const> src) noexcept -> void;

    enum class graph_eval_order : bool {
        left_to_right,
        right_to_left
    };

    template <const graph_eval_order Ord, typename T, typename F, typename... Args>
    auto graph_visit(
        T* root,
        F&& callback,
        Args&&... args
    ) -> void {
        if (root->opcode() == opcode::nop) return;
        auto&& operands {root->operands()};
        for (std::size_t i {}; i < operands.size(); ++i) {
            std::size_t ii;
            if constexpr (Ord == graph_eval_order::left_to_right) ii = i;
            else ii = operands.size() - i - 1;
            graph_visit<Ord>(operands[ii], std::forward<F>(callback), std::forward<Args>(args)...);
        }
        std::invoke(callback, root, std::forward<Args>(args)...);
    }

    // all validation functions go here
    namespace validators {
        #define rtml_verify(expr, msg, ...) \
            if (!(expr)) [[unlikely]] { \
                rtml_log_error("{}:{} Validation failed: " #expr "\t<-\t" msg, __FILE__, __LINE__, ## __VA_ARGS__); \
                return false; \
            }

        template <typename S> requires is_dtype<S>
        [[nodiscard]] constexpr auto validate_unary_op(
            const tensor<S>* const r,
            const std::span<const tensor<S>* const> src
        ) -> bool {
            rtml_verify(r, "Result tensor is null");
            const auto num_operands {static_cast<std::size_t>(r->opcode())};
            rtml_verify(k_operands[num_operands] == src.size(), "Number of operands mismatch, expected {} got {}", k_operands[num_operands], src.size());
            rtml_verify(src[0], "Source tensor is null");
            rtml_verify(src[0]->is_dense_except_dim1(), "Source tensor is not dense except dim1");
            rtml_verify(r->is_dense_except_dim1(), "Result tensor is not dense except dim1");
            rtml_verify(r->is_shape_eq(src[0]), "Result tensor shape mismatch");
            return true;
        }

        template <typename S> requires is_dtype<S>
        [[nodiscard]] constexpr auto validate_binary_op(
            const tensor<S>* const r,
            const std::span<const tensor<S>* const> src
        ) -> bool {
            rtml_verify(r, "Result tensor is null");
            const auto num_operands {static_cast<std::size_t>(r->opcode())};
            rtml_verify(k_operands[num_operands] == src.size(), "Number of operands mismatch, expected {} got {}", k_operands[num_operands], src.size());
            rtml_verify(src[0], "Source tensor 0 is null");
            rtml_verify(src[1], "Source tensor 1 is null");
            rtml_verify(src[0]->strides()[0] == dtype_traits<S>::k_size, "Source tensor 0 stride mismatch");
            rtml_verify(r->strides()[0] == dtype_traits<S>::k_size, "Result tensor stride mismatch");
            rtml_verify(src[1]->can_repeat(src[0]), "Source tensor 1 cannot repeat source tensor 0");
            rtml_verify(src[0]->is_shape_eq(r), "Source tensor 0 shape mismatch");
            return true;
        }

        #undef rtml_verify
    }

    template <typename S> requires is_dtype<S>
    struct routines;

    template <>
    struct routines<dtypes::f32> {
        static constexpr std::array<validate_function<dtypes::f32>*, static_cast<std::size_t>(opcode::$count)> validators {
            []() consteval -> auto { // Autogenerate table of validation functions for unary and binary ops
                std::array<validate_function<dtypes::f32>*, static_cast<std::size_t>(opcode::$count)> result {};
                for (std::size_t i {}; i < static_cast<std::size_t>(opcode::$count); ++i) {
                    if (k_operands[i] == 1) { // unary op
                        result[i] = &validators::validate_unary_op<dtypes::f32>;
                    } else { // binary op
                        result[i] = &validators::validate_binary_op<dtypes::f32>;
                    }
                }
                return result;
            }()
        };
        static constexpr std::array<eval_function<dtypes::f32>*, static_cast<std::size_t>(opcode::$count)> evaluators {
            +[]([[maybe_unused]] const blas::compute_ctx& ctx, [[maybe_unused]] tensor<dtypes::f32>* const r, [[maybe_unused]] const std::span<const tensor<dtypes::f32>* const> src) noexcept -> void {
                // nop
            },
            +[](const blas::compute_ctx& ctx, tensor<dtypes::f32>* const r, const std::span<const tensor<dtypes::f32>* const> src) noexcept -> void {
                blas::softmax(ctx, *r, *src[0]);
            },
            +[](const blas::compute_ctx& ctx, tensor<dtypes::f32>* const r, const std::span<const tensor<dtypes::f32>* const> src) noexcept -> void {
                blas::sigmoid(ctx, *r, *src[0]);
            },
            +[](const blas::compute_ctx& ctx, tensor<dtypes::f32>* const r, const std::span<const tensor<dtypes::f32>* const> src) noexcept -> void {
                blas::tanh(ctx, *r, *src[0]);
            },
            +[](const blas::compute_ctx& ctx, tensor<dtypes::f32>* const r, const std::span<const tensor<dtypes::f32>* const> src) noexcept -> void {
                blas::relu(ctx, *r, *src[0]);
            },
            +[](const blas::compute_ctx& ctx, tensor<dtypes::f32>* const r, const std::span<const tensor<dtypes::f32>* const> src) noexcept -> void {
                blas::gelu(ctx, *r, *src[0]);
            },
            +[](const blas::compute_ctx& ctx, tensor<dtypes::f32>* const r, const std::span<const tensor<dtypes::f32>* const> src) noexcept -> void {
                blas::silu(ctx, *r, *src[0]);
            },
            +[](const blas::compute_ctx& ctx, tensor<dtypes::f32>* const r, const std::span<const tensor<dtypes::f32>* const> src) noexcept -> void {
                blas::add(ctx, *r, *src[0], *src[1]);
            },
            +[](const blas::compute_ctx& ctx, tensor<dtypes::f32>* const r, const std::span<const tensor<dtypes::f32>* const> src) noexcept -> void {
                blas::sub(ctx, *r, *src[0], *src[1]);
            },
            +[](const blas::compute_ctx& ctx, tensor<dtypes::f32>* const r, const std::span<const tensor<dtypes::f32>* const> src) noexcept -> void {
                blas::mul(ctx, *r, *src[0], *src[1]);
            },
            +[](const blas::compute_ctx& ctx, tensor<dtypes::f32>* const r, const std::span<const tensor<dtypes::f32>* const> src) noexcept -> void {
                blas::div(ctx, *r, *src[0], *src[1]);
            },
            +[](const blas::compute_ctx& ctx, tensor<dtypes::f32>* const r, const std::span<const tensor<dtypes::f32>* const> src) noexcept -> void {
                blas::matmul(ctx, *r, *src[0], *src[1]);
            }
        };
    };

    template <typename S> requires is_dtype<S>
    auto RTML_HOT compute(tensor<S>* const root) -> void {
        blas::compute_ctx ctx {};
        graph_visit<graph_eval_order::left_to_right, const tensor<S>>(root, [&ctx](const tensor<S>* const t) noexcept -> void {
            const auto op_idx {static_cast<std::size_t>(t->opcode())};
            const std::span<const tensor<S>* const> operands {t->operands().data(), t->operands().size()};
            rtml_assert(routines<S>::validators[op_idx](t, operands), "Validation failed");
            routines<S>::evaluators[op_idx](ctx, const_cast<tensor<S>*>(t), operands);
        });
    }
}
