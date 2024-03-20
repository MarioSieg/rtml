// Copyright Mario "Neo" Sieg 2024. All rights reserved. mario.sieg.64@gmail.com

#pragma once

#include <functional>
#include <span>

#include "tensor_base.hpp"

namespace rtml::graph {
    #define rtml_co ,
    /* Opcodes: Mnemonic, Operands, Info Mnemonic */
    #define rtml_opcode_def(_, __) \
        /* Unary ops */\
        _(softmax, 1, "softmax")__\
        _(sigmoid, 1, "sigmoid")__\
        _(tanh, 1, "tanh")__\
        _(relu, 1, "relu")__\
        _(gelu, 1, "gelu")__\
        _(silu, 1, "silu")__\
        /* Binary ops */\
        _(add , 2, "+")__\
        _(sub , 2, "-")__\
        _(mul , 2, "*")__\
        _(div , 2, "/")__\
        _(matmul, 2, "matmul")__

    #define _(mnemonic, operands, name) mnemonic
    enum opcode : std::uint32_t {
        rtml_opcode_def(_, rtml_co)
        $count
    };
    #undef _

    #define _(mnemonic, operands, name) ((operands)&0xff)
    constexpr std::array<std::uint32_t, $count> k_operands {
        rtml_opcode_def(_, rtml_co)
    };
    #undef _

    #define _(mnemonic, operands, name) name
    constexpr std::array<std::string_view, $count> k_names {
        rtml_opcode_def(_, rtml_co)
    };
    #undef _

    template <typename S> requires is_dtype<S>
    using validate_function = auto (const tensor<S>* dst, std::span<const tensor<S>*> src) -> bool;
    template <typename S> requires is_dtype<S>
    using eval_function = auto (tensor<S>* dst, std::span<const tensor<S>*> src) noexcept -> void;

    enum class graph_eval_order : bool {
        left_to_right,
        right_to_left
    };

    template <const graph_eval_order order, typename S, typename F, typename... Args>
        requires is_dtype<S> && std::is_invocable_r_v<void, const tensor<S>*, Args...>
    auto graph_visit(
        const tensor<S>* root,
        F&& callback,
        Args&&... args
    ) -> void {
        const auto& operands {root->operands()};
        for (std::size_t i {}; i < operands.size(); ++i) {
            std::size_t ii;
            if constexpr (order == graph_eval_order::left_to_right) ii = i;
            else ii = operands.size() - i - 1;
            graph_visit(operands[ii], order, std::forward<F>(callback), std::forward<Args>(args)...);
        }
        std::invoke(callback, root, std::forward<Args>(args)...);
    }

    // all validation functions go here
    namespace validators {
        template <typename S> requires is_dtype<S>
        [[nodiscard]] constexpr auto validate_unary_op(
            const tensor<S>* const dst,
            const std::span<const tensor<S>*> src
        ) -> bool {
            const tensor<S>* const r {dst};
            if (!r) [[unlikely]] {
                return false;
            }
            if (const auto num {static_cast<std::size_t>(r->opcode())}; k_operands[num] != src.size() && src.size() == 1) [[unlikely]] {
                rtml_log_error("Operand count mismatch, expected {}, got {}", num, src.size());
                return false;
            }
            const tensor<S>* const x {src[0]};
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

        template <typename S> requires is_dtype<S>
        [[nodiscard]] constexpr auto validate_binary_op(
            const tensor<S>* const dst,
            const std::span<const tensor<S>*> src
        ) -> bool {
            const tensor<S>* const r {dst};
            if (!r) [[unlikely]] {
                return false;
            }
            if (const auto num {static_cast<std::size_t>(r->opcode())}; k_operands[num] != src.size() && src.size() == 2) [[unlikely]] {
                rtml_log_error("Operand count mismatch, expected {}, got {}", num, src.size());
                return false;
            }
            const tensor<S>* const x {src[0]};
            const tensor<S>* const y {src[1]};
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
    }

    namespace evaluators {
        template <typename S> requires is_dtype<S>
        constexpr auto evaluate_unary_op(
            const tensor<S>* const dst,
            const std::span<const tensor<S>*> src
        ) -> void {

        }
        template <typename S> requires is_dtype<S>
        constexpr auto evaluate_binary_op(
            const tensor<S>* const dst,
            const std::span<const tensor<S>*> src
        ) -> void {

        }
    }

    template <typename S> requires is_dtype<S>;
    struct routines;

    template <>
    struct routines final {
        static constexpr std::array<validate_function<dtypes::f32>*, $count> validators {
            [] consteval { // Autogenerate table of validation functions for unary and binary ops
                std::array<validate_function<dtypes::f32>*, $count> result {};
                for (std::size_t i {}; i < $count; ++i) {
                    if (k_operands[i] == 1) { // unary op
                        result[i] = &validators::validate_unary_op<dtypes::f32>;
                    } else { // binary op
                        result[i] = &validators::validate_binary_op<dtypes::f32>;
                    }
                }
                return result;
            }()
        };
        static constexpr std::array<eval_function<dtypes::f32>*, $count> evaluators {

        };
    };
}
