// Copyright Mario "Neo" Sieg 2024. All rights reserved. mario.sieg.64@gmail.com

#pragma once

#include <sstream>
#include <functional>
#include <span>
#include <unordered_set>
#include <fstream>

#include "base.hpp"
#include "blas.hpp"
#include "tensor.hpp"

namespace rtml::graph {
#if 0
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
            else ii = operands.size()-i-1;
            graph_visit<Ord>(operands[ii], std::forward<F>(callback), std::forward<Args>(args)...);
        }
        std::invoke(callback, root, std::forward<Args>(args)...);
    }

    // all validation functions go here
    namespace validators {
        #define rtml_verify(expr, msg, ...) \
            if (!(expr)) [[unlikely]] { \
                rtml_log_error("Graph validation failed: " #expr "\t<-\t" msg, ## __VA_ARGS__); \
                if (r) rtml_log_error("Result: {}", r->to_string(0)); \
                if (src.size() > 0 && src[0]) rtml_log_error("Source 1: {}", src[0]->to_string(0)); \
                if (src.size() > 1 && src[1]) rtml_log_error("Source 2: {}", src[1]->to_string(0)); \
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
            rtml_verify(src[0]->shape().is_dense_except_dim1(), "Source tensor is not dense except dim1");
            rtml_verify(r->shape().is_dense_except_dim1(), "Result tensor is not dense except dim1");
            rtml_verify(r->shape() == src[0]->shape(), "Result tensor shape mismatch");
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
            rtml_verify(src[0]->shape().strides()[0] == dtype_traits<S>::k_size, "Source tensor 0 '{}' stride mismatch", src[0]->name());
            rtml_verify(r->shape().strides()[0] == dtype_traits<S>::k_size, "Result tensor '{}' stride mismatch", r->name());
            rtml_verify(src[1]->shape().can_repeat(src[0]->shape()), "Source tensor 1 '{}' cannot repeat source tensor 0 '{}'", src[1]->name(), src[0]->name());
            rtml_verify(src[0]->shape() == r->shape(), "Source tensor 0 '{}' shape mismatch with result tensor '{}'", src[0]->name(), r->name());
            return true;
        }

        template <typename S> requires is_dtype<S>
        [[nodiscard]] constexpr auto validate_matmul(
            const tensor<S>* const r,
            const std::span<const tensor<S>* const> src
        ) -> bool {
            rtml_verify(r, "Result tensor is null");
            const auto num_operands {static_cast<std::size_t>(r->opcode())};
            rtml_verify(k_operands[num_operands] == src.size(), "Number of operands mismatch, expected {} got {}", k_operands[num_operands], src.size());
            rtml_verify(src[0], "Source tensor 0 is null");
            rtml_verify(src[1], "Source tensor 1 is null");
            rtml_verify(
                src[0]->shape().is_matmul_compatible(src[1]->shape()),
                "Source tensor 0 '{}' and source tensor 1 '{}' are not matmul compatible",
                src[0]->name(),
                src[1]->name()
            );
            rtml_verify(!src[0]->shape().is_transposed(), "Source tensor 0 '{}' is transposed", src[0]->name());
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
                    if (k_operands[i] == 1) // unary op
                        result[i] = &validators::validate_unary_op<dtypes::f32>;
                    else if (static_cast<opcode>(i) == opcode::matmul) // matmul
                        result[i] = &validators::validate_matmul<dtypes::f32>;
                    else // binary op
                        result[i] = &validators::validate_binary_op<dtypes::f32>;
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
    auto RTML_COLD generate_graphviz_dot_code(std::stringstream& ss, const tensor<S>* const root) -> void {
        ss << "digraph ComputationGraph {{\n";
        ss << "rankdir=LR;\n";
        std::unordered_set<const tensor<S>*> visited {};
        // ported from RTML v.1 which was written in C. TODO: port to C++
        graph_visit<graph_eval_order::left_to_right, const tensor<S>>(root, [&visited, &ss](const tensor<S>* const t) -> void {
            if (visited.contains(t)) return;
            visited.insert(t);
            const auto tensor_id {fmt::format("t_{:x}", std::bit_cast<std::uintptr_t>(t))};
            const char* const color {t->opcode() == opcode::nop ? "springgreen2" : "lightskyblue"};
            ss << fmt::format("{} [label=\"{}\", shape=box, style=\"rounded, filled\", color={}, fillcolor={}];\n", tensor_id, t->name(), color, color);
            if (t->opcode() != opcode::nop) {
                const auto op_id {fmt::format("op_{:x}", std::bit_cast<std::uintptr_t>(t))};
                ss << fmt::format("{} [label=\"{}\", shape=circle, style=filled, color=orchid1, fillcolor=orchid1];\n", op_id, k_op_names[static_cast<std::size_t>(t->opcode())]);
                for (std::size_t i {}; i < t->operands().size(); ++i) {
                    const auto input_id {fmt::format("t_{:x}", std::bit_cast<std::uintptr_t>(t->operands()[i]))};
                    ss << fmt::format("{} -> {} [arrowhead=vee];\n", input_id, op_id);
                }
                ss << fmt::format("{} -> {} [arrowhead=vee];\n", op_id, tensor_id);
            }
        });
        ss << "}}\n";
    }

    template <typename S> requires is_dtype<S>
    [[nodiscard]] auto RTML_COLD generate_graphviz_dot_code(const std::string& file_name, const tensor<S>* const root) -> bool {
        std::ofstream out {file_name};
        if (!out.is_open()) [[unlikely]] return false;
        std::stringstream ss {};
        graph::generate_graphviz_dot_code(ss, &*root);
        out << ss.str();
        out.close();
        return true;
    }

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
#endif
}
