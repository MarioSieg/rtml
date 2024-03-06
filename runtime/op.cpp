// Copyright Mario "Neo" Sieg 2024. All rights reserved. mario.sieg.64@gmail.com

#include "op.hpp"

namespace rtml::op {
    // Note we use C-style arrays instead of std::array to use array designators

    static constexpr validate_function* const k_validators[] {
        [nop] = [](const tensor* const dst, const std::span<const tensor*> src) -> bool {
            return true;
        },
        [softmax] = [](const tensor* const dst, const std::span<const tensor*> src) -> bool {
            return true;
        },
        [sigmoid] = [](const tensor* const dst, const std::span<const tensor*> src) -> bool {
            return true;
        },
        [tanh] = [](const tensor* const dst, const std::span<const tensor*> src) -> bool {
            return true;
        },
        [relu] = [](const tensor* const dst, const std::span<const tensor*> src) -> bool {
            return true;
        },
        [gelu] = [](const tensor* const dst, const std::span<const tensor*> src) -> bool {
            return true;
        },
        [add] = [](const tensor* const dst, const std::span<const tensor*> src) -> bool {
            return true;
        },
        [sub] = [](const tensor* const dst, const std::span<const tensor*> src) -> bool {
            return true;
        },
        [mul] = [](const tensor* const dst, const std::span<const tensor*> src) -> bool {
            return true;
        },
        [div] = [](const tensor* const dst, const std::span<const tensor*> src) -> bool {
            return true;
        },
        [matmul] = [](const tensor* const dst, const std::span<const tensor*> src) -> bool {
            return true;
        },
    };
    static_assert(std::size(k_validators) == $count);

    static constexpr eval_function* const k_evaluators[] {
        [nop] = [](tensor* const dst, const std::span<const tensor*> src) noexcept -> void {
        },
        [softmax] = [](tensor* const dst, const std::span<const tensor*> src) noexcept -> void {
        },
        [sigmoid] = [](tensor* const dst, const std::span<const tensor*> src) noexcept -> void {
        },
        [tanh] = [](tensor* const dst, const std::span<const tensor*> src) noexcept -> void {
        },
        [relu] = [](tensor* const dst, const std::span<const tensor*> src) noexcept -> void {
        },
        [gelu] = [](tensor* const dst, const std::span<const tensor*> src) noexcept -> void {
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
}
