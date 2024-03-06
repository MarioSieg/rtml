// Copyright Mario "Neo" Sieg 2024. All rights reserved. mario.sieg.64@gmail.com

#include "op.hpp"
#include "tensor.hpp"

namespace rtml::op {
    // Note we use C-style arrays instead of std::array to use array designators

    [[nodiscard]] constexpr auto validate_base(const tensor* const dst, const std::span<const tensor*> src) -> bool {
        return k_operands[static_cast<std::size_t>(dst->get_op())] == src.size();
    }

    static constexpr validate_function* const k_validators[] {
        [nop] = [](const tensor* const dst, const std::span<const tensor*> src) -> bool {
            return validate_base(dst, src);
        },
        [softmax] = [](const tensor* const dst, const std::span<const tensor*> src) -> bool {
            return validate_base(dst, src);
        },
        [sigmoid] = [](const tensor* const dst, const std::span<const tensor*> src) -> bool {
            return validate_base(dst, src);
        },
        [tanh] = [](const tensor* const dst, const std::span<const tensor*> src) -> bool {
            return validate_base(dst, src);
        },
        [relu] = [](const tensor* const dst, const std::span<const tensor*> src) -> bool {
            return validate_base(dst, src);
        },
        [gelu] = [](const tensor* const dst, const std::span<const tensor*> src) -> bool {
            return validate_base(dst, src);
        },
        [add] = [](const tensor* const dst, const std::span<const tensor*> src) -> bool {
            return validate_base(dst, src);
        },
        [sub] = [](const tensor* const dst, const std::span<const tensor*> src) -> bool {
            return validate_base(dst, src);
        },
        [mul] = [](const tensor* const dst, const std::span<const tensor*> src) -> bool {
            return validate_base(dst, src);
        },
        [div] = [](const tensor* const dst, const std::span<const tensor*> src) -> bool {
            return validate_base(dst, src);
        },
        [matmul] = [](const tensor* const dst, const std::span<const tensor*> src) -> bool {
            return validate_base(dst, src);
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
