// Copyright Mario "Neo" Sieg 2024. All rights reserved. mario.sieg.64@gmail.com

#pragma once

#include "base.hpp"
#include "fixed_vector.hpp"

namespace rtml {
    namespace dtypes {
        using f32 = float;
    }

    template <typename S>
    concept is_dtype = requires {
        !std::is_pointer_v<S>;
        !std::is_reference_v<S>;
        std::is_same_v<S, dtypes::f32>;
    };

    using dim = std::int64_t; // Dimension scalar used for dims, indices and strides.

    template <typename T> requires is_dtype<T>
    struct dtype_traits;

    template <>
    struct dtype_traits<float> {
        using type = float;
        static constexpr std::size_t k_size {sizeof(float)};
        static constexpr std::size_t k_align {alignof(float)};
        static constexpr std::string_view k_name {"f32"};
        static constexpr float k_one {1.0f};
    };

    template <typename T> requires is_dtype<T>
    class tensor;

    static constexpr dim k_max_dims {4};
    static constexpr std::size_t k_max_operands {2};
    static constexpr std::size_t k_max_name {128};

    class tensor_base {
    public:
        tensor_base() = default;
        template <typename... Ops>
           requires (sizeof...(Ops) > 0) && (sizeof...(Ops) <= k_max_operands)
               && (std::is_pointer_v<std::remove_reference_t<Ops>> && ...)
        auto op(const enum opcode opc, Ops&&... ops) -> void {
            m_op = opc;
            for (auto&& op : std::initializer_list<std::common_type_t<Ops...>>{ops...})
                m_operands.emplace_back(dynamic_cast<const tensor_base*>(op));
        }
        [[nodiscard]] auto op_code() const noexcept -> opcode { return m_op; }
        [[nodiscard]] auto raw_operands() const noexcept -> const fixed_vector<const tensor_base*, k_max_operands>& { return m_operands; }

    protected:
        opcode m_op {opcode::nop}; // Operation code
        fixed_vector<const tensor_base*, k_max_operands> m_operands {}; // Tensor operation operands
    };
}
