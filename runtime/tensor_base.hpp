// Copyright Mario "Neo" Sieg 2024. All rights reserved. mario.sieg.64@gmail.com

#pragma once

#include <type_traits>
#include <string_view>

namespace rtml {
    namespace dtypes {
        using f32 = float;
    }

    template <typename T>
    concept is_dtype = requires {
        requires std::is_same_v<T, dtypes::f32>;
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
}
