// Copyright Mario "Neo" Sieg 2024. All rights reserved. mario.sieg.64@gmail.com

#pragma once

#include <array>
#include <cstdint>
#include <cstddef>
#include <mutex>
#include <span>

namespace rtml {
    class context;

    class tensor final {
    public:
        using id = std::uint32_t;
        static constexpr std::int64_t k_max_dims = 4;

        enum class dtype : std::uint32_t {
            f32 = 0,
            $count
        };

        struct dtype_trait final {
            std::size_t size;
            std::size_t align;
        };

        static constexpr std::array<dtype_trait, static_cast<std::size_t>(dtype::$count)> k_stype_traits {
            { sizeof(float), alignof(float) }
        };

        tensor(const tensor&) = delete;
        tensor(tensor&&) = delete;
        auto operator=(const tensor&) -> tensor& = delete;
        auto operator=(tensor&&) -> tensor& = delete;
        ~tensor() = default;

    private:
        friend class context;
        friend class pool;
        tensor(
            context& ctx,
            std::uint32_t id,
            dtype type,
            std::span<const std::int64_t> dims,
            tensor* slice,
            std::size_t slice_offset
        ) noexcept;

        context& m_ctx;
        const std::uint32_t m_id;
        const dtype m_stype; // Tensor scalar data type
        std::size_t m_size {}; // Tensor data size in bytes
        std::array<std::int64_t, k_max_dims> m_dims {};
        std::array<std::int64_t, k_max_dims> m_strides {};
        tensor* m_slice {};
        std::size_t m_slice_offset {};
        union {
            float* m_s {};
            std::uint8_t* m_u8;
        };
    };
}
