// Copyright Mario "Neo" Sieg 2024. All rights reserved. mario.sieg.64@gmail.com

#pragma once

#include <array>
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <type_traits>

namespace rtml {
    static_assert(std::numeric_limits<float>::is_iec559);
    static_assert(std::numeric_limits<double>::is_iec559);

    class tensor;

    template <typename T>
    concept is_pool_allocateable = requires {
        requires std::is_trivially_destructible_v<T>;
    };

    class pool final {
    public:
        explicit pool(std::size_t size);
        pool(const pool&) = delete;
        pool(pool&&) = delete;
        auto operator=(const pool&) -> pool& = delete;
        auto operator=(pool&&) -> pool& = delete;
        ~pool() = default;

        [[nodiscard]] auto alloc_raw(std::size_t size) noexcept -> void*;
        [[nodiscard]] auto alloc_raw(std::size_t size, std::size_t align) noexcept -> void*;
        template <typename T, typename... Args> requires
            is_pool_allocateable<T> && std::is_constructible_v<T, Args...>
        [[nodiscard]] auto alloc(Args&&... args) noexcept(std::is_nothrow_constructible_v<T, Args...>) -> T* {
            T* p {static_cast<T*>(alloc_raw(sizeof(T), alignof(T)))};
            return new (p) T{std::forward<Args>(args)...};
        }
        auto print_info() const -> void;

    private:
        const std::size_t m_size;
        const std::unique_ptr<std::uint8_t[]> m_storage;
        std::uint8_t* m_top {};
        std::uint8_t* m_bot {};
        std::size_t m_num_allocs {};
    };

    class tensor final {
    public:
        static constexpr std::int64_t k_max_dims = 4;
        static constexpr std::int64_t k_max_elems_per_dim = 1ull<<53; // 2^53

        enum class stype {
            f32,
            $count
        };

        struct stype_trait final {
            std::size_t size;
            std::size_t align;
        };

        static constexpr std::array<stype_trait, static_cast<std::size_t>(stype::$count)> k_stype_traits {
            { sizeof(float), alignof(float) }
        };

        tensor(const tensor&) = delete;
        tensor(tensor&&) = delete;
        auto operator=(const tensor&) -> tensor& = delete;
        auto operator=(tensor&&) -> tensor& = delete;
        ~tensor() = default;

        [[nodiscard]] static auto create(
            pool& pool,
            stype type,
            std::initializer_list<const std::int64_t> dims,
            tensor* slice = nullptr,
            std::size_t slice_offset = 0
        ) -> tensor*;

    private:
        tensor(
            pool& pool,
            stype type,
            std::initializer_list<const std::int64_t> dims,
            tensor* slice,
            std::size_t slice_offset
        ) noexcept;

        pool* const m_pool;
        const stype m_stype; // Tensor scalar data type
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
