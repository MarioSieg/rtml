// Copyright Mario "Neo" Sieg 2024. All rights reserved. mario.sieg.64@gmail.com

#pragma once

#include <array>
#include <cstdint>
#include <cstddef>
#include <mutex>
#include <span>

namespace rtml {
    class isolate;

    class tensor final {
    public:
        using id = std::uint32_t;
        static constexpr std::int64_t k_max_dims = 4;
        static constexpr std::size_t k_max_name = 128;

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

        [[nodiscard]] auto get_style() const noexcept -> dtype { return m_dtype; }
        [[nodiscard]] auto get_id() const noexcept -> id { return m_id; }
        [[nodiscard]] auto get_size() const noexcept -> std::size_t { return m_size; }
        [[nodiscard]] auto get_dims() const noexcept -> std::span<const std::int64_t, k_max_dims> { return m_dims; }
        [[nodiscard]] auto get_strides() const noexcept -> std::span<const std::int64_t, k_max_dims> { return m_strides; }
        [[nodiscard]] auto get_slice() const noexcept -> tensor* { return m_slice; }
        [[nodiscard]] auto get_slice_offset() const noexcept -> std::size_t { return m_slice_offset; }
        [[nodiscard]] auto get_data() const noexcept -> void* { return m_s; }
        [[nodiscard]] auto get_name() const noexcept -> const char* { return m_name.data(); }
        auto set_name(const char* name) -> void { std::strncpy(m_name.data(), name, k_max_name); }
        [[nodiscard]] auto print() -> std::string;

    private:
        friend class isolate;
        friend class pool;
        tensor(
            isolate& ctx,
            std::uint32_t id,
            dtype type,
            std::span<const std::int64_t> dims,
            tensor* slice,
            std::size_t slice_offset
        ) noexcept;

        isolate& m_ctx;
        const std::uint32_t m_id;
        std::array<char, k_max_name> m_name {}; // Tensor name - cannot use std::string because we must be trivially destructable
        const dtype m_dtype; // Tensor scalar data type
        std::size_t m_size {}; // Tensor data size in bytes
        std::uint32_t m_num_dims {}; // Number of dimensions (1-k_max_dims)
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
