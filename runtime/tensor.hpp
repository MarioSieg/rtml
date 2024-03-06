// Copyright Mario "Neo" Sieg 2024. All rights reserved. mario.sieg.64@gmail.com

#pragma once

#include <array>
#include <cstdint>
#include <cstddef>
#include <mutex>
#include <span>

#include "spdlog/fmt/bundled/format.h"

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
            std::string_view name {};
            std::size_t size {};
            std::size_t align {};
        };

        static constexpr std::array<dtype_trait, static_cast<std::size_t>(dtype::$count)> k_stype_traits {
            { "f32", sizeof(float), alignof(float) }
        };

        tensor(const tensor&) = delete;
        tensor(tensor&&) = delete;
        auto operator=(const tensor&) -> tensor& = delete;
        auto operator=(tensor&&) -> tensor& = delete;
        ~tensor() = default;

        [[nodiscard]] auto get_data_type() const noexcept -> dtype { return m_dtype; }
        [[nodiscard]] auto get_data_type_traits() const noexcept -> const dtype_trait& { return k_stype_traits[static_cast<std::size_t>(m_dtype)]; }
        [[nodiscard]] auto get_id() const noexcept -> id { return m_id; }
        [[nodiscard]] auto get_data_size() const noexcept -> std::size_t { return m_datasize; }
        [[nodiscard]] auto get_num_dims() const noexcept -> std::uint32_t { return m_num_dims; }
        [[nodiscard]] auto get_dims() const noexcept -> std::span<const std::int64_t, k_max_dims> { return m_dims; }
        [[nodiscard]] auto get_active_dims() const noexcept -> std::span<const std::int64_t> { return {m_dims.cbegin(), m_num_dims}; }
        [[nodiscard]] auto get_strides() const noexcept -> std::span<const std::int64_t, k_max_dims> { return m_strides; }
        [[nodiscard]] auto get_slice() const noexcept -> tensor* { return m_slice; }
        [[nodiscard]] auto get_slice_offset() const noexcept -> std::size_t { return m_slice_offset; }
        [[nodiscard]] auto get_data() const noexcept -> void* { return m_x.u8; }
        [[nodiscard]] auto get_name() const noexcept -> const char* { return m_name.data(); }
        [[nodiscard]] auto is_contiguous() const noexcept -> bool;
        [[nodiscard]] auto can_repeat(const tensor* other) const noexcept -> bool;
        [[nodiscard]] auto row_count() const noexcept -> std::int64_t;
        auto unroll_index(std::int64_t i, std::array<std::int64_t, k_max_dims>& dims) const noexcept -> void;

        [[nodiscard]] auto isomorph() noexcept -> tensor*;
        [[nodiscard]] auto clone() noexcept -> tensor*;

        auto fill_zero() const -> void;
        auto fill_ones() const -> void;

        auto set_name(const char* name) -> void;
        template<typename... Args>
        auto format_name(const fmt::format_string<Args...>& fmt, Args&&... args) -> void {
            const std::string formatted {fmt::format(fmt, std::forward<Args>(args)...)}; // TODO: avoid clone
            set_name(formatted.c_str());
        }
        [[nodiscard]] auto to_string() -> std::string;

        tensor( // Do NOT use this constructor directly, use isolate::create_tensor instead
          isolate& ctx,
          std::uint32_t id,
          dtype type,
          std::span<const std::int64_t> dims,
          tensor* slice,
          std::size_t slice_offset
        ) noexcept;

    private:
        isolate& m_ctx;
        const std::uint32_t m_id;
        std::array<char, k_max_name> m_name {}; // Tensor name - cannot use std::string because we must be trivially destructable
        const dtype m_dtype; // Tensor scalar data type
        std::size_t m_datasize {}; // Tensor data size in bytes
        std::uint32_t m_num_dims {}; // Number of dimensions (1-k_max_dims)
        std::array<std::int64_t, k_max_dims> m_dims {};
        std::array<std::int64_t, k_max_dims> m_strides {};
        tensor* m_slice {};
        std::size_t m_slice_offset {};
        union {
            float* f32;
            std::uint8_t* u8 {};
        } m_x {};
    };
}
