// Copyright Mario "Neo" Sieg 2024. All rights reserved. mario.sieg.64@gmail.com

#pragma once

#include <array>
#include <cstdint>
#include <cstddef>
#include <mutex>
#include <span>

#include "fixed_vector.hpp"
#include "graph.hpp"

#include <spdlog/spdlog.h>

namespace rtml {
    class isolate;

    using dim = std::int64_t; // Dimension scalar used for dims, indices and strides.

    template <typename T>
    concept is_dtype = requires {
        requires std::is_same_v<T, float>;
    };

    // Represents an N-dimensional (1-k_max_dims) tensor, which is also a vertex in the computation DAG.
    // The dimensionality and data type of a tensor are dynamically handled at runtime.
    class tensor final {
    public:
        static constexpr dim k_max_dims {4};
        static constexpr std::size_t k_max_operands {2};
        static constexpr std::size_t k_max_name {128};

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

        [[nodiscard]] auto datatype() const noexcept -> dtype { return m_dtype; }
        [[nodiscard]] auto datatype_traits() const noexcept -> const dtype_trait& { return k_stype_traits[static_cast<std::size_t>(m_dtype)]; }
        [[nodiscard]] auto size() const noexcept -> std::size_t { return m_datasize; }
        [[nodiscard]] auto dim_count() const noexcept -> std::uint32_t { return m_num_dims; }
        [[nodiscard]] auto dims() const noexcept -> const std::array<dim, k_max_dims>& { return m_shape; }
        [[nodiscard]] auto used_dims() const noexcept -> std::span<const dim> { return {m_shape.cbegin(), m_num_dims}; }
        [[nodiscard]] auto strides() const noexcept -> const std::array<dim, k_max_dims>& { return m_strides; }
        [[nodiscard]] auto slice_base() const noexcept -> tensor* { return m_slice; }
        [[nodiscard]] auto slice_offset() const noexcept -> std::size_t { return m_slice_offset; }
        [[nodiscard]] auto operands() noexcept -> fixed_vector<const tensor*, k_max_operands>& { return m_operands; }
        [[nodiscard]] auto operands() const noexcept -> const fixed_vector<const tensor*, k_max_operands>& { return m_operands; }
        [[nodiscard]] auto ptr() const noexcept -> std::uint8_t* { return m_x.u8; }
        template <typename T = float> requires is_dtype<T>
        [[nodiscard]] auto data() const noexcept -> std::span<T> { return {reinterpret_cast<T*>(m_x.u8), m_datasize / sizeof(T)}; }
        [[nodiscard]] auto name() const noexcept -> const char* { return m_name.data(); }
        [[nodiscard]] auto opcode() const noexcept -> graph::opcode { return m_op; }
        [[nodiscard]] auto is_dense() const noexcept -> bool;
        [[nodiscard]] auto is_dense_except_dim1() const noexcept -> bool;
        [[nodiscard]] auto is_shape_eq(const tensor* other) const noexcept -> bool;
        [[nodiscard]] auto is_matmul_compatible(const tensor* other) const noexcept -> bool;
        [[nodiscard]] auto is_transposed() const noexcept -> bool;
        [[nodiscard]] auto is_permuted() const noexcept -> bool;
        [[nodiscard]] auto can_repeat(const tensor* other) const noexcept -> bool;
        [[nodiscard]] auto row_count() const noexcept -> dim;
        [[nodiscard]] auto col_count() const noexcept -> dim;
        [[nodiscard]] auto elem_count() const noexcept -> dim;
        [[nodiscard]] auto unroll_index(dim i) const noexcept -> std::array<dim, k_max_dims>;

        [[nodiscard]] auto isomorphic_clone() noexcept -> tensor*;
        [[nodiscard]] auto sliced_clone() noexcept -> tensor*;
        [[nodiscard]] auto clone() noexcept -> tensor*;

        auto splat_zero() const -> void;
        auto splat_one() const -> void;
        auto splat(float x) const -> void;
        auto push_operand(const tensor* x) -> void;

        [[nodiscard]] auto operator()(const std::array<dim, k_max_dims>& indices) const noexcept -> float&;
        [[nodiscard]] auto operator()(dim i) const noexcept -> float&;

        auto set_name(const char* name) -> void;
        template<typename... Args>
        auto format_name(const fmt::format_string<Args...>& fmt, Args&&... args) -> void {
            const std::string formatted {fmt::format(fmt, std::forward<Args>(args)...)}; // TODO: avoid clone
            set_name(formatted.c_str());
        }
        [[nodiscard]] auto to_string(std::size_t with_data_elems = 0) const -> std::string;
        auto print(std::size_t with_data_elems = 0) const -> void;

    private:
        friend class pool;

        tensor(
             isolate& ctx,
             dtype type,
             std::span<const dim> dims,
             tensor* slice,
             std::size_t slice_offset
        ) noexcept;

        isolate& m_ctx; // Associated isolate
        std::array<char, k_max_name> m_name {}; // Tensor name - cannot use std::string because we must be trivially destructable
        const dtype m_dtype; // Tensor scalar data type
        std::size_t m_datasize {}; // Tensor data size in bytes
        std::uint32_t m_num_dims {}; // Number of dimensions (1-k_max_dims)
        graph::opcode m_op {graph::nop}; // Operation code
        std::array<dim, k_max_dims> m_shape {}; // 4D dimensions - tensor shape
        std::array<dim, k_max_dims> m_strides {}; // 4D byte strides
        fixed_vector<const tensor*, k_max_operands> m_operands {}; // Tensor operation operands
        tensor* m_slice {}; // Sliced base tensor, if any
        std::size_t m_slice_offset {}; // Memory offset into sliced base tensor's data
        union {
            float* f32;
            std::uint8_t* u8 {};
        } m_x {};
    };
}
