// Copyright Mario "Neo" Sieg 2024. All rights reserved. mario.sieg.64@gmail.com

#pragma once

#include <array>
#include <cstdint>
#include <cstddef>
#include <iostream>
#include <ranges>
#include <random>
#include <span>

#include "fixed_vector.hpp"
#include "tensor_base.hpp"

#include <spdlog/spdlog.h>

namespace rtml {
    enum class opcode : std::uint32_t;

    template <typename T, typename... Args>
    [[nodiscard]] constexpr auto tuple_to_array(std::tuple<Args...>&& tup) -> std::array<T, sizeof...(Args)> {
        std::array<T, sizeof...(Args)> result {};
        std::apply([&result, i=0](auto&&... v) mutable {((result[i++] = v), ...);}, tup);
        return result;
    }

    constexpr bool k_clone_set_name {true}; // If true, some operations like clone or slice add this to the tensor's name.

    // Represents an N-dimensional (1-k_max_dims) tensor, which is also a vertex in the computation DAG.
    // The dimensionality and data type of a tensor are dynamically handled at runtime.
    template <typename S = dtypes::f32> requires is_dtype<S>
    class tensor final {
    public:
        using dtype = S;
        static constexpr dim k_max_dims {4};
        static constexpr std::size_t k_max_operands {2};
        static constexpr std::size_t k_max_name {128};

        tensor(const tensor&) = delete;
        tensor(tensor&&) = delete;
        auto operator=(const tensor&) -> tensor& = delete;
        auto operator=(tensor&&) -> tensor& = delete;
        ~tensor() = default;

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
        [[nodiscard]] auto data() const noexcept -> std::span<S> { return {reinterpret_cast<S*>(m_x.u8), m_datasize / dtype_traits<S>::k_size}; }
        [[nodiscard]] auto name() const noexcept -> const char* { return m_name.data(); }
        [[nodiscard]] auto opcode() const noexcept -> opcode { return m_op; }
        [[nodiscard]] auto is_dense() const noexcept -> bool {
            static_assert(k_max_dims == 4);
            return
                m_strides[0] == dtype_traits<S>::k_size && // Check if strides are contiguous
                m_strides[1] == m_strides[0] * m_shape[0] &&
                m_strides[2] == m_strides[1] * m_shape[1] &&
                m_strides[3] == m_strides[2] * m_shape[2];
        }
        [[nodiscard]] auto is_dense_except_dim1() const noexcept -> bool {
            return
            m_strides[0] == dtype_traits<S>::k_size && // Check if strides are contiguous
            m_strides[2] == m_strides[1] * m_shape[1] &&
            m_strides[3] == m_strides[2] * m_shape[2];
        }
        [[nodiscard]] auto is_shape_eq(const tensor* const other) const noexcept -> bool {
            if (this == other) [[unlikely]]
                return true;
            if (m_num_dims == other->m_num_dims) [[likely]]
                return std::equal(m_shape.cbegin(), m_shape.cbegin()+m_num_dims, other->m_shape.cbegin());
            return false;
        }
        [[nodiscard]] auto is_matmul_compatible(const tensor* const other) const noexcept -> bool {
            static_assert(k_max_dims == 4);
            return m_shape[0] == other->m_shape[0] && // Check if matrix multiplication is compatible
                other->m_shape[2] % m_shape[2] == 0 &&
                other->m_shape[3] % m_shape[3] == 0;
        }
        [[nodiscard]] auto is_transposed() const noexcept -> bool {
            static_assert(k_max_dims == 4);
            return m_strides[0] > m_strides[1];
        }
        [[nodiscard]] auto is_permuted() const noexcept -> bool {
            static_assert(k_max_dims == 4);
            return
                m_strides[0] > m_strides[1] ||
                m_strides[1] > m_strides[2] ||
                m_strides[2] > m_strides[3];
        }
        [[nodiscard]] auto can_repeat(const tensor* const other) const noexcept -> bool {
            static_assert(k_max_dims == 4);
            return
                other->m_shape[0] % m_shape[0] == 0 && // Check if other's dimensions are divisible by this tensor's dimensions
                other->m_shape[1] % m_shape[1] == 0 &&
                other->m_shape[2] % m_shape[2] == 0 &&
                other->m_shape[3] % m_shape[3] == 0;
        }
        [[nodiscard]] auto row_count() const noexcept -> dim {
            static_assert(k_max_dims == 4);
            return m_shape[1] * m_shape[2] * m_shape[3];
        }
        [[nodiscard]] auto col_count() const noexcept -> dim { return m_shape[0]; }
        [[nodiscard]] auto elem_count() const noexcept -> dim {
            static_assert(k_max_dims == 4);
            return m_shape[0] * m_shape[1] * m_shape[2] * m_shape[3];
        }
        [[nodiscard]] auto unroll_index(const dim i) const noexcept -> std::array<dim, k_max_dims> {
            static_assert(k_max_dims == 4);
            const dim d0 {m_shape[0]};
            const dim d1 {m_shape[1]};
            const dim d2 {m_shape[2]};
            const dim lambda {i / (d2*d1*d0)}; // Unroll index into dimensions
            const dim zeta {(i - lambda*d2*d1*d0) / (d1*d0)};
            const dim eta {(i - lambda*d2*d1*d0 - zeta*d1*d0) / d0};
            const dim xi {i - lambda*d2*d1*d0 - zeta*d1*d0 - eta * d0};
            return {
                xi,
                zeta,
                eta,
                lambda
            };
        }

        [[nodiscard]] auto isomorphic_clone() noexcept -> tensor* {
            auto* const ts {m_ctx.new_tensor<S>(used_dims())};
            if constexpr (k_clone_set_name)
                ts->format_name("{} (isomorph)", m_name.data());
            return ts;
        }
        [[nodiscard]] auto sliced_clone() noexcept -> tensor* {
            auto* const ts {m_ctx.new_tensor<S>(
              used_dims(),
              this,
              0
            )};
            std::ranges::copy(m_strides.cbegin(), m_strides.cend(), ts->m_strides.begin());
            if constexpr (k_clone_set_name)
                ts->format_name("{} (slice)", m_name.data());
            return ts;
        }
        [[nodiscard]] auto clone() noexcept -> tensor* {
            auto* const ts {m_ctx.new_tensor(used_dims())};
            std::ranges::copy(data(), ts->data().begin());
            if constexpr (k_clone_set_name)
                ts->format_name("{} (clone)", m_name.data());
            return ts;
        }
        auto fill_zero() -> tensor* {
            std::memset(m_x.u8, 0, m_datasize);
            return this;
        }
        auto fill_random(const S min = -dtype_traits<S>::k_one, const S max = dtype_traits<S>::k_one) -> tensor* {
            // todo: use tausworthe prng
            std::mt19937_64 rng {std::random_device{}()};
            std::uniform_real_distribution<S> dist {min, max};
            const std::span<S> ref {data()};
            std::generate(ref.begin(), ref.end(), [&dist, &rng] () noexcept -> S { return dist(rng); });
            return this;
        }
        auto fill_one() -> tensor* {
            fill(1.0f);
            return this;
        }
        auto fill(const S x) -> tensor* {
            std::ranges::fill(data(), x);
            return this;
        }
        auto fill_data(const std::span<const S> data) -> tensor* {
            rtml_dassert1(data.size() == elem_count());
            std::ranges::copy(data, this->data().begin());
            return this;
        }
        template <typename... Ops>
            requires (sizeof...(Ops) <= k_max_operands)
                && ((sizeof...(Ops) > 0) || (std::is_same_v<std::remove_cvref_t<Ops>, tensor*> && ...))
        auto op(const enum opcode opc, Ops&&... ops) noexcept -> tensor* {
            auto emit_op {[=](tensor& dst, auto&&... g_ops) -> tensor* {
                dst.m_op = opc;
                for (auto* const op : {g_ops...})
                    dst.m_operands.emplace_back(op);
                return &dst;
            }};
            return emit_op(*isomorphic_clone(), this, ops...);
        }

        [[nodiscard]] auto operator()(const std::array<dim, k_max_dims>& indices) const noexcept -> S& {
            return *reinterpret_cast<S*>(
                m_x.u8 +
                indices[0]*m_strides[0] +
                indices[1]*m_strides[1] +
                indices[2]*m_strides[2] +
                indices[3]*m_strides[3]
            );
        }
        [[nodiscard]] auto operator()(const dim i) const noexcept -> S& {
            if (is_dense()) return reinterpret_cast<S&>(m_x.u8[i*dtype_traits<S>::k_size]);
            return (*this)(unroll_index(i));
        }

        auto RTML_COLD set_name(const char* name) -> tensor* {
            std::strncpy(m_name.data(), name, k_max_name);
            m_name[k_max_name-1] = '\0';
            return this;
        }
        auto RTML_COLD set_name(const std::string& name) -> tensor* {
            std::strncpy(m_name.data(), name.c_str(), k_max_name);
            m_name[k_max_name-1] = '\0';
            return this;
        }
        template<typename... Args>
        auto RTML_COLD format_name(const fmt::format_string<Args...>& fmt, Args&&... args) -> void {
            const std::string formatted {fmt::format(fmt, std::forward<Args>(args)...)}; // TODO: avoid clone
            set_name(formatted.c_str());
        }
        [[nodiscard]] auto RTML_COLD to_string(const std::size_t with_data_elems = 0) const -> std::string {
            static_assert(k_max_dims == 4);
            const std::size_t total_size = m_datasize+sizeof(*this);
            auto size {static_cast<double>(total_size)};
            std::string_view unit {"B"};
            if (total_size > 1<<30) {
                size /= static_cast<double>(1<<30);
                unit = "GiB";
            } else if (total_size > 1<<20) {
                size /= static_cast<double>(1<<20);
                unit = "MiB";
            } else if (total_size > 1<<10) {
                size /= static_cast<double>(1<<10);
                unit = "KiB";
            }
            std::string fmt {};
            fmt.reserve(0x100+sizeof("2.000")*with_data_elems);
            fmt += fmt::format(
                "Tensor {}{}{} * {}D, Shape [{} X {} X {} X {}], Strides [{}B X {}B X {}B X {}B] {:.01f}{}",
                m_name.data(),
                m_name[0] ? ": " : "",
                dtype_traits<S>::k_name,
                m_num_dims,
                m_shape[0],
                m_shape[1],
                m_shape[2],
                m_shape[3],
                m_strides[0],
                m_strides[1],
                m_strides[2],
                m_strides[3],
                size,
                unit
            );
            if (with_data_elems > 0) {
                fmt += "\n[\n";
                for (dim i3 {}; i3 < m_shape[2]; ++i3) {
                    for (dim i2 {}; i2 < m_shape[1]; ++i2) {
                        fmt.push_back('\t');
                        for (dim i1 {}; i1 < m_shape[0]; ++i1) {
                            const S x {reinterpret_cast<S&>(m_x.u8[dtype_traits<S>::k_size*(i3*m_shape[1]*m_shape[0] + i2*m_shape[0] + i1)])};
                            fmt += fmt::format("{:.03f} ", x);
                        }
                        fmt.push_back('\n');
                    }
                }
                fmt += "\t...\n]";
            }
            return fmt;
        }
        auto RTML_COLD print(const std::size_t with_data_elems = std::numeric_limits<std::size_t>::max()) const -> void {
            std::cout << to_string(with_data_elems) << std::endl;
        }

    private:
        friend class pool;

        tensor(
             isolate& ctx,
             std::span<const dim> dims,
             tensor* slice,
             std::size_t slice_offset
        ) noexcept : m_ctx{ctx} {
            assert(!dims.empty() && dims.size() <= k_max_dims);
            if (slice && slice->m_slice) { // Account for if slice itself is also a slice
                slice_offset += slice->m_slice_offset;
                slice = slice->m_slice;
            }
            std::size_t datasize {dtype_traits<S>::k_size}; // Tensor data size to allocate
            for (std::size_t i {0}; i < dims.size(); ++i) { // Accumulate total data size
                assert(dims[i] > 0); // Check if dimensions are valid
                datasize *= dims[i];
            }
            assert(!slice || datasize+slice_offset <= slice->m_datasize); // Check if slice has enough space
            static constexpr bool k_align_scalar = false; // Aligned data address to scalar alignment?
            m_x.u8 = slice
                ? slice->m_x.u8+slice_offset
                : static_cast<std::uint8_t*>(k_align_scalar
                    ? m_ctx.pool().alloc_raw(datasize, dtype_traits<S>::k_align)
                    : m_ctx.pool().alloc_raw(datasize)); // Point into slice data (if slice) or allocate new data from pool.
            m_datasize = datasize;
            m_num_dims = dims.size();
            m_slice = slice;
            m_slice_offset = slice_offset;
            std::ranges::fill(m_shape.begin(), m_shape.end(), 1); // Splat identity dimensions
            std::ranges::copy(dims.begin(), dims.end(), m_shape.begin()); // Copy dimensions
            m_strides[0] = static_cast<dim>(dtype_traits<S>::k_size);
            for (std::size_t i {1}; i < k_max_dims; ++i) // Compute strides
                m_strides[i] = m_strides[i-1]*m_shape[i-1];
        }

        isolate& m_ctx; // Associated isolate
        std::array<char, k_max_name> m_name {}; // Tensor name - cannot use std::string because we must be trivially destructable
        std::size_t m_datasize {}; // Tensor data size in bytes
        std::uint32_t m_num_dims {}; // Number of dimensions (1-k_max_dims)
        enum opcode m_op {opcode::nop}; // Operation code
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

    template <typename S = dtypes::f32> requires is_dtype<S>
    class tensor_ref final {
    public:
        constexpr tensor_ref() noexcept = default;
        constexpr tensor_ref(tensor<S>* const t) noexcept : m_t{t} {}
        auto operator * () const noexcept -> tensor<S>& { return *m_t; }
        auto operator -> () const noexcept -> tensor<S>* { return m_t; }
        auto operator + (const tensor_ref& other) const noexcept -> tensor_ref {
            return m_t->op(opcode::add, other.m_t);
        }
        auto operator - (const tensor_ref& other) const noexcept -> tensor_ref {
            return m_t->op(opcode::sub, other.m_t);
        }
        auto operator * (const tensor_ref& other) const noexcept -> tensor_ref {
            return m_t->op(opcode::mul, other.m_t);
        }
        auto operator / (const tensor_ref& other) const noexcept -> tensor_ref {
            return m_t->op(opcode::div, other.m_t);
        }
        auto operator & (const tensor_ref& other) const noexcept -> tensor_ref {
            return m_t->op(opcode::matmul, other.m_t);
        }
        auto softmax() const noexcept -> tensor_ref {
            return m_t->op(opcode::softmax);
        }
        auto sigmoid() const noexcept -> tensor_ref {
            return m_t->op(opcode::sigmoid);
        }
        auto tanh() const noexcept -> tensor_ref {
            return m_t->op(opcode::tanh);
        }
        auto relu() const noexcept -> tensor_ref {
            return m_t->op(opcode::relu);
        }
        auto gelu() const noexcept -> tensor_ref {
            return m_t->op(opcode::gelu);
        }
        auto silu() const noexcept -> tensor_ref {
            return m_t->op(opcode::silu);
        }


    private:
        tensor<S>* m_t {};
    };
}
