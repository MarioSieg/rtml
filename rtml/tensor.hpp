// Copyright Mario "Neo" Sieg 2024. All rights reserved. mario.sieg.64@gmail.com

#pragma once

#include <array>
#include <cstdint>
#include <cstddef>
#include <iostream>
#include <ranges>
#include <random>
#include <span>

#include "base.hpp"
#include "blas.hpp"

#include <spdlog/spdlog.h>

namespace rtml {
    enum class opcode : std::uint32_t;

    template <typename T, typename... Args>
    [[nodiscard]] constexpr auto tuple_to_array(std::tuple<Args...>&& tup) -> std::array<T, sizeof...(Args)> {
        std::array<T, sizeof...(Args)> result {};
        std::apply([&result, i=0](auto&&... v) mutable {((result[i++] = v), ...);}, tup);
        return result;
    }

    constexpr bool k_clone_set_name {false}; // If true, some operations like clone or slice add this to the tensor's name.

    constexpr dim k_max_dims {4};

    template <const std::size_t dtype_size, const dim lim = k_max_dims>
        requires (dtype_size > 0) && (dtype_size <= 4) && (lim > 0 && (lim&-128) == 0)
    class fixed_shape {
    public:
        constexpr explicit fixed_shape(std::span<const dim> shape) noexcept {
            rtml_assert(!shape.empty() && shape.size() <= lim, "Invalid tensor shape must be within 1-{} dimensions", lim);
            m_rank = static_cast<std::uint32_t>(shape.size());
            std::ranges::fill(m_shape.begin(), m_shape.end(), 1); // Splat identity dimensions
            std::ranges::copy(shape.begin(), shape.end(), m_shape.begin()); // Copy dimensions
            m_strides[0] = static_cast<dim>(dtype_size);
            for (std::size_t i {1}; i < k_max_dims; ++i) // Compute strides
                m_strides[i] = m_strides[i-1] * m_shape[i-1];
        }

        [[nodiscard]] constexpr auto rank() const noexcept -> std::uint32_t { return m_rank; }
        [[nodiscard]] constexpr auto dims() const noexcept -> const std::array<dim, lim>& { return m_shape; }
        [[nodiscard]] constexpr auto used_dims() const noexcept -> std::span<const dim> { return {m_shape.cbegin(), m_rank}; }
        [[nodiscard]] constexpr auto strides() const noexcept -> const std::array<dim, lim>& { return m_strides; }
        [[nodiscard]] constexpr auto is_scalar() const noexcept -> bool {
            return std::all_of(m_shape.cbegin(), m_shape.cend(), [](const dim dimension) noexcept -> bool { return dimension == 1; });
        }
        [[nodiscard]] constexpr auto is_vector() const noexcept -> bool {
            return std::all_of(m_shape.cbegin()+1, m_shape.cend(), [](const dim dimension) noexcept -> bool { return dimension == 1; });
        }
        [[nodiscard]] constexpr auto is_matrix() const noexcept -> bool {
            return std::all_of(m_shape.cbegin()+2, m_shape.cend(), [](const dim dimension) noexcept -> bool { return dimension == 1; });
        }
        [[nodiscard]] constexpr auto is_dense() const noexcept -> bool {
            if (m_strides[0] != dtype_size)
                return false;
            for (dim i {1}; i < lim; ++i)
                if (m_strides[i] != m_strides[i - 1] * m_shape[i - 1])
                    return false;
            return true;
        }
        [[nodiscard]] constexpr auto is_dense_except_dim1() const noexcept -> bool {
            if (m_strides[0] != dtype_size)
                return false;
            for (dim i {2}; i < lim; ++i)
                if (m_strides[i] != m_strides[i - 1] * m_shape[i - 1])
                    return false;
            return true;
        }
        [[nodiscard]] constexpr auto is_matmul_compatible(const fixed_shape& other) const noexcept -> bool {
            return m_shape[1] == other.m_shape[0];
        }
        [[nodiscard]] constexpr auto is_transposed() const noexcept -> bool {
            return m_strides[0] > m_strides[1];
        }
        [[nodiscard]] constexpr auto is_permuted() const noexcept -> bool {
            for (dim i {}; i < lim-1; ++i)
                if (m_strides[i] > m_strides[i + 1])
                    return true;
            return false;
        }
        [[nodiscard]] constexpr auto can_repeat(const fixed_shape& other) const noexcept -> bool {
            for (std::size_t i {}; i < lim; ++i)
                if (other.m_shape[i] % m_shape[i] != 0)
                    return false;
            return true;
        }
        [[nodiscard]] constexpr auto can_repeat_rows(const fixed_shape& other) const noexcept -> bool {
            return m_shape[0] == other.m_shape[0] && can_repeat(other);
        }
        [[nodiscard]] constexpr auto row_count() const noexcept -> dim {
            return std::accumulate(m_shape.cbegin()+1, m_shape.cend(), 1, std::multiplies());
        }
        [[nodiscard]] constexpr auto col_count() const noexcept -> dim { return m_shape[0]; }
        [[nodiscard]] constexpr auto elem_count() const noexcept -> dim {
            return std::accumulate(m_shape.cbegin(), m_shape.cend(), 1, std::multiplies());
        }
        [[nodiscard]] constexpr auto unroll_index(const dim i) const noexcept -> std::array<dim, lim> {
            static_assert(lim == 4);
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
        [[nodiscard]] constexpr auto offset(const std::array<dim, lim>& indices) const noexcept -> std::size_t {
            return std::inner_product(indices.cbegin(), indices.cend(), m_strides.cbegin(), 0);
        }
        constexpr auto transpose(fixed_shape& other) noexcept -> void {
            m_shape[0] = other.m_shape[1];
            m_shape[1] = other.m_shape[0];
            m_strides[0] = other.m_strides[1];
            m_strides[1] = other.m_strides[0];
        }
        [[nodiscard]] constexpr auto operator == (const fixed_shape& other) const noexcept -> bool {
            if (this == &other)
                return true;
            if (m_rank == other.m_rank)
                return std::ranges::equal(m_shape, other.m_shape);
            return false;
        }
        [[nodiscard]] constexpr auto operator != (const fixed_shape& other) const noexcept -> bool {
            return !(*this == other);
        }

    private:
        std::uint32_t m_rank {}; // Number of dimensions (1-k_max_dims)
        std::array<dim, k_max_dims> m_shape {}; // 4D dimensions - tensor shape
        std::array<dim, k_max_dims> m_strides {}; // 4D byte strides
    };

    // Represents an N-dimensional (1-k_max_dims) tensor, which is also a vertex in the computation DAG.
    // The dimensionality and data type of a tensor are dynamically handled at runtime.
    template <typename S = dtypes::f32> requires is_dtype<S>
    class tensor final {
    public:
        using dtype = S;
        static constexpr std::size_t k_max_operands {2};
        static constexpr std::size_t k_max_name {128};

        tensor(const tensor&) = delete;
        tensor(tensor&&) = delete;
        auto operator=(const tensor&) -> tensor& = delete;
        auto operator=(tensor&&) -> tensor& = delete;
        ~tensor() = default;

        [[nodiscard]] auto size() const noexcept -> std::size_t { return m_datasize; }
        [[nodiscard]] auto shape() const noexcept -> const fixed_shape<sizeof(S)>& { return m_shape; }
        [[nodiscard]] auto slice_base() const noexcept -> tensor* { return m_slice; }
        [[nodiscard]] auto slice_offset() const noexcept -> std::size_t { return m_slice_offset; }
        [[nodiscard]] auto ptr() const noexcept -> std::uint8_t* { return m_x.u8; }
        [[nodiscard]] auto data() const noexcept -> std::span<S> { return {reinterpret_cast<S*>(m_x.u8), reinterpret_cast<S*>(m_x.u8+m_datasize)}; }
        [[nodiscard]] auto name() const noexcept -> const char* { return m_name.data(); }

        [[nodiscard]] auto isomorphic_clone() noexcept -> tensor* {
            auto* const ts {m_ctx.new_tensor<S>(m_shape.used_dims())};
            if constexpr (k_clone_set_name)
                ts->format_name("{} (isomorph)", m_name.data());
            return ts;
        }
        [[nodiscard]] auto sliced_clone() noexcept -> tensor* {
            auto* const ts {m_ctx.new_tensor<S>(
              m_shape.used_dims(),
              this,
              0
            )};
            std::ranges::copy(
               m_shape.strides().cbegin(),
               m_shape.strides().cend(),
               const_cast<std::array<dim, k_max_dims>&>(ts->shape().strides()).begin()
            );
            if constexpr (k_clone_set_name)
                ts->format_name("{} (slice)", m_name.data());
            return ts;
        }
        [[nodiscard]] auto transposed_clone() noexcept -> tensor* {
            auto* const ts {m_ctx.new_tensor<S>(
              m_shape.used_dims(),
              this,
              0
            )};
            std::ranges::copy(
                m_shape.strides().cbegin(),
                m_shape.strides().cend(),
                const_cast<std::array<dim, k_max_dims>&>(ts->shape().strides()).begin()
            );
            if constexpr (k_clone_set_name)
                ts->format_name("{} (transposed)", m_name.data());
            ts->m_shape.transpose(m_shape);
            return ts;
        }
        [[nodiscard]] auto clone() noexcept -> tensor* {
            auto* const ts {m_ctx.new_tensor<S>(m_shape.used_dims())};
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
            rtml_assert1(data.size() == static_cast<std::size_t>(m_shape.elem_count()));
            std::ranges::copy(data, this->data().begin());
            return this;
        }
        auto fill_data(const std::initializer_list<const S> data) -> tensor* {
            rtml_assert1(data.size() == static_cast<std::size_t>(m_shape.elem_count()));
            std::ranges::copy(data, this->data().begin());
            return this;
        }

        [[nodiscard]] auto add(const tensor* other) noexcept -> tensor* {
            rtml_assert1(validators::validate_binary_op(this, this, other));
            constexpr blas::compute_ctx ctx {};
            blas::add(ctx, *this, *this, *other);
            return this;
        }

        [[nodiscard]] auto sub(const tensor* other) noexcept -> tensor* {
            rtml_assert1(validators::validate_binary_op(this, this, other));
            constexpr blas::compute_ctx ctx {};
            blas::sub(ctx, *this, *this, *other);
            return this;
        }

        [[nodiscard]] auto mul(const tensor* other) noexcept -> tensor* {
            rtml_assert1(validators::validate_binary_op(this, this, other));
            constexpr blas::compute_ctx ctx {};
            blas::mul(ctx, *this, *this, *other);
            return this;
        }

        [[nodiscard]] auto div(const tensor* other) noexcept -> tensor* {
            rtml_assert1(validators::validate_binary_op(this, this, other));
            constexpr blas::compute_ctx ctx {};
            blas::div(ctx, *this, *this, *other);
            return this;
        }

        [[nodiscard]] auto matmul_clone(const tensor* other) noexcept -> tensor* {
            rtml_assert1(validators::validate_matmul(this, this, other));
            constexpr blas::compute_ctx ctx {};
            tensor* const r {m_ctx.new_tensor<S>({m_shape.dims()[0], other->m_shape.dims()[1]})};
            blas::matmul(ctx, *r, *this, *other);
            return r;
        }

        [[nodiscard]] auto sigmoid() noexcept -> tensor* {
            rtml_assert1(validators::validate_unary_op(this, this));
            constexpr blas::compute_ctx ctx {};
            blas::sigmoid(ctx, *this, *this);
            return this;
        }

        [[nodiscard]] auto sigmoid_derivative() noexcept -> tensor* {
            rtml_assert1(validators::validate_unary_op(this, this));
            constexpr blas::compute_ctx ctx {};
            blas::sigmoid_derivative(ctx, *this, *this);
            return this;
        }

#if 0
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
#endif

        [[nodiscard]] auto operator()(const std::array<dim, k_max_dims>& indices) const noexcept -> S& {
            return *reinterpret_cast<S*>(m_x.u8+m_shape.offset(indices));
        }
        [[nodiscard]] auto operator()(const dim i) const noexcept -> S& {
            if (m_shape.is_dense()) return reinterpret_cast<S&>(m_x.u8[i*dtype_traits<S>::k_size]);
            return (*this)(m_shape.unroll_index(i));
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
                "Tensor {}{}{}{} * {}D, Shape [{} X {} X {} X {}], Strides [{}B X {}B X {}B X {}B] {:.01f}{}",
                m_name[0] ? "'" : "",
                m_name.data(),
                m_name[0] ? "': " : "",
                dtype_traits<S>::k_name,
                m_shape.rank(),
                m_shape.dims()[0],
                m_shape.dims()[1],
                m_shape.dims()[2],
                m_shape.dims()[3],
                m_shape.strides()[0],
                m_shape.strides()[1],
                m_shape.strides()[2],
                m_shape.strides()[3],
                size,
                unit
            );
            if (with_data_elems > 0) {
                fmt += "\n[\n";
                for (dim i3 {}; i3 < m_shape.dims()[2]; ++i3) {
                    for (dim i2 {}; i2 < m_shape.dims()[1]; ++i2) {
                        fmt.push_back('\t');
                        for (dim i1 {}; i1 < m_shape.dims()[0]; ++i1) {
                            const S x {reinterpret_cast<S&>(m_x.u8[dtype_traits<S>::k_size*(i3*m_shape.dims()[1]*m_shape.dims()[0] + i2*m_shape.dims()[0] + i1)])};
                            fmt += fmt::format("{} ", x);
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
             std::span<const dim> shape,
             tensor* slice,
             std::size_t slice_offset
        ) noexcept : m_ctx{ctx}, m_shape{shape}{
            if (slice && slice->m_slice) { // Account for if slice itself is also a slice
                slice_offset += slice->m_slice_offset;
                slice = slice->m_slice;
            }
            std::size_t datasize {dtype_traits<S>::k_size}; // Tensor data size to allocate
            constexpr auto size_lim {std::numeric_limits<std::size_t>::max()};
            for (std::size_t i {0}; i < shape.size(); ++i) { // Accumulate total data size
                const std::size_t lim {size_lim / datasize};
                rtml_dassert(shape[i] > 0, "Invalid tensor shape dimension {}, must be > 0", shape[i]);
                rtml_dassert(static_cast<std::size_t>(shape[i]) <= lim, "Tensor size exceeds maximum limit");
                datasize *= static_cast<dim>(std::min<std::size_t>(lim, std::max<dim>(shape[i], 1)));
            }
            rtml_assert(!slice || datasize+slice_offset <= slice->m_datasize, "Slice tensor out of range"); // Check if slice has enough space
            static constexpr bool k_align_scalar = false; // Aligned data address to scalar alignment?
            m_x.u8 = slice
                ? slice->m_x.u8+slice_offset
                : static_cast<std::uint8_t*>(k_align_scalar
                    ? m_ctx.pool().alloc_raw(datasize, dtype_traits<S>::k_align)
                    : m_ctx.pool().alloc_raw(datasize)); // Point into slice data (if slice) or allocate new data from pool.
            m_datasize = datasize;
            m_slice = slice;
            m_slice_offset = slice_offset;
        }

        isolate& m_ctx; // Associated isolate
        std::array<char, k_max_name> m_name {}; // Tensor name - cannot use std::string because we must be trivially destructable
        std::size_t m_datasize {}; // Tensor data size in bytes
        class fixed_shape<sizeof(S)> m_shape; // Tensor shape
        tensor* m_slice {}; // Sliced base tensor, if any
        std::size_t m_slice_offset {}; // Memory offset into sliced base tensor's data
        union {
            float* f32;
            std::uint8_t* u8 {};
        } m_x {};
    };
}
