// Copyright Mario "Neo" Sieg 2024. All rights reserved. mario.sieg.64@gmail.com

#include "tensor.hpp"
#include "isolate.hpp"

#include <cassert>
#include <iostream>

namespace rtml {
    auto tensor::is_dense() const noexcept -> bool {
        static_assert(k_max_dims == 4);
        return
            m_strides[0] == datatype_traits().size && // Check if strides are contiguous
            m_strides[1] == m_strides[0] * m_shape[0] &&
            m_strides[2] == m_strides[1] * m_shape[1] &&
            m_strides[3] == m_strides[2] * m_shape[2];
    }

    auto tensor::is_dense_except_dim1() const noexcept -> bool {
        return
            m_strides[0] == datatype_traits().size && // Check if strides are contiguous
            m_strides[2] == m_strides[1] * m_shape[1] &&
            m_strides[3] == m_strides[2] * m_shape[2];
    }

    auto tensor::can_repeat(const tensor* const other) const noexcept -> bool {
        static_assert(k_max_dims == 4);
        return
            other->m_shape[0] % m_shape[0] == 0 && // Check if other's dimensions are divisible by this tensor's dimensions
            other->m_shape[1] % m_shape[1] == 0 &&
            other->m_shape[2] % m_shape[2] == 0 &&
            other->m_shape[3] % m_shape[3] == 0;
    }

    auto tensor::is_shape_eq(const tensor* const other) const noexcept -> bool {
        if (this == other)
            return true;
        if (m_num_dims == other->m_num_dims)
            return std::equal(m_shape.cbegin(), m_shape.cbegin()+m_num_dims, other->m_shape.cbegin());
        return false;
    }

    auto tensor::is_matmul_compatible(const tensor* const other) const noexcept -> bool {
        return m_shape[0] == other->m_shape[0] && // Check if matrix multiplication is compatible
            other->m_shape[2] % m_shape[2] == 0 &&
            other->m_shape[3] % m_shape[3] == 0;
    }

    auto tensor::row_count() const noexcept -> dim {
        static_assert(k_max_dims == 4);
        return m_shape[1] * m_shape[2] * m_shape[3];
    }

    auto tensor::col_count() const noexcept -> dim {
        return m_shape[0];
    }

    auto tensor::elem_count() const noexcept -> dim {
        static_assert(k_max_dims == 4);
        return m_shape[0] * m_shape[1] * m_shape[2] * m_shape[3];
    }

    auto tensor::unroll_index(const dim i) const noexcept -> std::array<dim, k_max_dims> {
        static_assert(k_max_dims == 4);
        const dim d0 {m_shape[0]};
        const dim d1 {m_shape[1]};
        const dim d2 {m_shape[2]};
        std::array<dim, k_max_dims> dims {};
        dims[3] =  i / (d2*d1*d0); // Unroll index into dimensions
        dims[2] = (i - dims[3]*d2*d1*d0) / (d1*d0);
        dims[1] = (i - dims[3]*d2*d1*d0 - dims[2]*d1*d0) / d0;
        dims[0] =  i - dims[3]*d2*d1*d0 - dims[2]*d1*d0 - dims[1] * d0;
        return dims;
    }

    auto tensor::operator()(const std::array<dim, k_max_dims>& indices) const noexcept -> float& {
        return *reinterpret_cast<float*>(
            m_x.u8 +
            indices[3]*m_strides[3] +
            indices[2]*m_strides[2] +
            indices[1]*m_strides[1] +
            indices[0]*m_strides[0]
        );
    }

    auto tensor::operator()(const dim i) const noexcept -> float& {
        if (is_dense()) {
            return m_x.f32[i];
        }
        return (*this)(unroll_index(i));
    }

    tensor::tensor(
        isolate& ctx,
        dtype type,
        const std::span<const dim> dims,
        tensor* slice,
        std::size_t slice_offset
    ) noexcept : m_ctx{ctx}, m_dtype{type} {
        assert(!dims.empty() && dims.size() <= k_max_dims);
        if (slice && slice->m_slice) { // Account for if slice itself is also a slice
            slice_offset += slice->m_slice_offset;
            slice = slice->m_slice;
        }
        const dtype_trait& dtype_inf {k_stype_traits[static_cast<std::size_t>(type)]};
        std::size_t datasize {dtype_inf.size}; // Tensor data size to allocate
        for (const auto dim : dims) // Accumulate total size of tensor
            datasize *= std::max<decltype(dim)>(1, dim);
        assert(!slice || datasize+slice_offset <= slice->m_datasize); // Check if slice has enough space
        static constexpr bool k_align_scalar = false; // Aligned data address to scalar alignment?
        m_x.u8 = slice
            ? slice->m_x.u8+slice_offset
            : static_cast<std::uint8_t*>(k_align_scalar
                ? m_ctx.pool().alloc_raw(datasize, dtype_inf.align)
                : m_ctx.pool().alloc_raw(datasize)); // Point into slice data (if slice) or allocate new data from pool.
        m_datasize = datasize;
        m_num_dims = dims.size();
        m_slice = slice;
        m_slice_offset = slice_offset;
        std::ranges::fill(m_shape.begin(), m_shape.end(), 1); // Splat identity dimensions
        std::ranges::copy(dims.begin(), dims.end(), m_shape.begin()); // Copy dimensions
        m_strides[0] = static_cast<dim>(dtype_inf.size);
        for (std::size_t i {1}; i < k_max_dims; ++i) // Compute strides
            m_strides[i] = m_strides[i-1]*m_shape[i-1];
    }

    auto tensor::isomorphic() noexcept -> tensor* {
        auto* const ts {m_ctx.create_tensor(
            m_dtype,
            used_dims()
        )};
        ts->format_name("{} (isomorph)", m_name.data());
        return ts;
    }

    auto tensor::clone() noexcept -> tensor* {
        auto* const ts {m_ctx.create_tensor(
            m_dtype,
            used_dims()
        )};
        std::memcpy(ts->m_x.u8, m_x.u8, m_datasize);
        ts->format_name("{} (clone)", m_name.data());
        return ts;
    }

    auto tensor::splat_zero() const -> void {
        std::memset(m_x.u8, 0, m_datasize);
    }

    auto tensor::splat_one() const -> void {
        splat(1.0f);
    }

    auto tensor::splat(const float x) const -> void {
        switch (m_dtype) {
            case dtype::f32: std::ranges::fill(m_x.f32, m_x.f32+m_datasize/datatype_traits().size, x); break;
            case dtype::$count: std::abort();
        }
    }

    auto tensor::push_operand(const tensor* const x) -> void {
        assert(x != nullptr && m_operands.size() < k_max_operands);
        m_operands.emplace_back(x);
    }

    auto tensor::set_name(const char* const name) -> void {
        std::strncpy(m_name.data(), name, k_max_name);
        m_name[k_max_name-1] = '\0';
    }

    auto tensor::to_string(const std::size_t with_data_elems) const -> std::string {
        static_assert(k_max_dims == 4);
        const std::size_t total_size = m_datasize+sizeof(*this);
        auto size {static_cast<double>(total_size)};
        std::string_view unit {"B"};
        const auto cvt_nit {[&](const std::size_t lim, const std::string_view name) {
            if (total_size > lim) {
                size /= static_cast<double>(lim);
                unit = name;
            }
        }};
        cvt_nit(1<<10, "KiB");
        cvt_nit(1<<20, "MiB");
        cvt_nit(1<<30, "GiB");
        std::string fmt {};
        fmt.reserve(0x100+sizeof("2.000")*with_data_elems);
        fmt += fmt::format(
            "Tensor '{}', {} * {}D, Shape [{} X {} X {} X {}], Strides [{} X {} X {} X {}] {:.01f}{}",
            m_name.data(),
            datatype_traits().name,
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
                        const float x {m_x.f32[i3*m_shape[1]*m_shape[0] + i2*m_shape[0] + i1]};
                        fmt += fmt::format("{:.03f} ", x);
                    }
                    fmt.push_back('\n');
                }
            }
            fmt += "\t...\n]";
        }
        return fmt;
    }

    auto tensor::print(const std::size_t with_data_elems) const -> void {
        std::cout << to_string(with_data_elems) << std::endl;
    }
}
