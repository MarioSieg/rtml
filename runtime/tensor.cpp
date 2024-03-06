// Copyright Mario "Neo" Sieg 2024. All rights reserved. mario.sieg.64@gmail.com

#include "tensor.hpp"
#include "isolate.hpp"

#include <cassert>
#include <iostream>

namespace rtml {
    auto tensor::is_contiguous() const noexcept -> bool {
        static_assert(k_max_dims == 4);
        return
            m_strides[0] == get_data_type_traits().size &&
            m_strides[1] == m_strides[0] * m_dims[0]    &&
            m_strides[2] == m_strides[1] * m_dims[1]    &&
            m_strides[3] == m_strides[2] * m_dims[2];
    }

    auto tensor::can_repeat(const tensor* const other) const noexcept -> bool {
        static_assert(k_max_dims == 4);
        return
            other->m_dims[0] % m_dims[0] == 0 &&
            other->m_dims[1] % m_dims[1] == 0 &&
            other->m_dims[2] % m_dims[2] == 0 &&
            other->m_dims[3] % m_dims[3] == 0;
    }

    auto tensor::row_count() const noexcept -> dim {
        static_assert(k_max_dims == 4);
        return m_dims[1] * m_dims[2] * m_dims[3];
    }

    auto tensor::unroll_index(const dim i) const noexcept -> std::array<dim, k_max_dims> {
        static_assert(k_max_dims == 4);
        const dim d0 {m_dims[0]};
        const dim d1 {m_dims[1]};
        const dim d2 {m_dims[2]};
        std::array<dim, k_max_dims> dims {};
        dims[3] =  i / (d2 * d1 * d0);
        dims[2] = (i - dims[3] * d2 * d1 * d0) / (d1 * d0);
        dims[1] = (i - dims[3] * d2 * d1 * d0 - dims[2] * d1 * d0) / d0;
        dims[0] =  i - dims[3] * d2 * d1 * d0 - dims[2] * d1 * d0 - dims[1] * d0;
        return dims;
    }

    tensor::tensor(
        isolate& ctx,
        const std::uint32_t id,
        dtype type,
        const std::span<const dim> dims,
        tensor* slice,
        std::size_t slice_offset
    ) noexcept : m_ctx{ctx}, m_id{id}, m_dtype{type} {
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
        std::ranges::fill(m_dims.begin(), m_dims.end(), 1); // Splat identity dimensions
        std::ranges::copy(dims.begin(), dims.end(), m_dims.begin()); // Copy dimensions
        m_strides[0] = static_cast<dim>(dtype_inf.size);
        for (std::size_t i {1}; i < k_max_dims; ++i) // Compute strides
            m_strides[i] = m_strides[i-1]*m_dims[i-1];
    }

    auto tensor::isomorph() noexcept -> tensor* {
        auto* const ts {m_ctx.create_tensor(
            m_dtype,
            get_active_dims()
        )};
        ts->format_name("{} (isomorph)", m_name.data());
        return ts;
    }

    auto tensor::clone() noexcept -> tensor* {
        auto* const ts {m_ctx.create_tensor(
            m_dtype,
            get_active_dims()
        )};
        std::memcpy(ts->m_x.u8, m_x.u8, m_datasize);
        ts->format_name("{} (clone)", m_name.data());
        return ts;
    }

    auto tensor::fill_zero() const -> void {
        std::memset(m_x.u8, 0, m_datasize);
    }

    auto tensor::fill_ones() const -> void {
        switch (m_dtype) {
            case dtype::f32: std::ranges::fill(m_x.f32, m_x.f32+m_datasize/get_data_type_traits().size, 1.0f); break;
            case dtype::$count: std::abort();
        }
    }

    auto tensor::set_name(const char* const name) -> void {
        std::strncpy(m_name.data(), name, k_max_name);
        m_name[k_max_name-1] = '\0';
    }

    auto tensor::to_string() -> std::string {
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
        return fmt::format(
            "Tensor '{}' {}D {} [{} X {} X {} X {}] {:.03f} {}",
            m_name.data(),
            m_num_dims,
            get_data_type_traits().name,
            m_dims[0],
            m_dims[1],
            m_dims[2],
            m_dims[3],
            size,
            unit
        );
    }
}
