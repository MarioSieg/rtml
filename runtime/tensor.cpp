// Copyright Mario "Neo" Sieg 2024. All rights reserved. mario.sieg.64@gmail.com

#include "tensor.hpp"
#include "isolate.hpp"

#include <cassert>
#include <iostream>

namespace rtml {
    auto tensor::set_name(const char* const name) -> void {
        std::strncpy(m_name.data(), name, k_max_name);
        m_name[k_max_name-1] = '\0';
    }

    auto tensor::to_string() -> std::string {
        const std::size_t total_size = m_size+sizeof(*this);
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

    tensor::tensor(
        isolate& ctx,
        const std::uint32_t id,
        dtype type,
        const std::span<const std::int64_t> dims,
        tensor* slice,
        std::size_t slice_offset
    ) noexcept : m_ctx{ctx}, m_id{id}, m_dtype{type} {
        assert(!dims.empty() && dims.size() <= k_max_dims);
        if (slice && slice->m_slice) { // Account for if slice itself is also a slice
            slice_offset += slice->m_slice_offset;
            slice = slice->m_slice;
        }
        const dtype_trait& trait {k_stype_traits[static_cast<std::size_t>(type)]};
        std::size_t datasize {trait.size};
        for (const auto dim : dims) { // Accumulate total size of tensor
            datasize *= std::max<decltype(dim)>(1, dim);
        }
        assert(!slice || datasize+slice_offset <= slice->m_size); // Check if slice has enough space
        static constexpr bool k_align_scalar = false; // Aligned data address to scalar alignment?
        m_u8 = slice
            ? slice->m_u8+slice_offset
            : static_cast<std::uint8_t*>(k_align_scalar
                ? m_ctx.pool().alloc_raw(datasize, trait.align)
                : m_ctx.pool().alloc_raw(datasize)); // Point into slice data or allocate new data from pool.
        m_size = datasize;
        m_num_dims = dims.size();
        m_slice = slice;
        m_slice_offset = slice_offset;
        std::ranges::fill(m_dims.begin(), m_dims.end(), 1); // Splat identity dimensions
        std::ranges::copy(dims.begin(), dims.end(), m_dims.begin()); // Copy dimensions
        m_strides[0] = static_cast<std::int64_t>(trait.size);
        for (std::size_t i {1}; i < k_max_dims; ++i) // Compute strides
            m_strides[i] = m_strides[i-1]*m_dims[i-1];
    }
}
