// Copyright Mario "Neo" Sieg 2024. All rights reserved. mario.sieg.64@gmail.com

#include "tensor.hpp"
#include "context.hpp"

#include <cassert>
#include <iostream>

namespace rtml {
    tensor::tensor(
        context& ctx,
        const std::uint32_t id,
        dtype type,
        const std::span<const std::int64_t> dims,
        tensor* slice,
        std::size_t slice_offset
    ) noexcept : m_ctx{ctx}, m_id{id}, m_stype{type} {
        assert(!dims.empty() && dims.size() <= k_max_dims);
        if (slice && slice->m_slice) { // Account for if slice itself is also a slice
            slice_offset += slice->m_slice_offset;
            slice = slice->m_slice;
        }
        const auto [ssize, salign] {k_stype_traits[static_cast<std::size_t>(type)]};
        std::size_t datasize {ssize};
        for (const auto dim : dims) { // Accumulate total size of tensor
            datasize *= std::max<decltype(dim)>(1, dim);
        }
        assert(!slice || datasize+slice_offset <= slice->m_size); // Check if slice has enough space
        static constexpr bool k_align_scalar = false; // Aligned data address to scalar alignment?
        m_u8 = slice
            ? slice->m_u8+slice_offset
            : static_cast<std::uint8_t*>(k_align_scalar
                ? m_ctx.pool().alloc_raw(datasize, salign)
                : m_ctx.pool().alloc_raw(datasize)); // Point into slice data or allocate new data from pool.
        m_size = datasize;
        m_slice = slice;
        m_slice_offset = slice_offset;
        std::ranges::fill(m_dims.begin(), m_dims.end(), 1); // Splat identity dimensions
        std::ranges::copy(dims.begin(), dims.end(), m_dims.begin()); // Copy dimensions
        m_strides[0] = static_cast<std::int64_t>(ssize);
        for (int i {}; i < k_max_dims; ++i) // Compute strides
            m_strides[i] = m_strides[i-1]*m_dims[i-1];
    }
}
