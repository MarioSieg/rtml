// Copyright Mario "Neo" Sieg 2024. All rights reserved. mario.sieg.64@gmail.com

#include "rtml.hpp"

#include <cassert>

namespace rtml {
    pool::pool(const std::size_t size) : m_size{size}, m_storage{new std::uint8_t[size]} {
        m_top = m_storage.get();
        m_bot = m_top + size;
    }

    auto pool::alloc_raw(const std::size_t size) noexcept -> void* {
        m_bot -= size;
        if (m_bot < m_top) [[unlikely]]
            std::abort();
        ++m_num_allocs;
        return m_bot;
    }

    auto pool::alloc_raw(const std::size_t size, const std::size_t align) noexcept -> void* {
        assert((align & align - 1) == 0);
        const std::size_t mask {align - 1};
        auto* p {static_cast<std::uint8_t*>(alloc_raw(size+mask))};
        return std::bit_cast<void*>(std::bit_cast<std::uintptr_t>(p)+mask&~mask);
    }

    auto pool::print_info() const -> void {
        const std::ptrdiff_t used {std::max<decltype(used)>(0, m_bot - m_top)};
        const double perc {100.0*static_cast<double>(m_size - used)/static_cast<double>(m_size)};
        std::printf(
            "Pool: %.03f/%.01f MiB, used: %.03f%%, %zu allocs\n",
            static_cast<double>(m_size-used)/std::pow(1024.0, 2.0),
           static_cast<double>(m_size)/std::pow(1024.0, 2.0), perc, m_num_allocs
        );
        std::printf("Mem: [&%p,  &%p]\n", static_cast<void*>(m_top), static_cast<void*>(m_bot));
    }

    auto tensor::create(
        pool& pool,
        stype type,
        const std::initializer_list<const std::int64_t> dims,
        tensor* slice,
        std::size_t slice_offset
    ) -> tensor* {
        return pool.alloc<tensor>(pool, type, dims, slice, slice_offset);
    }

    tensor::tensor(
        pool& pool,
        stype type,
        const std::initializer_list<const std::int64_t> dims,
        tensor* slice,
        std::size_t slice_offset
    ) noexcept : m_pool{&pool}, m_stype{type} {
        assert(dims.size() > 0 && dims.size() <= k_max_dims);
        if (slice && slice->m_slice) { // Account for if slice itself is also a slice
            slice_offset += slice->m_slice_offset;
            slice = slice->m_slice;
        }
        const auto [ssize, salign] {k_stype_traits[static_cast<std::size_t>(type)]};
        std::size_t datasize {ssize};
        for (const auto dim : dims) { // Accumulate total size of tensor
            assert(dim > 0 && dim < k_max_elems_per_dim);
            datasize *= dim;
        }
        assert(!slice || datasize+slice_offset <= slice->m_size); // Check if slice has enough space
        static constexpr bool k_align_scalar = false; // Aligned data address to scalar alignment?
        m_u8 = slice
            ? slice->m_u8+slice_offset
            : static_cast<std::uint8_t*>(k_align_scalar
                ? pool.alloc_raw(datasize, salign)
                : pool.alloc_raw(datasize)); // Point into slice data or allocate new data from pool.
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
