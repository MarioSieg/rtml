// Copyright Mario "Neo" Sieg 2024. All rights reserved. mario.sieg.64@gmail.com

#include "rtml.hpp"

#include <cassert>
#include <iostream>

#include <spdlog/logger.h>
#include <spdlog/sinks/stdout_color_sinks.h>

namespace rtml {
    struct context_proxy final : context {
        template <typename... Args>
        explicit context_proxy(Args&&... args) : context{std::forward<Args>(args)...} {}
    };

    auto context::create(
        std::string&& name,
        const compute_device device,
        const std::size_t pool_mem
    ) -> std::shared_ptr<context> {
        std::unique_lock lock {s_contexts_mutex};
        return s_contexts.emplace(std::string{name}, std::make_shared<context_proxy>(std::move(name), device, pool_mem)).first->second;
    }

    auto context::exists(const std::string& name) -> bool {
        std::unique_lock lock {s_contexts_mutex};
        return s_contexts.contains(name);
    }

    auto context::get(const std::string& name) -> std::shared_ptr<context> {
        std::unique_lock lock {s_contexts_mutex};
        return s_contexts.at(name);
    }

    auto context::get_all() -> const std::unordered_map<std::string, std::shared_ptr<context>>& {
        return s_contexts;
    }

    auto context::create_tensor(
        const tensor::stype type,
        const std::span<const std::int64_t> dims,
        tensor* const slice,
        const std::size_t slice_offset,
        tensor::id* const out_id
    ) -> tensor* {
        assert(m_tensors.size() <= std::numeric_limits<tensor::id>::max());
        const auto id {static_cast<tensor::id>(m_tensors.size())};
        auto* tensor = m_pool.alloc<class tensor>(*this, static_cast<tensor::id>(id), type, dims, slice, slice_offset);
        m_tensors.emplace_back(tensor);
        if (out_id) *out_id = id;
        return tensor;
    }

    auto context::create_tensor(
        const tensor::stype type,
        const std::initializer_list<const std::int64_t> dims,
        tensor* const slice,
        const std::size_t slice_offset,
        tensor::id* const out_id
    ) -> tensor* {
        return create_tensor(type, std::span{dims.begin(), dims.size()}, slice, slice_offset, out_id);
    }

    auto context::get_tensor(const tensor::id id) const -> tensor* {
        if (id >= m_tensors.size()) [[unlikely]]
            return nullptr;
        return &*m_tensors[id];
    }

    auto context::global_init() -> bool {
        std::iostream::sync_with_stdio(false);
        const std::shared_ptr<spdlog::logger> logger = spdlog::stdout_color_mt("rtml_runtime");
        logger->set_pattern("%H:%M:%S:%e %s:%# %^[%l]%$ T:%t %v");
        spdlog::set_default_logger(logger);
        rtml_log_info("RTML runtime initialized");
        return true;
    }

    auto context::global_shutdown() -> void {
        rtml_log_info("RTML runtime shutdown");
    }

    context::context(std::string&& name, compute_device device, const std::size_t pool_mem)
        : m_name{std::move(name)}, m_device{device}, m_pool{pool_mem} {
        rtml_log_info(
            "Creating context '{}', Device: '{}', Pool memory: {:.01f} GiB",
            m_name.c_str(),
            context::k_compute_device_names[static_cast<std::size_t>(m_device)],
            static_cast<double>(pool_mem)/std::pow(1024.0, 3.0)
        );
    }

    pool::pool(const std::size_t size) : m_size{size}, m_storage{new std::uint8_t[size]} {
        if (!size) [[unlikely]]
            std::abort();
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

    tensor::tensor(
        context& ctx,
        const std::uint32_t id,
        stype type,
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
