// Copyright Mario "Neo" Sieg 2024. All rights reserved. mario.sieg.64@gmail.com

#include "isolate.hpp"

#include <istream>

#if RTML_LOG_ENABLE
#include <spdlog/logger.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#endif

namespace rtml {
    pool::pool(const std::size_t size) : m_size{size}, m_storage{new std::uint8_t[size]} {
        if (!size) [[unlikely]]
            std::abort();
        rtml_log_info("Created linear memory pool of size {:.01f} MiB", static_cast<double>(size)/std::pow(1024.0, 2.0));
        m_bot = m_storage.get() + size;
    }

    auto pool::alloc_raw(const std::size_t size) noexcept -> void* {
        m_bot -= size;
        if (m_bot < m_storage.get()) [[unlikely]]
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
        const std::ptrdiff_t used {std::max<decltype(used)>(0, m_bot-m_storage.get())};
        const double perc {100.0*static_cast<double>(m_size - used)/static_cast<double>(m_size)};
        std::printf(
            "Pool: %.03f/%.01f MiB, used: %.03f%%, %zu allocs\n",
            static_cast<double>(m_size-used)/std::pow(1024.0, 2.0),
           static_cast<double>(m_size)/std::pow(1024.0, 2.0), perc, m_num_allocs
        );
        std::printf("Mem: &[%p, %p]\n", static_cast<void*>(m_storage.get()), static_cast<void*>(m_bot));
    }

    struct context_proxy final : isolate {
        template <typename... Args>
        explicit context_proxy(Args&&... args) : isolate{std::forward<Args>(args)...} {}
    };

    auto isolate::create(
        std::string&& name,
        const compute_device device,
        const std::size_t pool_mem
    ) -> std::shared_ptr<isolate> {
        if (!s_runtime_initialized.load(std::memory_order::seq_cst)) [[unlikely]] {
            rtml_log_warn("RTML runtime not initialized");
            std::abort();
        }
        return std::make_shared<context_proxy>(std::move(name), device, pool_mem);
    }

    auto isolate::create_tensor(
        const tensor::dtype type,
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

    auto isolate::create_tensor(
        const tensor::dtype type,
        const std::initializer_list<const std::int64_t> dims,
        tensor* const slice,
        const std::size_t slice_offset,
        tensor::id* const out_id
    ) -> tensor* {
        return create_tensor(type, std::span{dims.begin(), dims.size()}, slice, slice_offset, out_id);
    }

    auto isolate::get_tensor(const tensor::id id) const -> tensor* {
        if (id >= m_tensors.size()) [[unlikely]]
            return nullptr;
        return &*m_tensors[id];
    }

    auto isolate::init_rtml_runtime() -> bool {
        if (s_runtime_initialized.load(std::memory_order::seq_cst)) {
            rtml_log_warn("RTML runtime already initialized");
            return true;
        }
#if RTML_LOG_ENABLE
        std::iostream::sync_with_stdio(false);
        const std::shared_ptr<spdlog::logger> logger = spdlog::stdout_color_mt("rtml_runtime");
        logger->set_pattern("%H:%M:%S:%e %s:%# %^[%l]%$ T:%t %v");
        spdlog::set_default_logger(logger);
#endif
        s_runtime_initialized.store(true, std::memory_order::seq_cst);
        rtml_log_info("RTML runtime initialized");
        return true;
    }

    auto isolate::shutdown_rtml_runtime() -> void {
        if (!s_runtime_initialized.load(std::memory_order::seq_cst)) {
            rtml_log_warn("RTML runtime not initialized");
            return;
        }
        rtml_log_info("RTML runtime shutdown");
#if RTML_LOG_ENABLE
        spdlog::shutdown();
#endif
        s_runtime_initialized.store(false, std::memory_order::seq_cst);
    }

    isolate::isolate(std::string&& name, compute_device device, const std::size_t pool_mem)
        : m_name{std::move(name)}, m_device{device}, m_pool{pool_mem} {
        rtml_log_info(
            "Creating isolate '{}', Device: '{}', Pool memory: {:.01f} GiB",
            m_name.c_str(),
            isolate::k_compute_device_names[static_cast<std::size_t>(m_device)],
            static_cast<double>(pool_mem)/std::pow(1024.0, 3.0)
        );
    }
}
