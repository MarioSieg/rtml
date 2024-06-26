// Copyright Mario "Neo" Sieg 2024. All rights reserved. mario.sieg.64@gmail.com
// Isolates represent a single isolated contexts with their own memory pool, which can be used to allocate tensors
// Tensors are alive as long as the isolate they got allocated from is alive

#include "isolate.hpp"

#include <istream>

#if RTML_LOG_ENABLE
#include <spdlog/logger.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#endif

namespace rtml {
    pool::pool(const std::size_t size) : m_size{size} {
        if (!size) [[unlikely]]
            std::abort();
        m_buf = static_cast<std::uint8_t*>(std::malloc(size)); // Malloc to avoid initialization of memory and to use lazy mapping
        if (!m_buf) [[unlikely]]
            std::abort();
        rtml_log_info("Created linear memory pool of size {:.01f} MiB", static_cast<double>(size)/std::pow(1024.0, 2.0));
        m_bot = m_buf + size;
    }

    pool::~pool() {
        std::free(m_buf);
    }

    auto pool::alloc_raw(const std::size_t size) noexcept -> void* {
        m_bot -= size;
        if (m_bot < m_buf) [[unlikely]]
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
        const std::ptrdiff_t used {std::max<decltype(used)>(0, m_bot-m_buf)};
        const double perc {100.0*static_cast<double>(m_size - used)/static_cast<double>(m_size)};
        std::printf(
            "Pool: %.03f/%.01f MiB, used: %.03f%%, %zu allocs\n",
            static_cast<double>(m_size-used)/std::pow(1024.0, 2.0),
           static_cast<double>(m_size)/std::pow(1024.0, 2.0), perc, m_num_allocs
        );
        std::printf("Mem: &[%p, %p]\n", static_cast<void*>(m_buf), static_cast<void*>(m_bot));
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
