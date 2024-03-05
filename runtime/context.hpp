// Copyright Mario "Neo" Sieg 2024. All rights reserved. mario.sieg.64@gmail.com

#pragma once

#define RTML_LOG_ENABLE true

#if RTML_LOG_ENABLE
#include <spdlog/spdlog.h>
#define rtml_log_info SPDLOG_INFO
#define rtml_log_warn SPDLOG_WARN
#define rtml_log_error SPDLOG_ERROR
#else
#define rtml_log_info(...)
#define rtml_log_warn(...)
#define rtml_log_error(...)
#endif

#include "tensor.hpp"

namespace rtml {
    template <typename T>
    concept is_pool_allocateable = requires {
        requires std::is_trivially_destructible_v<T>;
    };

    class pool final {
    public:
        explicit pool(std::size_t size);
        pool(const pool&) = delete;
        pool(pool&&) = delete;
        auto operator=(const pool&) -> pool& = delete;
        auto operator=(pool&&) -> pool& = delete;
        ~pool() = default;

        [[nodiscard]] auto alloc_raw(std::size_t size) noexcept -> void*;
        [[nodiscard]] auto alloc_raw(std::size_t size, std::size_t align) noexcept -> void*;
        template <typename T, typename... Args> requires
            is_pool_allocateable<T> // && std::is_constructible_v<T, Args...>
        [[nodiscard]] auto alloc(Args&&... args) noexcept(std::is_nothrow_constructible_v<T, Args...>) -> T* {
            T* p {static_cast<T*>(alloc_raw(sizeof(T), alignof(T)))};
            return new (p) T{std::forward<Args>(args)...};
        }
        auto print_info() const -> void;
        [[nodiscard]] auto size() const noexcept -> std::size_t { return m_size; }
        [[nodiscard]] auto num_allocs() const noexcept -> std::size_t { return m_num_allocs; }
        [[nodiscard]] auto data() const noexcept -> std::uint8_t* { return m_storage.get(); }
        [[nodiscard]] auto needle() const noexcept -> std::uint8_t* { return m_bot; }
        [[nodiscard]] auto bytes_allocated() const noexcept -> std::size_t { return m_size-(m_bot-m_storage.get()); }

    private:
        const std::size_t m_size;
        const std::unique_ptr<std::uint8_t[]> m_storage;
        std::uint8_t* m_bot {};
        std::size_t m_num_allocs {};
    };

    class context : public std::enable_shared_from_this<context> {
    public:
        enum class compute_device : std::uint32_t {
            auto_select = 0,
            cpu,
            gpu,
            tpu,
            $count
        };
        static constexpr std::array<const char*, static_cast<std::size_t>(compute_device::$count)> k_compute_device_names {
            "Auto Select",
            "CPU",
            "GPU",
            "TPU"
        };

        [[nodiscard]] static auto create(
            std::string&& name,
            compute_device device,
            std::size_t pool_mem
        ) -> std::shared_ptr<context>;
        [[nodiscard]] static auto exists(const std::string& name) -> bool;
        [[nodiscard]] static auto get(const std::string& name) -> std::shared_ptr<context>;
        [[nodiscard]] static auto get_all() -> const std::unordered_map<std::string, std::shared_ptr<context>>&;
        [[nodiscard]] auto create_tensor(
            tensor::dtype type,
            std::span<const std::int64_t> dims,
            tensor* slice = nullptr,
            std::size_t slice_offset = 0,
            tensor::id* out_id = nullptr
        ) -> tensor*;
        [[nodiscard]] auto create_tensor(
            tensor::dtype type,
            std::initializer_list<const std::int64_t> dims,
            tensor* slice = nullptr,
            std::size_t slice_offset = 0,
            tensor::id* out_id = nullptr
        ) -> tensor*;
        [[nodiscard]] auto get_tensor(tensor::id id) const -> tensor*;
        [[nodiscard]] auto get_all_tensors() noexcept -> std::span<tensor*> { return m_tensors; }

        context(const context&) = delete;
        context(context&&) = delete;
        auto operator=(const context&) -> context& = delete;
        auto operator=(context&&) -> context& = delete;
        virtual ~context() = default;

        [[nodiscard]] static auto global_init() -> bool;
        static auto global_shutdown() -> void;

        [[nodiscard]] auto name() const noexcept -> const std::string& { return m_name; }
        [[nodiscard]] auto device() const noexcept -> compute_device { return m_device; }
        [[nodiscard]] auto pool() const noexcept -> const pool& { return m_pool; }
        [[nodiscard]] auto pool() noexcept -> class pool& { return m_pool; }

    private:
        static inline std::unordered_map<std::string, std::shared_ptr<context>> s_contexts;
        static inline std::mutex s_contexts_mutex;
        static inline constinit std::atomic_bool s_initialized;
        const std::string m_name;
        const compute_device m_device;
        class pool m_pool;
        std::vector<tensor*> m_tensors {}; // All tensors in this context. Memory us owned by m_pool.

    protected:
        context(std::string&& name, compute_device device, std::size_t pool_mem);
    };
}
