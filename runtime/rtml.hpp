// Copyright Mario "Neo" Sieg 2024. All rights reserved. mario.sieg.64@gmail.com

#pragma once

#include <array>
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <mutex>
#include <type_traits>
#include <span>

#include <spdlog/spdlog.h>

#define rtml_log_info SPDLOG_INFO
#define rtml_log_warn SPDLOG_WARN
#define rtml_log_error SPDLOG_ERROR

namespace rtml {
    static_assert(std::numeric_limits<float>::is_iec559);
    static_assert(std::numeric_limits<double>::is_iec559);

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
            is_pool_allocateable<T> // && std::is_constructible_v<T, std::decay_t<Args>...>
        [[nodiscard]] auto alloc(Args&&... args) noexcept(std::is_nothrow_constructible_v<T, Args...>) -> T* {
            T* p {static_cast<T*>(alloc_raw(sizeof(T), alignof(T)))};
            return new (p) T{std::forward<Args>(args)...};
        }
        auto print_info() const -> void;

    private:
        const std::size_t m_size;
        const std::unique_ptr<std::uint8_t[]> m_storage;
        std::uint8_t* m_top {};
        std::uint8_t* m_bot {};
        std::size_t m_num_allocs {};
    };

    class context;
    class tensor final {
    public:
        using id = std::uint32_t;
        static constexpr std::int64_t k_max_dims = 4;

        enum class stype {
            f32,
            $count
        };

        struct stype_trait final {
            std::size_t size;
            std::size_t align;
        };

        static constexpr std::array<stype_trait, static_cast<std::size_t>(stype::$count)> k_stype_traits {
            { sizeof(float), alignof(float) }
        };

        tensor(const tensor&) = delete;
        tensor(tensor&&) = delete;
        auto operator=(const tensor&) -> tensor& = delete;
        auto operator=(tensor&&) -> tensor& = delete;
        ~tensor() = default;

    private:
        friend class context;
        friend class pool;
        tensor(
            context& ctx,
            std::uint32_t id,
            stype type,
            std::span<const std::int64_t> dims,
            tensor* slice,
            std::size_t slice_offset
        ) noexcept;

        context& m_ctx;
        const std::uint32_t m_id;
        const stype m_stype; // Tensor scalar data type
        std::size_t m_size {}; // Tensor data size in bytes
        std::array<std::int64_t, k_max_dims> m_dims {};
        std::array<std::int64_t, k_max_dims> m_strides {};
        tensor* m_slice {};
        std::size_t m_slice_offset {};
        union {
            float* m_s {};
            std::uint8_t* m_u8;
        };
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
            tensor::stype type,
            std::span<const std::int64_t> dims,
            tensor* slice = nullptr,
            std::size_t slice_offset = 0,
            tensor::id* out_id = nullptr
        ) -> tensor*;
        [[nodiscard]] auto create_tensor(
            tensor::stype type,
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
        const std::string m_name;
        const compute_device m_device;
        class pool m_pool;
        std::vector<tensor*> m_tensors {}; // All tensors in this context. Memory us owned by m_pool.

    protected:
        context(std::string&& name, compute_device device, std::size_t pool_mem);
    };
}
