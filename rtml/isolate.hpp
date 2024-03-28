// Copyright Mario "Neo" Sieg 2024. All rights reserved. mario.sieg.64@gmail.com
// Isolates represent a single isolated contexts with their own memory pool, which can be used to allocate tensors
// Tensors are alive as long as the isolate they got allocated from is alive

#pragma once

#include "base.hpp"

namespace rtml {
    template <typename T>
    concept is_pool_allocateable = requires {
        requires std::is_trivially_destructible_v<T>;
    };

    // Sequencial linear pool allocator with fixed size
    class pool final {
    public:
        explicit pool(std::size_t size);
        pool(const pool&) = delete;
        pool(pool&&) = delete;
        auto operator=(const pool&) -> pool& = delete;
        auto operator=(pool&&) -> pool& = delete;
        ~pool();

        static constexpr auto k_natural_align {alignof(std::max_align_t) > 8 ? alignof(std::max_align_t) : 8};
        static constexpr bool k_force_align {true}; // Always force correct alignment of allocated types
        [[nodiscard]] auto alloc_raw(std::size_t size) noexcept -> void*;
        [[nodiscard]] auto alloc_raw(std::size_t size, std::size_t align) noexcept -> void*;
        template <typename T, typename... Args>
            requires is_pool_allocateable<T> //&& std::is_constructible_v<T, Args...>
        [[nodiscard]] auto alloc(Args&&... args) noexcept(std::is_nothrow_constructible_v<T, Args...>) -> T* {
            if constexpr (alignof(T) > k_natural_align || k_force_align) {
                return new(static_cast<T*>(alloc_raw(sizeof(T), alignof(T)))) T{std::forward<Args>(args)...};
            } else {
                return new(static_cast<T*>(alloc_raw(sizeof(T)))) T{std::forward<Args>(args)...};
            }
        }
        auto print_info() const -> void;
        [[nodiscard]] auto size() const noexcept -> std::size_t { return m_size; }
        [[nodiscard]] auto num_allocs() const noexcept -> std::size_t { return m_num_allocs; }
        [[nodiscard]] auto data() const noexcept -> std::uint8_t* { return m_buf; }
        [[nodiscard]] auto needle() const noexcept -> std::uint8_t* { return m_bot; }
        [[nodiscard]] auto bytes_allocated() const noexcept -> std::size_t { return m_size-(m_bot-m_buf); }

    private:
        const std::size_t m_size;
        std::uint8_t* m_buf {};
        std::uint8_t* m_bot {};
        std::size_t m_num_allocs {};
    };

    // Represents a single isolated context with its own memory pool, which can be used to allocate tensors
    // Tensors are alive as long as the isolate is alive
    // So the isolate should be kept alive as long as the tensors are used
    // TODO: Automatic memory management for tensors lol
    class isolate : public std::enable_shared_from_this<isolate> {
    public:
        enum class compute_device : std::uint32_t {
            auto_select = 0,
            cpu,
            $count
        };
        static constexpr std::array<const char*, static_cast<std::size_t>(compute_device::$count)> k_compute_device_names {
            "Auto Select",
            "CPU"
        };

        [[nodiscard]] static auto create(
            std::string&& name,
            compute_device device,
            std::size_t pool_mem
        ) -> std::shared_ptr<isolate>;

        template <typename S> requires is_dtype<S>
        [[nodiscard]] auto new_tensor(
            std::initializer_list<const dim> shape,
            std::initializer_list<const S> data = {},
            tensor<S>* slice = nullptr,
            std::size_t slice_offset = 0
        ) -> tensor<S>* {
            auto* t {m_pool.alloc<tensor<S>>(*this, shape, slice, slice_offset)};
            if (data.size() > 0)
                t->fill_data(std::span<const S>{data.begin(), data.end()});
            return t;
        }

        template <typename T> requires is_dtype<T>
        [[nodiscard]] auto new_tensor(
            std::span<const dim> shape,
            tensor<T>* slice = nullptr,
            std::size_t slice_offset = 0
        ) -> tensor<T>* {
            return m_pool.alloc<tensor<T>>(*this, shape, slice, slice_offset);
        }

        isolate(const isolate&) = delete;
        isolate(isolate&&) = delete;
        auto operator=(const isolate&) -> isolate& = delete;
        auto operator=(isolate&&) -> isolate& = delete;
        virtual ~isolate() = default;

        [[nodiscard]] static auto init_rtml_runtime() -> bool;
        static auto shutdown_rtml_runtime() -> void;

        [[nodiscard]] auto name() const noexcept -> const std::string& { return m_name; }
        [[nodiscard]] auto device() const noexcept -> compute_device { return m_device; }
        [[nodiscard]] auto pool() const noexcept -> const pool& { return m_pool; }
        [[nodiscard]] auto pool() noexcept -> class pool& { return m_pool; }

    private:
        static inline constinit std::atomic_bool s_runtime_initialized;
        const std::string m_name;
        const compute_device m_device;
        class pool m_pool;

    protected:
        isolate(std::string&& name, compute_device device, std::size_t pool_mem);
    };
}
