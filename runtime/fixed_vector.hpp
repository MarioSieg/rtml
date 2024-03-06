// Copyright Mario "Neo" Sieg 2024. All rights reserved. mario.sieg.64@gmail.com

#pragma once

#include <array>
#include <cstddef>
#include <span>
#include <type_traits>

namespace rtml {
    template <typename T, const std::size_t N>
    class fixed_vector final {
    public:
        using value_type = T;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using reference = value_type&;
        using const_reference = const value_type&;
        using pointer = value_type*;
        using const_pointer = const value_type*;
        using iterator = pointer;
        using const_iterator = const_pointer;
        using reverse_iterator = std::reverse_iterator<iterator>;
        using const_reverse_iterator = std::reverse_iterator<const_iterator>;
        static constexpr size_type k_capacity = N;

        [[nodiscard]] auto size() const noexcept -> std::size_t {
            return m_len;
        }
        [[nodiscard]] auto begin() noexcept -> iterator {
            return reinterpret_cast<T*>(m_storage.data());
        }
        [[nodiscard]] auto begin() const noexcept -> const_iterator {
            return reinterpret_cast<const T*>(m_storage.data());
        }
        [[nodiscard]] auto cbegin() const noexcept -> const_iterator {
            return reinterpret_cast<const T*>(m_storage.data());
        }
        [[nodiscard]] auto end() noexcept -> iterator {
            return reinterpret_cast<T*>(m_storage.data()+m_len*sizeof(T));
        }
        [[nodiscard]] auto end() const noexcept -> const_iterator {
            return reinterpret_cast<const T*>(m_storage.data()+m_len*sizeof(T));
        }
        [[nodiscard]] auto cend() const noexcept -> const_iterator {
            return reinterpret_cast<const T*>(m_storage.data()+m_len*sizeof(T));
        }
        [[nodiscard]] auto rbegin() noexcept -> reverse_iterator {
            return reverse_iterator{end()};
        }
        [[nodiscard]] auto rbegin() const noexcept -> const_reverse_iterator {
            return const_reverse_iterator{end()};
        }
        [[nodiscard]] auto crbegin() const noexcept -> const_reverse_iterator {
            return const_reverse_iterator{cend()};
        }
        [[nodiscard]] auto rend() noexcept -> reverse_iterator {
            return reverse_iterator{begin()};
        }
        [[nodiscard]] auto rend() const noexcept -> const_reverse_iterator {
            return const_reverse_iterator{begin()};
        }
        [[nodiscard]] auto crend() const noexcept -> const_reverse_iterator {
            return const_reverse_iterator{cbegin()};
        }
        [[nodiscard]] auto operator[](const std::size_t index) noexcept -> reference {
            return reinterpret_cast<T&>(m_storage[sizeof(T)*index]);
        }
        [[nodiscard]] auto operator[](const std::size_t index) const noexcept -> const_reference {
            return reinterpret_cast<T&>(m_storage[sizeof(T)*index]);
        }
        [[nodiscard]] auto front() noexcept -> reference {
            if (empty()) [[unlikely]] std::abort();
            return (*this)[0];
        }
        [[nodiscard]] auto front() const noexcept -> const_reference {
            if (empty()) [[unlikely]] std::abort();
            return (*this)[0];
        }
        [[nodiscard]] auto back() noexcept -> reference {
            if (empty()) [[unlikely]] std::abort();
            return (*this)[m_len-1];
        }
        [[nodiscard]] auto back() const noexcept -> const_reference {
            if (empty()) [[unlikely]] std::abort();
            return (*this)[m_len-1];
        }
        [[nodiscard]] auto data() noexcept -> pointer { return reinterpret_cast<T*>(m_storage.data()); }
        [[nodiscard]] auto data() const noexcept -> const_pointer { return reinterpret_cast<const T*>(m_storage.data()); }
        [[nodiscard]] auto empty() const noexcept -> bool { return m_len == 0; }
        [[nodiscard]] auto full() const noexcept -> bool { return m_len == N; }
        [[nodiscard]] static auto max_size() noexcept -> std::size_t { return N; }
        [[nodiscard]] static auto capacity() noexcept -> std::size_t { return N; }
        template <typename... Args> requires std::is_constructible_v<T, Args...>
        auto emplace_back(Args&&... args) noexcept(std::is_nothrow_constructible_v<T, Args...>) -> reference {
            if (m_len >= N) [[unlikely]] std::abort();
            return *new(reinterpret_cast<T*>(m_storage.data()+sizeof(T)*m_len++)) T{std::forward<Args>(args)...};
        }
        operator std::span<T, std::dynamic_extent> () noexcept {
            return std::span<T, std::dynamic_extent>{data(), m_len};
        }
        auto clear() noexcept -> void {
            if constexpr (!std::is_trivially_destructible_v<T>)
                for (auto&& v : *this)
                    v.~T();
            m_storage = {};
            m_len = 0;
        }

    private:
        alignas(T) std::array<std::byte, N*sizeof(T)> m_storage {};
        std::size_t m_len {};
    };
}
