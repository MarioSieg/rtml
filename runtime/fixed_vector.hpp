// Copyright Mario "Neo" Sieg 2024. All rights reserved. mario.sieg.64@gmail.com

#pragma once

#include <array>
#include <cstddef>
#include <span>
#include <type_traits>

#include "base.hpp"

namespace rtml {
    template <typename T, const std::size_t N>
    class fixed_vector final {
    public:
        using value_type = T;
        using iterator = T*;
        using const_iterator = const T*;
        using reverse_iterator = std::reverse_iterator<iterator>;
        using const_reverse_iterator = std::reverse_iterator<const_iterator>;
        static constexpr std::size_t k_capacity {N};

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
        [[nodiscard]] auto operator[](const std::size_t i) noexcept -> T& {
            return reinterpret_cast<T*>(m_storage.data())[i];
        }
        [[nodiscard]] auto operator[](const std::size_t i) const noexcept -> const T& {
            return reinterpret_cast<const T*>(m_storage.data())[i];
        }
        [[nodiscard]] auto front() noexcept -> T& {
            rtml_dassert(!empty(), "empty vector");
            return (*this)[0];
        }
        [[nodiscard]] auto front() const noexcept -> const T& {
            rtml_dassert(!empty(), "empty vector");
            return (*this)[0];
        }
        [[nodiscard]] auto back() noexcept -> T& {
            rtml_dassert(!empty(), "empty vector");
            return (*this)[m_len-1];
        }
        [[nodiscard]] auto back() const noexcept -> const T& {
            rtml_dassert(!empty(), "empty vector");
            return (*this)[m_len-1];
        }
        [[nodiscard]] auto data() noexcept -> T* { return reinterpret_cast<T*>(m_storage.data()); }
        [[nodiscard]] auto data() const noexcept -> const T* { return reinterpret_cast<const T*>(m_storage.data()); }
        [[nodiscard]] auto empty() const noexcept -> bool { return m_len == 0; }
        [[nodiscard]] auto full() const noexcept -> bool { return m_len == N; }
        [[nodiscard]] static auto max_size() noexcept -> std::size_t { return N; }
        [[nodiscard]] static auto capacity() noexcept -> std::size_t { return N; }
        template <typename... Args> requires std::is_constructible_v<T, Args...>
        auto emplace_back(Args&&... args) noexcept(std::is_nothrow_constructible_v<T, Args...>) -> T& {
            rtml_assert(m_len < N, "vector full: {}", N);
            return *new(m_len+++reinterpret_cast<T*>(m_storage.data())) T{std::forward<Args>(args)...};
        }
        operator std::span<T> () noexcept {
            return std::span<T>{data(), m_len};
        }
        operator std::span<const T> () const noexcept {
            return std::span<const T>{data(), m_len};
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
