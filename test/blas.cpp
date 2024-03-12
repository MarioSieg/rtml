// Copyright Mario "Neo" Sieg 2024. All rights reserved. mario.sieg.64@gmail.com

#include <gtest/gtest.h>

#include <blas.hpp>

#include <vector>
#include <random>

using namespace rtml;

#define impl_blas_test(name, op) \
    TEST(blas, name) { \
        std::vector<float> a {}, b{}, c{}; \
        constexpr std::size_t N {0xffff}; \
        a.reserve(N); \
        b.reserve(N); \
        std::mt19937_64 prng {}; \
        std::uniform_real_distribution<float> dist{-1.0f, 1.0f}; \
        for (std::size_t i {}; i < N; ++i) { \
            a.emplace_back(dist(prng)); \
            b.emplace_back(dist(prng)); \
        } \
        c.resize(N); \
        std::fill(c.begin(), c.end(), 0.0f); \
        blas::v_##name(N, c.data(), a.data(), b.data()); \
        for (std::size_t i {}; i < N; ++i) { \
            ASSERT_FLOAT_EQ(c[i], a[i] op b[i]); \
        } \
    }

impl_blas_test(f32_add, +)
impl_blas_test(f32_sub, -)
impl_blas_test(f32_mul, *)
impl_blas_test(f32_div, /)

TEST(blas, dot) {
    std::vector<float> a {}, b{};
    constexpr std::size_t N {0xffff};
    a.reserve(N);
    b.reserve(N);
    std::mt19937_64 prng {};
    std::uniform_real_distribution<float> dist{-1.0f, 1.0f};
    for (std::size_t i {}; i < N; ++i) {
        a.emplace_back(dist(prng));
        b.emplace_back(dist(prng));
    }
    float r {};
    double acc {};
    blas::v_f32_dot(N, &r, a.data(), b.data());
    for (std::size_t i {}; i < N; ++i) {
        acc += static_cast<double>(a[i] * b[i]);
    }
    ASSERT_FLOAT_EQ(r, static_cast<float>(acc));
}
