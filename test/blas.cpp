// Copyright Mario "Neo" Sieg 2024. All rights reserved. mario.sieg.64@gmail.com

#include <gtest/gtest.h>

#include <blas.hpp>
#include <isolate.hpp>
#include <tensor.hpp>

#include <vector>
#include <random>

using namespace rtml;

#define impl_blas_test(name, op) \
    TEST(blas, vec_##name) { \
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

#if 0

impl_blas_test(f32_add, +)
impl_blas_test(f32_sub, -)
impl_blas_test(f32_mul, *)
impl_blas_test(f32_div, /)

TEST(blas, vec_dot) {
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

#endif

TEST(blas, tensor_add) {
    std::mt19937_64 prng {};
    std::uniform_real_distribution<float> dist{-1.0f, 1.0f};
    const float x {dist(prng)};
    const float y {dist(prng)};
    std::array<dim, tensor<>::k_max_dims> shape {4, 4, 8, 3};
    auto ctx = isolate::create("test", isolate::compute_device::cpu, 0x1000<<3);
    tensor<float>* a = ctx->new_tensor<float>(shape);
    a->splat(x);
    tensor<float>* b = ctx->new_tensor<float>(shape);
    b->splat(y);
    tensor<float>* c = ctx->new_tensor<float>(shape);
    c->splat_zero();
    blas::t_f32_add(*c, *a, *b);
    for (dim i {}; i < shape[0]; ++i) {
        for (dim j {}; j < shape[1]; ++j) {
            for (dim k {}; k < shape[2]; ++k) {
                for (dim l {}; l < shape[3]; ++l) {
                    const float actual {(*c)({i, j, k, l})};
                    ASSERT_FLOAT_EQ(actual, x+y);
                }
            }
        }
    }
}

TEST(blas, tensor_sub) {
    std::mt19937_64 prng {};
    std::uniform_real_distribution<float> dist{-1.0f, 1.0f};
    const float x {dist(prng)};
    const float y {dist(prng)};
    std::array<dim, tensor<float>::k_max_dims> shape {4, 4, 8, 3};
    auto ctx = isolate::create("test", isolate::compute_device::cpu, 0x1000<<3);
    tensor<float>* a = ctx->new_tensor<float>(shape);
    a->splat(x);
    tensor<float>* b = ctx->new_tensor<float>(shape);
    b->splat(y);
    tensor<float>* c = ctx->new_tensor<float>(shape);
    c->splat_zero();
    blas::t_f32_sub(*c, *a, *b);
    for (dim i {}; i < shape[0]; ++i) {
        for (dim j {}; j < shape[1]; ++j) {
            for (dim k {}; k < shape[2]; ++k) {
                for (dim l {}; l < shape[3]; ++l) {
                    const float actual {(*c)({i, j, k, l})};
                    ASSERT_FLOAT_EQ(actual, x-y);
                }
            }
        }
    }
}

TEST(blas, tensor_mul) {
    std::mt19937_64 prng {};
    std::uniform_real_distribution<float> dist{-1.0f, 1.0f};
    const float x {dist(prng)};
    const float y {dist(prng)};
    std::array<dim, tensor<float>::k_max_dims> shape {4, 4, 8, 3};
    auto ctx = isolate::create("test", isolate::compute_device::cpu, 0x1000<<3);
    tensor<float>* a = ctx->new_tensor<float>(shape);
    a->splat(x);
    tensor<float>* b = ctx->new_tensor<float>(shape);
    b->splat(y);
    tensor<float>* c = ctx->new_tensor<float>(shape);
    c->splat_zero();
    blas::t_f32_mul(*c, *a, *b);
    for (dim i {}; i < shape[0]; ++i) {
        for (dim j {}; j < shape[1]; ++j) {
            for (dim k {}; k < shape[2]; ++k) {
                for (dim l {}; l < shape[3]; ++l) {
                    const float actual {(*c)({i, j, k, l})};
                    ASSERT_FLOAT_EQ(actual, x*y);
                }
            }
        }
    }
}

TEST(blas, tensor_div) {
    std::mt19937_64 prng {};
    std::uniform_real_distribution<float> dist{-1.0f, 1.0f};
    const float x {dist(prng)};
    const float y {dist(prng)};
    std::array<dim, tensor<float>::k_max_dims> shape {4, 4, 8, 3};
    auto ctx = isolate::create("test", isolate::compute_device::cpu, 0x1000<<3);
    tensor<float>* a = ctx->new_tensor<float>(shape);
    a->splat(x);
    tensor<float>* b = ctx->new_tensor<float>(shape);
    b->splat(y);
    tensor<float>* c = ctx->new_tensor<float>(shape);
    c->splat_zero();
    blas::t_f32_div(*c, *a, *b);
    for (dim i {}; i < shape[0]; ++i) {
        for (dim j {}; j < shape[1]; ++j) {
            for (dim k {}; k < shape[2]; ++k) {
                for (dim l {}; l < shape[3]; ++l) {
                    const float actual {(*c)({i, j, k, l})};
                    ASSERT_FLOAT_EQ(actual, x/y);
                }
            }
        }
    }
}

TEST(blas, tensor_matmul2) {
    constexpr std::size_t M {4}, N {16}, K {36};

    // matrix A (4 X 36)
    static constexpr std::array<float, M * K> A {
       2.0f, 9.0f, 2.0f, 10.0f, 6.0f, 4.0f, 3.0f, 6.0f, 3.0f, 6.0f, 9.0f, 7.0f, 8.0f, 8.0f, 3.0f, 3.0f, 10.0f, 5.0f, 2.0f, 10.0f, 7.0f, 10.0f, 9.0f, 3.0f, 6.0f, 6.0f, 5.0f, 10.0f, 2.0f, 3.0f, 6.0f, 1.0f, 9.0f, 4.0f, 10.0f, 4.0f,
        10.0f, 7.0f, 8.0f, 10.0f, 10.0f, 8.0f, 7.0f, 10.0f, 4.0f, 6.0f, 8.0f, 7.0f, 7.0f, 6.0f, 9.0f, 3.0f, 6.0f, 5.0f, 5.0f, 2.0f, 7.0f, 2.0f, 7.0f, 4.0f, 4.0f, 6.0f, 6.0f, 4.0f, 3.0f, 9.0f, 3.0f, 6.0f, 4.0f, 7.0f, 2.0f, 9.0f,
        7.0f, 3.0f, 2.0f, 5.0f, 7.0f, 3.0f, 10.0f, 2.0f, 6.0f, 1.0f, 4.0f, 7.0f, 5.0f, 10.0f, 3.0f, 10.0f, 4.0f, 5.0f, 5.0f, 1.0f, 6.0f, 10.0f, 7.0f, 4.0f, 5.0f, 3.0f, 9.0f, 9.0f, 8.0f, 6.0f, 9.0f, 2.0f, 3.0f, 6.0f, 8.0f, 5.0f,
        5.0f, 5.0f, 5.0f, 5.0f, 3.0f, 10.0f, 4.0f, 1.0f, 8.0f, 8.0f, 9.0f, 8.0f, 4.0f, 1.0f, 4.0f, 9.0f, 3.0f, 6.0f, 3.0f, 1.0f, 4.0f, 8.0f, 3.0f, 10.0f, 8.0f, 6.0f, 4.0f, 5.0f, 4.0f, 3.0f, 2.0f, 2.0f, 4.0f, 3.0f, 6.0f, 4.0f,
    };

    // matrix B (16 X 36)
   static constexpr std::array<float, N * K> B {
        9.0f, 7.0f, 1.0f, 3.0f, 5.0f, 9.0f, 7.0f, 6.0f, 1.0f, 10.0f, 1.0f, 1.0f, 7.0f, 2.0f, 4.0f, 9.0f, 10.0f, 4.0f, 5.0f, 5.0f, 7.0f, 1.0f, 7.0f, 7.0f, 2.0f, 9.0f, 5.0f, 10.0f, 7.0f, 4.0f, 8.0f, 9.0f, 9.0f, 3.0f, 10.0f, 2.0f,
        4.0f, 6.0f, 10.0f, 9.0f, 5.0f, 1.0f, 8.0f, 7.0f, 4.0f, 7.0f, 2.0f, 6.0f, 5.0f, 3.0f, 1.0f, 10.0f, 8.0f, 4.0f, 8.0f, 3.0f, 7.0f, 1.0f, 2.0f, 7.0f, 6.0f, 8.0f, 6.0f, 5.0f, 2.0f, 3.0f, 1.0f, 1.0f, 2.0f, 5.0f, 7.0f, 1.0f,
        8.0f, 2.0f, 8.0f, 8.0f, 8.0f, 8.0f, 4.0f, 4.0f, 6.0f, 10.0f, 10.0f, 9.0f, 2.0f, 9.0f, 3.0f, 7.0f, 7.0f, 1.0f, 4.0f, 9.0f, 1.0f, 2.0f, 3.0f, 6.0f, 1.0f, 10.0f, 5.0f, 8.0f, 9.0f, 4.0f, 6.0f, 2.0f, 3.0f, 1.0f, 2.0f, 7.0f,
        5.0f, 1.0f, 7.0f, 2.0f, 9.0f, 10.0f, 9.0f, 5.0f, 2.0f, 5.0f, 4.0f, 10.0f, 9.0f, 9.0f, 1.0f, 9.0f, 8.0f, 8.0f, 9.0f, 4.0f, 9.0f, 4.0f, 8.0f, 2.0f, 1.0f, 8.0f, 4.0f, 5.0f, 10.0f, 7.0f, 6.0f, 2.0f, 1.0f, 10.0f, 10.0f, 7.0f,
        9.0f, 4.0f, 5.0f, 9.0f, 5.0f, 10.0f, 10.0f, 3.0f, 6.0f, 6.0f, 4.0f, 4.0f, 4.0f, 8.0f, 5.0f, 4.0f, 9.0f, 1.0f, 9.0f, 9.0f, 1.0f, 7.0f, 9.0f, 2.0f, 10.0f, 9.0f, 10.0f, 8.0f, 3.0f, 3.0f, 9.0f, 3.0f, 9.0f, 10.0f, 1.0f, 8.0f,
        9.0f, 2.0f, 6.0f, 9.0f, 7.0f, 2.0f, 3.0f, 5.0f, 3.0f, 6.0f, 9.0f, 7.0f, 3.0f, 7.0f, 6.0f, 4.0f, 10.0f, 3.0f, 5.0f, 7.0f, 2.0f, 9.0f, 3.0f, 2.0f, 2.0f, 10.0f, 8.0f, 7.0f, 3.0f, 10.0f, 6.0f, 3.0f, 1.0f, 1.0f, 4.0f, 10.0f,
        2.0f, 9.0f, 2.0f, 10.0f, 6.0f, 4.0f, 3.0f, 6.0f, 3.0f, 6.0f, 9.0f, 7.0f, 8.0f, 8.0f, 3.0f, 3.0f, 10.0f, 5.0f, 2.0f, 10.0f, 7.0f, 10.0f, 9.0f, 3.0f, 6.0f, 6.0f, 5.0f, 10.0f, 2.0f, 3.0f, 6.0f, 1.0f, 9.0f, 4.0f, 10.0f, 4.0f,
        10.0f, 7.0f, 8.0f, 10.0f, 10.0f, 8.0f, 7.0f, 10.0f, 4.0f, 6.0f, 8.0f, 7.0f, 7.0f, 6.0f, 9.0f, 3.0f, 6.0f, 5.0f, 5.0f, 2.0f, 7.0f, 2.0f, 7.0f, 4.0f, 4.0f, 6.0f, 6.0f, 4.0f, 3.0f, 9.0f, 3.0f, 6.0f, 4.0f, 7.0f, 2.0f, 9.0f,
        7.0f, 3.0f, 2.0f, 5.0f, 7.0f, 3.0f, 10.0f, 2.0f, 6.0f, 1.0f, 4.0f, 7.0f, 5.0f, 10.0f, 3.0f, 10.0f, 4.0f, 5.0f, 5.0f, 1.0f, 6.0f, 10.0f, 7.0f, 4.0f, 5.0f, 3.0f, 9.0f, 9.0f, 8.0f, 6.0f, 9.0f, 2.0f, 3.0f, 6.0f, 8.0f, 5.0f,
        5.0f, 5.0f, 5.0f, 5.0f, 3.0f, 10.0f, 4.0f, 1.0f, 8.0f, 8.0f, 9.0f, 8.0f, 4.0f, 1.0f, 4.0f, 9.0f, 3.0f, 6.0f, 3.0f, 1.0f, 4.0f, 8.0f, 3.0f, 10.0f, 8.0f, 6.0f, 4.0f, 5.0f, 4.0f, 3.0f, 2.0f, 2.0f, 4.0f, 3.0f, 6.0f, 4.0f,
        6.0f, 2.0f, 3.0f, 3.0f, 3.0f, 7.0f, 5.0f, 1.0f, 8.0f, 1.0f, 4.0f, 5.0f, 1.0f, 1.0f, 6.0f, 4.0f, 2.0f, 1.0f, 7.0f, 8.0f, 6.0f, 1.0f, 1.0f, 5.0f, 6.0f, 5.0f, 10.0f, 6.0f, 7.0f, 5.0f, 9.0f, 3.0f, 2.0f, 7.0f, 9.0f, 4.0f,
        2.0f, 5.0f, 9.0f, 5.0f, 10.0f, 3.0f, 1.0f, 8.0f, 1.0f, 7.0f, 1.0f, 8.0f, 1.0f, 6.0f, 7.0f, 8.0f, 4.0f, 9.0f, 5.0f, 10.0f, 3.0f, 7.0f, 6.0f, 8.0f, 8.0f, 5.0f, 6.0f, 8.0f, 10.0f, 9.0f, 4.0f, 1.0f, 3.0f, 3.0f, 4.0f, 7.0f,
        8.0f, 2.0f, 6.0f, 6.0f, 5.0f, 1.0f, 3.0f, 7.0f, 1.0f, 7.0f, 2.0f, 2.0f, 2.0f, 8.0f, 4.0f, 1.0f, 1.0f, 5.0f, 9.0f, 4.0f, 1.0f, 2.0f, 3.0f, 10.0f, 1.0f, 4.0f, 9.0f, 9.0f, 6.0f, 8.0f, 8.0f, 1.0f, 9.0f, 10.0f, 4.0f, 1.0f,
        8.0f, 5.0f, 8.0f, 9.0f, 4.0f, 8.0f, 2.0f, 1.0f, 1.0f, 9.0f, 4.0f, 5.0f, 6.0f, 1.0f, 2.0f, 5.0f, 6.0f, 7.0f, 3.0f, 1.0f, 4.0f, 6.0f, 7.0f, 7.0f, 7.0f, 8.0f, 7.0f, 8.0f, 8.0f, 2.0f, 10.0f, 2.0f, 7.0f, 3.0f, 8.0f, 3.0f,
        8.0f, 7.0f, 6.0f, 2.0f, 4.0f, 10.0f, 10.0f, 6.0f, 10.0f, 3.0f, 7.0f, 6.0f, 4.0f, 3.0f, 5.0f, 5.0f, 5.0f, 3.0f, 8.0f, 10.0f, 3.0f, 4.0f, 8.0f, 4.0f, 2.0f, 6.0f, 8.0f, 9.0f, 6.0f, 9.0f, 4.0f, 3.0f, 5.0f, 2.0f, 2.0f, 6.0f,
        10.0f, 6.0f, 2.0f, 1.0f, 7.0f, 5.0f, 6.0f, 4.0f, 1.0f, 9.0f, 10.0f, 2.0f, 4.0f, 5.0f, 8.0f, 5.0f, 7.0f, 4.0f, 7.0f, 6.0f, 3.0f, 9.0f, 2.0f, 1.0f, 4.0f, 2.0f, 6.0f, 6.0f, 3.0f, 3.0f, 2.0f, 8.0f, 5.0f, 9.0f, 3.0f, 4.0f,
    };

    // matrix C (4 x 16)
    static constexpr std::array<float, M * N> result {
        1224.0f, 1023.0f, 1158.0f,1259.0f,1359.0f,1194.0f,1535.0f,1247.0f,1185.0f,1029.0f,889.0f,1182.0f,955.0f,1179.0f,1147.0f,1048.0f,
        1216.0f, 1087.0f, 1239.0f,1361.0f,1392.0f,1260.0f,1247.0f,1563.0f,1167.0f,1052.0f,942.0f,1214.0f,1045.0f,1134.0f,1264.0f,1126.0f,
        1125.0f, 966.0f, 1079.0f,1333.0f,1287.0f,1101.0f,1185.0f,1167.0f,1368.0f,990.0f,967.0f,1121.0f,971.0f,1086.0f,1130.0f,980.0f,
        999.0f, 902.0f, 1020.0f,1056.0f,1076.0f,929.0f,1029.0f,1052.0f,990.0f,1108.0f,823.0f,989.0f,759.0f,1041.0f,1003.0f,870.0f
    };

    auto ctx = isolate::create("test", isolate::compute_device::cpu, 0x1000<<3);
    tensor<float>* a = ctx->new_tensor<float>({M, K});
    std::memcpy(a->ptr(), A.data(), A.size() * sizeof(float));
    tensor<float>* b = ctx->new_tensor<float>({N, K});
    std::memcpy(b->ptr(), B.data(), B.size() * sizeof(float));
    tensor<float>* c = ctx->new_tensor<float>({M, N});
    c->splat_zero();
    blas::t_f32_matmul(*c, *a, *b);
    for(dim i {}; i < result.size(); ++i) {
        ASSERT_FLOAT_EQ((*c)(i), result[i]);
    }
}
