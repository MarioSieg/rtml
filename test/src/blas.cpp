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
    blas::compute_ctx cctx {};
    blas::t_f32_add(cctx, *c, *a, *b);
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
    blas::compute_ctx cctx {};
    blas::t_f32_sub(cctx, *c, *a, *b);
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
    blas::compute_ctx cctx {};
    blas::t_f32_mul(cctx, *c, *a, *b);
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
    blas::compute_ctx cctx {};
    blas::t_f32_div(cctx, *c, *a, *b);
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
    constexpr std::size_t M {4}, N {4}, K {4};

    // matrix A (4 X 36)
    static constexpr std::array<float, M * K> A {
       2.0f, 9.0f, 2.0f, 10.0f,
        6.0f, 4.0f, 3.0f, 6.0f,
        3.0f, 6.0f, 9.0f, 7.0f,
        8.0f, 8.0f, 3.0f, 3.0f
    };

    // matrix B (16 X 36)
   static constexpr std::array<float, N * K> B {
        9.0f, 7.0f, 1.0f, 3.0f,
       5.0f, 9.0f, 7.0f, 6.0f,
       1.0f, 10.0f, 1.0f, 1.0f,
       7.0f, 2.0f, 4.0f, 9.0f
    };

    // matrix C (4 x 16)
    static constexpr std::array<float, M * N> result {
        135., 135., 107., 152.,
        119., 120.,  61.,  99.,
        115., 179.,  82., 117.,
        136., 164.,  79., 102.
    };

    auto ctx = isolate::create("test", isolate::compute_device::cpu, 0x1000<<3);
    tensor<float>* a = ctx->new_tensor<float>({M, K});
    std::memcpy(a->ptr(), A.data(), A.size() * sizeof(float));
    tensor<float>* b = ctx->new_tensor<float>({N, K});
    std::memcpy(b->ptr(), B.data(), B.size() * sizeof(float));
    tensor<float>* c = ctx->new_tensor<float>({M, N});
    c->splat_zero();
    blas::compute_ctx cctx {};
    blas::t_f32_matmul(cctx, *c, *a, *b);
    c->print();
    for(dim i {}; i < result.size(); ++i) {
        ASSERT_FLOAT_EQ((*c)(i), result[i]);
    }
}
