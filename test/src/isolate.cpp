// Copyright Mario "Neo" Sieg 2024. All rights reserved. mario.sieg.64@gmail.com

#include <gtest/gtest.h>

#include <isolate.hpp>

using namespace rtml;

TEST(isolate, create) {
    auto ctx = isolate::create("test", isolate::compute_device::cpu, 0x1000);
    ASSERT_NE(ctx, nullptr);
    ASSERT_EQ(ctx->name(), "test");
    ASSERT_EQ(ctx->device(), isolate::compute_device::cpu);
    ASSERT_EQ(ctx->pool().size(), 0x1000);
}
