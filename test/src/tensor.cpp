// Copyright Mario "Neo" Sieg 2024. All rights reserved. mario.sieg.64@gmail.com

#include <gtest/gtest.h>

#include <isolate.hpp>
#include <tensor.hpp>

using namespace rtml;

TEST(tensor, create_1d) {
    auto ctx = isolate::create("test", isolate::compute_device::cpu, 0x1000);
    tensor<float>* tensor = ctx->new_tensor<float>({25});
    ASSERT_EQ(tensor->rank(), 1);
    ASSERT_EQ(tensor->size(), 25*sizeof(float));
    ASSERT_EQ(tensor->data().size(), 25);
    ASSERT_EQ(tensor->shape()[0], 25);
    ASSERT_EQ(tensor->shape()[1], 1);
    ASSERT_EQ(tensor->shape()[2], 1);
    ASSERT_EQ(tensor->shape()[3], 1);
    ASSERT_EQ(tensor->strides()[0], sizeof(float));
    ASSERT_EQ(tensor->strides()[1], 25*sizeof(float));
    ASSERT_EQ(tensor->strides()[2], 25*sizeof(float));
    ASSERT_EQ(tensor->strides()[3], 25*sizeof(float));
}

TEST(tensor, create_2d) {
    auto ctx = isolate::create("test", isolate::compute_device::cpu, 0x1000);
    tensor<float>* tensor = ctx->new_tensor<float>({4, 4});
    ASSERT_EQ(tensor->rank(), 2);
    ASSERT_EQ(tensor->size(), 4*4*sizeof(float));
    ASSERT_EQ(tensor->data().size(), 4*4);
    ASSERT_EQ(tensor->shape()[0], 4);
    ASSERT_EQ(tensor->shape()[1], 4);
    ASSERT_EQ(tensor->shape()[2], 1);
    ASSERT_EQ(tensor->shape()[3], 1);
    ASSERT_EQ(tensor->strides()[0], sizeof(float));
    ASSERT_EQ(tensor->strides()[1], 4*sizeof(float));
    ASSERT_EQ(tensor->strides()[2], 4*4*sizeof(float));
    ASSERT_EQ(tensor->strides()[3], 4*4*sizeof(float));
}

TEST(tensor, create_3d) {
    auto ctx = isolate::create("test", isolate::compute_device::cpu, 0x1000);
    tensor<float>* tensor = ctx->new_tensor<float>({4, 4, 8});
    ASSERT_EQ(tensor->rank(), 3);
    ASSERT_EQ(tensor->size(), 4*4*8*sizeof(float));
    ASSERT_EQ(tensor->data().size(), 4*4*8);
    ASSERT_EQ(tensor->shape()[0], 4);
    ASSERT_EQ(tensor->shape()[1], 4);
    ASSERT_EQ(tensor->shape()[2], 8);
    ASSERT_EQ(tensor->shape()[3], 1);
    ASSERT_EQ(tensor->strides()[0], sizeof(float));
    ASSERT_EQ(tensor->strides()[1], 4*sizeof(float));
    ASSERT_EQ(tensor->strides()[2], 4*4*sizeof(float));
    ASSERT_EQ(tensor->strides()[3], 4*4*8*sizeof(float));
}

TEST(tensor, create_4d) {
    auto ctx = isolate::create("test", isolate::compute_device::cpu, 0x1000);
    tensor<float>* tensor = ctx->new_tensor<float>({4, 4, 8, 3});
    ASSERT_EQ(tensor->rank(), 4);
    ASSERT_EQ(tensor->size(), 4*4*8*3*sizeof(float));
    ASSERT_EQ(tensor->data().size(), 4*4*8*3);
    ASSERT_EQ(tensor->shape()[0], 4);
    ASSERT_EQ(tensor->shape()[1], 4);
    ASSERT_EQ(tensor->shape()[2], 8);
    ASSERT_EQ(tensor->shape()[3], 3);
    ASSERT_EQ(tensor->strides()[0], sizeof(float));
    ASSERT_EQ(tensor->strides()[1], 4*sizeof(float));
    ASSERT_EQ(tensor->strides()[2], 4*4*sizeof(float));
    ASSERT_EQ(tensor->strides()[3], 4*4*8*sizeof(float));
}
