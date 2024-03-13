// Copyright Mario "Neo" Sieg 2024. All rights reserved. mario.sieg.64@gmail.com

#include <gtest/gtest.h>

#include <isolate.hpp>

using namespace rtml;

TEST(tensor, create_1d) {
    auto ctx = isolate::create("test", isolate::compute_device::cpu, 0x1000);
    tensor* tensor = ctx->create_tensor(tensor::dtype::f32, {25});
    ASSERT_EQ(tensor->get_num_dims(), 1);
    ASSERT_EQ(tensor->get_data_size(), 25*sizeof(float));
    ASSERT_EQ(tensor->get_dims()[0], 25);
    ASSERT_EQ(tensor->get_dims()[1], 1);
    ASSERT_EQ(tensor->get_dims()[2], 1);
    ASSERT_EQ(tensor->get_dims()[3], 1);
    ASSERT_EQ(tensor->get_strides()[0], sizeof(float));
    ASSERT_EQ(tensor->get_strides()[1], 25*sizeof(float));
    ASSERT_EQ(tensor->get_strides()[2], 25*sizeof(float));
    ASSERT_EQ(tensor->get_strides()[3], 25*sizeof(float));
}

TEST(tensor, create_2d) {
    auto ctx = isolate::create("test", isolate::compute_device::cpu, 0x1000);
    tensor* tensor = ctx->create_tensor(tensor::dtype::f32, {4, 4});
    ASSERT_EQ(tensor->get_num_dims(), 2);
    ASSERT_EQ(tensor->get_data_size(), 4*4*sizeof(float));
    ASSERT_EQ(tensor->get_dims()[0], 4);
    ASSERT_EQ(tensor->get_dims()[1], 4);
    ASSERT_EQ(tensor->get_dims()[2], 1);
    ASSERT_EQ(tensor->get_dims()[3], 1);
    ASSERT_EQ(tensor->get_strides()[0], sizeof(float));
    ASSERT_EQ(tensor->get_strides()[1], 4*sizeof(float));
    ASSERT_EQ(tensor->get_strides()[2], 4*4*sizeof(float));
    ASSERT_EQ(tensor->get_strides()[3], 4*4*sizeof(float));
}

TEST(tensor, create_3d) {
    auto ctx = isolate::create("test", isolate::compute_device::cpu, 0x1000);
    tensor* tensor = ctx->create_tensor(tensor::dtype::f32, {4, 4, 8});
    ASSERT_EQ(tensor->get_num_dims(), 3);
    ASSERT_EQ(tensor->get_data_size(), 4*4*8*sizeof(float));
    ASSERT_EQ(tensor->get_dims()[0], 4);
    ASSERT_EQ(tensor->get_dims()[1], 4);
    ASSERT_EQ(tensor->get_dims()[2], 8);
    ASSERT_EQ(tensor->get_dims()[3], 1);
    ASSERT_EQ(tensor->get_strides()[0], sizeof(float));
    ASSERT_EQ(tensor->get_strides()[1], 4*sizeof(float));
    ASSERT_EQ(tensor->get_strides()[2], 4*4*sizeof(float));
    ASSERT_EQ(tensor->get_strides()[3], 4*4*8*sizeof(float));
}

TEST(tensor, create_4d) {
    auto ctx = isolate::create("test", isolate::compute_device::cpu, 0x1000);
    tensor* tensor = ctx->create_tensor(tensor::dtype::f32, {4, 4, 8, 3});
    ASSERT_EQ(tensor->get_num_dims(), 4);
    ASSERT_EQ(tensor->get_data_size(), 4*4*8*3*sizeof(float));
    ASSERT_EQ(tensor->get_dims()[0], 4);
    ASSERT_EQ(tensor->get_dims()[1], 4);
    ASSERT_EQ(tensor->get_dims()[2], 8);
    ASSERT_EQ(tensor->get_dims()[3], 3);
    ASSERT_EQ(tensor->get_strides()[0], sizeof(float));
    ASSERT_EQ(tensor->get_strides()[1], 4*sizeof(float));
    ASSERT_EQ(tensor->get_strides()[2], 4*4*sizeof(float));
    ASSERT_EQ(tensor->get_strides()[3], 4*4*8*sizeof(float));
}
