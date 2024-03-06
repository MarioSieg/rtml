// Copyright Mario "Neo" Sieg 2024. All rights reserved. mario.sieg.64@gmail.com

#include <gtest/gtest.h>

#include <fixed_vector.hpp>

using namespace rtml;

TEST(fixed_vector, basic) {
    fixed_vector<int, 4> vec;
    vec.emplace_back(1);
    vec.emplace_back(2);
    vec.emplace_back(3);
    vec.emplace_back(4);
    ASSERT_EQ(vec.size(), 4);
    ASSERT_EQ(vec[0], 1);
    ASSERT_EQ(vec[1], 2);
    ASSERT_EQ(vec[2], 3);
    ASSERT_EQ(vec[3], 4);
    vec.clear();
    ASSERT_EQ(vec.size(), 0);
}

TEST(fixed_vector, iterator) {
    fixed_vector<int, 4> vec;
    vec.emplace_back(1);
    vec.emplace_back(2);
    vec.emplace_back(3);
    vec.emplace_back(4);
    int i = 1;
    for (const auto& v : vec) {
        ASSERT_EQ(v, i++);
    }
    i = 1;
    for (auto it = vec.begin(); it != vec.end(); ++it) {
        ASSERT_EQ(*it, i++);
    }
    i = 4;
    for (auto it = vec.rbegin(); it != vec.rend(); ++it) {
        ASSERT_EQ(*it, i--);
    }
    i = 4;
    for (auto it = vec.crbegin(); it != vec.crend(); ++it) {
        ASSERT_EQ(*it, i--);
    }
}

