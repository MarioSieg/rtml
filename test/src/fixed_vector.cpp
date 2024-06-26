// Copyright Mario "Neo" Sieg 2024. All rights reserved. mario.sieg.64@gmail.com

#include <gtest/gtest.h>

#include <fixed_vector.hpp>

using namespace rtml;

TEST(fixed_vector, basic) {
    fixed_vector<int, 4> vec {};
    static_assert(sizeof(vec) == sizeof(int) * 4 + sizeof(std::size_t));
    ASSERT_TRUE(vec.empty());
    ASSERT_FALSE(vec.full());
    vec.emplace_back(1);
    vec.emplace_back(2);
    vec.emplace_back(3);
    vec.emplace_back(4);
    ASSERT_FALSE(vec.empty());
    ASSERT_TRUE(vec.full());
    ASSERT_EQ(vec.size(), 4);
    ASSERT_EQ(vec[0], 1);
    ASSERT_EQ(vec[1], 2);
    ASSERT_EQ(vec[2], 3);
    ASSERT_EQ(vec[3], 4);
    const std::span<int> span {vec};
    ASSERT_EQ(span.size(), 4);
    ASSERT_EQ(span[0], 1);
    ASSERT_EQ(span[1], 2);
    ASSERT_EQ(span[2], 3);
    ASSERT_EQ(span[3], 4);
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

