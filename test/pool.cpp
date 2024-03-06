#include <gtest/gtest.h>

#include <isolate.hpp>

using namespace rtml;

TEST(pool, new) {
    pool p {0xff};
    ASSERT_EQ(p.num_allocs(), 0);
    ASSERT_EQ(p.size(), 0xff);
    ASSERT_EQ(p.data()+0xff, p.needle());
}

TEST(pool, alloc_unaligned) {
    pool p {0xff};
    ASSERT_EQ(p.num_allocs(), 0);
    ASSERT_EQ(p.size(), 0xff);
    ASSERT_EQ(p.bytes_allocated(), 0);
    int* a {static_cast<int*>(p.alloc_raw(sizeof(int)))};
    ASSERT_EQ(p.num_allocs(), 1);
    ASSERT_EQ(p.size(), 0xff);
    ASSERT_EQ(p.bytes_allocated(), sizeof(int));
    ASSERT_EQ(p.data()+0xff-sizeof(int), p.needle());
    *a = 0xdeadbeef;
    ASSERT_EQ(*a, 0xdeadbeef);
}

TEST(pool, alloc_aligned) {
    pool p {0xff};
    ASSERT_EQ(p.num_allocs(), 0);
    ASSERT_EQ(p.size(), 0xff);
    ASSERT_EQ(p.bytes_allocated(), 0);
    int* a {static_cast<int*>(p.alloc_raw(sizeof(int), 32))};
    ASSERT_EQ(p.num_allocs(), 1);
    ASSERT_EQ(p.size(), 0xff);
    ASSERT_EQ(p.bytes_allocated(), 32+sizeof(int)-1);
    ASSERT_EQ(p.data()+0xff-(32+sizeof(int)-1), p.needle());
    ASSERT_EQ(std::bit_cast<std::uintptr_t>(a) % 32, 0);
    *a = 0xdeadbeef;
    ASSERT_EQ(*a, 0xdeadbeef);
}

TEST(pool, alloc_type) {
    static constinit bool called = false;
    called = false;
    struct alignas(64) test {
        test() { called = true; }
        const int vv {-128};
    };
    pool p {0xff};
    ASSERT_EQ(p.num_allocs(), 0);
    ASSERT_EQ(p.size(), 0xff);
    ASSERT_EQ(p.bytes_allocated(), 0);
    test* a {p.alloc<test>()};
    ASSERT_TRUE(called);
    ASSERT_EQ(a->vv, -128);
    ASSERT_EQ(p.num_allocs(), 1);
    ASSERT_EQ(p.size(), 0xff);
    ASSERT_EQ(p.bytes_allocated(), sizeof(test)+alignof(test)-1);
    ASSERT_EQ(std::bit_cast<std::uintptr_t>(a) % alignof(test), 0);
}
