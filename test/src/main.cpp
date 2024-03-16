#include <gtest/gtest.h>
#include <isolate.hpp>

auto main(int argc, char** argv) -> int {
    if (!rtml::isolate::init_rtml_runtime()) {
        return -1;
    }
    testing::InitGoogleTest(&argc, argv);
    int r = RUN_ALL_TESTS();
    rtml::isolate::shutdown_rtml_runtime();
    return r;
}
