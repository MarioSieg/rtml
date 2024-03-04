#include "rtml.hpp"

auto main () -> int {
    rtml::context::global_init();

    auto ctx = rtml::context::create("test", rtml::context::compute_device::cpu, 1024*1024*1024);
    auto tensor = ctx->create_tensor(rtml::tensor::stype::f32, {1, 1});

    rtml::context::global_shutdown();
    return 0;
}