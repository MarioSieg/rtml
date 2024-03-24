// Copyright Mario "Neo" Sieg 2024. All rights reserved. mario.sieg.64@gmail.com

#include <gtest/gtest.h>

#include <isolate.hpp>
#include <tensor.hpp>
#include <graph.hpp>
#include <net.hpp>

using namespace rtml;

TEST(net, with_ass) {
    std::shared_ptr a {isolate::create("alex", isolate::compute_device::cpu, 0x100000)};
    net xor_network {*a, {2, 2, 1}};
    xor_network.forward();


}
