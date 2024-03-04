// Copyright Mario "Neo" Sieg 2024. All rights reserved. mario.sieg.64@gmail.com

#include "rtml_capi.h"
#include "rtml.hpp"

using namespace rtml;

extern "C" {
    auto rtml_global_init() -> bool {
        return context::global_init();
    }

    auto rtml_global_shutdown() -> void {
        context::global_shutdown();
    }

    auto rtml_context_create(
        const char* const name,
        const std::uint32_t device,
        const std::size_t memory_budged
    ) -> void {
        context::create(name, static_cast<context::compute_device>(device), memory_budged);
    }

    auto rtml_context_exists(const char* const name) -> bool {
        return context::exists(name);
    }
}
