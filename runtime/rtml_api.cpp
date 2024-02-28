// Copyright Mario "Neo" Sieg 2024. All rights reserved. mario.sieg.64@gmail.com

#include "rtml.hpp"

#ifdef _MSVC_VER
#define RTML_API __declspec(dllexport)
#else
#define RTML_API __attribute__((visibility("default")))
#endif

using namespace rtml;

extern "C" RTML_API auto rtml_global_init() -> bool {
    return context::global_init();
}

extern "C" RTML_API auto rtml_global_shutdown() -> void {
    context::global_shutdown();
}

extern "C" RTML_API auto rtml_context_create(
    const char* const name,
    const context::compute_device device,
    const std::size_t memory_budged
) -> void {
    context::create(name, device, memory_budged);
}

extern "C" RTML_API auto rtml_context_exists(const char* const name) -> bool {
    return context::exists(name);
}
