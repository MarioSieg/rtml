// Copyright Mario "Neo" Sieg 2024. All rights reserved. mario.sieg.64@gmail.com

#include "capi.h"
#include "isolate.hpp"

using namespace rtml;

extern "C" {
    static_assert(std::is_same_v<tensor::id, rtml_tensor_id_t>);

    auto rtml_global_init() -> bool {
        return isolate::global_init();
    }

    auto rtml_global_shutdown() -> void {
        isolate::global_shutdown();
    }

    auto rtml_isolate_create(
        const char* const name,
        const std::uint32_t device,
        const std::size_t memory_budged
    ) -> void {
        isolate::create(name, static_cast<isolate::compute_device>(device), memory_budged);
    }

    auto rtml_isolate_exists(const char* const name) -> bool {
        return isolate::exists(name);
    }

    auto rtml_isolate_create_tensor(
        const char* const isolate_name,
        const std::uint32_t data_type,
        const std::int64_t d1, const std::int64_t d2,
        const std::int64_t d3, const std::int64_t d4,
        const std::uint32_t shape_len,
        const rtml_tensor_id_t view,
        const std::size_t slice_offset
    ) -> rtml_tensor_id_t {
        tensor::id id {};
        std::vector<std::int64_t> shape {d1};
        switch (shape_len) {
            case 1: break;
            case 2: shape.emplace_back(d2); break;
            case 3: shape.emplace_back(d2); shape.emplace_back(d3); break;
            case 4: shape.emplace_back(d2); shape.emplace_back(d3); shape.emplace_back(d4); break;
            default: assert(false);
        }
        if (!isolate::exists(isolate_name)) [[unlikely]] {
            rtml_log_error("context {} does not exist", isolate_name);
            std::abort();
        }
        isolate::get(isolate_name)->create_tensor(
            static_cast<tensor::dtype>(data_type),
            shape,
            view == 0 ? nullptr : isolate::get(isolate_name)->get_tensor(view),
            slice_offset,
            &id
        );
        return id;
    }

    auto rtml_tensor_print(const char* const isolate_name, const rtml_tensor_id_t id) -> const char* {
        if (!isolate::exists(isolate_name)) [[unlikely]] {
            rtml_log_error("context {} does not exist", isolate_name);
            std::abort();
        }
        static std::string proxy;
        proxy = isolate::get(isolate_name)->get_tensor(id)->print();
        return proxy.c_str();
    }
}
