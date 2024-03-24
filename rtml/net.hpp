// Copyright Mario "Neo" Sieg 2024. All rights reserved. mario.sieg.64@gmail.com

#pragma once

#include "tensor.hpp"
#include "isolate.hpp"

namespace rtml {
    class net final {
    public:
        explicit net(isolate& ctx, std::vector<dim>&& arch);
        net(const net&) = delete;
        net(net&&) = delete;
        auto operator=(const net&) -> net& = delete;
        auto operator=(net&&) -> net& = delete;
        ~net() = default;

        auto forward() const -> void;

    private:
        std::vector<dim> m_arch {};
        std::vector<tensor_ref<>> m_weights {};
        std::vector<tensor_ref<>> m_biases {};
        std::vector<tensor_ref<>> m_ass {};
    };
}
