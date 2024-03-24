// Copyright Mario "Neo" Sieg 2024. All rights reserved. mario.sieg.64@gmail.com

#pragma once

#include "tensor.hpp"
#include "isolate.hpp"

namespace rtml {
    class net final {
    public:
        explicit net(isolate& ctx, std::vector<dim>&& layers);
        net(const net&) = delete;
        net(net&&) = delete;
        auto operator=(const net&) -> net& = delete;
        auto operator=(net&&) -> net& = delete;
        ~net() = default;

        [[nodiscard]] auto forward_propagate(tensor_ref<> input) const -> tensor_ref<>;
        auto backward_propagate(tensor_ref<> outputs, tensor_ref<> targets) const -> void;
        auto train(tensor_ref<> outputs, tensor_ref<> targets, dim epochs, float learning_rate) const -> void;

    private:
        auto build_forward_graph() -> void;
        auto build_backward_graph() -> void;

        std::vector<dim> m_layers {};
        std::vector<tensor_ref<>> m_weights {};
        std::vector<tensor_ref<>> m_biases {};
        std::vector<tensor_ref<>> m_data {};
    };
}
