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

        [[nodiscard]] auto forward_propagate(tensor<>* inputs) -> tensor<>*;
        auto backward_propagate(tensor<>* outputs, tensor<>* targets, float learning_rate) -> void;
        auto train(std::span<tensor<>* const> inputs, std::span<tensor<>* const> targets, std::size_t epochs, float learning_rate) -> void;

    private:
        auto build_forward_graph() -> void;
        auto build_backward_graph() -> void;

        [[maybe_unused]] isolate& m_ctx;
        std::vector<dim> m_layers {};
        std::vector<tensor<>*> m_weights {};
        std::vector<tensor<>*> m_biases {};
        std::vector<tensor<>*> m_data {};
    };
}
