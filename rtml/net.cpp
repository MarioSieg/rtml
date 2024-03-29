// Copyright Mario "Neo" Sieg 2024. All rights reserved. mario.sieg.64@gmail.com

#include "isolate.hpp"
#include "net.hpp"
#include "graph.hpp"

namespace rtml {
    net::net(isolate& ctx, std::vector<dim>&& layers) : m_ctx{ctx}, m_layers{std::move(layers)} {
        m_weights.reserve(m_layers.size()-1);
        m_biases.reserve(m_layers.size()-1);
        m_data.reserve(m_layers.size()-1);
        for (std::size_t i {}; i < m_layers.size()-1; ++i) {
            m_weights.emplace_back(ctx.new_tensor<dtypes::f32>({m_layers[i+1], m_layers[i]})->fill_random());
            m_biases.emplace_back(ctx.new_tensor<dtypes::f32>({m_layers[i+1], 1})->fill_random());
        }
        build_forward_graph();
    }

    auto net::forward_propagate(tensor<>* const inputs) -> tensor<>* {
        tensor<>* current {inputs->transposed_clone()};
        m_data.clear();
        m_data.emplace_back(current);
        for (std::size_t i {}; i < m_layers.size()-1; ++i) {
            current = m_weights[i]->matmul_clone(current)->add(m_biases[i])->sigmoid();
            m_data.emplace_back(current);
        }
        tensor<>* r {current->clone()};
        rtml_dassert1(r->shape().is_scalar());
        return r;
    }

    auto net::backward_propagate(tensor<>* const outputs, tensor<>* const targets, const float learning_rate) -> void {
        rtml_assert1(targets->shape().is_vector() && targets->shape().col_count() == m_layers.back());
        rtml_assert1(outputs->shape().is_vector());
        tensor<>* const parsed {outputs->clone()};
        tensor<>* errors {targets->clone()->sub(parsed)};
        tensor<>* gradients {parsed->clone()->sigmoid_derivative()};
        for (std::size_t i {m_layers.size()-1}; i --> 0; ) {
            gradients = gradients->mul(errors)->mul(gradients->isomorphic_clone()->fill(learning_rate));
            m_weights[i] = m_weights[i]->clone()->add(gradients->matmul_clone(m_data[i]->transposed_clone()));
            m_biases[i] = m_biases[i]->clone()->add(gradients);
            errors = m_weights[i]->transposed_clone()->matmul_clone(errors);
            gradients = m_data[i]->clone()->sigmoid_derivative();
        }
    }

    auto net::train(const std::span<tensor<>* const> inputs, const std::span<tensor<>* const> targets, const std::size_t epochs, const float learning_rate) -> void {
        rtml_assert1(inputs.size() == targets.size());
        const auto now {std::chrono::system_clock::now()};
        rtml_log_info("Training network with {} epochs and learning rate {}", epochs, learning_rate);
        for (std::size_t e {}; e < epochs; ++e) {
            for (std::size_t i {}; i < inputs.size(); ++i) {
                auto* const outputs {forward_propagate(inputs[i])};
                backward_propagate(outputs, targets[i], learning_rate);
            }
        }
        rtml_log_info("Training took {} ms", std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(std::chrono::system_clock::now() - now).count());
    }

    auto net::build_forward_graph() -> void {

    }

    auto net::build_backward_graph() -> void {

    }
}
