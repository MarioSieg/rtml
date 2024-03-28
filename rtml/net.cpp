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

    auto net::forward_propagate(const std::span<const dtypes::f32> inputs) -> std::span<const dtypes::f32> {
        tensor<>* current {m_ctx.new_tensor<dtypes::f32>({1, static_cast<dim>(inputs.size())})->transposed_clone()};
        current->fill_data(inputs);
        for (std::size_t i {}; i < m_layers.size()-1; ++i) {
            current = m_weights[i]->clone()->matmul_clone(current)->add(m_biases[i])->sigmoid();
            m_data.emplace_back(current);
        }
        return current->transposed_clone()->data();
    }

    auto net::backward_propagate([[maybe_unused]] tensor<>* outputs, [[maybe_unused]] tensor<>* targets) -> void {

    }

    auto net::train([[maybe_unused]] tensor<>* outputs, [[maybe_unused]] tensor<>* targets, [[maybe_unused]] const dim epochs, [[maybe_unused]] const float learning_rate) -> void {
        //const auto now {std::chrono::system_clock::now()};
        //rtml_log_info("Training network with {} epochs and learning rate {}", epochs, learning_rate);
        //for (dim i {}; i < epochs; ++i) {
        //    tensor_ref<> output {forward_propagate(outputs)};
        //    backward_propagate(output, targets);
        //    if (i % 1000 == 0) {
        //        const double percent {static_cast<double>(i) / static_cast<double>(epochs) * 100.0};
        //        rtml_log_info("Epoch {} of {} ({:.01f}%)", i, epochs, percent);
        //    }
        //}
        //rtml_log_info("Training took {} ms", std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(std::chrono::system_clock::now() - now).count());
    }

    auto net::build_forward_graph() -> void {

    }

    auto net::build_backward_graph() -> void {

    }
}
