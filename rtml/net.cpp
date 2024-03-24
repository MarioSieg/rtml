// Copyright Mario "Neo" Sieg 2024. All rights reserved. mario.sieg.64@gmail.com

#include "isolate.hpp"
#include "net.hpp"
#include "graph.hpp"

namespace rtml {
    net::net(isolate& ctx, std::vector<dim>&& layers) : m_layers{std::move(layers)} {
        m_weights.reserve(m_layers.size());
        m_biases.reserve(m_layers.size());
        m_data.reserve(m_layers.size() - 1);
        for (std::size_t i {}; i < m_layers.size()-1; ++i) {
            m_weights.emplace_back(ctx.new_tensor<dtypes::f32>({m_layers[i+1], m_layers[i]})->fill_random());
            m_biases.emplace_back(ctx.new_tensor<dtypes::f32>({m_layers[i+1], 1})-> fill_random());
        }
        build_forward_graph();
    }

    auto net::forward_propagate(tensor_ref<> input) const -> tensor_ref<> {

    }

    auto net::backward_propagate(tensor_ref<> outputs, tensor_ref<> targets) const -> void {

    }

    auto net::train(tensor_ref<> outputs, tensor_ref<> targets, const dim epochs, const float learning_rate) const -> void {
        const auto now {std::chrono::system_clock::now()};
        rtml_log_info("Training network with {} epochs and learning rate {}", epochs, learning_rate);
        for (dim i {}; i < epochs; ++i) {
            std::this_thread::sleep_for(std::chrono::microseconds{8});
            tensor_ref<> output {forward_propagate(output)};
            backward_propagate(output, targets);
            if (i % 1000 == 0) {
                const double percent {static_cast<double>(i) / static_cast<double>(epochs) * 100.0};
                rtml_log_info("Epoch {} of {} ({:.01f}%)", i, epochs, percent);
            }
        }
        rtml_log_info("Training took {} ms", std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(std::chrono::system_clock::now() - now).count());
    }

    auto net::build_forward_graph() -> void {

    }

    auto net::build_backward_graph() -> void {

    }
}