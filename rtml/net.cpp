// Copyright Mario "Neo" Sieg 2024. All rights reserved. mario.sieg.64@gmail.com

#include "isolate.hpp"
#include "net.hpp"

#include "graph.hpp"

namespace rtml {
    net::net(isolate& ctx, std::vector<dim>&& arch) : m_arch{std::move(arch)} {
        m_weights.resize(m_arch.size()-1);
        m_biases.resize(m_arch.size()-1);
        m_ass.resize(m_arch.size());
        m_ass[0] = ctx.new_tensor<dtypes::f32>({1, m_arch[0]});
        for (std::size_t i {1}; i < m_arch.size(); ++i) {
            m_weights[i-1] = ctx.new_tensor<dtypes::f32>({m_ass[i-1]->col_count(), m_arch[i]})
                ->splat_zero()
                ->set_name(fmt::format("weight {}", i));
            m_biases[i-1] = ctx.new_tensor<dtypes::f32>({1, m_arch[i]})
                ->splat_zero()
                ->set_name(fmt::format("bias {}", i));
            m_ass[i] = ctx.new_tensor<dtypes::f32>({1, m_arch[i]})
                ->splat_zero()
                ->set_name(fmt::format("ass {}", i));
        }
        for (std::size_t i {}; i < m_arch.size()-1; ++i) {
            m_ass[i+1] = (m_ass[i] & m_weights[i])->set_name(fmt::format("ass @ weight"));
            m_ass[i+1] = (m_ass[i+1] + m_biases[i])->set_name(fmt::format("ass + bias"));
            m_ass[i+1] = m_ass[i+1].sigmoid()->set_name(fmt::format("result {}"));
            graph::generate_graphviz_dot_code(&*m_ass[i+1]);
        }
    }

    auto net::forward() const -> void {
        for (std::size_t i {}; i < m_arch.size()-1; ++i) {
            graph::compute(&*m_ass[i+1]);
        }
    }
}
