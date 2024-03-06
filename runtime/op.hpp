// Copyright Mario "Neo" Sieg 2024. All rights reserved. mario.sieg.64@gmail.com

#pragma once

#include <array>
#include <cstdint>
#include <string_view>
#include <span>

namespace rtml {
    class tensor;
}

namespace rtml::op {
    #define rtml_co ,
    /* Opcodes: Mnemonic, Operands, Info Mnemonic */
    #define rtml_opcode_def(_, __) \
        /* Nullary ops */\
        _(nop, 0, "nop")__\
        /* Unary ops */\
        _(softmax, 1, "softmax")__\
        _(sigmoid, 1, "sigmoid")__\
        _(tanh, 1, "tanh")__\
        _(relu, 1, "relu")__\
        _(gelu, 1, "gelu")__\
        /* Binary ops */\
        _(add , 2, "+")__\
        _(sub , 2, "-")__\
        _(mul , 2, "*")__\
        _(div , 2, "/")__\
        _(matmul, 2, "matmul")__

    #define _(mnemonic, operands, name) mnemonic
    enum opcode : std::uint32_t {
        rtml_opcode_def(_, rtml_co)
        $count
    };
    #undef _

    #define _(mnemonic, operands, name) ((operands)&0xff)
    static constexpr std::array<std::uint32_t, $count> k_operands {
        rtml_opcode_def(_, rtml_co)
    };
    #undef _

    #define _(mnemonic, operands, name) name
    static constexpr std::array<std::string_view, $count> k_names {
        rtml_opcode_def(_, rtml_co)
    };
    #undef _

    using validate_function = auto (const tensor* dst, std::span<const tensor*> src) -> bool;
    using eval_function = auto (tensor* dst, std::span<const tensor*> src) noexcept -> void;
}
