// Copyright Mario "Neo" Sieg 2024. All rights reserved. mario.sieg.64@gmail.com

#pragma once

#include <span>
#include <string>
#include <cassert>

#include <spdlog/spdlog.h> // For logging and fmt::format but can be removed later

#ifdef _MSC_VER
#    define RTML_AINLINE __forceinline
#    define RTML_COLD
#    define RTML_HOT
#    define RTML_EXPORT __declspec(dllexport)
#else
#    define RTML_AINLINE __attribute__((always_inline))
#    define RTML_COLD __attribute__((cold))
#    define RTML_HOT __attribute__((hot))
#    define RTML_EXPORT __attribute__((visibility("default")))
#endif

#define RTML_LOG_ENABLE true

#if RTML_LOG_ENABLE
#    define rtml_log_info SPDLOG_INFO
#    define rtml_log_warn SPDLOG_WARN
#    define rtml_log_error SPDLOG_ERROR
#else
#    define rtml_log_info(...)
#    define rtml_log_warn(...)
#    define rtml_log_error(...)
#endif

#define RTML_CCRED "\x1b[31m"
#define RTML_CCGREEN "\x1b[32m"
#define RTML_CCYELLOW "\x1b[33m"
#define RTML_CCBLUE "\x1b[34m"
#define RTML_CCMAGENTA "\x1b[35m"
#define RTML_CCCYAN "\x1b[36m"
#define RTML_CCRESET "\x1b[0m"

namespace rtml {
    [[noreturn]] extern auto RTML_COLD panic(std::string_view msg) -> void;

    constexpr auto operator ""_kib(const unsigned long long int x) noexcept -> unsigned long long int  {
        return x << 10;
    }
    constexpr auto operator ""_mib(const unsigned long long int x) noexcept -> unsigned long long int  {
        return x << 20;
    }
    constexpr auto operator ""_gib(const unsigned long long int x) noexcept -> unsigned long long int  {
        return x << 30;
    }

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
    _(silu, 1, "silu")__\
    /* Binary ops */\
    _(add , 2, "+")__\
    _(sub , 2, "-")__\
    _(mul , 2, "*")__\
    _(div , 2, "/")__\
    _(matmul, 2, "matmul")__

#define _(mnemonic, operands, name) mnemonic
    enum class opcode : std::uint32_t {
        rtml_opcode_def(_, rtml_co)
        $count
    };
#undef _

#define _(mnemonic, operands, name) ((operands)&0xff)
    constexpr std::array<std::uint32_t, static_cast<std::size_t>(opcode::$count)> k_operands {
        rtml_opcode_def(_, rtml_co)
    };
#undef _

#define _(mnemonic, operands, name) name
    constexpr std::array<std::string_view, static_cast<std::size_t>(opcode::$count)> k_names {
        rtml_opcode_def(_, rtml_co)
    };
#undef _

    constexpr opcode k_first_binary_op {opcode::add};
    static_assert(k_operands[static_cast<std::size_t>(k_first_binary_op)] == 2);
    static_assert(k_operands[static_cast<std::size_t>(k_first_binary_op)-1] == 1);
    static_assert(k_operands[static_cast<std::size_t>(k_first_binary_op)+1] == 2);
}

// Assert for debug and release builds.
#define rtml_assert(expr, msg, ...) \
    if (!(expr)) [[unlikely]] { \
        ::rtml::panic(::fmt::format("{}:{} Assertion failed: " #expr "\t<-\t" msg, __FILE__, __LINE__, ## __VA_ARGS__)); \
    }

#define rtml_assert1(expr) rtml_assert(expr, "")

// Assert for debug builds only.
#if defined(NDEBUG)
#    define rtml_dassert(expr, msg, ...)
#    define rtml_dassert1(expr)
#else
#    define rtml_dassert(expr, msg, ...) rtml_assert(expr, msg, ## __VA_ARGS__)
#    define rtml_dassert1(expr) rtml_assert1(expr)
#endif
