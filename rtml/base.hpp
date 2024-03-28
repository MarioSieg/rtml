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
    constexpr std::array<std::string_view, static_cast<std::size_t>(opcode::$count)> k_op_names {
        rtml_opcode_def(_, rtml_co)
    };
#undef _

    constexpr opcode k_first_binary_op {opcode::add};
    static_assert(k_operands[static_cast<std::size_t>(k_first_binary_op)] == 2);
    static_assert(k_operands[static_cast<std::size_t>(k_first_binary_op)-1] == 1);
    static_assert(k_operands[static_cast<std::size_t>(k_first_binary_op)+1] == 2);

    namespace dtypes {
        using f32 = float;
    }

    template <typename S>
    concept is_dtype = requires {
        !std::is_pointer_v<S>;
        !std::is_reference_v<S>;
        std::is_same_v<S, dtypes::f32>;
    };

    using dim = std::int64_t; // Dimension scalar used for dims, indices and strides.

    template <typename T> requires is_dtype<T>
    struct dtype_traits;

    template <>
    struct dtype_traits<float> {
        using type = float;
        static constexpr std::size_t k_size {sizeof(float)};
        static constexpr std::size_t k_align {alignof(float)};
        static constexpr std::string_view k_name {"f32"};
        static constexpr float k_one {1.0f};
    };

    template <typename T> requires is_dtype<T>
    class tensor;

     // all validation functions go here
    namespace validators {
    #define rtml_verify_un(expr, msg, ...) \
        if (!(expr)) [[unlikely]] { \
            rtml_log_error("Graph validation failed: " #expr "\t<-\t" msg, ## __VA_ARGS__); \
            if (r) rtml_log_error("R: {}", r->to_string(0)); \
            if (x) rtml_log_error("X: {}", x->to_string(0)); \
            return false; \
        }
        #define rtml_verify_bi(expr, msg, ...) \
            if (!(expr)) [[unlikely]] { \
                rtml_log_error("Graph validation failed: " #expr "\t<-\t" msg, ## __VA_ARGS__); \
                if (r) rtml_log_error("R: {}", r->to_string(0)); \
                if (x) rtml_log_error("X: {}", x->to_string(0)); \
                if (y) rtml_log_error("Y: {}", y->to_string(0)); \
                return false; \
            }

        template <typename S> requires is_dtype<S>
        [[nodiscard]] constexpr auto validate_unary_op(
            const tensor<S>* const r,
            const tensor<S>* const x
        ) -> bool {
            rtml_verify_un(r, "Result tensor is null");
            rtml_verify_un(x, "Source tensor is null");
            rtml_verify_un(x->shape().is_dense_except_dim1(), "Source tensor is not dense except dim1");
            rtml_verify_un(r->shape().is_dense_except_dim1(), "Result tensor is not dense except dim1");
            rtml_verify_un(r->shape() == x->shape(), "Result tensor shape mismatch");
            return true;
        }

        template <typename S> requires is_dtype<S>
        [[nodiscard]] constexpr auto validate_binary_op(
            const tensor<S>* const r,
            const tensor<S>* const x,
            const tensor<S>* const y
        ) -> bool {
            rtml_verify_bi(r, "R is null");
            rtml_verify_bi(x, "X is null");
            rtml_verify_bi(y, "Y is null");
            rtml_verify_bi(x->shape().strides()[0] == dtype_traits<S>::k_size, "X '{}' stride mismatch", x->name());
            rtml_verify_bi(r->shape().strides()[0] == dtype_traits<S>::k_size, "R '{}' stride mismatch", r->name());
            //rtml_verify_bi(y->shape().can_repeat_rows(x->shape()), "Y '{}' cannot repeat X '{}'", y->name(), x->name());
            rtml_verify_bi(x->shape() == r->shape(), "X '{}' shape mismatch with R '{}'", x->name(), r->name());
            return true;
        }

        template <typename S> requires is_dtype<S>
        [[nodiscard]] constexpr auto validate_matmul(
            const tensor<S>* const r,
            const tensor<S>* const x,
            const tensor<S>* const y
        ) -> bool {
            rtml_verify_bi(r, "R is null");
            rtml_verify_bi(x, "X 0 is null");
            rtml_verify_bi(y, "Y 1 is null");
            rtml_verify_bi(
                x->shape().is_matmul_compatible(y->shape()),
                "X 0 '{}' and Y '{}' are not matmul compatible",
                x->name(),
                y->name()
            );
            //rtml_verify_bi(!x->shape().is_transposed(), "X '{}' is transposed", x->name());
            return true;
        }

        #undef rtml_verify
    }
}

// Assert for debug and release builds.
#define rtml_assert(expr, msg, ...) \
    if (!(expr)) [[unlikely]] { \
        ::rtml::panic(::fmt::format("{}:{} Assertion failed: " #expr "\t<-\t" msg, __FILE__, __LINE__, ## __VA_ARGS__)); \
    }

#define rtml_assert1(expr) rtml_assert(expr, "Error")

// Assert for debug builds only.
#if defined(NDEBUG)
#    define rtml_dassert(expr, msg, ...)
#    define rtml_dassert1(expr)
#else
#    define rtml_dassert(expr, msg, ...) rtml_assert(expr, msg, ## __VA_ARGS__)
#    define rtml_dassert1(expr) rtml_assert1(expr)
#endif
