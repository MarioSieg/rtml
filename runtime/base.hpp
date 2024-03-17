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

#define RTML_LOG_ENABLE false

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
}

// Assert for debug and release builds.
#define rtml_assert(expr, msg, ...) \
    if (!(expr)) [[unlikely]] { \
        ::rtml::panic(::fmt::format("{}:{} Assertion failed: " #expr "\t<-\t" msg, __FILE__, __LINE__, ## __VA_ARGS__)); \
    }
