// Copyright Mario "Neo" Sieg 2024. All rights reserved. mario.sieg.64@gmail.com

#include "base.hpp"

#include <iostream>

namespace rtml {
    auto panic(const std::string_view msg) -> void {
        spdlog::default_logger_raw()->flush();
        spdlog::shutdown();
        std::cerr << (RTML_CCRED "!! RTML runtime panic !!\n" RTML_CCRESET) << msg << std::endl;
        std::abort();
    }
}
