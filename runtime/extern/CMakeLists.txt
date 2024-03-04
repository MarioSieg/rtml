cmake_policy(SET CMP0079 NEW)

add_subdirectory(spdlog)
target_include_directories(rtml_runtime PUBLIC spdlog/include)
target_link_libraries(rtml_runtime PUBLIC spdlog::spdlog)
