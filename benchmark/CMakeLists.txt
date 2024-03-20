add_subdirectory(gbench)
file(GLOB_RECURSE SOURCES src/*.cpp src/*.hpp)
add_executable(rtml_benchmark ${SOURCES})
target_link_libraries(
    rtml_benchmark
    rtml_runtime benchmark::benchmark
)
target_include_directories(rtml_benchmark PRIVATE "gbench/include")
target_include_directories(rtml_benchmark PRIVATE "../runtime")
