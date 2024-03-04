set(SOURCES
    rtml.cpp
    rtml.hpp
    rtml_capi.cpp
    rtml_capi.h
)
add_compile_options(-march=native)
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-O0 -g)
else()
    add_compile_options(-Ofast)
endif()
add_compile_options(-fPIC)
add_library(rtml_runtime SHARED ${SOURCES})
add_subdirectory(extern)

add_executable(rtml_test rtml_tests.cpp)
target_link_libraries(rtml_test rtml_runtime)
