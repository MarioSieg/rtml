project(rtml LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOURCES
    runtime/rtml.cpp
    runtime/rtml.hpp
    runtime/rtml_capi.cpp
    runtime/rtml_capi.h
)
add_compile_options(-march=native)
add_compile_options(-Ofast)
add_compile_options(-fPIC -rdynamic)
add_library(rtml_runtime SHARED ${SOURCES})
