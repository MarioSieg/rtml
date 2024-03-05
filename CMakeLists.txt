cmake_minimum_required(VERSION 3.10)

project(rtml LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(RTML_BUILD_TESTS "Build tests" ON)

add_subdirectory(runtime)

if (RTML_BUILD_TESTS)
    add_subdirectory(test)
endif()
