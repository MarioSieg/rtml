add_subdirectory(gtest)
enable_testing()
file(GLOB_RECURSE SOURCES src/*.cpp src/*.hpp)
add_executable(rtml_test ${SOURCES})
target_link_libraries(
    rtml_test
    rtml_runtime GTest::gtest_main
)
target_include_directories(rtml_test PRIVATE "../rtml")
