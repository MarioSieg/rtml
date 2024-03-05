include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
FetchContent_MakeAvailable(googletest)
enable_testing()
file(GLOB_RECURSE SOURCES *.cpp *.hpp)
add_executable(rtml_test ${SOURCES})
target_link_libraries(
    rtml_test
    rtml_runtime GTest::gtest_main
)
target_include_directories(rtml_test PRIVATE "../runtime")
include(GoogleTest)
gtest_discover_tests(rtml_test)
