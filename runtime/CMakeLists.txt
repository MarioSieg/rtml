set(SOURCES
    rtml.cpp
    rtml.hpp
    rtml_capi.cpp
    rtml_capi.h
)
add_compile_options(-march=native)
add_compile_options(-Ofast)
add_compile_options(-fPIC)
add_library(rtml_runtime SHARED ${SOURCES})
add_subdirectory(extern)
