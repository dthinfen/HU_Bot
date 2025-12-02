cmake_minimum_required(VERSION 3.14)
project(cpp_vec_env)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")

# Allow specifying Python executable
if(DEFINED ENV{PYTHON_EXECUTABLE})
    set(Python3_EXECUTABLE $ENV{PYTHON_EXECUTABLE})
endif()

# Find Python and pybind11
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG)

if(NOT pybind11_FOUND)
    # Try to find pybind11 via pip
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
        OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(PYBIND11_CMAKE_DIR)
        set(pybind11_DIR ${PYBIND11_CMAKE_DIR})
        find_package(pybind11 CONFIG REQUIRED)
    else()
        message(FATAL_ERROR "pybind11 not found. Install with: pip install pybind11")
    endif()
endif()

# Source files
set(SOURCES
    src/game_state.cpp
    src/bindings.cpp
)

# Create Python module
pybind11_add_module(cpp_vec_env ${SOURCES})

target_include_directories(cpp_vec_env PRIVATE src)

# Copy to parent directory for easy import
add_custom_command(TARGET cpp_vec_env POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    $<TARGET_FILE:cpp_vec_env>
    ${CMAKE_SOURCE_DIR}/../cpp_vec_env$<TARGET_FILE_SUFFIX:cpp_vec_env>
)
