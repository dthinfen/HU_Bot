cmake_minimum_required(VERSION 3.16)
project(ares_hu_solver VERSION 1.0.0 LANGUAGES CXX)

# C++17 required
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -fsanitize=address,undefined")
endif()

# Options
option(ARES_BUILD_TESTS "Build unit tests" ON)
option(ARES_BUILD_BENCHMARKS "Build benchmarks" ON)
option(ARES_USE_OPENMP "Enable OpenMP parallelization" ON)
option(ARES_USE_LIBTORCH "Enable LibTorch for neural networks" OFF)

# Find OpenMP
if(ARES_USE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        message(STATUS "OpenMP found - enabling parallel CFR")
    else()
        message(WARNING "OpenMP not found - CFR will be single-threaded")
    endif()
endif()

# Find LibTorch (optional)
if(ARES_USE_LIBTORCH)
    find_package(Torch QUIET)
    if(Torch_FOUND)
        message(STATUS "LibTorch found - neural network support enabled")
    else()
        message(WARNING "LibTorch not found - neural features disabled")
        set(ARES_USE_LIBTORCH OFF)
    endif()
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

# Collect source files
file(GLOB_RECURSE CORE_SOURCES "src/core/*.cpp")
file(GLOB_RECURSE GAME_SOURCES "src/game/*.cpp")
file(GLOB_RECURSE CFR_SOURCES "src/cfr/*.cpp")
file(GLOB_RECURSE BELIEF_SOURCES "src/belief/*.cpp")
file(GLOB_RECURSE ABSTRACTION_SOURCES "src/abstraction/*.cpp")
file(GLOB_RECURSE SEARCH_SOURCES "src/search/*.cpp")
file(GLOB_RECURSE UTILS_SOURCES "src/utils/*.cpp")

# Main library
add_library(ares_solver STATIC
    ${CORE_SOURCES}
    ${GAME_SOURCES}
    ${CFR_SOURCES}
    ${BELIEF_SOURCES}
    ${ABSTRACTION_SOURCES}
    ${SEARCH_SOURCES}
    ${UTILS_SOURCES}
)

# Link OpenMP
if(ARES_USE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(ares_solver PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(ares_solver PUBLIC ARES_USE_OPENMP)
endif()

# Link LibTorch
if(ARES_USE_LIBTORCH AND Torch_FOUND)
    file(GLOB_RECURSE NEURAL_SOURCES "src/neural/*.cpp")
    target_sources(ares_solver PRIVATE ${NEURAL_SOURCES})
    target_link_libraries(ares_solver PUBLIC ${TORCH_LIBRARIES})
    target_compile_definitions(ares_solver PUBLIC ARES_USE_LIBTORCH)
endif()

# Main training executable
add_executable(train_cfr src/main_train.cpp)
target_link_libraries(train_cfr PRIVATE ares_solver)

# Agent executable (for playing) - TODO: create main_play.cpp
# add_executable(play_agent src/main_play.cpp)
# target_link_libraries(play_agent PRIVATE ares_solver)

# Tests
if(ARES_BUILD_TESTS)
    enable_testing()

    # Find or fetch GoogleTest
    find_package(GTest QUIET)
    if(NOT GTest_FOUND)
        include(FetchContent)
        FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG v1.14.0
        )
        FetchContent_MakeAvailable(googletest)
    endif()

    file(GLOB TEST_SOURCES "tests/*.cpp")
    add_executable(run_tests ${TEST_SOURCES})
    target_link_libraries(run_tests PRIVATE ares_solver GTest::gtest_main)

    include(GoogleTest)
    gtest_discover_tests(run_tests)
endif()

# Benchmarks
if(ARES_BUILD_BENCHMARKS)
    find_package(benchmark QUIET)
    if(benchmark_FOUND)
        file(GLOB BENCHMARK_SOURCES "benchmarks/*.cpp")
        add_executable(run_benchmarks ${BENCHMARK_SOURCES})
        target_link_libraries(run_benchmarks PRIVATE ares_solver benchmark::benchmark)
    else()
        message(STATUS "Google Benchmark not found - skipping benchmarks")
    endif()
endif()

# Installation
install(TARGETS ares_solver train_cfr
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/ DESTINATION include)

# Print configuration summary
message(STATUS "")
message(STATUS "=== ARES-HU Solver Configuration ===")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "  OpenMP: ${ARES_USE_OPENMP}")
message(STATUS "  LibTorch: ${ARES_USE_LIBTORCH}")
message(STATUS "  Tests: ${ARES_BUILD_TESTS}")
message(STATUS "  Benchmarks: ${ARES_BUILD_BENCHMARKS}")
message(STATUS "=====================================")
