#!/bin/bash
# Build script for ARES-HU C++ solver

set -e

# Directory setup
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
SRC_DIR="$SCRIPT_DIR/src"
INCLUDE_DIR="$SCRIPT_DIR/include"
TESTS_DIR="$SCRIPT_DIR/tests"

# Compiler settings
CXX="${CXX:-g++}"
CXXFLAGS="-std=c++17 -O3 -Wall -Wextra -I$INCLUDE_DIR"
DEBUG_FLAGS="-g -O0 -fsanitize=address,undefined"

# Create build directory
mkdir -p "$BUILD_DIR"

# Source files
CORE_SOURCES=(
    "$SRC_DIR/core/hand_evaluator.cpp"
    "$SRC_DIR/game/holdem_state.cpp"
    "$SRC_DIR/belief/public_belief_state.cpp"
    "$SRC_DIR/cfr/dcfr_solver.cpp"
)

echo "Building ARES-HU C++ Solver..."
echo "  Compiler: $CXX"
echo "  Build dir: $BUILD_DIR"

# Build training executable
echo ""
echo "Building train_cfr..."
$CXX $CXXFLAGS \
    "${CORE_SOURCES[@]}" \
    "$SRC_DIR/main_train.cpp" \
    -o "$BUILD_DIR/train_cfr"
echo "  Built: $BUILD_DIR/train_cfr"

# Build evaluation executable
echo ""
echo "Building eval_cfr..."
$CXX $CXXFLAGS \
    "${CORE_SOURCES[@]}" \
    "$SRC_DIR/main_eval.cpp" \
    -o "$BUILD_DIR/eval_cfr"
echo "  Built: $BUILD_DIR/eval_cfr"

# Build tests
echo ""
echo "Building tests..."
$CXX $CXXFLAGS \
    "${CORE_SOURCES[@]}" \
    "$TESTS_DIR/test_basic.cpp" \
    -o "$BUILD_DIR/test_runner"
echo "  Built: $BUILD_DIR/test_runner"

# Run tests
echo ""
echo "Running tests..."
"$BUILD_DIR/test_runner"

echo ""
echo "Build complete!"
echo ""
echo "Usage:"
echo "  $BUILD_DIR/train_cfr --iterations 10000 --stack 20"
echo "  $BUILD_DIR/train_cfr --help"
