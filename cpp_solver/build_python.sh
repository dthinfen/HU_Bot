#!/bin/bash
# Build script for ARES-HU pybind11 Python module

set -e

# Directory setup
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="$SCRIPT_DIR/src"
INCLUDE_DIR="$SCRIPT_DIR/include"
BUILD_DIR="$SCRIPT_DIR/build"

# Use miniforge3 Python (has pybind11 installed)
PYTHON="${PYTHON:-/Users/daniel/miniforge3/bin/python3}"

# Get pybind11 includes and extension suffix
PYBIND_INCLUDES=$($PYTHON -m pybind11 --includes)
EXTENSION_SUFFIX=$($PYTHON -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

# LibTorch setup
TORCH_DIR="/Users/daniel/miniforge3/lib/python3.12/site-packages/torch"
USE_LIBTORCH=1

# Compiler settings
CXX="${CXX:-g++}"
CXXFLAGS="-std=c++17 -O3 -Wall -Wextra -shared -fPIC -I$INCLUDE_DIR $PYBIND_INCLUDES"

# Add LibTorch if available
if [[ "$USE_LIBTORCH" == "1" && -d "$TORCH_DIR" ]]; then
    CXXFLAGS="$CXXFLAGS -DARES_USE_LIBTORCH"
    CXXFLAGS="$CXXFLAGS -I$TORCH_DIR/include"
    CXXFLAGS="$CXXFLAGS -I$TORCH_DIR/include/torch/csrc/api/include"
    LDFLAGS="-L$TORCH_DIR/lib -ltorch -ltorch_cpu -lc10"
    # Set rpath for macOS
    if [[ "$(uname)" == "Darwin" ]]; then
        LDFLAGS="$LDFLAGS -Wl,-rpath,$TORCH_DIR/lib"
    fi
    echo "LibTorch enabled: $TORCH_DIR"
else
    LDFLAGS=""
    echo "LibTorch disabled"
fi

# macOS-specific flags
if [[ "$(uname)" == "Darwin" ]]; then
    CXXFLAGS="$CXXFLAGS -undefined dynamic_lookup"
fi

# Create build directory
mkdir -p "$BUILD_DIR"

# Source files
CORE_SOURCES=(
    "$SRC_DIR/core/hand_evaluator.cpp"
    "$SRC_DIR/game/holdem_state.cpp"
    "$SRC_DIR/belief/public_belief_state.cpp"
    "$SRC_DIR/cfr/dcfr_solver.cpp"
    "$SRC_DIR/search/realtime_search.cpp"
)

# Add neural evaluator if LibTorch is enabled
if [[ "$USE_LIBTORCH" == "1" && -d "$TORCH_DIR" ]]; then
    CORE_SOURCES+=("$SRC_DIR/neural/neural_evaluator.cpp")
fi

OUTPUT="$BUILD_DIR/ares_solver$EXTENSION_SUFFIX"

echo "Building ARES-HU Python Module..."
echo "  Python: $PYTHON"
echo "  Extension: $EXTENSION_SUFFIX"
echo "  Output: $OUTPUT"
echo ""

# Build the module
echo "Compiling..."
$CXX $CXXFLAGS \
    "${CORE_SOURCES[@]}" \
    "$SRC_DIR/python_bindings.cpp" \
    $LDFLAGS \
    -o "$OUTPUT"

echo ""
echo "Build successful!"
echo "  Output: $OUTPUT"
echo ""
echo "Usage from Python:"
echo "  import sys"
echo "  sys.path.insert(0, '$BUILD_DIR')"
echo "  import ares_solver"
echo "  solver = ares_solver.Solver()"
echo "  solver.train(iterations=10000)"
