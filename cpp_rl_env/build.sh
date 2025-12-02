#!/bin/bash

# Build C++ vectorized environment

set -e

cd "$(dirname "$0")"

# Find Python - prefer miniforge, then system python
if [ -x "/Users/daniel/miniforge3/bin/python3" ]; then
    export PYTHON_EXECUTABLE="/Users/daniel/miniforge3/bin/python3"
elif [ -x "/root/miniconda3/bin/python" ]; then
    export PYTHON_EXECUTABLE="/root/miniconda3/bin/python"
elif command -v python3 &> /dev/null; then
    export PYTHON_EXECUTABLE=$(which python3)
fi

echo "Using Python: $PYTHON_EXECUTABLE"

# Create build directory
rm -rf build
mkdir -p build
cd build

# Configure with Python hint
cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DPython3_EXECUTABLE="$PYTHON_EXECUTABLE"

# Build
cmake --build . -j$(nproc 2>/dev/null || sysctl -n hw.ncpu)

echo ""
echo "Build complete! Module copied to parent directory."
echo "Test with: python -c 'import cpp_vec_env; print(cpp_vec_env.NUM_ACTIONS)'"
