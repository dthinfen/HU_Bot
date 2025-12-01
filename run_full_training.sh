#!/bin/bash
# Full multi-stack CFR training for best quality
# Each stack: 5M iterations, 1M training samples

set -e

OUTPUT_DIR="blueprints/full_training"
mkdir -p "$OUTPUT_DIR"

ITERATIONS=5000000
SAMPLES=1000000

echo "=========================================="
echo "ARES-HU Full Multi-Stack CFR Training"
echo "=========================================="
echo "Output: $OUTPUT_DIR"
echo "Iterations per stack: $ITERATIONS"
echo "Samples per stack: $SAMPLES"
echo ""

# Stack depths to train
STACKS=(10 20 50 100 200)

for STACK in "${STACKS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Training ${STACK}bb - $ITERATIONS iterations"
    echo "=========================================="

    time ./cpp_solver/build/train_cfr \
        --iterations $ITERATIONS \
        --stack $STACK \
        --threads 8 \
        --qre-tau 1.0 \
        --output "$OUTPUT_DIR/strategy_${STACK}bb.bin" \
        --export-training-data "$OUTPUT_DIR/training_data_${STACK}bb.bin" \
        --num-samples $SAMPLES

    echo "Completed ${STACK}bb"
done

echo ""
echo "=========================================="
echo "All CFR Training Complete!"
echo "=========================================="
ls -lh "$OUTPUT_DIR"

echo ""
echo "Next: Combine data and train value network"
echo "  python combine_training_data.py --input-dir $OUTPUT_DIR"
