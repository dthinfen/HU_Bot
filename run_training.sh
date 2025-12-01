#!/bin/bash
# ARES-HU Multi-Stack Training Script
# Run with: nohup ./run_training.sh > training.log 2>&1 &
# Monitor with: tail -f training.log

cd /Users/daniel/Desktop/Code/HU_Bot

echo "=========================================="
echo "ARES-HU Multi-Stack CFR Training"
echo "Started: $(date)"
echo "=========================================="

mkdir -p blueprints/multi_stack

# Stack configurations: stack_bb iterations samples
# Total estimated time: ~6 hours on 8 threads
CONFIGS=(
    "10 1000000 200000"
    "20 1000000 200000"
    "50 500000 200000"
    "100 300000 200000"
)

for config in "${CONFIGS[@]}"; do
    read -r stack iters samples <<< "$config"

    echo ""
    echo "============================================"
    echo "Training ${stack}bb - ${iters} iterations"
    echo "Started: $(date)"
    echo "============================================"

    # Check if already done (skip if strategy file exists and is recent)
    if [ -f "blueprints/multi_stack/strategy_${stack}bb.bin" ]; then
        echo "Strategy file exists, skipping ${stack}bb"
        continue
    fi

    ./cpp_solver/build/train_cfr \
        --iterations $iters \
        --stack $stack \
        --threads 8 \
        --qre-tau 1.0 \
        --output "blueprints/multi_stack/strategy_${stack}bb.bin" \
        --export-training-data "blueprints/multi_stack/training_data_${stack}bb.bin" \
        --num-samples $samples

    echo "Completed ${stack}bb at $(date)"
    echo ""
done

echo ""
echo "=========================================="
echo "All training complete!"
echo "Finished: $(date)"
echo "=========================================="

echo ""
echo "Output files:"
ls -lh blueprints/multi_stack/

echo ""
echo "Next steps:"
echo "  python combine_training_data.py --input-dir blueprints/multi_stack"
echo "  python train_value_network.py --data data/multi_stack_training.npz"
