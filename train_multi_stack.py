#!/usr/bin/env python3
"""
Multi-Stack CFR Training for ARES-HU

Trains CFR at multiple stack depths to generate diverse training data
for neural networks that can generalize across any stack size.

Stack depths and their characteristics:
- 10bb:  Push/fold poker, simple game tree
- 20bb:  Short stack, limited postflop play
- 50bb:  Medium stack, standard tournament depth
- 100bb: Deep stack, full postflop complexity
- 200bb: Very deep, maximum strategic complexity (Slumbot depth)

Usage:
    python train_multi_stack.py --iterations 100000 --output-dir blueprints/multi_stack
"""

import argparse
import subprocess
import os
import sys
import time
from pathlib import Path


# Stack configurations
# Format: (stack_bb, iterations, description)
STACK_CONFIGS = [
    (10,  500000,  "Push/fold"),
    (20,  1000000, "Short stack"),
    (50,  1000000, "Medium stack"),
    (100, 500000,  "Deep stack"),
    (200, 500000,  "Very deep (Slumbot)"),
]


def run_training(
    stack_bb: int,
    iterations: int,
    output_dir: str,
    threads: int = 8,
    qre_tau: float = 1.0,
    export_samples: int = 100000
):
    """Run CFR training for a single stack depth."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    strategy_file = output_path / f"strategy_{stack_bb}bb.bin"
    training_data_file = output_path / f"training_data_{stack_bb}bb.bin"

    print(f"\n{'='*60}")
    print(f"Training {stack_bb}bb ({iterations:,} iterations)")
    print(f"{'='*60}")

    cmd = [
        "./cpp_solver/build/train_cfr",
        "--iterations", str(iterations),
        "--stack", str(stack_bb),
        "--threads", str(threads),
        "--qre-tau", str(qre_tau),
        "--output", str(strategy_file),
        "--export-training-data", str(training_data_file),
        "--num-samples", str(export_samples),
    ]

    print(f"Command: {' '.join(cmd)}\n")

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            check=True
        )
        elapsed = time.time() - start_time
        print(f"\nCompleted {stack_bb}bb in {elapsed/60:.1f} minutes")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Training failed for {stack_bb}bb")
        print(f"Exit code: {e.returncode}")
        return False
    except FileNotFoundError:
        print("\nERROR: C++ solver not found. Run: cd cpp_solver && ./build.sh")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Train CFR at multiple stack depths for neural network generalization"
    )
    parser.add_argument(
        "--output-dir", type=str, default="blueprints/multi_stack",
        help="Output directory for strategies and training data"
    )
    parser.add_argument(
        "--stacks", type=str, default="10,20,50,100,200",
        help="Comma-separated list of stack depths to train (default: 10,20,50,100,200)"
    )
    parser.add_argument(
        "--iterations", type=int, default=None,
        help="Override iterations for all stacks (default: use per-stack defaults)"
    )
    parser.add_argument(
        "--threads", type=int, default=8,
        help="Number of threads (default: 8)"
    )
    parser.add_argument(
        "--qre-tau", type=float, default=1.0,
        help="QRE temperature (default: 1.0, 0=Nash)"
    )
    parser.add_argument(
        "--samples", type=int, default=100000,
        help="Training samples to export per stack depth (default: 100000)"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: 10k iterations per stack (for testing)"
    )

    args = parser.parse_args()

    # Parse requested stacks
    requested_stacks = [int(s) for s in args.stacks.split(",")]

    # Filter configs to only requested stacks
    configs = [(s, i, d) for s, i, d in STACK_CONFIGS if s in requested_stacks]

    if not configs:
        print(f"ERROR: No valid stacks found in: {args.stacks}")
        print(f"Available: {[s for s, _, _ in STACK_CONFIGS]}")
        return 1

    # Override iterations if specified
    if args.iterations:
        configs = [(s, args.iterations, d) for s, _, d in configs]
    elif args.quick:
        configs = [(s, 10000, d) for s, _, d in configs]

    print("=" * 60)
    print("ARES-HU Multi-Stack CFR Training")
    print("=" * 60)
    print(f"\nOutput directory: {args.output_dir}")
    print(f"QRE tau: {args.qre_tau}")
    print(f"Threads: {args.threads}")
    print(f"Samples per stack: {args.samples:,}")
    print(f"\nTraining schedule:")

    total_iterations = 0
    for stack_bb, iterations, desc in configs:
        print(f"  {stack_bb:3d}bb: {iterations:>10,} iterations ({desc})")
        total_iterations += iterations

    print(f"\nTotal iterations: {total_iterations:,}")

    # Estimate time (rough: ~14k iter/sec for 20bb, slower for deeper)
    estimated_seconds = sum(
        iters / (14000 * (20 / stack))  # Deeper stacks are slower
        for stack, iters, _ in configs
    )
    print(f"Estimated time: {estimated_seconds/3600:.1f} hours")

    print("\n" + "-" * 60)
    input("Press Enter to start training (Ctrl+C to cancel)...")

    # Run training for each stack
    start_time = time.time()
    results = []

    for stack_bb, iterations, desc in configs:
        success = run_training(
            stack_bb=stack_bb,
            iterations=iterations,
            output_dir=args.output_dir,
            threads=args.threads,
            qre_tau=args.qre_tau,
            export_samples=args.samples
        )
        results.append((stack_bb, success))

    total_time = time.time() - start_time

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nTotal time: {total_time/3600:.1f} hours")
    print(f"\nResults:")
    for stack_bb, success in results:
        status = "OK" if success else "FAILED"
        print(f"  {stack_bb:3d}bb: {status}")

    # List output files
    output_path = Path(args.output_dir)
    if output_path.exists():
        print(f"\nOutput files in {args.output_dir}/:")
        for f in sorted(output_path.glob("*.bin")):
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.name}: {size_mb:.1f} MB")

    print("\nNext step: Combine training data and train neural networks")
    print("  python combine_training_data.py --input-dir blueprints/multi_stack")

    return 0 if all(s for _, s in results) else 1


if __name__ == "__main__":
    sys.exit(main())
