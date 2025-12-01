#!/usr/bin/env python3
"""
Combine Multi-Stack Training Data for ARES-HU

Combines training data from multiple stack depths and creates
normalized PBS encodings that generalize across stack sizes.

Key insight: Use RELATIVE features instead of absolute values:
- SPR (stack-to-pot ratio) instead of raw stack/pot
- Pot fractions instead of absolute chip counts
- This allows networks to interpolate between trained depths

Usage:
    python combine_training_data.py --input-dir blueprints/multi_stack --output data/combined_training.npz
"""

import argparse
import sys
import os
import struct
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import json


# Training data binary format magic number
TRAINING_DATA_MAGIC = 0x54524E44  # "TRND"


def read_training_data_file(filepath: str) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Read training data exported from C++ solver.

    C++ format:
    - Header: magic (4), version (4), num_samples (4), pbs_dim (4) = 16 bytes
    - For each sample:
      - PBS encoding: pbs_dim floats (already normalized by starting_stack)
      - Values: 2 floats (value_p0, value_p1)

    PBS encoding format (indices):
      [0]: pot / starting_stack
      [1]: stack[0] / starting_stack
      [2]: stack[1] / starting_stack
      [3]: bet[0] / starting_stack
      [4]: bet[1] / starting_stack
      [5-8]: street one-hot (preflop, flop, turn, river)
      [9]: current_player
      [10-19]: board cards (5 cards × 2 values: norm_rank, norm_suit)
      [20+]: padding to pbs_dim

    Returns:
        (pbs_encodings, target_values, metadata)
    """
    with open(filepath, 'rb') as f:
        # Read header (16 bytes)
        magic = struct.unpack('I', f.read(4))[0]
        if magic != TRAINING_DATA_MAGIC:
            raise ValueError(f"Invalid magic number: {hex(magic)}, expected {hex(TRAINING_DATA_MAGIC)}")

        version = struct.unpack('I', f.read(4))[0]
        num_samples = struct.unpack('I', f.read(4))[0]
        pbs_dim = struct.unpack('I', f.read(4))[0]

        target_dim = 2  # value_p0, value_p1

        # Read samples
        pbs_list = []
        target_list = []

        for _ in range(num_samples):
            # Read PBS encoding
            pbs = np.frombuffer(f.read(pbs_dim * 4), dtype=np.float32)
            pbs_list.append(pbs)

            # Read target values
            values = np.frombuffer(f.read(target_dim * 4), dtype=np.float32)
            target_list.append(values)

        pbs_data = np.stack(pbs_list)
        target_data = np.stack(target_list)

        # Extract starting stack from filename if possible (e.g., training_data_20bb.bin)
        import re
        match = re.search(r'_(\d+)bb\.bin$', filepath)
        starting_stack = float(match.group(1)) if match else 20.0

        metadata = {
            'version': version,
            'num_samples': num_samples,
            'pbs_dim': pbs_dim,
            'target_dim': target_dim,
            'starting_stack': starting_stack,
            'filepath': filepath
        }

        return pbs_data, target_data, metadata


def create_enhanced_pbs(
    raw_pbs: np.ndarray,
    starting_stack: float
) -> np.ndarray:
    """
    Enhance PBS encoding with additional derived features for better generalization.

    C++ export format (already normalized by starting_stack):
    [0]: pot / starting_stack
    [1]: stack[0] / starting_stack (P0 stack fraction)
    [2]: stack[1] / starting_stack (P1 stack fraction)
    [3]: bet[0] / starting_stack
    [4]: bet[1] / starting_stack
    [5-8]: street one-hot
    [9]: current_player
    [10-19]: board cards
    [20+]: padding

    Enhanced output (additional derived features):
    [0]: SPR = min(stack0, stack1) / pot (stack-to-pot ratio)
    [1-N]: original features
    """
    n_samples = raw_pbs.shape[0]

    # Extract already-normalized features
    pot_frac = raw_pbs[:, 0].clip(min=0.01)  # pot / starting_stack
    stack0_frac = raw_pbs[:, 1]  # stack[0] / starting_stack
    stack1_frac = raw_pbs[:, 2]  # stack[1] / starting_stack

    # Calculate additional derived features
    effective_stack_frac = np.minimum(stack0_frac, stack1_frac)
    spr = effective_stack_frac / pot_frac  # Stack-to-pot ratio (key for decision making)

    # Stack ratio (for asymmetric stack situations)
    stack_ratio = stack0_frac / stack1_frac.clip(min=0.01)

    # Clamp SPR to reasonable range
    spr = np.clip(spr, 0, 50).reshape(-1, 1)
    stack_ratio = np.clip(stack_ratio, 0.1, 10).reshape(-1, 1)

    # Prepend derived features to original encoding
    enhanced_pbs = np.hstack([
        spr,
        stack_ratio,
        raw_pbs  # Keep all original features
    ]).astype(np.float32)

    return enhanced_pbs


def combine_datasets(
    data_files: List[str],
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Combine multiple training data files into one dataset.
    """
    all_pbs = []
    all_targets = []
    all_metadata = []

    for filepath in data_files:
        print(f"Loading {filepath}...")
        try:
            pbs, targets, meta = read_training_data_file(filepath)
            print(f"  Samples: {len(pbs):,}, Stack: {meta['starting_stack']}bb")

            if normalize:
                pbs = create_enhanced_pbs(pbs, meta['starting_stack'])
                print(f"  Normalized PBS dim: {pbs.shape[1]}")

            all_pbs.append(pbs)
            all_targets.append(targets)
            all_metadata.append(meta)

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    if not all_pbs:
        raise ValueError("No valid training data files found")

    # Ensure all PBS dimensions match
    pbs_dims = [p.shape[1] for p in all_pbs]
    if len(set(pbs_dims)) > 1:
        print(f"\nWARNING: PBS dimensions differ: {pbs_dims}")
        # Pad to max dimension
        max_dim = max(pbs_dims)
        all_pbs = [
            np.pad(p, ((0, 0), (0, max_dim - p.shape[1])))
            for p in all_pbs
        ]

    # Combine
    combined_pbs = np.vstack(all_pbs)
    combined_targets = np.vstack(all_targets)

    # Shuffle
    indices = np.random.permutation(len(combined_pbs))
    combined_pbs = combined_pbs[indices]
    combined_targets = combined_targets[indices]

    combined_meta = {
        'sources': [m['filepath'] for m in all_metadata],
        'stacks': [m['starting_stack'] for m in all_metadata],
        'samples_per_stack': [m['num_samples'] for m in all_metadata],
        'total_samples': len(combined_pbs),
        'pbs_dim': combined_pbs.shape[1],
        'target_dim': combined_targets.shape[1],
        'normalized': normalize
    }

    return combined_pbs, combined_targets, combined_meta


def main():
    parser = argparse.ArgumentParser(
        description="Combine multi-stack training data with normalization"
    )
    parser.add_argument(
        "--input-dir", type=str, default="blueprints/multi_stack",
        help="Directory containing training data files"
    )
    parser.add_argument(
        "--output", type=str, default="data/multi_stack_training.npz",
        help="Output file path"
    )
    parser.add_argument(
        "--no-normalize", action="store_true",
        help="Skip normalization (not recommended)"
    )
    parser.add_argument(
        "--train-split", type=float, default=0.9,
        help="Fraction of data for training (default: 0.9)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("ARES-HU Multi-Stack Training Data Combiner")
    print("=" * 60)

    # Find all training data files
    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f"ERROR: Input directory not found: {args.input_dir}")
        return 1

    data_files = sorted(input_path.glob("training_data_*.bin"))
    if not data_files:
        print(f"ERROR: No training data files found in {args.input_dir}")
        print("Expected files like: training_data_20bb.bin")
        return 1

    print(f"\nFound {len(data_files)} training data files:")
    for f in data_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name}: {size_mb:.1f} MB")

    # Combine datasets
    print("\nCombining datasets...")
    pbs, targets, meta = combine_datasets(
        [str(f) for f in data_files],
        normalize=not args.no_normalize
    )

    print(f"\nCombined dataset:")
    print(f"  Total samples: {meta['total_samples']:,}")
    print(f"  PBS dimension: {meta['pbs_dim']}")
    print(f"  Target dimension: {meta['target_dim']}")
    print(f"  Stacks included: {meta['stacks']}")

    # Split into train/val
    n = len(pbs)
    n_train = int(n * args.train_split)

    X_train, X_val = pbs[:n_train], pbs[n_train:]
    y_train, y_val = targets[:n_train], targets[n_train:]

    print(f"\nSplit:")
    print(f"  Training:   {len(X_train):,} samples")
    print(f"  Validation: {len(X_val):,} samples")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        args.output,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        metadata=json.dumps(meta)
    )

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nSaved to: {args.output} ({size_mb:.1f} MB)")

    # Feature statistics
    print("\nNormalized feature statistics (training set):")
    feature_names = [
        "SPR (stack/pot)",
        "Pot fraction",
        "Stack ratio",
        "To-call fraction",
        "Effective stack frac"
    ]
    for i, name in enumerate(feature_names[:min(5, X_train.shape[1])]):
        col = X_train[:, i]
        print(f"  {name:20s}: mean={col.mean():.3f}, std={col.std():.3f}, "
              f"min={col.min():.3f}, max={col.max():.3f}")

    print("\nNext step: Train neural networks on combined data")
    print("  python train_value_network.py --data data/multi_stack_training.npz")
    print("  python train_policy_network.py --data data/multi_stack_training.npz")

    return 0


if __name__ == "__main__":
    sys.exit(main())
