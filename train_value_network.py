#!/usr/bin/env python3
"""
Train Value Network for ARES-HU

Trains a neural network to predict counterfactual values from PBS encodings.
This is the foundation for real-time search (ReBeL architecture).

Usage:
    python train_value_network.py --data data/training_100k.bin --epochs 100
"""

import argparse
import os
import sys
import time
import torch
import numpy as np

from src.neural.cpp_data_loader import CppTrainingDataset
from src.neural.value_network import ValueNetwork, ValueNetworkTrainer


def main():
    parser = argparse.ArgumentParser(description="Train ARES-HU value network")

    parser.add_argument('--data', type=str, default='data/training_100k.bin',
                        help='Path to training data')
    parser.add_argument('--output', type=str, default='models/value_network.pt',
                        help='Output model path')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Hidden layer dimension')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='Number of hidden layers')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')

    args = parser.parse_args()

    print("=" * 60)
    print("ARES-HU Value Network Training")
    print("=" * 60)

    # Check for data
    if not os.path.exists(args.data):
        print(f"ERROR: Training data not found: {args.data}")
        print("Generate training data first:")
        print("  python -c \"import ares_solver; s = ares_solver.Solver(); "
              "s.load('blueprints/cpp_1M/strategy_1M.bin'); "
              "s.export_training_data('data/training_100k.bin', 100000)\"")
        sys.exit(1)

    # Load data - handle both .bin and .npz formats
    print(f"\nLoading training data from: {args.data}")

    if args.data.endswith('.npz'):
        # Load combined multi-stack data
        data = np.load(args.data, allow_pickle=True)
        X_train = torch.from_numpy(data['X_train']).float()
        X_val = torch.from_numpy(data['X_val']).float()
        # Target values are [value_p0, value_p1] - use p0 for value network
        y_train_raw = data['y_train']
        y_val_raw = data['y_val']
        y_train = torch.from_numpy(y_train_raw[:, 0:1]).float()  # Player 0 value
        y_val = torch.from_numpy(y_val_raw[:, 0:1]).float()
        input_dim = X_train.shape[1]
        print(f"  Format: Multi-stack NPZ")
        print(f"  Train samples: {len(X_train):,}")
        print(f"  Val samples: {len(X_val):,}")
        print(f"  PBS dimension: {input_dim}")
        print(f"  Value range: [{y_train.min():.2f}, {y_train.max():.2f}]")
    else:
        # Load single-stack binary data
        dataset = CppTrainingDataset(args.data, player=0)
        print(f"  Format: Binary")
        print(f"  Samples: {len(dataset):,}")
        print(f"  PBS dimension: {dataset.pbs.shape[1]}")
        print(f"  Value range: [{dataset.values.min():.2f}, {dataset.values.max():.2f}]")

        # Split data
        (train_pbs, train_vals), (val_pbs, val_vals) = dataset.split(train_ratio=0.9)
        print(f"  Train: {len(train_pbs):,}, Val: {len(val_pbs):,}")

        # Convert to tensors
        X_train = torch.from_numpy(train_pbs).float()
        y_train = torch.from_numpy(train_vals.reshape(-1, 1)).float()
        X_val = torch.from_numpy(val_pbs).float()
        y_val = torch.from_numpy(val_vals.reshape(-1, 1)).float()
        input_dim = dataset.pbs.shape[1]

    # Create model
    model = ValueNetwork(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        output_dim=1
    )

    print(f"\nModel architecture:")
    print(f"  Input: {input_dim}")
    print(f"  Hidden: {args.hidden_dim} × {args.num_layers} layers")
    print(f"  Output: 1 (value)")
    print(f"  Parameters: {model.num_parameters:,}")

    # Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"  Device: {device}")

    # Create trainer
    trainer = ValueNetworkTrainer(model, learning_rate=args.lr, device=device)

    # Train
    print(f"\nTraining for up to {args.epochs} epochs...")
    print("-" * 60)

    start_time = time.time()
    stats = trainer.train(
        X_train, y_train,
        X_val, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        verbose=True
    )
    elapsed = time.time() - start_time

    print("-" * 60)
    print(f"\nTraining complete in {elapsed:.1f}s")
    print(f"  Best epoch: {stats['best_epoch']}")
    print(f"  Best val loss: {stats['best_val_loss']:.6f}")
    print(f"  Final RMSE: {stats['final_rmse']:.3f} bb")

    # Save model
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    trainer.save(args.output)
    print(f"\nModel saved to: {args.output}")

    # Quick sanity check
    print("\nSanity check predictions:")
    model.eval()
    with torch.no_grad():
        sample_idx = [0, len(X_val)//2, len(X_val)-1]
        for idx in sample_idx:
            pred = model(X_val[idx:idx+1].to(device)).item()
            actual = y_val[idx].item()
            print(f"  Sample {idx}: pred={pred:+.3f}, actual={actual:+.3f}, diff={abs(pred-actual):.3f}")

    print("\n" + "=" * 60)
    print("Value network ready for real-time search!")
    print("=" * 60)


if __name__ == "__main__":
    main()
