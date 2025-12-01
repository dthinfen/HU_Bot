#!/usr/bin/env python3
"""
Train Policy Network for ARES-HU

Trains a neural network to predict action probabilities from PBS encodings.
This enables instant strategy lookup for any game state.

Usage:
    python train_policy_network.py --data data/policy_training.npz --epochs 100
"""

import argparse
import os
import sys
import time
import torch
import numpy as np

from src.neural.policy_network import PolicyNetwork, PolicyNetworkTrainer, bucket_to_action_name


def main():
    parser = argparse.ArgumentParser(description="Train ARES-HU policy network")

    parser.add_argument('--data', type=str, default='data/policy_training.npz',
                        help='Path to training data')
    parser.add_argument('--output', type=str, default='models/policy_network.pt',
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
    print("ARES-HU Policy Network Training")
    print("=" * 60)

    # Check for data
    if not os.path.exists(args.data):
        print(f"ERROR: Training data not found: {args.data}")
        print("Generate training data first:")
        print("  python generate_policy_data.py --samples 100000")
        sys.exit(1)

    # Load data
    print(f"\nLoading training data from: {args.data}")
    data = np.load(args.data)
    pbs = data['pbs']
    policy = data['policy']
    masks = data['masks']

    print(f"  Samples: {len(pbs):,}")
    print(f"  PBS dimension: {pbs.shape[1]}")
    print(f"  Action buckets: {policy.shape[1]}")

    # Split data
    n = len(pbs)
    indices = np.random.permutation(n)
    split_idx = int(n * 0.9)

    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]

    X_train = torch.from_numpy(pbs[train_idx]).float()
    y_train = torch.from_numpy(policy[train_idx]).float()
    m_train = torch.from_numpy(masks[train_idx]).bool()

    X_val = torch.from_numpy(pbs[val_idx]).float()
    y_val = torch.from_numpy(policy[val_idx]).float()
    m_val = torch.from_numpy(masks[val_idx]).bool()

    print(f"  Train: {len(X_train):,}, Val: {len(X_val):,}")

    # Create model
    input_dim = pbs.shape[1]
    num_actions = policy.shape[1]

    model = PolicyNetwork(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_actions=num_actions
    )

    print(f"\nModel architecture:")
    print(f"  Input: {input_dim}")
    print(f"  Hidden: {args.hidden_dim} x {args.num_layers} layers")
    print(f"  Output: {num_actions} action buckets")
    print(f"  Parameters: {model.num_parameters:,}")

    # Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"  Device: {device}")

    # Create trainer
    trainer = PolicyNetworkTrainer(model, learning_rate=args.lr, device=device)

    # Train
    print(f"\nTraining for up to {args.epochs} epochs...")
    print("-" * 60)

    start_time = time.time()
    stats = trainer.train(
        X_train, y_train,
        X_val, y_val,
        masks_train=m_train,
        masks_val=m_val,
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
    print(f"  Final accuracy: {stats['final_accuracy']:.1%}")

    # Save model
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    trainer.save(args.output)
    print(f"\nModel saved to: {args.output}")

    # Sanity check predictions
    print("\nSanity check predictions:")
    model.eval()
    with torch.no_grad():
        for idx in [0, len(X_val)//2, len(X_val)-1]:
            x = X_val[idx:idx+1].to(device)
            mask = m_val[idx:idx+1].to(device)

            probs = model.get_action_probs(x, mask)
            pred_action = probs.argmax(dim=-1).item()
            target_action = y_val[idx].argmax().item()

            print(f"  Sample {idx}:")
            print(f"    Predicted: {bucket_to_action_name(pred_action)} ({probs[0, pred_action]:.1%})")
            print(f"    Target:    {bucket_to_action_name(target_action)} ({y_val[idx, target_action]:.1%})")

    print("\n" + "=" * 60)
    print("Policy network ready for real-time inference!")
    print("=" * 60)


if __name__ == "__main__":
    main()
