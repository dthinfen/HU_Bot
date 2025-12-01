#!/usr/bin/env python3
"""
Generate Policy Training Data for ARES-HU

This script generates training data for the policy network by:
1. Loading the trained CFR solver
2. Sampling random game states
3. Querying CFR strategies
4. Mapping to action buckets
5. Saving as training data

Usage:
    python generate_policy_data.py --output data/policy_training.npz --samples 100000
"""

import argparse
import sys
import os
import numpy as np
from typing import List, Tuple, Dict
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cpp_solver', 'build'))

# Action bucket mapping - GTO-standard (must match policy_network.py)
# Reference: https://blog.gtowizard.com/pot-geometry/
ACTION_FOLD = 0
ACTION_CHECK_CALL = 1
ACTION_BET_SMALL = 2      # 25-40% pot
ACTION_BET_MEDIUM = 3     # 40-75% pot
ACTION_BET_LARGE = 4      # 75-125% pot
ACTION_BET_OVERBET = 5    # 125%+ pot
ACTION_ALL_IN = 6
NUM_ACTION_BUCKETS = 7


def map_action_to_bucket(action_str: str, pot: float) -> int:
    """
    Map CFR action string to bucket index using GTO-standard sizing.

    Buckets:
    - Small: 25-40% pot (probe, block, range bet)
    - Medium: 40-75% pot (standard value/bluff)
    - Large: 75-125% pot (geometric, polarized)
    - Overbet: 125%+ pot
    """
    action_str = action_str.lower()

    if action_str == 'fold':
        return ACTION_FOLD
    elif action_str in ('check', 'call'):
        return ACTION_CHECK_CALL
    elif action_str.startswith('allin'):
        return ACTION_ALL_IN
    elif action_str.startswith(('bet_', 'raise_')):
        # Extract bet size
        try:
            bet_size = float(action_str.split('_')[1])
            if pot > 0:
                ratio = bet_size / pot
                if ratio <= 0.40:
                    return ACTION_BET_SMALL      # 25-40% pot
                elif ratio <= 0.75:
                    return ACTION_BET_MEDIUM     # 40-75% pot
                elif ratio <= 1.25:
                    return ACTION_BET_LARGE      # 75-125% pot
                else:
                    return ACTION_BET_OVERBET    # 125%+ pot
            return ACTION_BET_MEDIUM
        except (ValueError, IndexError):
            return ACTION_BET_MEDIUM

    return ACTION_CHECK_CALL  # Default


def create_pbs_encoding(
    hero_cards: List[str],
    board_cards: List[str],
    hero_stack: float,
    villain_stack: float,
    pot: float,
    street: int,  # 0=preflop, 1=flop, 2=turn, 3=river
    starting_stack: float = 20.0
) -> np.ndarray:
    """
    Create PBS (Public Belief State) encoding for neural network input.

    Returns a fixed-size feature vector encoding the game state.
    """
    features = []

    # Stack/pot features (normalized)
    features.append(pot / starting_stack)
    features.append(hero_stack / starting_stack)
    features.append(villain_stack / starting_stack)

    # Street one-hot
    for s in range(4):
        features.append(1.0 if s == street else 0.0)

    # Hero hole cards encoding (1326 canonical indices -> 52 card features)
    hero_card_vec = np.zeros(52)
    for card_str in hero_cards:
        idx = card_to_index(card_str)
        if 0 <= idx < 52:
            hero_card_vec[idx] = 1.0
    features.extend(hero_card_vec)

    # Board cards encoding (52 features)
    board_card_vec = np.zeros(52)
    for card_str in board_cards:
        idx = card_to_index(card_str)
        if 0 <= idx < 52:
            board_card_vec[idx] = 1.0
    features.extend(board_card_vec)

    return np.array(features, dtype=np.float32)


def card_to_index(card_str: str) -> int:
    """Convert card string to 0-51 index."""
    if len(card_str) != 2:
        return -1

    ranks = "23456789TJQKA"
    suits = "cdhs"

    rank = ranks.find(card_str[0].upper())
    suit = suits.find(card_str[1].lower())

    if rank == -1 or suit == -1:
        return -1

    return suit * 13 + rank


def generate_random_hand() -> Tuple[List[str], List[str]]:
    """Generate random hero cards and optionally board cards."""
    ranks = "23456789TJQKA"
    suits = "cdhs"

    # Create deck
    deck = [f"{r}{s}" for r in ranks for s in suits]
    random.shuffle(deck)

    # Deal hero cards
    hero_cards = [deck.pop(), deck.pop()]

    # Random street (0=preflop, 1=flop, 2=turn, 3=river)
    # Bias towards preflop and flop for training
    street_probs = [0.4, 0.35, 0.15, 0.10]
    street = random.choices([0, 1, 2, 3], weights=street_probs)[0]

    # Deal board based on street
    board_cards = []
    if street >= 1:  # Flop
        board_cards.extend([deck.pop(), deck.pop(), deck.pop()])
    if street >= 2:  # Turn
        board_cards.append(deck.pop())
    if street >= 3:  # River
        board_cards.append(deck.pop())

    return hero_cards, board_cards, street


def main():
    parser = argparse.ArgumentParser(description="Generate policy training data")
    parser.add_argument('--output', type=str, default='data/policy_training.npz',
                        help='Output file path')
    parser.add_argument('--samples', type=int, default=100000,
                        help='Number of training samples')
    parser.add_argument('--strategy', type=str, default='blueprints/cpp_1M/strategy_1M.bin',
                        help='Path to trained CFR strategy')
    parser.add_argument('--stack', type=float, default=20.0,
                        help='Starting stack size in BB')
    args = parser.parse_args()

    # Import solver
    try:
        import ares_solver
    except ImportError:
        print("ERROR: C++ solver not built. Run: cd cpp_solver && ./build_python.sh")
        sys.exit(1)

    print("=" * 60)
    print("ARES-HU Policy Training Data Generator")
    print("=" * 60)

    # Load solver
    print(f"\nLoading strategy from: {args.strategy}")
    solver = ares_solver.Solver()
    solver.load(args.strategy)
    print(f"Loaded {solver.num_info_sets():,} info sets")

    # Storage for training data
    pbs_dim = 7 + 52 + 52  # pot/stacks + street + hero cards + board
    pbs_data = []
    policy_data = []
    mask_data = []

    print(f"\nGenerating {args.samples:,} training samples...")

    valid_samples = 0
    attempts = 0
    max_attempts = args.samples * 10

    while valid_samples < args.samples and attempts < max_attempts:
        attempts += 1

        # Generate random hand
        hero_cards, board_cards, street = generate_random_hand()

        # Random stacks and pot (reasonable ranges)
        hero_stack = random.uniform(5.0, args.stack)
        villain_stack = random.uniform(5.0, args.stack)
        pot = random.uniform(1.5, min(hero_stack, villain_stack) * 0.5)

        # Query solver for strategy
        try:
            strategy = solver.get_strategy(
                hero_cards, board_cards,
                hero_stack, villain_stack, pot
            )
        except Exception:
            continue

        # Skip if we only got debug info (key not found)
        if '_debug_key' in strategy:
            continue

        # Check if strategy is meaningful (not uniform)
        probs = [v for k, v in strategy.items() if not k.startswith('_')]
        if len(probs) < 2:
            continue

        # Create PBS encoding
        pbs = create_pbs_encoding(
            hero_cards, board_cards,
            hero_stack, villain_stack, pot,
            street, args.stack
        )

        # Map actions to buckets
        policy_vec = np.zeros(NUM_ACTION_BUCKETS, dtype=np.float32)
        mask_vec = np.zeros(NUM_ACTION_BUCKETS, dtype=np.float32)

        for action_name, prob in strategy.items():
            if action_name.startswith('_'):
                continue
            bucket = map_action_to_bucket(action_name, pot)
            policy_vec[bucket] += prob
            mask_vec[bucket] = 1.0

        # Renormalize (in case multiple actions mapped to same bucket)
        total = policy_vec.sum()
        if total > 0:
            policy_vec /= total
        else:
            continue

        pbs_data.append(pbs)
        policy_data.append(policy_vec)
        mask_data.append(mask_vec)
        valid_samples += 1

        if valid_samples % 10000 == 0:
            print(f"  Generated {valid_samples:,}/{args.samples:,} samples")

    print(f"\nGenerated {valid_samples:,} valid samples from {attempts:,} attempts")

    # Convert to numpy arrays
    pbs_array = np.stack(pbs_data)
    policy_array = np.stack(policy_data)
    mask_array = np.stack(mask_data)

    print(f"\nData shapes:")
    print(f"  PBS: {pbs_array.shape}")
    print(f"  Policy: {policy_array.shape}")
    print(f"  Masks: {mask_array.shape}")

    # Save
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    np.savez_compressed(
        args.output,
        pbs=pbs_array,
        policy=policy_array,
        masks=mask_array
    )
    print(f"\nSaved to: {args.output}")

    # Show sample statistics
    print("\nPolicy distribution statistics:")
    mean_policy = policy_array.mean(axis=0)
    bucket_names = ['Fold', 'Check/Call', 'Bet 0.5x', 'Bet 1x', 'Bet 2x', 'All-in']
    for i, name in enumerate(bucket_names):
        print(f"  {name:12}: {mean_policy[i]:.1%}")

    print("\n" + "=" * 60)
    print("Policy training data ready!")
    print("=" * 60)


if __name__ == "__main__":
    main()
