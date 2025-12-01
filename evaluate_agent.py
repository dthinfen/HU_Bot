#!/usr/bin/env python3
"""
Evaluate ARES-HU Agent Performance

Uses the C++ solver (via pybind11) for fast evaluation.

Usage:
    python evaluate_agent.py --strategy blueprints/cpp_1M/strategy_1M.bin
    python evaluate_agent.py --opponent callfold --hands 100000
"""

import argparse
import sys
import os

# Add C++ build directory to path
cpp_build_dir = os.path.join(os.path.dirname(__file__), 'cpp_solver', 'build')
if cpp_build_dir not in sys.path:
    sys.path.insert(0, cpp_build_dir)


def main():
    parser = argparse.ArgumentParser(description="Evaluate ARES-HU poker agent")

    parser.add_argument('--strategy', type=str,
                        default='blueprints/cpp_1M/strategy_1M.bin',
                        help='Path to C++ strategy file')
    parser.add_argument('--opponent', type=str, default='random',
                        choices=['random', 'callfold', 'alwayscall'],
                        help='Opponent type (default: random)')
    parser.add_argument('--hands', type=int, default=100000,
                        help='Number of hands to play (default: 100000)')
    parser.add_argument('--stack', type=float, default=20.0,
                        help='Starting stack in BB (default: 20)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    args = parser.parse_args()

    # Import C++ solver
    try:
        import ares_solver
    except ImportError:
        print("ERROR: C++ solver not built. Run:")
        print("  cd cpp_solver && ./build_python.sh")
        sys.exit(1)

    print("=" * 60)
    print("ARES-HU Agent Evaluation")
    print("=" * 60)

    # Load solver
    solver = ares_solver.Solver()
    print(f"\nLoading strategy: {args.strategy}")
    num_infosets = solver.load(args.strategy)
    print(f"Loaded {num_infosets:,} information sets")

    # Run evaluation
    print(f"\nEvaluating vs {args.opponent} ({args.hands:,} hands)...")
    results = solver.evaluate(
        opponent=args.opponent,
        num_hands=args.hands,
        stack_size=args.stack,
        seed=args.seed
    )

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Hands played:    {results['hands_played']:,}")
    print(f"  Total BB won:    {results['total_bb_won']:+,.1f}")
    print(f"  Win rate:        {results['win_rate_bb_per_100']:+.2f} bb/100")
    print(f"  Agent wins:      {results['agent_wins']:,}")
    print(f"  Opponent wins:   {results['opponent_wins']:,}")
    print(f"  Ties:            {results['ties']:,}")

    # Rating
    bb_per_100 = results['win_rate_bb_per_100']
    if args.opponent == 'random':
        if bb_per_100 > 5:
            rating = "GOOD - beating random"
        elif bb_per_100 > -5:
            rating = "OK - near Nash equilibrium vs random"
        else:
            rating = "POOR - losing to random"
    else:
        if bb_per_100 > 100:
            rating = "EXCELLENT - crushing weak opponent"
        elif bb_per_100 > 50:
            rating = "GOOD - exploiting weak opponent"
        elif bb_per_100 > 0:
            rating = "OK - winning"
        else:
            rating = "POOR - losing"

    print(f"\n  Rating: {rating}")
    print("=" * 60)


if __name__ == "__main__":
    main()
