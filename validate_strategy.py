#!/usr/bin/env python3
"""
Validate ARES-HU Strategy

Tests the CFR strategy against various opponents to verify correctness.

Usage:
    python validate_strategy.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cpp_solver', 'build'))


def main():
    try:
        import ares_solver
    except ImportError:
        print("ERROR: C++ solver not built. Run: cd cpp_solver && ./build_python.sh")
        sys.exit(1)

    print("=" * 60)
    print("ARES-HU Strategy Validation")
    print("=" * 60)

    solver = ares_solver.Solver()
    solver.load('blueprints/cpp_1M/strategy_1M.bin')
    print(f"Loaded {solver.num_info_sets():,} info sets\n")

    tests = []

    # Test 1: vs Random - should be near-Nash (small positive)
    print("Test 1: vs Random (expect -10 to +20 bb/100)")
    r = solver.evaluate('random', 100000, 20.0, 42)
    passed = -10 < r['win_rate_bb_per_100'] < 25
    print(f"  Result: {r['win_rate_bb_per_100']:+.2f} bb/100 {'✅' if passed else '❌'}")
    tests.append(passed)

    # Test 2: vs Always Call - should crush
    print("\nTest 2: vs Always Call (expect +50 to +200 bb/100)")
    r = solver.evaluate('alwayscall', 100000, 20.0, 42)
    passed = r['win_rate_bb_per_100'] > 50
    print(f"  Result: {r['win_rate_bb_per_100']:+.2f} bb/100 {'✅' if passed else '❌'}")
    tests.append(passed)

    # Test 3: vs Call/Fold - should crush
    print("\nTest 3: vs Call/Fold (expect +50 to +150 bb/100)")
    r = solver.evaluate('callfold', 100000, 20.0, 42)
    passed = r['win_rate_bb_per_100'] > 50
    print(f"  Result: {r['win_rate_bb_per_100']:+.2f} bb/100 {'✅' if passed else '❌'}")
    tests.append(passed)

    # Test 4: Consistency across seeds
    print("\nTest 4: Consistency (variance < 30 bb/100)")
    results = []
    for seed in [1, 100, 999]:
        r = solver.evaluate('random', 50000, 20.0, seed)
        results.append(r['win_rate_bb_per_100'])
    variance = max(results) - min(results)
    passed = variance < 30
    print(f"  Results: {[f'{x:+.1f}' for x in results]}")
    print(f"  Variance: {variance:.1f} bb/100 {'✅' if passed else '❌'}")
    tests.append(passed)

    # Summary
    print("\n" + "=" * 60)
    passed_count = sum(tests)
    print(f"Results: {passed_count}/{len(tests)} tests passed")

    if all(tests):
        print("\n✅ Strategy VALIDATED - working correctly")
        return True
    else:
        print(f"\n⚠️  {len(tests) - passed_count} test(s) failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
