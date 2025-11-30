"""
CFR solver implementations

This module provides both Python and C++ CFR solver implementations.

For high-performance usage (48M+ info sets, 500k+ hands/sec evaluation),
use the C++ solver via:

    from src.solver import CppSolver

    solver = CppSolver()
    solver.load('blueprints/cpp_1M/strategy_1M.bin')
    results = solver.evaluate(opponent='random', num_hands=100000)

For development/debugging, use the pure Python implementations:
    from src.solver.cfr import CFRSolver
    from src.solver.mccfr_holdem import MCCFRHoldem
"""

import sys
import os

# Try to import the C++ solver
_cpp_build_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    'cpp_solver', 'build'
)

if _cpp_build_dir not in sys.path:
    sys.path.insert(0, _cpp_build_dir)

try:
    from ares_solver import Solver as CppSolver
    HAS_CPP_SOLVER = True
except ImportError:
    CppSolver = None
    HAS_CPP_SOLVER = False


def get_cpp_solver():
    """
    Get the C++ solver class, or raise an error if not built.

    Returns:
        The CppSolver class from the C++ pybind11 module.

    Raises:
        ImportError: If the C++ module is not built.
    """
    if CppSolver is None:
        raise ImportError(
            "C++ solver not available. Build it first:\n"
            "  cd cpp_solver && ./build_python.sh"
        )
    return CppSolver


__all__ = ['CppSolver', 'HAS_CPP_SOLVER', 'get_cpp_solver']
