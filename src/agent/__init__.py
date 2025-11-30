"""
Agent components for ARES-HU.

Available agents:
- BaseAgent: Abstract base class
- RandomAgent: Baseline random player
- CallFoldAgent: Simple baseline (calls/folds only)
- CppCFRAgent: Uses C++ CFR blueprint strategy

Usage:
    from src.agent import CppCFRAgent, RandomAgent

    agent = CppCFRAgent('blueprints/cpp_1M/strategy_1M.bin')
"""

from src.agent.base_agent import BaseAgent
from src.agent.random_agent import RandomAgent, CallFoldAgent
from src.agent.cpp_cfr_agent import CppCFRAgent

__all__ = [
    'BaseAgent',
    'RandomAgent',
    'CallFoldAgent',
    'CppCFRAgent',
]
