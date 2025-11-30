"""
Neural network components for ARES-HU.

Provides:
- CppTrainingDataset: Load training data exported from C++ solver
- ValueNetwork: Neural network for value prediction
- ValueNetworkTrainer: Training loop with early stopping
- PolicyNetwork: Neural network for action probability prediction
- PolicyNetworkTrainer: Training loop for policy network
"""

from src.neural.cpp_data_loader import CppTrainingDataset
from src.neural.value_network import ValueNetwork, ValueNetworkTrainer
from src.neural.policy_network import (
    PolicyNetwork,
    PolicyNetworkTrainer,
    NUM_ACTION_BUCKETS,
    ACTION_FOLD,
    ACTION_CHECK_CALL,
    ACTION_BET_SMALL,
    ACTION_BET_MEDIUM,
    ACTION_BET_LARGE,
    ACTION_BET_OVERBET,
    ACTION_ALL_IN,
    map_action_to_bucket,
    bucket_to_action_name,
)

__all__ = [
    'CppTrainingDataset',
    'ValueNetwork',
    'ValueNetworkTrainer',
    'PolicyNetwork',
    'PolicyNetworkTrainer',
    'NUM_ACTION_BUCKETS',
    'ACTION_FOLD',
    'ACTION_CHECK_CALL',
    'ACTION_BET_SMALL',
    'ACTION_BET_MEDIUM',
    'ACTION_BET_LARGE',
    'ACTION_BET_OVERBET',
    'ACTION_ALL_IN',
    'map_action_to_bucket',
    'bucket_to_action_name',
]
