"""
Python loader for C++ CFR training data export.

The C++ solver exports binary training data in the following format:
- Header (16 bytes):
  - num_samples: int32
  - pbs_dim: int32
  - padding: int64
- Data (repeated for each sample):
  - pbs: float32[pbs_dim]
  - value_p0: float32
  - value_p1: float32

This module provides efficient loading and conversion to PyTorch tensors.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import struct


def load_cpp_training_data(
    path: str,
    max_samples: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load training data exported from C++ CFR solver.

    Args:
        path: Path to binary training data file
        max_samples: Maximum number of samples to load (None = all)

    Returns:
        Tuple of (pbs_data, values_p0, values_p1) numpy arrays
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Training data not found: {path}")

    with open(path, 'rb') as f:
        # Read header: magic(4) + version(4) + num_samples(4) + pbs_dim(4) = 16 bytes
        header = f.read(16)
        magic, version, num_samples, pbs_dim = struct.unpack('IIII', header)

        # Verify magic number
        if magic != 0x54524E44:  # "TRND"
            raise ValueError(f"Invalid file format: magic={hex(magic)}, expected 0x54524E44")

        if max_samples is not None:
            num_samples = min(num_samples, max_samples)

        # Calculate sample size
        sample_size = pbs_dim * 4 + 8  # pbs_dim floats + 2 value floats

        # Read all data
        pbs_data = np.zeros((num_samples, pbs_dim), dtype=np.float32)
        values_p0 = np.zeros(num_samples, dtype=np.float32)
        values_p1 = np.zeros(num_samples, dtype=np.float32)

        for i in range(num_samples):
            # Read PBS encoding
            pbs_bytes = f.read(pbs_dim * 4)
            pbs_data[i] = np.frombuffer(pbs_bytes, dtype=np.float32)

            # Read values
            val_bytes = f.read(8)
            values_p0[i], values_p1[i] = struct.unpack('ff', val_bytes)

    return pbs_data, values_p0, values_p1


def load_cpp_training_data_fast(
    path: str,
    max_samples: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fast loading using memory mapping (more efficient for large files).

    Args:
        path: Path to binary training data file
        max_samples: Maximum number of samples to load (None = all)

    Returns:
        Tuple of (pbs_data, values_p0, values_p1) numpy arrays
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Training data not found: {path}")

    # Read header first: magic(4) + version(4) + num_samples(4) + pbs_dim(4)
    with open(path, 'rb') as f:
        header = f.read(16)
        magic, version, num_samples, pbs_dim = struct.unpack('IIII', header)

        if magic != 0x54524E44:
            raise ValueError(f"Invalid file format: magic={hex(magic)}")

    if max_samples is not None:
        num_samples = min(num_samples, max_samples)

    # Memory map the file for fast reading
    # Data starts at offset 16 (after header)
    sample_size = pbs_dim + 2  # floats per sample

    # Load data using numpy
    with open(path, 'rb') as f:
        f.seek(16)  # Skip header

        # Read all data as flat float32 array
        total_floats = num_samples * sample_size
        flat_data = np.fromfile(f, dtype=np.float32, count=total_floats)

    # Reshape and extract components
    data = flat_data.reshape(num_samples, sample_size)
    pbs_data = data[:, :pbs_dim]
    values_p0 = data[:, pbs_dim]
    values_p1 = data[:, pbs_dim + 1]

    return pbs_data, values_p0, values_p1


class CppTrainingDataset:
    """
    PyTorch-compatible dataset for C++ exported training data.
    """

    def __init__(
        self,
        path: str,
        player: int = 0,
        max_samples: Optional[int] = None
    ):
        """
        Initialize dataset from C++ export file.

        Args:
            path: Path to binary training data
            player: Which player's values to use (0 or 1)
            max_samples: Maximum samples to load
        """
        self.pbs, self.values_p0, self.values_p1 = load_cpp_training_data_fast(
            path, max_samples
        )
        self.player = player

    def __len__(self) -> int:
        return len(self.pbs)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get (pbs, value) pair."""
        pbs = self.pbs[idx]
        value = self.values_p0[idx] if self.player == 0 else self.values_p1[idx]
        return pbs, np.array([value], dtype=np.float32)

    @property
    def values(self) -> np.ndarray:
        """Get all values for the selected player."""
        return self.values_p0 if self.player == 0 else self.values_p1

    def get_torch_tensors(self):
        """Convert to PyTorch tensors for training."""
        import torch
        X = torch.from_numpy(self.pbs).float()
        y = torch.from_numpy(self.values.reshape(-1, 1)).float()
        return X, y

    def split(self, train_ratio: float = 0.8):
        """Split into train/validation sets."""
        n = len(self)
        indices = np.random.permutation(n)
        split_idx = int(n * train_ratio)

        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]

        train_pbs = self.pbs[train_indices]
        train_vals = self.values[train_indices]

        val_pbs = self.pbs[val_indices]
        val_vals = self.values[val_indices]

        return (train_pbs, train_vals), (val_pbs, val_vals)


def test_cpp_data_loader():
    """Test the C++ data loader."""
    print("Testing C++ data loader...")

    # Create dummy test data
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "test_data.bin")

        # Write test data
        num_samples = 100
        pbs_dim = 256

        with open(test_file, 'wb') as f:
            # Header: magic + version + num_samples + pbs_dim
            magic = 0x54524E44  # "TRND"
            version = 1
            f.write(struct.pack('IIII', magic, version, num_samples, pbs_dim))

            # Data
            for i in range(num_samples):
                # PBS (random)
                pbs = np.random.randn(pbs_dim).astype(np.float32)
                f.write(pbs.tobytes())

                # Values
                v0 = float(i) / num_samples
                v1 = -float(i) / num_samples
                f.write(struct.pack('ff', v0, v1))

        # Test loading
        pbs, v0, v1 = load_cpp_training_data(test_file)
        print(f"  Loaded {len(pbs)} samples")
        print(f"  PBS shape: {pbs.shape}")
        print(f"  Values P0: min={v0.min():.3f}, max={v0.max():.3f}")
        print(f"  Values P1: min={v1.min():.3f}, max={v1.max():.3f}")

        # Test fast loading
        pbs2, v02, v12 = load_cpp_training_data_fast(test_file)
        assert np.allclose(pbs, pbs2), "Fast loading mismatch in PBS"
        assert np.allclose(v0, v02), "Fast loading mismatch in P0 values"
        assert np.allclose(v1, v12), "Fast loading mismatch in P1 values"
        print("  Fast loading: OK")

        # Test dataset
        dataset = CppTrainingDataset(test_file, player=0)
        print(f"  Dataset size: {len(dataset)}")

        pbs_sample, val_sample = dataset[0]
        print(f"  Sample: PBS {pbs_sample.shape}, value {val_sample.shape}")

        # Test torch conversion
        try:
            X, y = dataset.get_torch_tensors()
            print(f"  Torch tensors: X={X.shape}, y={y.shape}")
        except ImportError:
            print("  (PyTorch not available)")

        print("\n✅ C++ data loader test complete!")


if __name__ == "__main__":
    test_cpp_data_loader()
