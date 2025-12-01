# ARES-HU C++ Solver

High-performance C++ implementation of the ARES-HU poker solver.

## Overview

This is the production C++ implementation, designed for:
- **100x+ speedup** over Python prototype
- **Multi-threaded CFR** for parallel training
- **GPU acceleration** support (optional)
- **Memory-efficient** abstraction handling

## Architecture

```
cpp_solver/
├── src/
│   ├── core/           # Core data structures
│   │   ├── cards.cpp   # Card/deck representation
│   │   └── hand_evaluator.cpp  # 7-card evaluation (OMPEval)
│   ├── game/           # Game logic
│   │   ├── holdem_state.cpp    # Game state representation
│   │   └── action.cpp          # Action definitions
│   ├── abstraction/    # Card/betting abstraction
│   │   ├── card_abstraction.cpp     # Hand clustering
│   │   └── betting_abstraction.cpp  # Bet sizing
│   ├── cfr/            # CFR solver implementations
│   │   ├── cfr_base.cpp       # Base CFR interface
│   │   ├── dcfr_plus.cpp      # DCFR+ (Discounted CFR)
│   │   └── mccfr.cpp          # Monte Carlo CFR
│   ├── neural/         # Neural network integration
│   │   └── value_network.cpp  # PyTorch C++ (LibTorch)
│   └── utils/          # Utilities
│       ├── random.cpp         # RNG
│       └── serialization.cpp  # Save/load strategies
├── include/            # Public headers
├── tests/              # Unit tests
├── benchmarks/         # Performance benchmarks
├── third_party/        # External dependencies
│   ├── OMPEval/        # Hand evaluator
│   └── nlohmann/       # JSON library
└── docs/               # Documentation
```

## Dependencies

- **C++17** or later
- **CMake 3.16+**
- **OMPEval** - Fast hand evaluation (included as submodule)
- **LibTorch** (optional) - For neural network inference
- **OpenMP** - For multi-threaded training

## Building

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## Key Design Decisions

### 1. Hand Evaluator (OMPEval)
- Perfect hashing reduces lookup table from 36MB to 200kB
- Single lookup per 7-card hand evaluation
- Precomputed flush tables

### 2. CFR Implementation
- **DCFR+** as default (fastest convergence)
- Deferred policy averaging for better exploitability
- Regret matching with positive-only updates

### 3. Abstraction
- **Card Abstraction**: Earth Mover's Distance (EMD) clustering
- **Betting Abstraction**: Pot-relative sizing (0.5x, 1x, 2x, all-in)
- **Isomorphism**: Lossless suit canonicalization

### 4. Parallelization Strategy
- Game tree traversal parallelized across threads
- Each thread handles independent subtrees
- Lock-free regret accumulation with atomics

## Performance Targets

| Metric | Python | C++ Target |
|--------|--------|------------|
| CFR iterations/sec | 3-5 | 500-1000 |
| Memory per info set | ~1KB | ~100B |
| Startup time | 5s | <0.5s |

## Usage

```cpp
#include "cfr/dcfr_plus.hpp"
#include "game/holdem_state.hpp"

int main() {
    // Configure solver
    DCFRConfig config;
    config.iterations = 1000000;
    config.num_threads = 8;

    // Create and train
    DCFRPlusSolver solver(config);
    auto strategy = solver.train();

    // Save blueprint
    solver.save("blueprint.bin");

    return 0;
}
```

## References

- [Slumbot2019](https://github.com/ericgjackson/slumbot2019) - C++17 poker AI
- [OMPEval](https://github.com/zekyll/OMPEval) - Fast hand evaluator
- [OpenSpiel](https://github.com/google-deepmind/open_spiel) - CFR reference
- [ReBeL Paper](https://arxiv.org/abs/2007.13544) - Recursive belief-based learning
- [Pluribus](https://science.sciencemag.org/content/365/6456/885) - Superhuman multiplayer
