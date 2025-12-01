# Python to C++ Mapping Reference

This document maps the Python prototype to the C++ implementation.

## Component Mapping

| Python Module | C++ Component | Notes |
|---------------|---------------|-------|
| `src/game/cards.py` | `include/ares/core/cards.hpp` | Card representation |
| `src/game/holdem_state.py` | `include/ares/game/holdem_state.hpp` | Game state |
| `src/game/abstracted_game.py` | `src/abstraction/` | Action abstraction |
| `src/solver/mccfr_holdem.py` | `include/ares/cfr/dcfr_solver.hpp` | CFR solver |
| `src/card_abstraction/` | `src/abstraction/card_abstraction.cpp` | Hand clustering |
| `src/neural/value_network.py` | `src/neural/value_network.cpp` | LibTorch |

## Key Class Mappings

### Cards

```python
# Python
class Card(IntEnum):
    TWO_CLUBS = 0
    ...

def card_to_string(card: int) -> str: ...
```

```cpp
// C++
using Card = uint8_t;

constexpr Card make_card(Rank r, Suit s);
std::string card_to_string(Card c);
```

### Game State

```python
# Python
class HoldemState:
    def __init__(self, starting_stack: float = 20.0): ...
    def apply_action(self, action: Action) -> 'HoldemState': ...
    def get_legal_actions(self) -> List[Action]: ...
```

```cpp
// C++
class HoldemState {
public:
    static HoldemState create_initial(const Config& config = {});
    HoldemState apply_action(const Action& action) const;
    std::vector<Action> get_legal_actions() const;
};
```

### CFR Solver

```python
# Python
class DCFRPlusSolver:
    def __init__(self, game: AbstractedGame, config: DCFRConfig): ...
    def train(self) -> Dict[str, np.ndarray]: ...
```

```cpp
// C++
class DCFRPlusSolver {
public:
    explicit DCFRPlusSolver(const DCFRConfig& config);
    void train();
    const StrategyMap& get_strategy() const;
};
```

## Performance Differences

| Operation | Python | C++ (Expected) |
|-----------|--------|----------------|
| Card operations | ~1M/sec | ~1B/sec |
| Hand evaluation | ~1M/sec | ~200M/sec |
| CFR iteration | 3-5/sec | 500-1000/sec |
| Memory/info set | ~1KB | <100 bytes |

## Data Format Compatibility

### Blueprint Format

The C++ solver can read Python blueprints via:
1. Direct pickle loading (slow, for development)
2. Converted binary format (fast, for production)

```bash
# Convert Python blueprint to C++ format
python scripts/convert_blueprint.py \
    --input blueprints/20bb_100k/blueprint.pkl \
    --output blueprints/20bb_100k/blueprint.bin
```

### Strategy Exchange

Both use the same strategy representation:
- Key: Info set hash (uint64)
- Value: Action probabilities (float array)

## Training Workflow

### Development (Python)
```bash
# Quick iteration, debugging
python train_full.py --preset quick
```

### Production (C++)
```bash
# High-performance training
./build/train_cfr \
    --iterations 10000000 \
    --threads 8 \
    --checkpoint-dir output/
```

### Hybrid (Python + C++)
```python
# Use C++ solver from Python
import ares_solver

solver = ares_solver.DCFRPlusSolver(config)
solver.train()
strategy = solver.get_strategy()  # Returns Python dict
```

## Migration Checklist

- [ ] Verify card representation matches
- [ ] Test game state transitions
- [ ] Compare CFR values after N iterations
- [ ] Benchmark performance improvement
- [ ] Validate agent behavior matches
