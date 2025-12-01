# ARES-HU C++ Solver Architecture

## Overview

This document describes the architecture of the C++ solver, designed for
high-performance CFR training and inference.

## Design Philosophy

1. **Performance First**: Every design decision prioritizes speed
2. **Memory Efficiency**: Compact representations for billions of info sets
3. **Parallelization**: Multi-threaded from the ground up
4. **Modularity**: Components can be swapped (e.g., different CFR variants)

## Component Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Application Layer                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │  train_cfr   │  │  play_agent  │  │   Python Bindings    │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                         Solver Layer                             │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    CFR Engine                            │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐  │    │
│  │  │  DCFR+   │  │  MCCFR   │  │ External │  │ Subgame │  │    │
│  │  │ Solver   │  │ Solver   │  │ Sampling │  │ Solver  │  │    │
│  │  └──────────┘  └──────────┘  └──────────┘  └─────────┘  │    │
│  └─────────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────────┤
│                       Abstraction Layer                          │
│  ┌──────────────────┐  ┌─────────────────────────────────────┐  │
│  │  Card Abstraction │  │       Betting Abstraction          │  │
│  │  - EMD Clustering │  │  - Pot-relative sizes              │  │
│  │  - Isomorphism    │  │  - Raise multipliers               │  │
│  │  - Bucket LUT     │  │  - All-in threshold                │  │
│  └──────────────────┘  └─────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                          Game Layer                              │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐  │
│  │   HoldemState    │  │     Action       │  │   InfoSet    │  │
│  │  - Board         │  │  - Type          │  │  - Key       │  │
│  │  - Stacks        │  │  - Amount        │  │  - Regrets   │  │
│  │  - Pot           │  │  - All-in flag   │  │  - Strategy  │  │
│  └──────────────────┘  └──────────────────┘  └──────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                          Core Layer                              │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐  │
│  │      Cards       │  │  HandEvaluator   │  │     RNG      │  │
│  │  - Card repr     │  │  - 7-card eval   │  │  - XorShift  │  │
│  │  - Deck          │  │  - Perfect hash  │  │  - Thread-   │  │
│  │  - Canonicalize  │  │  - Flush tables  │  │    safe      │  │
│  └──────────────────┘  └──────────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. Card Representation (`core/cards.hpp`)

```cpp
// Compact card representation (6 bits)
// Bits 0-3: Rank (0-12 for 2-A)
// Bits 4-5: Suit (0-3 for c/d/h/s)
using Card = uint8_t;

// Hand is a bitset for fast operations
using Hand = uint64_t;  // 52 bits used

// Fast card construction
constexpr Card make_card(Rank r, Suit s) {
    return (s << 4) | r;
}
```

### 2. Hand Evaluator (`core/hand_evaluator.hpp`)

Based on OMPEval's perfect hash approach:

```cpp
class HandEvaluator {
public:
    // Evaluate 7-card hand to rank (lower is better)
    // Returns value in [1, 7462] representing hand strength
    static uint16_t evaluate(Hand hand);

    // Batch evaluation for SIMD
    static void evaluate_batch(const Hand* hands, uint16_t* ranks, size_t n);

private:
    // Perfect hash tables (loaded at startup)
    static std::array<uint16_t, 8192> flush_table;
    static std::array<uint16_t, 49205> noflush_table;
};
```

### 3. Game State (`game/holdem_state.hpp`)

```cpp
struct HoldemState {
    // Board cards (up to 5)
    std::array<Card, 5> board;
    uint8_t board_count;

    // Player hands
    std::array<std::array<Card, 2>, 2> hands;

    // Betting state
    std::array<float, 2> stacks;
    std::array<float, 2> bets;
    float pot;

    // Game flow
    uint8_t current_player;
    Street street;
    bool is_terminal;

    // Methods
    bool is_chance_node() const;
    std::vector<Action> get_legal_actions() const;
    HoldemState apply_action(Action action) const;
    float get_utility(int player) const;
};
```

### 4. Information Set (`game/info_set.hpp`)

```cpp
struct InfoSetKey {
    // Canonical representation for hashing
    uint64_t hand_bucket;     // Card abstraction bucket
    uint32_t action_sequence; // Betting history encoding

    bool operator==(const InfoSetKey& other) const;
};

// Custom hash for unordered_map
struct InfoSetHash {
    size_t operator()(const InfoSetKey& key) const;
};

struct InfoSetData {
    std::vector<float> regrets;   // Cumulative regrets
    std::vector<float> strategy;  // Cumulative strategy
    uint32_t visits;              // For importance sampling
};

using StrategyMap = std::unordered_map<InfoSetKey, InfoSetData, InfoSetHash>;
```

### 5. CFR Solver (`cfr/dcfr_plus.hpp`)

```cpp
class DCFRPlusSolver {
public:
    struct Config {
        int iterations = 1000000;
        int num_threads = 8;
        float alpha = 1.5;   // Discount factor for positive regrets
        float beta = 0.0;    // Discount factor for negative regrets
        float gamma = 1.0;   // Strategy contribution weight
        int defer_averaging = 1000;
        std::string checkpoint_dir;
    };

    explicit DCFRPlusSolver(const Config& config);

    // Main training loop
    void train();

    // Get average strategy
    const StrategyMap& get_strategy() const;

    // Serialization
    void save(const std::string& path) const;
    static DCFRPlusSolver load(const std::string& path);

private:
    // Parallel CFR iteration
    void iterate(int iteration);

    // Recursive tree traversal
    float cfr_traverse(
        HoldemState& state,
        int traverser,
        std::array<float, 2> reach_probs,
        int thread_id
    );

    // Regret matching
    std::vector<float> compute_strategy(const InfoSetData& data);

    Config config_;
    StrategyMap strategy_;
    std::vector<std::mt19937> rngs_;  // Per-thread RNG
};
```

### 6. Abstraction (`abstraction/card_abstraction.hpp`)

```cpp
class CardAbstraction {
public:
    struct Config {
        int preflop_buckets = 169;   // Can reduce for memory
        int flop_buckets = 200;
        int turn_buckets = 200;
        int river_buckets = 200;
    };

    explicit CardAbstraction(const Config& config);

    // Get bucket for hand at given street
    int get_bucket(const std::array<Card, 2>& hand,
                   const std::span<Card> board,
                   Street street) const;

    // Precompute all buckets
    void compute_buckets();

private:
    // EMD-based clustering
    void cluster_preflop();
    void cluster_postflop(Street street);

    // Equity calculation
    float calculate_equity(const std::array<Card, 2>& hand,
                          const std::span<Card> board) const;

    // Lookup tables
    std::array<int, 1326> preflop_lut_;  // 52c2 hands
    std::unordered_map<uint64_t, int> postflop_luts_[3];

    Config config_;
};
```

## Memory Layout

### Info Set Storage

For 10M info sets with 5 actions:
- Regrets: 5 × float = 20 bytes
- Strategy: 5 × float = 20 bytes
- Metadata: 8 bytes
- Total per info set: ~48 bytes

With 10M info sets: **480 MB**

### Optimization: Compressed Strategy

For large-scale training, use compressed representation:

```cpp
struct CompressedInfoSet {
    // Store only significant actions (regret > 0)
    std::vector<std::pair<uint8_t, int16_t>> nonzero_regrets;
    uint32_t total_visits;
};
```

This reduces memory by 60-80% for sparse strategies.

## Parallelization Strategy

### Thread Pool Architecture

```
Main Thread
    │
    ├── Worker Thread 0
    │   └── Traverse subtree (player 0 hands)
    ├── Worker Thread 1
    │   └── Traverse subtree (player 0 hands)
    ├── Worker Thread 2
    │   └── Traverse subtree (player 1 hands)
    └── Worker Thread N
        └── Traverse subtree (player 1 hands)
```

### Lock-Free Regret Updates

Using atomic operations for thread safety:

```cpp
void update_regrets(InfoSetData& data, const std::vector<float>& updates) {
    for (size_t i = 0; i < updates.size(); ++i) {
        // Atomic fetch-add
        std::atomic_ref<float> regret(data.regrets[i]);
        regret.fetch_add(updates[i], std::memory_order_relaxed);
    }
}
```

## File Formats

### Blueprint Format (`.bin`)

```
Header (64 bytes):
  - Magic: "ARES" (4 bytes)
  - Version: uint32
  - Iterations: uint64
  - Num info sets: uint64
  - Stack size: float
  - Reserved: remaining bytes

Info Set Entries:
  - Key: uint64 (packed)
  - Num actions: uint8
  - Strategy: float[] (normalized)
```

### Checkpoint Format

Same as blueprint but includes:
- Cumulative regrets
- Cumulative strategy sums
- Iteration count
- Discount parameters

## Integration with Python

Using pybind11 for Python bindings:

```cpp
PYBIND11_MODULE(ares_solver, m) {
    py::class_<DCFRPlusSolver>(m, "DCFRPlusSolver")
        .def(py::init<const DCFRPlusSolver::Config&>())
        .def("train", &DCFRPlusSolver::train)
        .def("save", &DCFRPlusSolver::save)
        .def("get_strategy", &DCFRPlusSolver::get_strategy);

    py::class_<HoldemState>(m, "HoldemState")
        .def("get_legal_actions", &HoldemState::get_legal_actions)
        .def("apply_action", &HoldemState::apply_action)
        .def("is_terminal", &HoldemState::is_terminal);
}
```

## Performance Optimizations

### 1. Cache-Friendly Traversal
- Sort hands by bucket to improve cache locality
- Use arena allocation for temporary states

### 2. SIMD Hand Evaluation
- Process 4/8 hands simultaneously with AVX2
- Batch equity calculations

### 3. Memory-Mapped Strategy
- Use mmap for large blueprints
- Demand paging for rarely-accessed info sets

### 4. Profile-Guided Optimization (PGO)
- Build with `-fprofile-generate`
- Run representative workload
- Rebuild with `-fprofile-use`

## Benchmarks Target

| Operation | Target | Notes |
|-----------|--------|-------|
| Hand evaluation | 200M/sec | Per thread |
| CFR iteration | 1K/sec | 8 threads, full game |
| Strategy lookup | 100M/sec | Cache-hot |
| Blueprint load | <500ms | 10M info sets |

## Future Enhancements

1. **GPU Acceleration**: CUDA kernels for regret/strategy updates
2. **Distributed Training**: MPI for multi-node training
3. **Mixed Precision**: FP16 for regrets to halve memory
4. **Sparse Representation**: Only store non-zero regrets
