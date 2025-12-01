# ARES-HU: Ultimate Poker AI Roadmap

## Mission: Build a World-Class Heads-Up No-Limit Hold'em Bot

**Target Architecture**: ReBeL-style Neural Search + QRE (Quantal Response Equilibrium)

**Goal**: Any stack, any bet, any action - with human exploitation

```
┌─────────────────────────────────────────────────────────────────┐
│                      ARES-HU Architecture                       │
├─────────────────────────────────────────────────────────────────┤
│   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐       │
│   │    Value     │   │    Policy    │   │  Real-time   │       │
│   │   Network    │   │   Network    │   │    Search    │       │
│   │  (6×1536)    │   │  (6×1536)    │   │  (CFR+QRE)   │       │
│   └──────────────┘   └──────────────┘   └──────────────┘       │
│                           │                                     │
│                    ┌──────▼──────┐                              │
│                    │ Any stack,  │                              │
│                    │ Any bet,    │                              │
│                    │ Any state   │                              │
│                    └─────────────┘                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Current Progress (November 2025)

### Completed
- [x] **Multi-stack CFR training** - 10bb, 20bb, 100bb (600k samples)
- [x] **Training data pipeline** - `combine_training_data.py` with SPR normalization
- [x] **Value network** - Trained on multi-stack data (20bb RMSE)
- [x] **C++ real-time search** - `realtime_search.cpp` with QRE support
- [x] **Hand strength leaf evaluation** - Heuristic for depth-limited search
- [x] **Slumbot benchmark** - `benchmark_slumbot.py` (using urllib)

### Verification Results
```
✓ Premium hands (AA, KK, AKs) never fold
✓ CFR converges with more iterations
✓ Search performance: 63 searches/sec
✓ Infrastructure working end-to-end
```

### Benchmark Status
| Agent | Win Rate | Hands | Status |
|-------|----------|-------|--------|
| Heuristic | -389 mbb/hand | 500 | Stable baseline |
| C++ Search | -935 mbb/hand | 616 | Crashes intermittently |
| ReBeL (target) | +45 mbb/hand | - | Reference |

### Known Issues (Fixed)
1. ~~**Memory crash** - Search crashes after ~20-50 hands~~ **FIXED** (unordered_map reference invalidation)
2. **Passive play** - Search prefers calling (leaf eval needs improvement)
3. **Preflop only** - Postflop falls back to heuristic

### Next Steps for Improvement
1. Debug crash bug (likely memory corruption in node map)
2. Integrate value network for neural leaf evaluation
3. More CFR iterations (200 → 1000+)
4. Full game tree search (not just preflop)

---

## How "Any Stack, Any Bet" Works

### The Problem with Tabular CFR
```
Tabular CFR stores exact states:
  key = "AsKs|pot=40|stack=200"  → Only works for EXACT state

  20bb training ≠ 200bb play (completely different game)
  - Different SPR (stack-to-pot ratio)
  - Different implied odds
  - Different range construction
```

### The Neural Network Solution
```
Neural networks learn RELATIONSHIPS, not exact states:

Input (PBS Encoding):
  - SPR = stack / pot           # 5.0 (works for 200/40 OR 100/20)
  - pot_fraction = pot / stacks # Normalized, stack-agnostic
  - street, position            # Categorical
  - hand strength vs board      # Relative features
  - board texture               # Flush/straight potential

Output:
  - Value Network  → Expected value (continuous)
  - Policy Network → Action probabilities (7 buckets)

Key insight: Network trained on 20bb AND 100bb can INTERPOLATE to 50bb
```

### The Training Recipe
```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Step 1: Multi-Stack CFR Training                               │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Run CFR at MULTIPLE stack depths:                       │  │
│  │    5bb, 10bb, 20bb, 50bb, 100bb, 200bb                   │  │
│  │                                                          │  │
│  │  For each state visited, export:                         │  │
│  │    → PBS encoding (NORMALIZED/relative features)         │  │
│  │    → CFR strategy (action probabilities)                 │  │
│  │    → Counterfactual values (EVs)                         │  │
│  └──────────────────────────────────────────────────────────┘  │
│                           │                                     │
│                           ▼                                     │
│  Step 2: Train Neural Networks                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Value Network:  PBS → EV for each player                │  │
│  │  Policy Network: PBS → action distribution               │  │
│  │                                                          │  │
│  │  Networks learn to GENERALIZE across stack depths        │  │
│  │  because inputs are normalized/relative                  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                           │                                     │
│                           ▼                                     │
│  Step 3: Real-Time Search (at play time)                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Given ANY game state (any stack, any bet history):      │  │
│  │    1. Build subgame tree (depth 2-3 streets)             │  │
│  │    2. Run CFR search (100-1000 iterations)               │  │
│  │    3. Use VALUE NETWORK for leaf node evaluation         │  │
│  │    4. Warm-start from POLICY NETWORK                     │  │
│  │    5. Apply QRE soft regret matching                     │  │
│  │    6. Return action from converged strategy              │  │
│  └──────────────────────────────────────────────────────────┘  │
│                           │                                     │
│                           ▼                                     │
│  Step 4: Self-Play Improvement Loop                            │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Self-play with current networks generates new data      │  │
│  │  → Retrain networks on expanded dataset                  │  │
│  │  → Networks improve, self-play improves                  │  │
│  │  → Iterate until convergence                             │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Where QRE Fits In
```python
# QRE is applied during SEARCH, not training

# Standard Nash CFR (unexploitable but doesn't exploit)
def regret_match_nash(regrets):
    positive = max(0, regrets)
    return positive / sum(positive)

# QRE CFR (nearly unexploitable + exploits mistakes)
def regret_match_qre(regrets, tau=1.0):
    exp_r = exp(regrets / tau)  # Softmax
    return exp_r / sum(exp_r)

# Temperature controls exploitation level:
#   τ → 0:   Pure Nash (unexploitable, no exploitation)
#   τ = 1.0: Balanced (recommended)
#   τ = 2.0: Aggressive exploitation
```

---

## Current Status (2025-11-27)

### What's Working

**C++ CFR Solver**
- 48.7M information sets trained (1M iterations, 20bb)
- 711,000 hands/second evaluation
- 14,000 iterations/second training
- QRE equilibrium targeting

**Results vs Simple Opponents (20bb)**
| Opponent | Win Rate | Status |
|----------|----------|--------|
| Random | +2.38 bb/100 | Near-Nash |
| Always-Call | +95 bb/100 | Exploiting |
| Call-Fold | +110 bb/100 | Exploiting |

**Slumbot Benchmark (200bb)**
| Agent | Win Rate | Notes |
|-------|----------|-------|
| Call-station | -740 mbb/hand | Baseline (bad) |
| Heuristic | -88 mbb/hand | Simple rules |
| ARES (20bb CFR) | -610 mbb/hand | Stack mismatch! |
| **Target (ReBeL)** | **+45 mbb/hand** | Goal |

**Key Insight**: 20bb tabular CFR cannot generalize to 200bb Slumbot.
This is why we need neural networks + real-time search.

**Neural Network Progress**
- Value network architecture (RMSE: 6.8bb on 20bb data)
- Policy network architecture (7 GTO-standard action buckets)
- Training data export pipeline
- Slumbot API integration for validation

### Current Limitations

1. **Single Stack Depth**: CFR only trained at 20bb
   - **Solution**: Multi-stack training (Phase 3A)

2. **No Real-Time Search**: Can't adapt to unseen states
   - **Solution**: Depth-limited search with neural leaf eval (Phase 4)

3. **PBS Encoding Needs Work**: Not fully normalized for stack generalization
   - **Solution**: Improve encoding with relative features (Phase 3B)

---

## Development Phases

### Phase 1: Core Engine ✅ COMPLETE
- [x] Card representation and evaluation (32M evals/sec)
- [x] HoldemState with full game tree
- [x] Legal action generation
- [x] Information set key generation

### Phase 2: CFR Solver ✅ COMPLETE
- [x] DCFR+ with External Sampling MCCFR
- [x] QRE regret matching: `σ(a) = exp(R(a)/τ) / Σexp(R/τ)`
- [x] Binary serialization (magic: 0x41524553)
- [x] Training data export (magic: 0x54524E44)
- [x] pybind11 Python bindings

### Phase 3: Neural Networks 🔄 IN PROGRESS

#### 3A: Multi-Stack CFR Training
- [ ] Add stack depth parameter to C++ solver
- [ ] Train CFR at: 10bb, 20bb, 50bb, 100bb, 200bb
- [ ] Export training data from all depths
- [ ] ~5M iterations per depth = ~25M total iterations

#### 3B: Improved PBS Encoding
- [ ] Use normalized/relative features (SPR, pot fractions)
- [ ] Board texture features (flush/straight potential)
- [ ] Hand strength relative to board
- [ ] Target: Features that generalize across stack depths

#### 3C: Train Networks on Multi-Stack Data
- [x] Value network architecture
- [x] Policy network architecture (7 GTO buckets)
- [ ] Train on combined multi-stack dataset
- [ ] Target: RMSE < 2bb, Accuracy > 60%

### Phase 4: Real-Time Search ← CRITICAL FOR "ANY STATE"
```
Current state → Build subgame → CFR (100-1000 iters) → Neural leaf eval → Action
```
- [ ] Depth-limited subgame tree builder
- [ ] CFR search with configurable iterations
- [ ] Neural network leaf evaluation
- [ ] Warm-start from policy network
- [ ] QRE integration at search time
- [ ] Target: < 1 second for easy decisions, up to 60s for hard ones

### Phase 5: Self-Play Training Loop
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Self-Play  │ ──► │   Buffer    │ ──► │  Training   │
│   Games     │     │  (1M PBS)   │     │   Loop      │
└─────────────┘     └─────────────┘     └─────────────┘
      │                                        │
      └────────────────────────────────────────┘
                   (updated networks)
```
- [ ] Parallel self-play workers
- [ ] Prioritized experience replay
- [ ] Network vs network evaluation
- [ ] Iterative improvement until convergence

### Phase 6: Scale & Deploy
- [ ] Multi-GPU training
- [ ] TensorRT inference optimization
- [ ] API server for real-time play
- [ ] Slumbot leaderboard submission

---

## Why This Architecture

### vs Pluribus (Facebook, 2019)
- Pluribus: Tabular CFR with card/action abstractions
- **ARES**: Neural networks → no abstraction loss, continuous state space

### vs Libratus (CMU, 2017)
- Libratus: Precomputed blueprints + subgame solving
- **ARES**: Same subgame solving + neural value estimation (no blueprint needed)

### vs ReBeL (Meta, 2020)
- ReBeL: Targets Nash equilibrium (unexploitable but doesn't exploit)
- **ARES**: Same architecture with QRE → more profit vs humans/suboptimal bots

### vs GTO Wizard (2025)
- GTO Wizard: QRE for human exploitation, precomputed solutions
- **ARES**: QRE + ReBeL real-time search → adapts to any situation

---

## Technical Specifications

### Neural Networks (Target: ReBeL-style)
```
Value Network:
  Input:  PBS encoding (~2500 dims, normalized)
  Hidden: 6 × [Linear(1536) → LayerNorm → ReLU]
  Output: Player EVs (2 values)
  Params: ~15M

Policy Network:
  Input:  PBS encoding
  Hidden: 6 × [Linear(1536) → LayerNorm → ReLU]
  Output: Action probabilities (7 buckets)
  Params: ~15M
```

### PBS Encoding (Stack-Agnostic)
```
Normalized Features (generalize across stacks):
  - SPR: stack / pot                    # Stack-to-pot ratio
  - pot_fraction: pot / starting_stacks # How much committed
  - to_call_fraction: to_call / pot     # Pot odds denominator
  - stack_ratio: my_stack / opp_stack   # Relative stack sizes

Categorical Features:
  - Street: [preflop, flop, turn, river]
  - Position: [IP, OOP]

Hand/Board Features:
  - Hero hole cards (52-dim one-hot or embedding)
  - Board cards (52-dim one-hot)
  - Hand strength bucket (0-1 normalized)
  - Board texture (paired, suited, connected, etc.)
```

### QRE Parameters
```
Temperature (τ):
  τ → 0:   Nash equilibrium (unexploitable)
  τ = 1.0: Balanced QRE (recommended default)
  τ = 2.0: More exploitative (vs weak opponents)

Current: τ = 1.0
```

### Bet Sizing Abstraction (GTO-Standard)
```
Action Buckets (7 total):
  0: Fold
  1: Check/Call
  2: Bet Small    (25-40% pot)  - probe, block, range bet
  3: Bet Medium   (40-75% pot)  - standard value/bluff
  4: Bet Large    (75-125% pot) - geometric, polarized
  5: Overbet      (125%+ pot)   - max pressure, nut advantage
  6: All-in

References:
  - GTO Wizard: https://blog.gtowizard.com/pot-geometry/
  - Geometric sizing: (The Mathematics of Poker, Chen & Ankenman)
```

### Real-Time Search Parameters
```
Iterations by decision difficulty:
  - Easy (clear value/bluff): 100-200 iterations, <1s
  - Medium (marginal spots): 500-1000 iterations, 5-10s
  - Hard (complex multi-way): 2000+ iterations, up to 60s

Depth limits:
  - Preflop: Search to river (full depth)
  - Flop: Search to river (3 streets)
  - Turn: Search to river (2 streets)
  - River: Full solve (1 street)
```

### Training Resources
| Option | GPUs | Training Time | Cost |
|--------|------|---------------|------|
| Local (dev) | 1× M1/M2 | 2-4 weeks | $0 |
| Cloud mid | 4× A100 | 3-5 days | ~$1000 |
| Cloud full | 8× A100 | 1-2 days | ~$2500 |

---

## Validation & Benchmarks

### Slumbot (Primary Benchmark)
- 2018 ACPC Champion
- 200bb deep, heads-up NLHE
- API: slumbot.com

Reference rates:
| Bot | Win Rate | Year |
|-----|----------|------|
| ReBeL (Meta) | +45 mbb/hand | 2020 |
| Ruse | +194 mbb/hand | 2024 |
| Human pro (estimate) | -50 to +50 mbb/hand | - |

### Exploitability Measurement
- GTO Wizard reports 0.12% pot exploitability
- Target: < 1% pot exploitability
- Measured via best-response calculation

---

## References

### Core Papers
- [ReBeL (Meta, 2020)](https://arxiv.org/abs/2007.13544) - RL + Search for imperfect information
- [Deep CFR (2019)](https://arxiv.org/abs/1811.00164) - Neural CFR without abstraction
- [Pluribus (2019)](https://www.science.org/doi/10.1126/science.aay2400) - 6-player superhuman poker
- [Libratus (2017)](https://www.science.org/doi/10.1126/science.aao1733) - First superhuman HU bot

### QRE Research
- [QRE Original (McKelvey, 1995)](https://www.jstor.org/stable/2950883)
- [GTO Wizard QRE (2025)](https://blog.gtowizard.com/introducing-quantal-response-equilibrium-the-next-evolution-of-gto/)

### Implementation Resources
- [PokerRL](https://github.com/EricSteinberger/PokerRL)
- [OpenSpiel](https://github.com/deepmind/open_spiel)

---

## Commands

```bash
# Build
cd cpp_solver && ./build.sh && ./build_python.sh

# Train CFR (C++) - single stack
./cpp_solver/build/train_cfr --iterations 1000000 --stack 20 --output strategy_20bb.bin

# Train CFR - multi-stack (TODO)
for stack in 10 20 50 100 200; do
    ./cpp_solver/build/train_cfr --iterations 1000000 --stack $stack --output strategy_${stack}bb.bin
done

# Evaluate vs simple opponents
python evaluate_agent.py --hands 100000

# Benchmark vs Slumbot
python benchmark_slumbot.py --hands 1000 --agent ares

# Train value network
python train_value_network.py --epochs 100

# Train policy network
python train_policy_network.py --epochs 100

# Validate strategy
python validate_strategy.py
```
