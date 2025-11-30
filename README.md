# ARES-HU: Heads-Up No-Limit Hold'em Poker AI

A high-performance poker AI using CFR (Counterfactual Regret Minimization) with real-time search and QRE (Quantal Response Equilibrium) targeting.

## Current Status (Nov 2024)

### What's Working
- **DCFR+ Algorithm**: Same as ReBeL/Pluribus, with QRE for human exploitation
- **C++ CFR Solver**: Fast training (14K iter/s), 711K hands/sec evaluation
- **Real-time Search**: From ANY game state (any stack, any bet, any board)
- **Neural Leaf Evaluation**: Value network integrated into C++ search via TorchScript
- **Multi-stack Training**: Blueprints for 10bb, 20bb, 100bb, 200bb
- **Slumbot Integration**: Can play hands against Slumbot (2018 ACPC Champion)

### Architecture vs State-of-the-Art

| Component | ARES-HU | GTO Wizard | ReBeL | Notes |
|-----------|---------|------------|-------|-------|
| CFR Algorithm | DCFR+ | CFR + NN | DCFR | All equivalent |
| Neural Network | Value net (leaf eval) | Value net (future streets) | Value + Policy | GTO Wizard predicts EVs for ALL hands |
| Real-time Search | Yes | Yes | Yes | Search at decision time |
| QRE Exploitation | Yes (τ tunable) | Yes (as of 2025) | Yes | **Same formula as GTO Wizard** |
| Card Abstraction | None | None (NN replaces it) | None | **Neural nets replace abstraction** |
| Solving Speed | ~100 iter/s | 3 sec/street | ~200 iter/s | GTO Wizard is fastest |
| Win vs Slumbot | TBD | +19.4 bb/100 | +4.5 bb/100 | GTO Wizard is current best |

## How Neural Networks Replace Abstraction

Traditional poker solvers use **card abstraction** (grouping similar hands into buckets) to reduce memory. Modern AI solvers like GTO Wizard and ReBeL use **neural networks instead**:

### Traditional Approach (Libratus/Pluribus)
```
1. Group hands into buckets using EMD clustering
2. Train CFR on abstracted game (smaller)
3. At play time, map real hands to buckets
4. Problem: Bucket boundaries create exploitable artifacts
```

### Modern Approach (GTO Wizard/ReBeL/ARES)
```
1. Train CFR on full game (no card abstraction)
2. Use neural network to predict EVs for future streets
3. Only solve current street exactly
4. Neural net "intuits" value of any hand in any situation
```

**Key insight**: The neural network learns a continuous representation of hand values, eliminating the need for discrete buckets. GTO Wizard states: "AI-solvers do not actually calculate the strategy on future streets. Rather, they predict the EV of future streets using neural nets."

### What This Means for ARES

We already have the right architecture:
- **Real-time search** with neural leaf evaluation
- **PBS encoding** captures full game state (3040 dims)
- **Value network** predicts EVs at leaf nodes

What we need:
1. **Better neural network** - Trained on more data, larger model
2. **More training iterations** - Generate better training labels
3. **Potentially**: Predict EVs for ALL hands in range (like GTO Wizard), not just our hand

## Benchmark Targets

| Bot | Win Rate vs Slumbot | Notes |
|-----|---------------------|-------|
| GTO Wizard AI | +19.4 bb/100 | Current world's best, 150K hands |
| ReBeL (Meta) | +4.5 bb/100 | Academic benchmark |
| Slumbot | 0 (baseline) | 2018 ACPC Champion |
| ARES-HU | TBD | Stack mismatch limits current tests |

## Optimization Roadmap (Target: <$5K Compute)

### Goal: Beat Slumbot (+10 bb/100) with minimal compute

**Reference**: AlphaHoldem beat Slumbot (+111 bb/100) and DeepStack (+16 bb/100) with only **3 days training on a single PC**.

### Current Optimizations ✅

| Optimization | Status | Impact |
|--------------|--------|--------|
| C++ Implementation | ✅ Done | Baseline |
| DCFR+ Algorithm | ✅ Done | Fastest CFR variant |
| -O3 -march=native | ✅ Done | Full compiler optimization |
| External Sampling MCCFR | ✅ Done | Variance reduction vs vanilla CFR |
| QRE Regret Matching | ✅ Done | Same as GTO Wizard (2025) |
| TorchScript Inference | ✅ Done | C++ neural eval |
| Deferred Strategy Averaging | ✅ Done | Skip first 1000 iterations |

### High-Impact Optimizations (TODO)

| Optimization | Status | Impact | Effort |
|--------------|--------|--------|--------|
| **VR-MCCFR (Variance Reduction)** | ❌ Missing | **100x speedup** | Medium |
| **Linear CFR Warm-up** | ⚠️ Code exists, not used | 2-4x faster convergence | Low |
| **Regret Pruning** | ❌ Missing | 10x speedup (Pluribus uses) | Low |
| **OpenMP Parallel CFR** | ⚠️ Config exists | 4-8x faster | Low |
| **OMPEval Hand Evaluator** | ❌ Missing | 10x faster (775M/s vs 711K/s) | Medium |
| **Knowledge Distillation** | ❌ Missing | 100x smaller model | Medium |

### Algorithm Optimizations (Research-Backed)

#### 1. VR-MCCFR (Variance Reduction) - **100x Speedup**
From [VR-MCCFR Paper](https://arxiv.org/abs/1809.03057):
> "VR-MCCFR brings an order of magnitude speedup, while empirical variance decreases by three orders of magnitude. The decreased variance allows CFR+ to be used with sampling, increasing speedup to two orders of magnitude."

**Implementation**: Add baseline value estimates that bootstrap from other estimates within the same episode.

#### 2. Linear CFR Warm-up (Pluribus) - **2-4x Faster**
From [Pluribus Paper](https://noambrown.github.io/papers/19-Science-Superhuman_Supp.pdf):
> "Pluribus used linear weighting for first 400 minutes. Linear CFR assigns weight proportional to T, so influence of first iteration decays faster."

**Status**: Code exists in `regret_minimizer.hpp:154` but not used in solver.

#### 3. Regret Pruning - **10x Speedup**
From Pluribus:
> "Actions with extremely negative regret are not explored in 95% of iterations."

**Implementation**: Skip actions where regret < -300M (Pluribus threshold).

#### 4. Final Iteration Strategy
From Pluribus:
> "Pluribus plays according to the strategy on the final iteration rather than the weighted average strategy."

**Benefit**: Avoids poor actions not fully eliminated in weighted average.

### Implementation Optimizations

#### 5. OpenMP Parallel CFR - **4-8x Speedup**
**Status**: CMakeLists.txt has `ARES_USE_OPENMP` option but not used in solver loop.

#### 6. OMPEval Hand Evaluator - **1000x Faster**
From [OMPEval](https://github.com/zekyll/OMPEval):
> "691-775 million evaluations per second" vs current 711K/s

**Implementation**: Replace custom evaluator with OMPEval's perfect hash lookup.

#### 7. SIMD Batch Evaluation - **4-8x Speedup**
Process multiple hands simultaneously with AVX2/NEON instructions.

### Neural Network Optimizations

#### 8. Knowledge Distillation - **100x Smaller Model**
Train large teacher network (30M params), distill to small student (300K params).
> "98.4% reduction in learnable parameters with minimal accuracy loss"

#### 9. Mixed Precision Training (FP16) - **2x Faster**
Use half-precision for training, full precision for inference.

#### 10. Efficient Architecture
- Current: 4×256 (1M params)
- Target: 4×512 with residual connections (4M params, better accuracy)
- Or: Distilled 4×128 (250K params, same speed as heuristic)

### Training Strategy

#### Phase 1: Optimize Implementation (1-2 days)
1. Enable OpenMP parallelization
2. Implement regret pruning
3. Use Linear CFR warm-up
4. Switch to OMPEval

**Expected**: 10-50x faster CFR training

#### Phase 2: VR-MCCFR (3-5 days)
1. Implement baseline value tracking
2. Bootstrap estimates along trajectory
3. Test on small game first

**Expected**: Additional 10-100x speedup

#### Phase 3: Neural Network (2-3 days)
1. Train larger network (4×512 or 6×1024)
2. Knowledge distillation to small model
3. Multi-stack training (10-200bb)

#### Phase 4: Scale Training (Cloud)
With all optimizations:
- **10M CFR iterations**: ~1-2 hours (vs 12 hours before)
- **50M CFR iterations**: ~6-12 hours (vs 3 days before)
- **GPU Cost**: ~$50-100 on Lambda/Vast.ai

### Compute Cost Estimate

| Configuration | Time | Cost |
|---------------|------|------|
| Current (no optimizations) | 3-4 days | ~$500 |
| With OpenMP + Pruning (10x) | 8-12 hours | ~$50 |
| With VR-MCCFR (100x) | 1-2 hours | ~$10 |
| Full training (multi-stack) | 12-24 hours | ~$100-200 |

**Total estimated cost for superhuman bot: $200-500**

### Priority Implementation Order

1. **OpenMP Parallelization** - Easy, 4-8x speedup
2. **Regret Pruning** - Easy, 10x speedup
3. **Linear CFR Warm-up** - Easy, 2x faster convergence
4. **OMPEval Integration** - Medium, 1000x faster eval
5. **VR-MCCFR** - Medium, 100x speedup
6. **Larger Neural Network** - Easy, better accuracy
7. **Knowledge Distillation** - Medium, faster inference

### References

- [VR-MCCFR Paper](https://arxiv.org/abs/1809.03057) - 100x speedup with variance reduction
- [AlphaHoldem](https://ojs.aaai.org/index.php/AAAI/article/view/20394) - Beat Slumbot with 3 days on 1 PC
- [OMPEval](https://github.com/zekyll/OMPEval) - 775M hands/sec evaluator
- [Pluribus Supplement](https://noambrown.github.io/papers/19-Science-Superhuman_Supp.pdf) - Linear CFR, pruning details
- [Deep CFR](https://arxiv.org/abs/1811.00164) - Neural network CFR

## CFR vs End-to-End RL: Architecture Decision

### Why We Use CFR (Not Pure RL like AlphaHoldem)

AlphaHoldem beat Slumbot (+111 bb/100) with only **3 days training on a single PC** using end-to-end reinforcement learning instead of CFR. So why do we use CFR?

### Comparison

| Aspect | CFR (ARES/GTO Wizard/ReBeL) | End-to-End RL (AlphaHoldem) |
|--------|------------------------------|------------------------------|
| **Training time** | Days-weeks | 3 days |
| **Inference speed** | 100-1000ms (search) | 2.9ms (forward pass) |
| **Nash guarantee** | ✅ Provable convergence | ❌ No guarantee |
| **Exploitability** | ✅ Measurable (0.12% pot) | ❌ Can't measure |
| **Bluffing** | Equilibrium-driven (balanced) | Q-value driven (under-bluffs) |
| **Best result** | +19.4 bb/100 (GTO Wizard) | +111 bb/100 (AlphaHoldem) |

### Why CFR Has Theoretical Advantages

From research:
> "CFR computes the Nash-theoretic equilibrium, which is guaranteed to be unexploitable, on average."

> "CFR attempts to bluff more than DQN because CFR is equilibrium-driven and must sometimes bluff to remain unpredictable. DQN only bluffs when the estimated Q-value says it is profitable."

> "If both players' average overall regret is less than ε, then the average strategy is an ε-Nash equilibrium."

### Why AlphaHoldem is Faster

From the AlphaHoldem paper:
> "The prohibitive computation cost of CFR iteration makes it difficult for subsequent researchers to learn the CFR model in HUNL. Under the CFR framework, the primary computation cost comes from the CFR iteration process performed in both the model training and testing stages."

> "AlphaHoldem does not perform any card information abstractions using human domain knowledge. Instead, it encodes the game information into tensors containing the current and historical poker information."

### Which Approach for Which Goal?

| Goal | Best Approach |
|------|---------------|
| Beat Slumbot quickly (any means) | AlphaHoldem-style RL |
| Theoretically unexploitable | CFR |
| Beat humans who exploit patterns | QRE-CFR (what ARES has) |
| Study tool with exploitability metrics | CFR (like GTO Wizard) |
| Maximum win rate in practice | Unclear (AlphaHoldem won more vs Slumbot) |

### Our Decision: Stay with CFR + Optimizations

**Reasons:**
1. **Already implemented** - CFR solver is working
2. **Theoretical guarantees** - Can prove convergence to Nash
3. **GTO Wizard uses CFR+NN** - They're the proven best (+19.4 bb/100)
4. **With VR-MCCFR** - CFR can be 100x faster (approaches RL speed)
5. **Measurable quality** - Can compute exploitability

**Alternative to consider later:**
- Hybrid approach: RL for fast training, CFR for refinement
- Or: Implement AlphaHoldem-style RL as separate experiment

### Key Insight

AlphaHoldem's +111 bb/100 vs Slumbot is impressive, but:
- Slumbot is from 2018 (older bot)
- GTO Wizard's +19.4 bb/100 is against current Slumbot with better methodology
- AlphaHoldem may have higher variance / be more exploitable

The safest path to a **reliable, unexploitable** bot is optimized CFR. The fastest path to **beating Slumbot specifically** might be RL.

## AlphaHoldem-Style RL: Implementation Plan

### Why Switch to RL?

Given our goal (beat humans, beat Slumbot, minimal compute), AlphaHoldem-style RL is the better path:
- **3 days training** on single PC (vs weeks for CFR)
- **+111 bb/100 vs Slumbot** (massive win rate)
- **2.9ms inference** (vs 100-1000ms for CFR search)
- **No abstraction needed** - learns directly from raw game state

### AlphaHoldem Architecture (from paper)

#### State Representation (Key Innovation)
AlphaHoldem encodes game state as **3D tensors** containing:
- Current card information (hole cards + board)
- Historical betting information (all actions in hand)
- Pot/stack information

This tensor representation enables **convolutional networks** to learn spatial patterns in the game state.

#### Network Architecture
- **Pseudo-Siamese architecture**: Two networks (current vs historical versions)
- **ConvNet backbone**: Processes 3D tensor state representation
- **Policy head**: Outputs action probabilities (fold, call, raise amounts)
- **Value head**: Estimates expected value of current state

#### Training Algorithm
- **Trinal-Clip PPO**: Modified PPO with triple clipping for stability
- **K-Best Self-Play**: Plays against K best historical versions for diversity
- **Multi-task loss**: Combines policy gradient + value estimation

### How We Can Improve on AlphaHoldem

#### 1. Better State Representation
- **Transformer attention** instead of CNN for betting history
- **Explicit hand strength features** (equity vs random hands)
- **Position encoding** for temporal action sequences

#### 2. Larger/Better Network
- AlphaHoldem: CNN-based
- Improvement: **Transformer encoder** for betting sequences
- Add **residual connections** for deeper networks

#### 3. More Training
- AlphaHoldem: 3 days, single PC
- Us: Can train longer with cloud GPU
- Unofficial implementation trained **1 billion+ games**, still improving

#### 4. Better Self-Play
- **Population-based training** (multiple diverse agents)
- **Curriculum learning** (start with simpler scenarios)
- **League training** (like AlphaStar)

### Existing Implementations

| Implementation | Notes |
|----------------|-------|
| [AlphaNLHoldem](https://github.com/bupticybee/AlphaNLHoldem) | Unofficial, TensorFlow+Ray, 1B+ games trained |
| [PokerRL](https://github.com/EricSteinberger/PokerRL) | Framework with NFSP, Deep CFR |
| [RLCard](https://github.com/datamllab/rlcard) | OpenAI Gym environment |

### Implementation Plan

#### Phase 1: Environment Setup (1-2 days)
1. Use existing game logic from ARES
2. Create OpenAI Gym-compatible environment
3. Implement proper state encoding (3D tensors)

#### Phase 2: Network Architecture (2-3 days)
1. Implement AlphaHoldem-style CNN
2. Add policy head (action probabilities)
3. Add value head (state value estimation)

#### Phase 3: Training Pipeline (3-5 days)
1. Implement PPO with clipping
2. Set up self-play with historical agents
3. Implement K-Best agent selection

#### Phase 4: Training & Evaluation (1-7 days)
1. Train on local GPU (initial)
2. Scale to cloud if needed
3. Benchmark against Slumbot

### Compute Estimates

| Configuration | Training Time | Estimated Cost |
|---------------|---------------|----------------|
| Local GPU (RTX 3080) | 3-7 days | $0 (electricity) |
| Cloud GPU (A100) | 1-2 days | $50-100 |
| Extended training (1B games) | 1-2 weeks | $200-500 |

### Potential Improvements Over AlphaHoldem

| Improvement | Expected Benefit |
|-------------|------------------|
| Transformer for betting history | Better temporal modeling |
| Explicit equity features | Faster learning |
| More training (weeks vs days) | Higher win rate |
| Population-based training | More robust strategy |
| QRE-style exploration | Better vs exploiters |

### State-of-the-Art Benchmarks (2024)

| Bot | Win Rate vs Slumbot | Method |
|-----|---------------------|--------|
| GTO Wizard AI | +19.4 bb/100 | CFR + Neural (real-time) |
| AlphaHoldem | +111.56 bb/100 | End-to-end RL |
| SpinGPT (LLM) | +13.4 bb/100 | GPT fine-tuning |
| World-class interpretable | Beats Slumbot | Decision trees + CFR |

Note: AlphaHoldem's +111 bb/100 vs GTO Wizard's +19.4 bb/100 may reflect different testing methodology or Slumbot versions.

### Next Steps

1. **Decide**: Full RL rewrite or hybrid approach?
2. **If RL**: Start with AlphaNLHoldem codebase as reference
3. **If hybrid**: Keep CFR for blueprint, add RL for fine-tuning

### References

- [AlphaHoldem Paper](https://cdn.aaai.org/ojs/20394/20394-13-24407-1-2-20220628.pdf) - Original AAAI 2022 paper
- [AlphaNLHoldem](https://github.com/bupticybee/AlphaNLHoldem) - Unofficial implementation
- [GTO Wizard Benchmarks](https://blog.gtowizard.com/gto-wizard-ai-benchmarks/) - +19.4 bb/100 vs Slumbot
- [PokerRL Framework](https://github.com/EricSteinberger/PokerRL) - Multi-agent RL framework

## Quick Start

```bash
# Build C++ solver with neural network support
cd cpp_solver && ./build.sh

# Build Python bindings (requires PyTorch)
./build_python.sh

# Test against Slumbot (requires internet)
python benchmark_slumbot.py --hands 100 --agent search --iterations 200

# Evaluate against simple opponents
python evaluate_agent.py --strategy blueprints/cpp_1M/strategy_1M.bin --hands 10000
```

## Project Structure

```
cpp_solver/           # C++ CFR solver (main codebase)
  src/
    core/             # Hand evaluator, cards
    game/             # HoldemState, actions
    cfr/              # DCFR+ solver
    search/           # Real-time search
    belief/           # PBS encoding
    neural/           # Neural evaluator (TorchScript)
  include/            # Headers
  build/              # Build output + Python module

src/                  # Python utilities
  neural/             # Neural network training

blueprints/           # Trained strategies
  multi_stack/        # 10bb, 20bb, 100bb, 200bb
  full_training/      # Latest training run

models/               # Neural network checkpoints
  value_net.torchscript    # Deployed value network
  value_network.pt         # Training checkpoint

benchmark_slumbot.py  # Slumbot benchmark
train_value_network.py    # Value network training
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    benchmark_slumbot.py                      │
│                  (Slumbot API integration)                   │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                  RealtimeSearchAgent                         │
│              (Python wrapper for C++ search)                 │
│  - Loads neural model: models/value_net.torchscript          │
│  - Configures search: iterations, QRE tau, time limit        │
└─────────────────────────────────────────────────────────────┘
                              │ pybind11
┌─────────────────────────────────────────────────────────────┐
│                    C++ RealtimeSearch                        │
│  - Depth-limited CFR from any game state                     │
│  - QRE regret matching (τ configurable)                      │
│  - Neural leaf evaluation (if model loaded)                  │
│  - Fallback: hand strength heuristic                         │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                     C++ HoldemState                          │
│  - Game tree traversal                                       │
│  - Legal action generation                                   │
│  - create_from_position() for arbitrary states               │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   C++ NeuralEvaluator                        │
│  - Loads TorchScript model                                   │
│  - PBS encoding → value prediction                           │
│  - Batch evaluation support                                  │
└─────────────────────────────────────────────────────────────┘
```

## Technical Details

### CFR Algorithm
- **Variant**: DCFR+ (Discounted CFR with CFR+ regret clamping)
- **Discounting**: α=1.5, β=0 (CFR+ behavior), γ=1.0
- **Traversal**: External sampling MCCFR
- **Regret Matching**: Standard or QRE softmax

### PBS Encoding (3040 dimensions)
- Board cards: 52 × 5 = 260 dims (one-hot)
- Pot/stacks: 4 dims (normalized)
- Betting history: 100 dims
- Hand beliefs: 1326 × 2 = 2652 dims
- Board texture: 20 dims
- Position/street: 4 dims

### Performance Benchmarks
- CFR training: 14,000 iterations/sec (8 threads)
- Hand evaluation: 711,000 hands/sec
- Search: 50-100 iterations/sec (with neural eval)
- Info sets (20bb): 48.7M after 1M iterations

## Key Insights from State-of-the-Art

### Why No Card Abstraction Needed

GTO Wizard explains the modern approach:

> "AI-solvers do not actually calculate the strategy on future streets. Rather, they predict the EV of future streets using neural nets and lookahead functions."

> "Rivers in GTO Wizard are solved exactly, without any neural networks."

This means:
1. **Preflop/Flop/Turn**: Use neural network to estimate future value
2. **River**: Solve exactly (small enough tree)
3. **No bucket artifacts**: Continuous hand representation via neural net

The neural network learns to "intuit" the value of any hand, replacing discrete abstraction buckets with a continuous learned function.

### Why QRE Beats Nash

GTO Wizard switched from Nash to QRE in 2025, achieving **25% lower exploitability**. Here's why:

**The Ghostlines Problem**: Traditional Nash solvers only optimize spots they expect to happen. Once a line has 0% frequency, the solver stops improving responses to it. But real opponents make mistakes and take these "impossible" lines.

**QRE Solution**: Instead of assuming perfect play, QRE assumes players occasionally make mistakes with probability inversely proportional to how costly the mistake is.

```
Nash:  σ(a) = max(R(a), 0) / Σ max(R, 0)     # Hard max, ignores negative regret actions
QRE:   σ(a) = exp(R(a) / τ) / Σ exp(R / τ)   # Soft max, small prob to all actions
```

**ARES-HU implements exactly this formula** (see `regret_minimizer.hpp:68-91`):
- `τ → 0`: Approaches Nash (hard max)
- `τ = 1.0`: Balanced QRE (recommended)
- `τ → ∞`: Uniform random (pure exploration)

**Benefits**:
- Optimal responses to "ghostlines" (opponent mistakes)
- More robust strategies against imperfect opponents
- No node-locking needed for exploitation

## References

- [GTO Wizard AI Explained](https://blog.gtowizard.com/gto-wizard-ai-explained/) - How GTO Wizard uses neural networks
- [Introducing QRE](https://blog.gtowizard.com/introducing-quantal-response-equilibrium-the-next-evolution-of-gto/) - Why QRE beats Nash
- [GTO Wizard AI Benchmarks](https://blog.gtowizard.com/gto-wizard-ai-benchmarks/) - +19.4 bb/100 vs Slumbot
- [ReBeL: A general game-playing AI](https://ai.meta.com/blog/rebel-a-general-game-playing-ai-bot-that-excels-at-poker-and-more/) - Meta's approach
- [ReBeL Paper](https://arxiv.org/abs/2007.13544) - Brown et al., 2020
- [Libratus Paper](https://science.sciencemag.org/content/359/6374/418) - Brown & Sandholm, 2017
- [DCFR Paper](https://arxiv.org/abs/1809.04040) - Discounted Regret Minimization
