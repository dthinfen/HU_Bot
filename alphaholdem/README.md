# AlphaHoldem-Style RL for ARES-HU

End-to-end reinforcement learning for heads-up no-limit hold'em, inspired by the AlphaHoldem paper.

## Current Status (Nov 2024)

**Working:**
- Training loop with K-Best self-play pool
- Verified: Agent learns to beat random (~+700 bb/100), then transitions to self-play
- Self-play converges toward balanced play (~0 bb/100 vs itself)
- Code is CUDA-ready (auto-detects GPU)

**Next Steps:**
1. Run on cloud GPU for longer training (100M-1B steps)
2. Add Weights & Biases logging for remote monitoring
3. Test against Slumbot benchmark

**To Resume Training:**
```bash
# Local (CPU, ~380 steps/sec)
python -m alphaholdem.src.train --timesteps 1000000

# Cloud GPU (10-50x faster)
# Just run same command - auto-detects CUDA
python -m alphaholdem.src.train --timesteps 100000000
```

## Architecture

Based on AlphaHoldem (AAAI 2022) - verified against paper:
- **3D Tensor State Encoding**: Cards + betting history as 38x4x13 tensor
- **Model Size**: 8.1M params (paper: 8.6M = 1.8M Conv + 6.8M FC)
- **PPO with Trinal-Clip**: δ1=3.0 clip ratio (NOT standard PPO's 0.2)
- **Discount Factor**: γ=0.999 with GAE λ=0.95
- **K-Best Self-Play with ELO**: Pool of K=5 best agents selected by ELO scores

## Training Stages

1. **Warmup** (0-50K hands): Train vs random opponent, learn basic poker
2. **Self-Play** (50K+ hands): Evaluate for pool every 50 updates via round-robin
3. **Evolution**: Pool keeps K=5 best agents by ELO, strategy converges over time

**Expected Timeline:**
| Steps | Time (GPU) | Cost | Expected Strength |
|-------|------------|------|-------------------|
| 1M | 5-10 min | ~$0.10 | Beats random |
| 100M | 1-2 hours | ~$2-5 | Beats weak humans |
| 1B+ | 10-20 hours | ~$20-50 | Competitive with Slumbot |

## Files

```
alphaholdem/
├── src/
│   ├── env.py           # Poker environment (14 discrete actions)
│   ├── encoder.py       # State encoding (38x4x13 tensor)
│   ├── network.py       # Actor-Critic CNN (~8.1M params)
│   ├── ppo.py           # PPO with Trinal-Clip (δ1=3)
│   └── train.py         # Self-play training loop
├── checkpoints/         # Saved models & agent pool
└── logs/                # TensorBoard logs
```

## Key Config (verified against paper)

```python
# Network (paper: 8.6M params = 1.8M Conv + 6.8M FC)
fc_hidden_dim: int = 1024           # FC layer width
fc_num_layers: int = 4              # FC layer depth (~6.8M FC params)

# PPO (paper values)
gamma: float = 0.999                # Discount factor (paper: 0.999)
clip_ratio: float = 3.0             # Trinal-Clip δ1=3 (NOT 0.2!)
gae_lambda: float = 0.95            # GAE parameter (paper: λ=0.95)
batch_size: int = 2048              # Per GPU (paper: 2048 x 8 GPUs)
learning_rate: float = 3e-4         # Adam LR (paper: 0.0003)

# ELO-based K-Best selection
k_best: int = 5                     # Keep 5 best agents by ELO
elo_games_per_matchup: int = 100    # Games per matchup for ELO
initial_elo: float = 1500.0         # Starting ELO (standard)
```

**Pool Admission (AlphaHoldem ELO method):**
- New agent plays round-robin vs ALL agents in pool
- ELO score calculated from all matchup results
- Agent added to pool, K-best by ELO are kept
- Lowest ELO agent evicted when pool full (K=5)

## Action Space (14 discrete actions)

| Action | Description |
|--------|-------------|
| 0 | Fold |
| 1 | Check/Call |
| 2-11 | Raise (0.33x to 3x pot) |
| 12 | Min-raise |
| 13 | All-in |

## Cloud Deployment

**Recommended: RunPod or Vast.ai**
```bash
# SSH into cloud GPU instance
git clone <repo> && cd HU_Bot
pip install torch numpy

# Run training (disconnects safely with nohup)
nohup python -m alphaholdem.src.train --timesteps 100000000 > train.log 2>&1 &

# Monitor from anywhere
tail -f train.log
```

## Results Target

| Bot | Win Rate vs Slumbot |
|-----|---------------------|
| AlphaHoldem (paper) | +111.56 bb/100 |
| GTO Wizard AI | +19.4 bb/100 |
| Our target | +20-50 bb/100 |

## References

- **AlphaHoldem** (AAAI 2022): [Paper](https://cdn.aaai.org/ojs/20394/20394-13-24407-1-2-20220628.pdf)
- **PPO** (OpenAI 2017): [Paper](https://arxiv.org/abs/1707.06347)
- **Slumbot**: [slumbot.com](https://www.slumbot.com/) - Standard benchmark
