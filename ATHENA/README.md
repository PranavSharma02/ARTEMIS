# ATHENA: Adaptive Trading with Hierarchical Ensemble and Neural Attention

ATHENA is a modified version of [ARTEMIS](../README.md) that introduces **6 categories of improvements** across every layer of the trading system while maintaining the same overall pipeline structure.

## Modifications vs ARTEMIS

| # | Layer | ARTEMIS (Original) | ATHENA (Modified) | Reference |
|---|-------|--------------------|--------------------|-----------|
| 1a | Network | Ultra-1: straight GRU pipeline | + Residual connections (skip connections) | He et al., 2016 |
| 1b | Network | Return: no positional encoding | + Sinusoidal positional encoding | Shaw et al., 2018 |
| 2 | Features | 98+ hand-crafted indicators | + Wavelet decomposition (~15 new features) | Ramsey & Lampart, 1998 |
| 3 | Ensemble | `np.average(preds, weights)` | Cross-attention fusion (learned) | Vaswani et al., 2017 |
| 4 | Signal | Single-horizon output | Dual-horizon + learned gate | Dauphin et al., 2017 |
| 5 | RL | TD3 (deterministic) | SAC (stochastic + entropy) | Haarnoja et al., 2018 |
| 6 | Blend | Fixed 70/30 hardcoded | Adaptive confidence gate (learned) | Novel |

All 5 models also receive minor hyperparameter adjustments (wider hidden dimensions, lower dropout).

## Quick Start

```bash
# From the ARTEMIS root directory:

# Supervised mode (no RL)
python ATHENA/main.py GOOGL --mode supervised --epochs 100

# Hybrid mode (supervised + SAC RL)
python ATHENA/main.py GOOGL --mode hybrid --episodes 50

# With custom parameters
python ATHENA/main.py AAPL --mode hybrid --episodes 100 --lr 5e-5 --epochs 200
```

## Architecture

### 5 Neural Networks (same families, improved internals)

1. **ATHENA-Ultra-1** – GRU + Attention + **Residual Connections** (hidden=200)
2. **ATHENA-Momentum** – LSTM + TCN (hidden=180)
3. **ATHENA-Return** – Transformer + **Positional Encoding** (hidden=168)
4. **ATHENA-Trend** – CNN + BiLSTM (hidden=152)
5. **ATHENA-HF** – Multi-Scale CNN + GRU (hidden=136)

### Training Pipeline

```
Phase 0: System initialization
Phase 1: Data preparation (98+ indicators + wavelet features)
Phase 2: Model training
  2.1: Supervised ensemble (5 models)
  2.2: Baseline evaluation
  2.3: Cross-attention fusion training
  2.4: Dual-horizon signal generator training
  2.5: SAC RL agent training (hybrid mode only)
  2.6: Adaptive confidence gate training (hybrid mode only)
Phase 3: Signal generation
Phase 4: Performance evaluation
```

## Project Structure

```
ATHENA/
├── main.py              # Entry point
├── athena_core.py       # Core ensemble system (Mods 1-4)
├── athena_rl_system.py  # Hybrid RL system (Mods 5-6)
├── athena_algorithms.py # SAC agent + confidence gate
├── athena_networks.py   # 5 modified architectures
├── athena_fusion.py     # Cross-attention fusion module
├── athena_features.py   # Wavelet feature engineering
├── athena_signals.py    # Dual-horizon signal generator
├── athena_utils.py      # Loss functions, configs, helpers
└── README.md            # This file
```

## Dependencies

Same as ARTEMIS, plus:
- `PyWavelets>=1.4.0` (optional; fallback features used if unavailable)

## CLI Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `stock` | required | Stock symbol (e.g., GOOGL, AAPL) |
| `--mode` | hybrid | `supervised` or `hybrid` |
| `--epochs` | 250 | Supervised training epochs |
| `--episodes` | 500 | RL training episodes |
| `--models` | 5 | Number of ensemble models |
| `--lr` | 1e-4 | RL learning rate |
| `--balance` | 100000 | Initial portfolio balance |
