# 🚀 ARTEMIS - Advanced Reinforcement Learning Trading System

**Adaptive Reinforcement Trading Ensemble with Multi-Intelligence Systems**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.0+-green.svg)](https://developer.nvidia.com/cuda-downloads)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 Overview

ARTEMIS is a state-of-the-art **hybrid reinforcement learning trading system** that combines **5 specialized neural network architectures** with **advanced TD3 RL algorithms** and **multi-agent coordination**. The system achieves exceptional risk-adjusted performance through conservative RL integration and ultra-aggressive return optimization.

## 🏆 Performance Highlights

| Metric | Value |
|--------|-------|
| **Annual Return** | 30.07% |
| **Total Return** | 353.41% |
| **Sharpe Ratio** | 2.786 |
| **Max Drawdown** | -7.10% |
| **Volatility** | 10.79% |
| **RL Enhancement** | +0.177 Sharpe |

*Results achieved on AAPL (2012-2017) with conservative RL training (100 episodes)*

## 🧠 Multi-Intelligence Architecture

The system employs **5 specialized ARTEMIS neural networks**, each optimized for different market patterns:

### 1. **ARTEMIS-Ultra-1** - Advanced GRU + Multi-Head Attention
- **Architecture:** 3-layer hierarchical GRU + 16-head attention mechanism
- **Specialization:** Ultra-aggressive maximum return capture
- **Hidden Size:** 192
- **Innovation:** Attention-weighted sequence processing

### 2. **ARTEMIS-Momentum** - LSTM + Temporal Convolution Network
- **Architecture:** LSTM + TCN layers for momentum pattern detection
- **Specialization:** Momentum persistence and trend continuation
- **Hidden Size:** 176
- **Innovation:** Temporal convolution for momentum signals

### 3. **ARTEMIS-Return** - Transformer Architecture
- **Architecture:** Multi-layer Transformer with positional encoding
- **Specialization:** Pure return optimization and pattern recognition
- **Hidden Size:** 160
- **Innovation:** Self-attention for complex pattern learning

### 4. **ARTEMIS-Trend** - CNN + Bidirectional LSTM
- **Architecture:** Convolutional layers + BiLSTM for trend analysis
- **Specialization:** Trend identification and directional prediction
- **Hidden Size:** 144
- **Innovation:** Bidirectional trend pattern recognition

### 5. **ARTEMIS-HF** - Multi-Scale CNN + GRU
- **Architecture:** Multi-scale 1D CNN + GRU for high-frequency signals
- **Specialization:** Short-term pattern detection and micro-trends
- **Hidden Size:** 128
- **Innovation:** Multi-scale feature extraction

## 🤖 Advanced Reinforcement Learning

### TD3 Multi-Agent System
- **Coordinator Agent:** Ensemble model coordination
- **Position Sizer Agent:** Dynamic risk-adjusted position sizing
- **Regime Agent:** Enhanced market regime detection
- **Advanced RL Agent:** TD3 with risk management

### Conservative RL Integration
- **Performance Preservation:** Maintains strong supervised baseline
- **Risk Management:** Conservative training approach
- **Adaptive Learning:** Dynamic learning rate adjustment
- **Fallback Mechanism:** Automatic fallback to supervised mode

## 🔄 Intelligent Regime Detection

ARTEMIS automatically detects and adapts to 5 market regimes:

- **Bull Market:** Growth-focused model weighting
- **Bear Market:** Defensive positioning and risk control
- **Sideways Market:** Balanced strategy allocation
- **High Volatility:** Enhanced risk management protocols
- **Strong Momentum:** Momentum-based model emphasis

## 📊 Enhanced Features

### 98+ Advanced Technical Indicators
- **Multi-timeframe momentum indicators** (6 Fibonacci periods)
- **Volatility-adjusted returns** (3 timeframes)
- **Price acceleration and jerk indicators**
- **Support/resistance breakthrough signals**
- **Market regime transition indicators**
- **Volume-price relationship metrics**

### Dynamic Position Sizing
- **Neural network-based** position optimization
- **Risk-adjusted multipliers** with volatility consideration
- **Ultra-aggressive boost factors** for return enhancement
- **Conservative risk management** integration

### GPU Acceleration
- **CUDA optimization** for RTX 4070 and modern GPUs
- **Automatic Mixed Precision** for faster training
- **Batch processing** with optimized memory usage
- **Multi-threaded** data loading and preprocessing

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/PranavSharma02/ARTEMIS.git
cd ARTEMIS

# Install dependencies
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- PyTorch 1.9+
- CUDA 11.0+ (recommended for GPU acceleration)
- 8GB+ RAM (16GB recommended)
- GPU with 6GB+ VRAM (optional but recommended)

## 🚀 Quick Start

### Basic Hybrid RL Training
```bash
python main.py AAPL --mode hybrid --episodes 50
```

### Extended RL Training
```bash
python main.py AAPL --mode hybrid --episodes 100
```

### Supervised Baseline Only
```bash
python main.py AAPL --mode supervised
```

### Available Parameters
- `--mode`: `hybrid`, `supervised`, or `comparison`
- `--episodes`: Number of RL training episodes (default: 500)
- `--balance`: Initial investment amount (default: $100,000)
- `--lr`: RL learning rate (default: 1e-4)
- `--models`: Number of ensemble models (default: 5)
- `--epochs`: Supervised training epochs (default: 250)

## 📁 Project Structure

```
ARTEMIS/
├── main.py                     # Main entry point with clean phases
├── artemis_core.py             # Core ARTEMIS ensemble system
├── artemis_rl_system.py        # Hybrid RL-ARTEMIS implementation
├── artemis_algorithms.py       # Advanced RL algorithms (TD3, Multi-Agent)
├── artemis_networks.py         # 5 specialized neural architectures
├── artemis_utils.py            # Loss functions and utility models
├── pytorch_trading_system.py   # Base trading system framework
├── tradingPerformance.py       # Performance evaluation and analysis
├── requirements.txt            # Python dependencies
├── README.md                   # This documentation
└── Data/                       # Stock market datasets
    ├── AAPL_2012-1-1_2018-1-1.csv
    ├── GOOGL_2012-1-1_2018-1-1.csv
    └── ... (61 stock datasets)
```

## 🎯 Key Innovations

### 1. **Hybrid RL-Supervised Learning**
Conservative reinforcement learning integration that preserves supervised performance while enhancing risk-adjusted returns.

### 2. **Multi-Agent Coordination**
Advanced TD3 RL with multiple specialized agents for different trading aspects (coordination, positioning, regime detection).

### 3. **Architecture Diversity**
5 completely different neural network architectures optimized for different market patterns and conditions.

### 4. **Performance Preservation**
Conservative training approach with automatic fallback mechanisms to maintain baseline performance.

### 5. **GPU Optimization**
Full CUDA acceleration with automatic mixed precision and optimized memory usage for modern GPUs.

## 🔬 Training Pipeline

ARTEMIS uses a comprehensive 5-phase training pipeline:

```
🔧 Phase 0: System initialization...
📊 Phase 1: Data preparation...
🧠 Phase 2: Model training...
   📚 Phase 2.1: Training supervised ensemble...
   📊 Phase 2.2: Baseline evaluation...
   🤖 Phase 2.3: Initializing RL agents...
   🎯 Phase 2.4: RL training (episodes)...
🎯 Phase 3: Signal generation...
📈 Phase 4: Performance evaluation...
```

## 📈 Performance Comparison

| Strategy | Annual Return | Sharpe Ratio | Max Drawdown | RL Enhancement |
|----------|---------------|--------------|--------------|----------------|
| **ARTEMIS (100 episodes)** | **30.07%** | **2.786** | **-7.10%** | **+0.177** |
| **ARTEMIS (50 episodes)** | **28.95%** | **2.435** | **-7.61%** | **+0.150** |
| **ARTEMIS (20 episodes)** | **20.59%** | **2.043** | **-7.10%** | **+0.170** |
| Supervised Baseline | ~30-35% | ~2.2-2.6 | ~7-9% | N/A |
| Buy & Hold (AAPL) | ~14.39% | ~0.85 | ~44% | N/A |

## 🎮 Usage Examples

### Training with Custom Parameters
```bash
# High-performance training
python main.py AAPL --mode hybrid --episodes 200 --lr 5e-5

# Quick evaluation
python main.py GOOGL --mode hybrid --episodes 20

# Comparison study
python main.py MSFT --mode comparison --trials 3
```

### Multiple Stock Analysis
```bash
# Tech stocks
python main.py AAPL --mode hybrid --episodes 50
python main.py GOOGL --mode hybrid --episodes 50
python main.py MSFT --mode hybrid --episodes 50

# Market ETFs
python main.py SPY --mode hybrid --episodes 100
python main.py QQQ --mode hybrid --episodes 100
```

## 📊 Available Datasets

ARTEMIS includes 61 high-quality stock datasets covering:

### US Technology
- **AAPL, GOOGL, MSFT, AMZN, TSLA, FB**

### US Finance
- **JPM, KO, XOM**

### ETFs
- **SPY, QQQ, DIA, EZU, EWJ**

### International
- **HSBC, NOK, SIE.DE, VOW3.DE, PHIA.AS, RDSA.AS**

### Asian Markets
- **BABA, BIDU, 0700.HK, 0939.HK, 2503.T, 6758.T, 7203.T**

## 🏗️ Architecture Details

### Neural Network Specifications
- **Input Features:** 86 enhanced technical indicators
- **Sequence Length:** 60 timesteps
- **Batch Size:** 96 (optimized for GPU memory)
- **Attention Heads:** 8-16 (architecture dependent)
- **Dropout Rates:** 0.12-0.18 (progressive regularization)

### RL Configuration
- **Algorithm:** TD3 (Twin Delayed Deep Deterministic Policy Gradient)
- **Experience Replay:** Enabled with priority sampling
- **Target Networks:** Soft updates with τ=0.005
- **Action Space:** Continuous [-1, 1] (position sizing)
- **State Space:** 86-dimensional feature vectors

## 🔬 Research Applications

ARTEMIS demonstrates cutting-edge techniques in:
- **Hybrid RL-Supervised Learning** for financial markets
- **Multi-agent reinforcement learning** coordination
- **Conservative RL integration** with performance preservation
- **GPU-accelerated deep learning** for trading systems
- **Risk-adjusted return optimization** through RL

## 🤝 Contributing

We welcome contributions! Please feel free to:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request
4. Report issues or suggestions

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

ARTEMIS is a research project demonstrating advanced RL and deep learning techniques for algorithmic trading. Past performance does not guarantee future results. This system is for educational and research purposes only. Use at your own risk and consult with financial advisors before making investment decisions.

## 🏆 Acknowledgments

- **Framework:** ARTEMIS (Adaptive Reinforcement Trading Ensemble with Multi-Intelligence Systems)
- **Innovation:** Hybrid RL-Supervised Learning Integration
- **Performance:** Conservative RL with Risk-Adjusted Enhancement
- **Technology:** GPU-Accelerated Multi-Agent Deep Learning

---

**🚀 Experience the future of algorithmic trading with ARTEMIS!** 

*Advanced Reinforcement Learning • Multi-Intelligence Systems • Ultra-Performance Trading* 