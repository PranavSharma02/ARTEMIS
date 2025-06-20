#!/usr/bin/env python3

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

from artemis_core import ARTEMISEnsembleSystem
from artemis_algorithms import (
    ARTEMISMultiAgentSystem,
    ARTEMISCoordinatorAgent,
    ARTEMISPositionSizerAgent,
    ARTEMISRegimeAgent,
    ARTEMISTD3Agent
)

class HybridRLARTEMISSystem(ARTEMISEnsembleSystem):
    """
    Hybrid RL-ARTEMIS System v2.0
    
    Advanced hybrid system combining supervised learning with reinforcement learning
    for enhanced trading performance while preserving baseline performance.
    
    Key Features:
    - Conservative RL integration with performance preservation
    - Multi-agent RL coordination system
    - Advanced TD3 with risk management
    - Hybrid signal generation with fallback mechanism
    - Real-time performance monitoring
    """
    
    def __init__(self, sequence_length=60, initial_balance=100000, device=None,
                 return_boost_factor=1.25, ultra_aggressive_mode=True, rl_learning_rate=1e-4):
        super().__init__(sequence_length, initial_balance, device, return_boost_factor, ultra_aggressive_mode)
        
        self.framework_name = "Hybrid-RL-ARTEMIS"
        self.version = "2.0"
        self.rl_learning_rate = rl_learning_rate
        
        # RL components
        self.multi_agent_system = None
        self.rl_agents_initialized = False
        
        # Performance tracking
        self.supervised_baseline_performance = None
        self.rl_performance_history = []
        self.performance_violations = 0
        self.max_violations = 3
        
        # Conservative RL parameters
        self.rl_exploration_factor = 0.3
        self.performance_threshold = 0.95
        self.fallback_mode = False
        
        print(f"🧠 {self.framework_name} v{self.version} initialized")
        print(f"   Hybrid RL-Supervised Learning Architecture")
        print(f"   Advanced TD3 RL agent integration")
        print(f"   Conservative RL integration for performance preservation")
        
        # Enhanced GPU info
        if self.device.type == 'cuda':
            gpu_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory
            capability = torch.cuda.get_device_capability(0)
            
            print(f"   🚀 Device: {self.device}")
            print(f"   💎 GPU: {gpu_name}")
            print(f"   🔋 VRAM: {total_memory / 1e9:.1f} GB")
            print(f"   ⚡ CUDA Capability: {capability}")

    def _ensure_tensor_device(self, tensor):
        """Ensure tensor is on the correct device"""
        if isinstance(tensor, torch.Tensor):
            return tensor.to(self.device, non_blocking=True)
        elif isinstance(tensor, (list, tuple)):
            return [self._ensure_tensor_device(t) for t in tensor]
        elif isinstance(tensor, dict):
            return {k: self._ensure_tensor_device(v) for k, v in tensor.items()}
        else:
            return tensor

    def train_hybrid_system(self, n_supervised_models=5, supervised_epochs=250, 
                          rl_episodes=500, evaluation_frequency=50):
        """
        Train the complete Hybrid RL-ARTEMIS system
        
        Training Pipeline:
        1. Train supervised ARTEMIS ensemble (baseline)
        2. Initialize and train RL agents
        3. Joint optimization with performance monitoring
        
        Args:
            n_supervised_models (int): Number of supervised models
            supervised_epochs (int): Epochs for supervised training
            rl_episodes (int): RL training episodes
            evaluation_frequency (int): Frequency of performance evaluation
        """
        print("   📚 Phase 2.1: Training supervised ensemble...")
        self.train_artemis_ultra_aggressive_ensemble(n_models=n_supervised_models, epochs=supervised_epochs)
        
        print("   📊 Phase 2.2: Baseline evaluation...")
        baseline_signals = self.generate_artemis_ultra_aggressive_signals()
        baseline_results = self.backtest_signals(baseline_signals)
        
        self.supervised_baseline_performance = {
            'annual_return': baseline_results['annual_return'],
            'sharpe_ratio': baseline_results['sharpe_ratio'],
            'max_drawdown': baseline_results['max_drawdown']
        }
        
        print("   🤖 Phase 2.3: Initializing RL agents...")
        self._initialize_rl_agents()
        
        print(f"   🎯 Phase 2.4: RL training ({rl_episodes} episodes)...")
        self._train_rl_agents_conservatively(rl_episodes, evaluation_frequency)
        
        print("   ✅ Model training complete!")

    def _initialize_rl_agents(self):
        """Initialize the multi-agent RL system"""
        # Calculate state and feature dimensions
        state_size = self.X.shape[2] if hasattr(self, 'X') and self.X is not None else 86
        n_models = len(self.models) if self.models else 5
        market_features = min(20, state_size)
        
        # Initialize multi-agent system
        self.multi_agent_system = ARTEMISMultiAgentSystem(
            state_size=state_size,
            n_models=n_models,
            market_features=market_features
        )
        
        self.rl_agents_initialized = True

    def _train_rl_agents_conservatively(self, episodes, evaluation_frequency):
        """Train RL agents with conservative approach and performance monitoring"""
        if not self.rl_agents_initialized:
            raise ValueError("RL agents must be initialized first")
        
        for episode in range(episodes):
            # Generate episode data
            episode_reward = self._run_rl_episode()
            
            # Evaluate performance periodically (silently)
            if (episode + 1) % evaluation_frequency == 0:
                performance_ratio = self._evaluate_hybrid_performance()
                
                # Performance preservation check
                if performance_ratio < self.supervised_baseline_performance['sharpe_ratio'] * self.performance_threshold:
                    self.performance_violations += 1
                    
                    if self.performance_violations >= self.max_violations:
                        self.fallback_mode = True
                        break
                else:
                    # Reset violations if performance is good
                    if self.performance_violations > 0:
                        self.performance_violations = max(0, self.performance_violations - 1)

    def _run_rl_episode(self):
        """Run a single RL training episode with proper tensor handling"""
        if not hasattr(self, 'X') or self.X is None:
            return 0.0
        
        total_reward = 0.0
        sequence_length = min(100, len(self.X))
        
        for i in range(sequence_length - 1):
            # Current state - use last timestep, flatten to 1D
            state = self.X[i, -1, :].cpu().numpy()  # Shape: (86,)
            
            # Get model predictions
            model_predictions = []
            for model in self.models:
                model.eval()
                with torch.no_grad():
                    pred = model(self.X[i:i+1])
                    model_predictions.append(pred.cpu().item())
            
            # Market features - last 20 features
            market_features = state[-20:] if len(state) >= 20 else np.pad(state, (0, 20-len(state)))
            
            # RL action
            action = self.multi_agent_system.act(state, model_predictions, market_features)
            
            # Simulate reward (based on next period return)
            next_return = self.y[i+1].cpu().item() if i+1 < len(self.y) else 0.0
            reward = action * next_return * 100  # Scale reward
            
            # Update agents
            next_state = self.X[i+1, -1, :].cpu().numpy() if i+1 < len(self.X) else state
            done = (i == sequence_length - 2)
            
            self.multi_agent_system.update(
                state, action, reward, next_state, done, 
                model_predictions, market_features
            )
            
            total_reward += reward
        
        return total_reward

    def _evaluate_hybrid_performance(self):
        """Evaluate current hybrid system performance"""
        try:
            # Generate hybrid signals
            hybrid_signals = self.generate_hybrid_rl_signals()
            
            # Quick performance evaluation
            if len(hybrid_signals) > 100:
                recent_signals = hybrid_signals[-100:]
                recent_returns = self.y[-100:].cpu().numpy() if hasattr(self, 'y') else np.random.randn(100) * 0.01
                
                portfolio_returns = recent_signals * recent_returns
                sharpe_ratio = np.mean(portfolio_returns) / (np.std(portfolio_returns) + 1e-8) * np.sqrt(252)
                
                self.rl_performance_history.append(sharpe_ratio)
                return sharpe_ratio
            
        except Exception as e:
            print(f"   ⚠️  Performance evaluation error: {str(e)}")
        
        return 0.0

    def generate_hybrid_rl_signals(self, start_idx=None, end_idx=None):
        """
        Generate hybrid RL-ARTEMIS trading signals with comprehensive error handling
        
        Combines supervised ARTEMIS signals with RL enhancement while maintaining
        performance preservation through fallback mechanism.
        
        Args:
            start_idx (int): Start index for signal generation
            end_idx (int): End index for signal generation
            
        Returns:
            np.ndarray: Hybrid RL-enhanced trading signals
        """
        # Fallback to supervised if RL performance is poor
        if self.fallback_mode or not self.rl_agents_initialized:
            return self.generate_artemis_ultra_aggressive_signals(start_idx, end_idx)
        
        # Generate base supervised signals
        base_signals = self.generate_artemis_ultra_aggressive_signals(start_idx, end_idx)
        
        if start_idx is None:
            start_idx = self.sequence_length
        if end_idx is None:
            end_idx = len(self.features_df)
        
        # Use the same sequences as the base signal generation
        X_sequences = self.X  # Use pre-computed sequences
        
        # RL enhancement
        enhanced_signals = np.zeros(len(base_signals))
        
        for i in range(len(base_signals)):
            try:
                if i < len(X_sequences):
                    # Use last timestep of sequence, flatten to 1D
                    state = X_sequences[i, -1, :].cpu().numpy()  # Shape: (86,)
                    
                    # Get model predictions
                    model_predictions = []
                    for model in self.models:
                        model.eval()
                        with torch.no_grad():
                            state_tensor = torch.FloatTensor(X_sequences[i:i+1]).to(self.device)
                            pred = model(state_tensor)
                            model_predictions.append(pred.cpu().item())
                    
                    # Market features - last 20 features
                    market_features = state[-20:] if len(state) >= 20 else np.pad(state, (0, 20-len(state)))
                    
                    # RL-enhanced action
                    rl_action = self.multi_agent_system.act(state, model_predictions, market_features)
                    
                    # Conservative combination (favor supervised baseline)
                    conservative_weight = 0.7  # 70% supervised, 30% RL
                    enhanced_signals[i] = (
                        conservative_weight * base_signals[i] + 
                        (1 - conservative_weight) * rl_action
                    )
                else:
                    enhanced_signals[i] = base_signals[i]
                    
            except Exception as e:
                # Fallback to supervised signal if RL fails
                enhanced_signals[i] = base_signals[i]
        
        # Apply risk management bounds
        enhanced_signals = np.clip(enhanced_signals, -0.8, 0.8)
        

        
        return enhanced_signals

    def backtest_signals(self, signals):
        """Backtest ARTEMIS trading signals"""
        if hasattr(self, 'features_df') and 'Returns' in self.features_df.columns:
            returns = self.features_df['Returns'].dropna()
            
            # Align signals with returns
            min_len = min(len(signals), len(returns))
            signals = signals[-min_len:]
            returns = returns.iloc[-min_len:]
            
            # Calculate portfolio returns
            portfolio_returns = signals * returns.values
            
            # Performance metrics
            total_return = (1 + portfolio_returns).prod() - 1
            annual_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
            volatility = portfolio_returns.std() * np.sqrt(252)
            sharpe_ratio = annual_return / volatility if volatility > 0 else 0
            
            # Max drawdown
            import pandas as pd
            cumulative = pd.Series(1 + portfolio_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
            
            return {
                'total_return': total_return,
                'annual_return': annual_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'volatility': volatility
            }
        else:
            # Default performance for missing data
            return {
                'total_return': 0.0,
                'annual_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'volatility': 0.0
            }

    def get_performance_comparison(self):
        """Get performance comparison between supervised and hybrid systems"""
        if not self.supervised_baseline_performance:
            return None
        
        return {
            'supervised_baseline': self.supervised_baseline_performance,
            'rl_violations': self.performance_violations,
            'rl_performance_history': self.rl_performance_history[-10:],  # Last 10 evaluations
            'fallback_mode': self.fallback_mode
        }

# Legacy compatibility
HybridRLMAREASystem = HybridRLARTEMISSystem 